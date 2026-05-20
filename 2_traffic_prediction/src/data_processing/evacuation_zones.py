from pathlib import Path


def _find_project_root(marker="data"):
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise RuntimeError(
        f"Cannot find project root (looked for '{marker}/' from {here})"
    )


PROJECT_ROOT = _find_project_root("data")
RAW  = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"

ROAD_GPKG    = PROC / "road_edges.gpkg"
PEMS_CSV     = PROC / "pems_congestion.csv"
SVI_CSV      = RAW  / "California_2020_SVI.csv"
EVAC_SHP     = RAW  / "evacuation_zones" / "evacuation_zones.shp"
OUT_FEATURES = PROC / "edge_features.csv"

# =============================================================================
# CONFIG
# =============================================================================
# CDC SVI standard column names -- adjust if your export differs
SVI_TRACT_COL = "FIPS"
SVI_POP_COL   = "E_TOTPOP"
SVI_SCORE_COL = "RPL_THEMES"   # 0-1, higher = more vulnerable

# BPR fallback label for edges without a PeMS station
# t_congested = t_freeflow * (1 + BPR_ALPHA * (V/C)^BPR_BETA)
BPR_ALPHA     = 0.15
BPR_BETA      = 4
TARGET_MED_VC = 0.70           # scale demand so median V/C = 0.70

EVAC_ZONE_BOOST = 1.5          # inside zone: demand * (1 + 1.5) = 2.5x

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

PROC.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Evacuation Zones -- Feature Engineering + Final Label")
print("=" * 60)
print(f"  Project root : {PROJECT_ROOT}")

# =============================================================================
# VALIDATE INPUTS
# =============================================================================
if not ROAD_GPKG.exists():
    print(f"\n[ERROR] Not found: {ROAD_GPKG}")
    print("        Run road_network.py first.")
    sys.exit(1)

if not SVI_CSV.exists():
    print(f"\n[ERROR] Not found: {SVI_CSV}")
    print("        Place California_SVI_2022.csv in data/raw/")
    sys.exit(1)

evac_available = EVAC_SHP.exists()
pems_available = PEMS_CSV.exists()

if not evac_available:
    print(f"\n[warn] Evacuation zones shapefile not found at:")
    print(f"       {EVAC_SHP}")
    print("       Download: https://data.lacounty.gov -> 'Evacuation Zones'")
    print("       Continuing -- in_evac_zone = 0 for all edges\n")

if not pems_available:
    print("[warn] pems_congestion.csv not found -- all edges will use BPR label")
    print("       Run caltrans_pems.py first for real PeMS labels\n")

# =============================================================================
# STEP 1  Load road edges
# =============================================================================
print("\n[1/5] Loading road edges...")

edges = gpd.read_file(ROAD_GPKG).to_crs(epsg=3310)
edges["edge_id"] = edges["edge_id"].astype(str)

print(f"      {len(edges):,} edges")
print(f"      Columns: {list(edges.columns)}")

# =============================================================================
# STEP 2  SVI -- population + social vulnerability
# =============================================================================
print("\n[2/5] SVI (population + vulnerability)...")

svi = pd.read_csv(SVI_CSV, dtype={SVI_TRACT_COL: str})
svi = svi[[SVI_TRACT_COL, SVI_POP_COL, SVI_SCORE_COL]].dropna()
svi.columns = ["FIPS", "population", "svi_score"]
svi["svi_score"]  = pd.to_numeric(svi["svi_score"],  errors="coerce").clip(0, 1)
svi["population"] = pd.to_numeric(svi["population"], errors="coerce")

print(f"      {len(svi):,} census tracts")
print(f"      Population range : {svi['population'].min():,.0f} - "
      f"{svi['population'].max():,.0f}")
print(f"      SVI score range  : {svi['svi_score'].min():.3f} - "
      f"{svi['svi_score'].max():.3f}")

# Join via FIPS column in road edges (TIGER-derived GeoJSON has GEOID / FIPS)
tract_col = next(
    (c for c in ["GEOID", "FIPS", "tract", "TRACTCE"] if c in edges.columns),
    None,
)
if tract_col:
    edges["FIPS"] = edges[tract_col].astype(str).str.zfill(11)
    edges = edges.merge(svi, on="FIPS", how="left")
    matched = edges["population"].notna().sum()
    print(f"      Joined via '{tract_col}': {matched:,}/{len(edges):,} edges matched")
else:
    edges["population"] = svi["population"].mean()
    edges["svi_score"]  = svi["svi_score"].mean()
    print("      [warn] no FIPS column in edges -- county-median SVI applied")

edges["population"] = edges["population"].fillna(svi["population"].median())
edges["svi_score"]  = edges["svi_score"].fillna(svi["svi_score"].median())

# vulnerability = svi_score only (no HVI)
edges["vulnerability"] = edges["svi_score"].copy()

print(f"      Vulnerability (SVI only): "
      f"{edges['vulnerability'].min():.3f} - {edges['vulnerability'].max():.3f}")

# =============================================================================
# STEP 3  Evacuation zone flag + tier
# =============================================================================
print("\n[3/5] Evacuation zone join...")

if evac_available:
    evac = gpd.read_file(EVAC_SHP).to_crs(epsg=3310)
    print(f"      {len(evac):,} zone polygons | columns: {list(evac.columns)}")

    edge_centroids = gpd.GeoDataFrame(
        edges[["edge_id"]], geometry=edges.geometry.centroid, crs=edges.crs
    )
    joined = gpd.sjoin(
        edge_centroids, evac, how="left", predicate="within"
    )
    joined = joined.reset_index().drop_duplicates(subset="edge_id", keep="first")

    edges = edges.merge(
        joined[["edge_id", "index_right"]].rename(
            columns={"index_right": "_zone_idx"}
        ),
        on="edge_id", how="left",
    )
    edges["in_evac_zone"] = edges["_zone_idx"].notna().astype(int)
    edges = edges.drop(columns=["_zone_idx"])

    # Zone tier -- encode if column exists
    tier_col = next(
        (c for c in ["ZONE", "TIER", "EV_ZONE", "EVAC_ZONE", "ORDER", "zone"]
         if c in evac.columns),
        None,
    )
    if tier_col:
        tier_map = joined[["edge_id", tier_col]].rename(
            columns={tier_col: "zone_tier"}
        )
        edges = edges.merge(tier_map, on="edge_id", how="left")
        try:
            edges["zone_tier"] = pd.to_numeric(
                edges["zone_tier"], errors="coerce"
            ).fillna(0)
        except Exception:
            tier_order = {
                v: i + 1
                for i, v in enumerate(sorted(edges["zone_tier"].dropna().unique()))
            }
            edges["zone_tier"] = edges["zone_tier"].map(tier_order).fillna(0)
        print(f"      Zone tier from '{tier_col}'")
    else:
        edges["zone_tier"] = edges["in_evac_zone"].astype(float)
        print("      No tier column found -- zone_tier = in_evac_zone (0/1)")

    n_in = edges["in_evac_zone"].sum()
    print(f"      {n_in:,}/{len(edges):,} edges inside evacuation zones "
          f"({n_in / len(edges) * 100:.1f}%)")
else:
    edges["in_evac_zone"] = 0
    edges["zone_tier"]    = 0.0
    print("      Shapefile not found -- in_evac_zone = 0, zone_tier = 0")

# =============================================================================
# STEP 4  BPR synthetic label + hybrid merge with PeMS
# =============================================================================
print("\n[4/5] BPR label + hybrid merge...")

# Demand V: population x vulnerability x evacuation zone boost
evac_boost = 1.0 + edges["in_evac_zone"] * EVAC_ZONE_BOOST
raw_demand = edges["population"] * edges["vulnerability"] * evac_boost
scale      = TARGET_MED_VC / (
    (raw_demand / edges["capacity_veh_hr"]).median() + 1e-9
)

edges["demand_V"]       = (raw_demand * scale).clip(lower=0)
edges["vc_ratio"]       = (
    edges["demand_V"] / edges["capacity_veh_hr"]
).clip(upper=3.0)
edges["bpr_multiplier"] = 1.0 + BPR_ALPHA * (edges["vc_ratio"] ** BPR_BETA)

# Merge PeMS labels if available
if pems_available:
    pems = pd.read_csv(PEMS_CSV, dtype={"edge_id": str})
    edges = edges.merge(
        pems[[
            "edge_id", "pems_multiplier", "pems_speed_event",
            "pems_speed_baseline", "pems_n_samples",
        ]],
        on="edge_id", how="left",
    )
    # Hybrid: real PeMS where matched, BPR elsewhere
    edges["congestion_multiplier"] = np.where(
        edges["pems_multiplier"].notna(),
        edges["pems_multiplier"],
        edges["bpr_multiplier"],
    )
    edges["label_source"] = np.where(
        edges["pems_multiplier"].notna(), "pems", "bpr"
    )
    n_pems = (edges["label_source"] == "pems").sum()
    n_bpr  = (edges["label_source"] == "bpr").sum()
    print(f"      PeMS labels : {n_pems:,} edges  ({n_pems / len(edges) * 100:.1f}%)")
    print(f"      BPR labels  : {n_bpr:,} edges  ({n_bpr / len(edges) * 100:.1f}%)")
else:
    edges["pems_multiplier"]     = np.nan
    edges["pems_speed_event"]    = np.nan
    edges["pems_speed_baseline"] = np.nan
    edges["pems_n_samples"]      = np.nan
    edges["congestion_multiplier"] = edges["bpr_multiplier"]
    edges["label_source"]          = "bpr"
    print("      No PeMS data -- all edges use BPR label")

# Congested travel time (edge weight for sub-problem 3 routing)
edges["congested_time_min"] = edges["freeflow_min"] * edges["congestion_multiplier"]

print(f"\n      congestion_multiplier -- "
      f"median {edges['congestion_multiplier'].median():.3f}  "
      f"max {edges['congestion_multiplier'].max():.3f}")
print(f"      congested_time_min   -- "
      f"median {edges['congested_time_min'].median():.2f} min  "
      f"(vs free-flow {edges['freeflow_min'].median():.2f} min)")

# =============================================================================
# STEP 5  Save final feature table
# =============================================================================
print("\n[5/5] Saving edge_features.csv...")

FEATURE_COLS = [
    # identifiers
    "edge_id",
    # road geometry  (from road_network.py)
    "length_m", "speed_kmh", "lanes", "capacity_veh_hr",
    "freeflow_min", "is_highway",
    # graph topology  (from road_network.py)
    "betweenness", "avg_degree",
    # vulnerability  (SVI only -- no HVI)
    "svi_score", "vulnerability", "population",
    # evacuation zone  (this script)
    "in_evac_zone", "zone_tier",
    # demand  (BPR inputs)
    "demand_V", "vc_ratio",
    # labels
    "bpr_multiplier",            # synthetic label (always present)
    "pems_multiplier",           # real label (NaN for unmatched edges)
    "congestion_multiplier",     # PRIMARY TARGET -- hybrid of above two
    "congested_time_min",        # edge weight for 3_route_optimization
    "label_source",              # "pems" | "bpr"
    # PeMS diagnostics
    "pems_speed_event", "pems_speed_baseline", "pems_n_samples",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in edges.columns]

edge_features = edges[FEATURE_COLS].copy()
edge_features.to_csv(OUT_FEATURES, index=False)

print(f"\n  [OK]  edge_features.csv -> data/processed/")
print(f"  Shape : {edge_features.shape[0]:,} rows x {edge_features.shape[1]} features")

print("\n-- Label summary --")
print(edge_features.groupby("label_source")["congestion_multiplier"]
      .describe().round(3))

print("\n-- Feature summary (key columns) --")
print(edge_features[[
    "congestion_multiplier", "bpr_multiplier",
    "vc_ratio", "vulnerability", "in_evac_zone",
]].describe().round(3))

print("\n  Pipeline complete.")