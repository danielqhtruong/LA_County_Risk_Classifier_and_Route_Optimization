# ── Paths ─────────────────────────────────────────────────────────────────────
from pathlib import Path
 
def _find_project_root(marker: str = "data") -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find project root (looked for '{marker}/' from {here})."
    )
 
PROJECT_ROOT  = _find_project_root("data")
GEOJSON_PATH  = PROJECT_ROOT / "data" / "external" / "escape_routes_roads.geojson"
GRAPHML_PATH  = PROJECT_ROOT / "data" / "external" / "road_network.graphml"
OUT_GPKG      = PROJECT_ROOT / "data" / "processed" / "road_edges.gpkg"
OUT_CSV       = PROJECT_ROOT / "data" / "processed" / "road_edges.csv"
OUT_GRAPH_PKL = PROJECT_ROOT / "data" / "cache"     / "road_graph.pkl"
 
# ── Betweenness approximation samples (lower = faster, higher = more accurate)
BC_SAMPLES = 200
 
# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
 
for d in [OUT_GPKG.parent, OUT_GRAPH_PKL.parent]:
    d.mkdir(parents=True, exist_ok=True)
 
print("=" * 60)
print("  Road Network — Feature Engineering")
print("=" * 60)
print(f"  Project root : {PROJECT_ROOT}")
 
if not GEOJSON_PATH.exists():
    print(f"\n[ERROR] Not found: {GEOJSON_PATH}")
    sys.exit(1)
 
 
# ── 1. Load GeoJSON ───────────────────────────────────────────────────────────
print("\n[1/5] Loading road edges...")
 
edges = gpd.read_file(GEOJSON_PATH).to_crs(epsg=3310)  # CA Albers — metres
edges = edges.reset_index(drop=True)
edges["edge_id"] = edges.index.astype(str)
 
print(f"      {len(edges):,} edges loaded")
print(f"      Columns: {list(edges.columns)}")
 
 
# ── 2. Geometry features ──────────────────────────────────────────────────────
print("\n[2/5] Geometry features...")
 
# Length
edges["length_m"] = edges.geometry.length
 
# Speed — scan common OSM / Caltrans column names
speed_col = next(
    (c for c in ["speed", "maxspeed", "SPEED", "SPD_LIMIT", "speed_kph"]
     if c in edges.columns), None
)
if speed_col:
    edges["speed_kmh"] = pd.to_numeric(edges[speed_col], errors="coerce").fillna(40)
    print(f"      Speed from '{speed_col}'")
else:
    edges["speed_kmh"] = 40.0
    print("      [warn] no speed column — defaulting to 40 km/h")
 
# Lanes
lane_col = next(
    (c for c in ["lanes", "LANES", "lane_count", "num_lanes"] if c in edges.columns),
    None
)
if lane_col:
    edges["lanes"] = pd.to_numeric(edges[lane_col], errors="coerce").fillna(2).astype(int)
    print(f"      Lanes from '{lane_col}'")
else:
    edges["lanes"] = 2
    print("      [warn] no lanes column — defaulting to 2")
 
# Road type → highway flag + capacity
rt_col = next(
    (c for c in ["highway", "road_type", "ROAD_TYPE", "fclass", "type"]
     if c in edges.columns), None
)
if rt_col:
    hwy_pattern = "motorway|trunk|primary|secondary|highway|freeway"
    edges["is_highway"] = (
        edges[rt_col].astype(str).str.lower()
        .str.contains(hwy_pattern, na=False).astype(int)
    )
    # HCM simplified: freeway 2000, arterial 1000 veh/hr/lane
    edges["capacity_veh_hr"] = np.where(
        edges["is_highway"] == 1, 2000, 1000
    ) * edges["lanes"]
    print(f"      Road type from '{rt_col}'")
else:
    edges["is_highway"]      = 0
    edges["capacity_veh_hr"] = 1000 * edges["lanes"]
    print("      [warn] no road type column — capacity = 1000 veh/hr/lane")
 
# Free-flow travel time (minutes)
edges["freeflow_min"] = (edges["length_m"] / 1000.0) / edges["speed_kmh"] * 60.0
 
# Edge midpoint coordinates (used later for spatial joins)
edges["mid_lon"] = edges.geometry.centroid.to_crs(epsg=4326).x
edges["mid_lat"] = edges.geometry.centroid.to_crs(epsg=4326).y
 
print(f"      Length  : {edges['length_m'].min():.0f} – {edges['length_m'].max():.0f} m")
print(f"      Speed   : {edges['speed_kmh'].min():.0f} – {edges['speed_kmh'].max():.0f} km/h")
print(f"      Capacity: {edges['capacity_veh_hr'].min():.0f} – {edges['capacity_veh_hr'].max():.0f} veh/hr")
 
 
# ── 3. Build NetworkX graph ───────────────────────────────────────────────────
print("\n[3/5] Building NetworkX graph...")
 
if GRAPHML_PATH.exists():
    print(f"      Loading existing graph from {GRAPHML_PATH.name}...")
    G = nx.read_graphml(GRAPHML_PATH)
    # Ensure edge weights exist; fall back to length_m
    for u, v, d in G.edges(data=True):
        if "weight" not in d:
            d["weight"] = d.get("length_m", 1.0)
    print(f"      {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")
else:
    print("      road_network.graphml not found — building from GeoJSON...")
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        coords = list(row.geometry.coords)
        u, v   = coords[0], coords[-1]
        G.add_edge(u, v,
                   edge_id=row["edge_id"],
                   weight=row["length_m"],
                   freeflow_min=row["freeflow_min"])
    print(f"      Built: {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")
 
# Persist for 3_route_optimization
with open(OUT_GRAPH_PKL, "wb") as f:
    pickle.dump(G, f)
print(f"      Saved → {OUT_GRAPH_PKL.relative_to(PROJECT_ROOT)}")
 
 
# ── 4. Graph topology features ────────────────────────────────────────────────
print("\n[4/5] Graph topology features...")
 
# Use undirected view for centrality (symmetric betweenness)
G_ud  = G.to_undirected() if G.is_directed() else G
n     = G_ud.number_of_nodes()
k     = min(BC_SAMPLES, n)
print(f"      Betweenness centrality (k={k} samples)...")
 
bc     = nx.betweenness_centrality(G_ud, k=k, weight="weight", normalized=True)
degree = dict(G_ud.degree())
 
# Map node-level centrality to edges via endpoint nodes
def _bc(row):
    coords = list(row.geometry.coords)
    u, v   = coords[0], coords[-1]
    return (bc.get(u, 0.0) + bc.get(v, 0.0)) / 2.0
 
def _deg(row):
    coords = list(row.geometry.coords)
    u, v   = coords[0], coords[-1]
    return (degree.get(u, 1) + degree.get(v, 1)) / 2.0
 
edges["betweenness"] = edges.apply(_bc,  axis=1)
edges["avg_degree"]  = edges.apply(_deg, axis=1)
 
print(f"      Betweenness: {edges['betweenness'].min():.5f} – {edges['betweenness'].max():.5f}")
print(f"      Avg degree : {edges['avg_degree'].min():.1f} – {edges['avg_degree'].max():.1f}")
 
 
# ── 5. Save ───────────────────────────────────────────────────────────────────
print("\n[5/5] Saving...")
 
OUTPUT_COLS = [
    "edge_id",
    "length_m", "speed_kmh", "lanes", "capacity_veh_hr",
    "freeflow_min", "is_highway",
    "betweenness", "avg_degree",
    "mid_lat", "mid_lon",
]
OUTPUT_COLS = [c for c in OUTPUT_COLS if c in edges.columns]
 
# GeoPackage (keeps geometry — used by evacuation_zones.py and routing)
edges[OUTPUT_COLS + ["geometry"]].to_file(OUT_GPKG, driver="GPKG")
 
# Flat CSV (used by caltrans_pems.py for spatial join + by ml_pipeline.py)
edges[OUTPUT_COLS].to_csv(OUT_CSV, index=False)
 
print(f"  ✓  road_edges.gpkg → data/processed/")
print(f"  ✓  road_edges.csv  → data/processed/")
print(f"  ✓  road_graph.pkl  → data/cache/")
print(f"\n  Shape : {len(edges):,} edges × {len(OUTPUT_COLS)} features")