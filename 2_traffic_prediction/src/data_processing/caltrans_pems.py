from pathlib import Path
 
 
def _find_project_root(marker="data"):
    """Walk up from this file until we find a folder containing `marker/`."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise RuntimeError(
        f"Cannot find project root (looked for '{marker}/' starting from {here})"
    )
 
 
PROJECT_ROOT = _find_project_root("data")
RAW  = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
 
PEMS_META_PATH = RAW  / "pems_meta.txt"
PEMS_5MIN_DIR  = RAW  / "pems_5min"
ROAD_EDGES_CSV = PROC / "road_edges.csv"
OUT_CSV        = PROC / "pems_congestion.csv"
 
# =============================================================================
# CONFIG  (all values confirmed by running diagnose_pems.py)
# =============================================================================
 
# LA wildfire evacuation events
WILDFIRE_EVENTS = {
    "woolsey_2018": ("2018-11-08", "2018-11-21"),
    "bobcat_2020":  ("2020-09-06", "2020-09-30"),
    "easy_2024":    ("2024-11-01", "2024-11-22"),
}
 
BASELINE_WEEKS   = 4     # weeks of non-event data for baseline speed
MIN_PCT_OBSERVED = 0.50  # drop 5-min rows with <50% real (non-imputed) samples
SNAP_MAX_M       = 500   # max metres to snap a PeMS station to a road edge
 
# --- 5-minute flow file columns (confirmed from diagnose_pems.py) ---
# col[ 0]  timestamp      "05/03/2026 00:00:00"
# col[ 1]  station_id     integer station ID
# col[ 5]  lane_type      "ML", "OR", "FR", etc.
# col[ 8]  pct_observed   fraction of 30-sec samples that are real (not imputed)
# col[ 9]  total_flow     total vehicles across all lanes
# col[11]  avg_speed      flow-weighted mean speed across all lanes (mph)
FLOW_USE_COLS  = [0, 1, 5, 8, 9, 11]
FLOW_COL_NAMES = ["timestamp", "station_id", "lane_type",
                  "pct_observed", "total_flow", "avg_speed"]
 
# --- Metadata file (confirmed from diagnose_pems.py) ---
# File HAS a header row: ID Fwy Dir District County City State_PM Abs_PM
#                        Latitude Longitude Length Type Lanes Name User_ID_*
# LA County code in this file = 37  (NOT 19)
LA_COUNTY_CODE = "37"
 
# =============================================================================
# IMPORTS
# =============================================================================
import sys
import gzip
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
 
PROC.mkdir(parents=True, exist_ok=True)
 
print("=" * 60)
print("  Caltrans PeMS -- Congestion Label Processing")
print("=" * 60)
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Raw data     : {RAW}")
 
# =============================================================================
# VALIDATE INPUTS
# =============================================================================
for p, hint in [
    (PEMS_META_PATH, "download d07_text_meta_*.txt from pems.dot.ca.gov"),
    (ROAD_EDGES_CSV, "run road_network.py first"),
]:
    if not p.exists():
        print(f"\n[ERROR] Not found: {p}")
        print(f"        Hint: {hint}")
        sys.exit(1)
 
gz_files = sorted(PEMS_5MIN_DIR.glob("*.gz")) + \
           sorted(PEMS_5MIN_DIR.glob("*.txt"))
if not gz_files:
    print(f"\n[ERROR] No 5-min files found in {PEMS_5MIN_DIR}")
    print("        Download d07_text_station_5min_YYYY_MM_DD.txt.gz")
    sys.exit(1)
 
print(f"\n  {len(gz_files)} day-file(s) found in pems_5min/")
print(f"  Files: {[f.name for f in gz_files[:5]]}")
 
# Check whether any event-date files exist
event_dates_all = set()
for _, (start, end) in WILDFIRE_EVENTS.items():
    event_dates_all.update(pd.date_range(start, end, freq="D"))
 
file_dates = set()
for f in gz_files:
    # filename: d07_text_station_5min_YYYY_MM_DD.txt[.gz]
    parts = f.stem.replace(".txt", "").split("_")
    try:
        file_dates.add(pd.Timestamp("-".join(parts[-3:])))
    except Exception:
        pass
 
event_files = file_dates & event_dates_all
if not event_files:
    print("\n[WARN] None of your files fall on wildfire event dates.")
    print("       Event dates needed:")
    for name, (start, end) in WILDFIRE_EVENTS.items():
        print(f"         {name}: {start} to {end}")
    print("       Current files cover:", sorted(str(d.date()) for d in file_dates))
    print("       The script will still run but congestion_multiplier will be NaN.")
    print("       Download event-date files from pems.dot.ca.gov to get real labels.\n")
 
 
# =============================================================================
# STEP 1  Station metadata
# =============================================================================
print("\n[1/5] Station metadata...")
 
# File has a real header row -- use header=0
meta = pd.read_csv(
    PEMS_META_PATH,
    sep="\t",
    header=0,           # <-- header row IS present
    dtype={"ID": str, "County": str},
)
 
# Normalize column names to lowercase for consistent access
meta.columns = [c.strip().lower() for c in meta.columns]
 
# Keep only mainline (ML) stations -- most reliable for congestion
meta = meta[meta["type"].astype(str).str.strip() == "ML"].copy()
 
# Filter to LA County -- code is 37 in this file (confirmed by diagnose_pems.py)
meta = meta[meta["county"].astype(str).str.strip() == LA_COUNTY_CODE].copy()
meta = meta.dropna(subset=["latitude", "longitude"])
meta = meta.rename(columns={"id": "station_id", "fwy": "freeway"})
meta["station_id"] = meta["station_id"].astype(str).str.strip()
 
print(f"      {len(meta):,} mainline stations in LA County (county={LA_COUNTY_CODE})")
if len(meta) > 0:
    print(f"      Freeways covered: {sorted(meta['freeway'].unique())}")
else:
    print("      [WARN] No stations found. Check LA_COUNTY_CODE constant.")
    print("      Unique county values in file:",
          pd.read_csv(PEMS_META_PATH, sep="\t", header=0)["County"]
          .unique()[:10].tolist())
 
# =============================================================================
# STEP 2  Load 5-minute flow files
# =============================================================================
print("\n[2/5] Loading 5-minute flow files...")
 
chunks, failed = [], []
for f in gz_files:
    try:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fh:
            chunk = pd.read_csv(
                fh,
                header=None,
                usecols=FLOW_USE_COLS,
                names=FLOW_COL_NAMES,
                dtype={"station_id": str},
            )
        chunks.append(chunk)
    except Exception as e:
        failed.append((f.name, str(e)))
 
if failed:
    print(f"      [warn] {len(failed)} file(s) failed to parse:")
    for name, err in failed[:3]:
        print(f"             {name}: {err}")
 
if not chunks:
    print("\n[ERROR] All 5-min files failed. Check file format.")
    sys.exit(1)
 
flow = pd.concat(chunks, ignore_index=True)
print(f"      {len(flow):,} raw rows from {len(chunks)} file(s)")
print(f"      Columns: {list(flow.columns)}")
 
# =============================================================================
# STEP 3  Clean + filter to mainline + split event vs baseline
# =============================================================================
print("\n[3/5] Cleaning and splitting event vs baseline...")
 
flow["timestamp"]    = pd.to_datetime(flow["timestamp"], errors="coerce")
flow["avg_speed"]    = pd.to_numeric(flow["avg_speed"],    errors="coerce")
flow["pct_observed"] = pd.to_numeric(flow["pct_observed"], errors="coerce")
flow["station_id"]   = flow["station_id"].astype(str).str.strip()
 
# Keep only mainline rows (lane_type == "ML")
before = len(flow)
flow = flow[flow["lane_type"].astype(str).str.strip() == "ML"].copy()
print(f"      After ML filter  : {len(flow):,} rows  "
      f"(dropped {before - len(flow):,} non-mainline)")
 
# Drop imputed / zero-speed rows
before = len(flow)
flow = flow[
    (flow["pct_observed"] >= MIN_PCT_OBSERVED) &
    (flow["avg_speed"] > 0)
].dropna(subset=["timestamp", "avg_speed"])
print(f"      After quality filter : {len(flow):,} rows  "
      f"(dropped {before - len(flow):,} imputed/invalid, "
      f"threshold pct_observed >= {MIN_PCT_OBSERVED})")
 
flow["date"]    = flow["timestamp"].dt.normalize()
flow["hour"]    = flow["timestamp"].dt.hour
flow["weekday"] = flow["timestamp"].dt.weekday   # 0=Mon, 6=Sun
 
# Identify event vs baseline rows
flow_event    = flow[flow["date"].isin(event_dates_all)].copy()
flow_baseline = flow[~flow["date"].isin(event_dates_all)].copy()
 
# Restrict baseline window
baseline_cutoff = min(
    pd.Timestamp(v[0]) - pd.Timedelta(weeks=BASELINE_WEEKS)
    for v in WILDFIRE_EVENTS.values()
)
flow_baseline = flow_baseline[flow_baseline["date"] >= baseline_cutoff].copy()
 
print(f"      Event rows    : {len(flow_event):,}  "
      f"({flow_event['date'].nunique()} event days)")
print(f"      Baseline rows : {len(flow_baseline):,}  "
      f"({flow_baseline['date'].nunique()} baseline days)")
 
if len(flow_event) == 0:
    print("\n[WARN] Zero event rows found.")
    print("       Your current files are baseline-only (non-wildfire dates).")
    print("       Download the event-date files listed above, then re-run.")
    print("       Saving empty pems_congestion.csv with NaN multipliers.")
 
# =============================================================================
# STEP 4  Congestion multiplier per station
#
#   baseline_speed  = mean speed for (station_id, weekday, hour)
#                     across non-event days in the baseline window
#
#   congestion_mult = baseline_speed / actual_event_speed
#       1.0 = free flow | 2.0 = half speed | 3.0+ = severe gridlock
# =============================================================================
print("\n[4/5] Computing congestion multipliers...")
 
station_data = meta.copy()  # default: no multiplier columns
 
if len(flow_event) > 0 and len(flow_baseline) > 0:
    baseline_speed = (
        flow_baseline
        .groupby(["station_id", "weekday", "hour"])["avg_speed"]
        .mean()
        .reset_index()
        .rename(columns={"avg_speed": "baseline_speed"})
    )
 
    flow_event = flow_event.merge(
        baseline_speed, on=["station_id", "weekday", "hour"], how="left"
    ).dropna(subset=["baseline_speed"])
 
    flow_event["congestion_mult"] = (
        flow_event["baseline_speed"] / flow_event["avg_speed"]
    ).clip(lower=0.5, upper=5.0)
 
    station_mult = (
        flow_event
        .groupby("station_id")
        .agg(
            pems_multiplier     = ("congestion_mult", "mean"),
            pems_speed_event    = ("avg_speed",       "mean"),
            pems_speed_baseline = ("baseline_speed",  "mean"),
            pems_n_samples      = ("congestion_mult", "count"),
        )
        .reset_index()
    )
    station_data = meta.merge(station_mult, on="station_id", how="inner")
    print(f"      {len(station_data):,} stations with multiplier")
    print(f"      Multiplier -- median {station_data['pems_multiplier'].median():.3f}  "
          f"max {station_data['pems_multiplier'].max():.3f}")
else:
    station_data["pems_multiplier"]     = float("nan")
    station_data["pems_speed_event"]    = float("nan")
    station_data["pems_speed_baseline"] = float("nan")
    station_data["pems_n_samples"]      = 0
    print("      Skipped -- no event rows available")
 
# =============================================================================
# STEP 5  Snap stations to road edges
# =============================================================================
print("\n[5/5] Snapping stations to road edges...")
 
road_edges = pd.read_csv(ROAD_EDGES_CSV, dtype={"edge_id": str})
 
# Build output: one row per edge, NaN where no station is nearby
out = road_edges[["edge_id"]].copy()
 
if len(station_data) > 0 and not station_data["pems_multiplier"].isna().all():
    stations_gdf = gpd.GeoDataFrame(
        station_data,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(station_data["longitude"], station_data["latitude"])
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3310)
 
    edges_gdf = gpd.GeoDataFrame(
        road_edges[["edge_id"]],
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(road_edges["mid_lon"], road_edges["mid_lat"])
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3310)
 
    snapped = gpd.sjoin_nearest(
        edges_gdf,
        stations_gdf[[
            "geometry", "station_id", "freeway",
            "pems_multiplier", "pems_speed_event",
            "pems_speed_baseline", "pems_n_samples",
        ]],
        how="left",
        distance_col="snap_dist_m",
    )
    snapped = (
        snapped[snapped["snap_dist_m"] <= SNAP_MAX_M]
        .sort_values("snap_dist_m")
        .drop_duplicates(subset="edge_id", keep="first")
    )
 
    out = out.merge(
        snapped[[
            "edge_id", "station_id", "freeway", "snap_dist_m",
            "pems_multiplier", "pems_speed_event",
            "pems_speed_baseline", "pems_n_samples",
        ]],
        on="edge_id", how="left",
    )
    n_matched = out["pems_multiplier"].notna().sum()
    print(f"      Matched : {n_matched:,} / {len(out):,} edges  "
          f"({n_matched / len(out) * 100:.1f}%)")
    print(f"      Unmatched {len(out) - n_matched:,} edges -> BPR fallback "
          f"in evacuation_zones.py")
else:
    # No multipliers yet -- write skeleton with all NaN
    for col in ["station_id", "freeway", "snap_dist_m",
                "pems_multiplier", "pems_speed_event",
                "pems_speed_baseline", "pems_n_samples"]:
        out[col] = float("nan")
    print("      No station multipliers -- skeleton written with NaN values")
    print("      Re-run after downloading wildfire event-date files")
 
out.to_csv(OUT_CSV, index=False)
 
print(f"\n  [OK]  pems_congestion.csv -> data/processed/")
print(f"  Shape : {out.shape[0]:,} rows x {out.shape[1]} columns")
 
if out["pems_multiplier"].notna().sum() > 0:
    print("\n-- Label distribution (matched edges only) --")
    print(out.dropna(subset=["pems_multiplier"])["pems_multiplier"]
            .describe().round(3).to_string())
else:
    print("\n-- No matched edges yet --")
    print("   Download these files from pems.dot.ca.gov -> Data Clearinghouse -> District 7:")
    for name, (start, end) in WILDFIRE_EVENTS.items():
        print(f"   {name}: {start} to {end}  (+ 4 weeks before for baseline)")
 