"""
Simulate a wildfire ignition near a suburb and identify high-risk surrounding
census tracts — without relying on historical perimeter data.

Given a suburb name or lat/lon, the script:
  1. Geocodes the location via Nominatim
  2. Creates concentric analysis rings around the ignition point
  3. Loads the trained risk classifier and predicts risk for all LA County tracts
  4. Tags each tract by zone (fire zone / immediate / nearby)
  5. Prints a ranked risk report to the console
  6. Saves an interactive Folium HTML map

Usage
-----
    python simulate.py --suburb "Chatsworth"
    python simulate.py --suburb "Malibu" --radius 8
    python simulate.py --lat 34.17 --lon -118.60 --radius 6
    python simulate.py --suburb "Altadena" --radius 5 --out outputs/maps/sim_altadena.html
"""

import argparse
import sys
from pathlib import Path

import folium
import geopandas as gpd
import joblib
import pandas as pd
import requests
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent                                       # 3_route_optimization/
DATA = ROOT.parent / "data" / "processed"                                 # project_root/data/processed
MODEL_DIR = ROOT.parent / "1_risk_classifier" / "outputs" / "models"
MAP_DIR = ROOT / "outputs" / "maps"

CRS   = "EPSG:3310"   # CA Albers Equal-Area (meters) — same as model training
WGS84 = "EPSG:4326"

FEATURES = [
    "dist_fire_station_m",
    "dist_hospital_m",
    "hydrant_density",
    "road_density",
    "pop_density",
    "rpl_theme_1",
    "rpl_theme_2",
    "rpl_theme_3",
    "rpl_theme_4",
]

TIER_LABELS = {0: "Low", 1: "Medium", 2: "High"}
TIER_COLORS = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}

ZONE_ORDER  = ["fire_zone", "immediate", "nearby"]
ZONE_OPACITY = {"fire_zone": 0.85, "immediate": 0.65, "nearby": 0.45}
ZONE_RING_COLOR = {
    "fire_zone": "#cc0000",
    "immediate": "#ff7700",
    "nearby":    "#ffaa00",
}


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_suburb(suburb: str) -> tuple[float, float]:
    """Return (lat, lon) for a suburb name using Nominatim."""
    headers = {"User-Agent": "wildfire-risk-simulator/1.0"}

    for query in [
        f"{suburb}, Los Angeles County, California",
        f"{suburb}, California",
        suburb,
    ]:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            print(f"Geocoded '{suburb}' → {lat:.5f}, {lon:.5f}")
            print(f"  Resolved as: {results[0].get('display_name', suburb)}")
            return lat, lon

    print(f"ERROR: Could not geocode '{suburb}'. Try --lat / --lon instead.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model & data
# ---------------------------------------------------------------------------

def load_model():
    rf     = joblib.load(MODEL_DIR / "risk_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    print(f"Model loaded: {type(rf).__name__}")
    return rf, scaler


def load_and_predict(rf, scaler) -> gpd.GeoDataFrame:
    tracts = gpd.read_file(DATA / "features.geojson").to_crs(CRS)
    valid  = tracts[FEATURES].dropna()
    X_sc   = scaler.transform(valid)
    tracts = tracts.copy()
    tracts.loc[valid.index, "predicted_tier"]  = rf.predict(X_sc).astype(float)
    tracts["predicted_label"] = tracts["predicted_tier"].map(TIER_LABELS)
    print(f"Tracts loaded and predicted: {len(tracts)}")
    return tracts


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------

def assign_zones(tracts: gpd.GeoDataFrame, center: Point, radius_m: float) -> gpd.GeoDataFrame:
    """
    Tag each tract with its zone based on centroid distance to the ignition point.
    Zones:
      fire_zone  — 0 to 1× radius
      immediate  — 1× to 2× radius
      nearby     — 2× to 3× radius
      outside    — beyond 3× radius
    """
    centroids = tracts.geometry.centroid
    tracts = tracts.copy()
    tracts["dist_to_fire_m"] = centroids.distance(center)

    def _zone(d):
        if d <= radius_m:
            return "fire_zone"
        elif d <= 2 * radius_m:
            return "immediate"
        elif d <= 3 * radius_m:
            return "nearby"
        return "outside"

    tracts["zone"] = tracts["dist_to_fire_m"].apply(_zone)
    return tracts


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(tracts: gpd.GeoDataFrame, suburb: str, radius_km: float) -> None:
    in_scope = tracts[tracts["zone"] != "outside"].dropna(subset=["predicted_tier"])
    in_scope = in_scope.sort_values(
        ["predicted_tier", "dist_to_fire_m"], ascending=[False, True]
    )

    print(f"\n{'='*62}")
    print(f"  Simulated Ignition Near: {suburb}")
    print(f"  Fire Zone: {radius_km} km  |  Analysis Radius: {3*radius_km:.0f} km")
    print(f"{'='*62}")

    zone_labels = {
        "fire_zone": f"INSIDE FIRE ZONE  (0 – {radius_km} km)",
        "immediate": f"IMMEDIATE RISK    ({radius_km} – {2*radius_km:.0f} km)",
        "nearby":    f"NEARBY            ({2*radius_km:.0f} – {3*radius_km:.0f} km)",
    }

    for zone_key in ZONE_ORDER:
        subset = in_scope[in_scope["zone"] == zone_key]
        if subset.empty:
            continue

        counts = subset["predicted_label"].value_counts()
        high   = counts.get("High",   0)
        med    = counts.get("Medium", 0)
        low    = counts.get("Low",    0)

        print(f"\n  {zone_labels[zone_key]}")
        print(f"  Total tracts: {len(subset)}   High: {high}   Medium: {med}   Low: {low}")
        print(f"  {'Tract':<14} {'Risk':<10} {'Dist (km)':<12} {'Pop/km²':<12} {'SVI Rank':<10}")
        print(f"  {'-'*57}")

        for _, row in subset.iterrows():
            tract_id = str(row.get("tract", row.name))[:13]
            tier     = row.get("predicted_label", "?")
            dist_km  = row["dist_to_fire_m"] / 1000
            pop      = row.get("pop_density", float("nan"))
            # Average across the 4 SVI theme percentiles as a composite vulnerability
            svi_cols = ["rpl_theme_1", "rpl_theme_2", "rpl_theme_3", "rpl_theme_4"]
            svi_avg  = row[svi_cols].mean() if all(c in row.index for c in svi_cols) else float("nan")
            flag     = " ⚠" if tier == "High" and svi_avg >= 0.75 else ""
            print(
                f"  {tract_id:<14} {tier:<10} {dist_km:<12.1f} "
                f"{pop:<12.0f} {svi_avg:<10.3f}{flag}"
            )

    total = len(in_scope)
    high_total = (in_scope["predicted_label"] == "High").sum()

    print(f"\n  SUMMARY")
    print(f"  {'─'*40}")
    for tier in ["High", "Medium", "Low"]:
        n = (in_scope["predicted_label"] == tier).sum()
        bar = "█" * int(20 * n / max(total, 1))
        print(f"  {tier:<8} {bar:<21} {n:>4} tracts ({100*n/max(total,1):.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total in analysis area: {total} tracts")
    print(
        f"\n  ⚠  {high_total} High-risk tract(s) within {3*radius_km:.0f} km "
        f"of the simulated ignition point."
    )


# ---------------------------------------------------------------------------
# Folium map
# ---------------------------------------------------------------------------

def build_map(
    tracts: gpd.GeoDataFrame,
    center: Point,
    radius_km: float,
    suburb: str,
    output_path: Path,
) -> None:
    # Project ignition center back to WGS84 for Folium
    center_gdf = gpd.GeoDataFrame(geometry=[center], crs=CRS).to_crs(WGS84)
    clat = center_gdf.geometry.iloc[0].y
    clon = center_gdf.geometry.iloc[0].x

    tracts_wgs  = tracts.to_crs(WGS84)
    in_scope    = tracts_wgs[tracts_wgs["zone"] != "outside"]
    background  = tracts_wgs[tracts_wgs["zone"] == "outside"]

    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")

    # --- Background tracts (muted) ---
    export_cols = [c for c in ["tract", "predicted_label", "geometry"] if c in background.columns]
    folium.GeoJson(
        background[export_cols].__geo_interface__,
        style_function=lambda f: {
            "fillColor": TIER_COLORS.get(
                f["properties"].get("predicted_label", "Low"), "#aaaaaa"
            ),
            "color": "white",
            "weight": 0.2,
            "fillOpacity": 0.12,
        },
        name="LA County (background)",
    ).add_to(m)

    # --- In-scope tracts by zone (back-to-front: nearby first) ---
    tooltip_fields = [c for c in ["tract", "predicted_label", "zone", "pop_density"] if c in in_scope.columns]
    tooltip_aliases = ["Tract", "Risk Tier", "Zone", "Pop Density (per km²)"][: len(tooltip_fields)]

    for zone_key in reversed(ZONE_ORDER):
        subset = in_scope[in_scope["zone"] == zone_key]
        if subset.empty:
            continue

        opacity = ZONE_OPACITY[zone_key]
        folium.GeoJson(
            subset[[c for c in tooltip_fields + ["geometry"] if c in subset.columns]].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": TIER_COLORS.get(
                    f["properties"].get("predicted_label", "Low"), "#aaaaaa"
                ),
                "color": "white",
                "weight": 0.5,
                "fillOpacity": op,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
            ),
            name=zone_key.replace("_", " ").title(),
        ).add_to(m)

    # --- Risk rings ---
    ring_meta = [
        (3 * radius_km * 1000, ZONE_RING_COLOR["nearby"],    f"Nearby boundary ({3*radius_km:.0f} km)"),
        (2 * radius_km * 1000, ZONE_RING_COLOR["immediate"], f"Immediate risk boundary ({2*radius_km:.0f} km)"),
        (    radius_km * 1000, ZONE_RING_COLOR["fire_zone"], f"Simulated fire zone ({radius_km} km)"),
    ]
    for radius_m, color, label in ring_meta:
        folium.Circle(
            location=[clat, clon],
            radius=radius_m,
            color=color,
            weight=2,
            fill=False,
            dash_array="6",
            tooltip=label,
        ).add_to(m)

    # --- Ignition marker ---
    folium.Marker(
        location=[clat, clon],
        popup=folium.Popup(f"<b>Simulated Ignition</b><br>{suburb}", max_width=200),
        icon=folium.Icon(color="red", icon="exclamation-sign"),
        tooltip="Simulated ignition point",
    ).add_to(m)

    # --- Legend ---
    legend_html = (
        "<div style='position:fixed;bottom:40px;left:40px;z-index:1000;"
        "background:white;padding:14px 16px;border-radius:8px;font-size:13px;"
        "box-shadow:2px 2px 8px rgba(0,0,0,0.3);min-width:220px;line-height:1.7'>"
        f"<b>Simulated Fire — {suburb}</b><br>"
        "<hr style='margin:6px 0'>"
        "<b>Predicted Risk Tier</b><br>"
    )
    for tier, color in TIER_COLORS.items():
        legend_html += (
            f"<span style='background:{color};display:inline-block;"
            f"width:13px;height:13px;margin-right:7px;border-radius:2px'></span>"
            f"{tier}<br>"
        )
    legend_html += (
        "<hr style='margin:6px 0'>"
        "<b>Analysis Zones</b><br>"
        f"<span style='color:{ZONE_RING_COLOR['fire_zone']};font-weight:bold'>●</span>"
        f" Fire Zone (0–{radius_km} km)<br>"
        f"<span style='color:{ZONE_RING_COLOR['immediate']};font-weight:bold'>●</span>"
        f" Immediate ({radius_km}–{2*radius_km:.0f} km)<br>"
        f"<span style='color:{ZONE_RING_COLOR['nearby']};font-weight:bold'>●</span>"
        f" Nearby ({2*radius_km:.0f}–{3*radius_km:.0f} km)<br>"
        "<hr style='margin:6px 0'>"
        "<small>Opacity = zone proximity<br>Darker = closer to ignition</small>"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Map saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate a wildfire near a suburb and identify high-risk tracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument("--suburb", metavar="NAME",
                     help="Suburb or neighborhood name (e.g. 'Chatsworth', 'Malibu')")
    loc.add_argument("--lat", type=float, metavar="LAT",
                     help="Latitude of ignition point (requires --lon)")

    parser.add_argument("--lon", type=float, metavar="LON",
                        help="Longitude of ignition point (required with --lat)")
    parser.add_argument("--radius", type=float, default=5.0, metavar="KM",
                        help="Fire zone radius in km (default: 5). "
                             "Analysis covers 3× this radius.")
    parser.add_argument("--out", metavar="PATH",
                        help="Output HTML map path "
                             "(default: outputs/maps/sim_<suburb>.html)")

    args = parser.parse_args()

    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when --lat is provided")

    # --- Resolve location ---
    if args.suburb:
        lat, lon = geocode_suburb(args.suburb)
        label = args.suburb
    else:
        lat, lon = args.lat, args.lon
        label = f"{lat:.4f}_{lon:.4f}"
        print(f"Ignition point: {lat:.5f}, {lon:.5f}")

    # --- Load model and tracts ---
    rf, scaler = load_model()
    tracts     = load_and_predict(rf, scaler)

    # --- Project ignition point and assign zones ---
    center_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=WGS84).to_crs(CRS)
    center     = center_gdf.geometry.iloc[0]
    radius_m   = args.radius * 1000

    tracts = assign_zones(tracts, center, radius_m)

    in_scope = tracts[tracts["zone"] != "outside"]
    print(f"Tracts within {3*args.radius:.0f} km analysis area: {len(in_scope)}")

    # --- Report + map ---
    print_report(tracts, label, args.radius)

    safe_label = label.lower().replace(" ", "_").replace(",", "").replace(".", "")
    map_path   = (
        Path(args.out) if args.out
        else MAP_DIR / f"sim_{safe_label}.html"
    )
    build_map(tracts, center, args.radius, label, map_path)


if __name__ == "__main__":
    main()
