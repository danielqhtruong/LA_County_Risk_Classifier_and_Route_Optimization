"""
Historic wildfire validation for the risk classifier.

Loads pre-saved CAL FIRE perimeters (data/processed/fire_perimeters.geojson),
tags which LA County census tracts were actually burned, then compares those
against the trained model's predicted risk tiers.

Run src/data_processing/fetch_calfire_perimeters.ipynb first to populate the
perimeters file.

Usage
-----
    python predict.py                              # all LA fires in the saved dataset
    python predict.py --fire EATON              # single named fire
    python predict.py --fire EATON --year 2025 # name + year
    python predict.py --fire EATON --year 2025 --out outputs/maps/eaton.html
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import folium
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent          # 1_risk_classifier/
DATA = ROOT.parent / "data" / "processed"   # project_root/data/processed
MODEL_DIR = ROOT / "outputs" / "models"
MAP_DIR = ROOT / "outputs" / "maps"
PERIMETERS_FILE = DATA / "fire_perimeters.geojson"

CRS = "EPSG:3310"  # CA Albers — same as model training

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


# ---------------------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------------------

def load_model():
    rf = joblib.load(MODEL_DIR / "risk_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    print(f"Model loaded: {type(rf).__name__}")
    return rf, scaler


def load_tracts():
    tracts = gpd.read_file(DATA / "features.geojson").to_crs(CRS)
    print(f"Tracts loaded: {len(tracts)}")
    return tracts


# ---------------------------------------------------------------------------
# CAL FIRE perimeters
# ---------------------------------------------------------------------------

def load_fire_perimeters():
    if not PERIMETERS_FILE.exists():
        print(
            f"\nERROR: {PERIMETERS_FILE} not found.\n"
            "Run first:  src/data_processing/fetch_calfire_perimeters.ipynb"
        )
        sys.exit(1)
    fires = gpd.read_file(PERIMETERS_FILE).to_crs(CRS)
    print(f"Perimeters loaded: {len(fires)} records")
    return fires


def filter_fires(fires, fire_name=None, year=None):
    filtered = fires.copy()
    if fire_name:
        filtered = filtered[filtered["FIRE_NAME"] == fire_name.strip().upper()]
    if year:
        year_col = next(
            (c for c in filtered.columns if c.upper().startswith("YEAR")), None
        )
        if year_col:
            filtered = filtered[pd.to_numeric(filtered[year_col], errors="coerce") == int(year)]
    if filtered.empty:
        year_col = next((c for c in fires.columns if c.upper().startswith("YEAR")), None)
        years = sorted(pd.to_numeric(fires[year_col], errors="coerce").dropna().unique().astype(int).tolist()) if year_col else []
        print(f"No fires found for name={fire_name!r} year={year}")
        print("Available fire names:", fires["FIRE_NAME"].dropna().unique().tolist()[:20])
        print("Available years in dataset:", years)
        sys.exit(1)
    print(f"Fires matched: {len(filtered)}")
    return filtered


# ---------------------------------------------------------------------------
# Tag burned tracts + predict
# ---------------------------------------------------------------------------

def tag_burned_tracts(tracts, fires):
    burned_idx = gpd.sjoin(
        tracts[["geometry"]],
        fires[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()

    tracts = tracts.copy()
    tracts["actually_burned"] = tracts.index.isin(burned_idx).astype(int)
    print(f"Tracts intersecting fire perimeter(s): {tracts['actually_burned'].sum()}")
    return tracts


def predict_risk(tracts, rf, scaler):
    valid = tracts[FEATURES].dropna()
    X_sc = scaler.transform(valid)
    tracts = tracts.copy()
    tracts.loc[valid.index, "predicted_tier"] = rf.predict(X_sc).astype(float)
    tracts["predicted_label"] = tracts["predicted_tier"].map(TIER_LABELS)
    return tracts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(tracts, fire_label, save_cm=True):
    df = tracts.dropna(subset=["predicted_tier", "actually_burned"])

    y_true = df["actually_burned"]
    y_pred = (df["predicted_tier"] == 2).astype(int)  # High-risk = at-risk

    burned = df[df["actually_burned"] == 1]

    print(f"\n{'='*50}")
    print(f"  Historic Validation — {fire_label}")
    print(f"{'='*50}")
    print(f"  Total tracts evaluated : {len(df)}")
    print(f"  Actually burned        : {y_true.sum()}")
    print(f"  Predicted High-risk    : {y_pred.sum()}")

    if len(burned) > 0:
        print(f"\n  Burned tracts by predicted tier:")
        tier_dist = burned["predicted_label"].value_counts()
        for label in ["High", "Medium", "Low"]:
            n = tier_dist.get(label, 0)
            pct = 100 * n / len(burned)
            print(f"    {label:8s}: {n:4d}  ({pct:.1f}%)")

    print(f"\n  Classification Report (High-risk vs rest):")
    print(classification_report(
        y_true, y_pred,
        target_names=["Not Burned", "Burned"],
        zero_division=0,
    ))

    if save_cm:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Not At-Risk", "Pred: High-Risk"],
            yticklabels=["Not Burned", "Burned"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {fire_label}")
        plt.tight_layout()
        cm_path = MAP_DIR / f"validation_cm_{fire_label.lower().replace(' ', '_')}.png"
        MAP_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=150)
        print(f"  Confusion matrix saved: {cm_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

OUTCOME_STYLE = {
    (1, 1): ("#d62728", "Burned + Predicted High (TP)"),
    (1, 0): ("#ff7f0e", "Burned + Predicted Low/Med (FN — missed)"),
    (0, 1): ("#1f77b4", "Not Burned + Predicted High (FP)"),
    (0, 0): ("#c7c7c7", "Not Burned + Predicted Low/Med (TN)"),
}


def _style_fn(feature):
    props = feature["properties"]
    burned = int(props.get("actually_burned", 0))
    high = int(props.get("predicted_tier", -1) == 2)
    color, _ = OUTCOME_STYLE.get((burned, high), ("#c7c7c7", ""))
    return {"fillColor": color, "color": "white", "weight": 0.3, "fillOpacity": 0.65}


def build_map(tracts, fires, output_path):
    tracts_wgs = tracts.to_crs("EPSG:4326")
    fires_wgs = fires.to_crs("EPSG:4326")

    m = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles="CartoDB positron")

    keep_cols = ["tract", "actually_burned", "predicted_tier", "predicted_label", "geometry"]
    export_cols = [c for c in keep_cols if c in tracts_wgs.columns]
    plot_data = tracts_wgs[export_cols].dropna(subset=["predicted_tier"])

    folium.GeoJson(
        plot_data.__geo_interface__,
        style_function=_style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in ["tract", "predicted_label", "actually_burned"] if c in plot_data.columns],
            aliases=["Tract", "Predicted Risk", "Burned (1=Yes)"],
        ),
    ).add_to(m)

    folium.GeoJson(
        fires_wgs.__geo_interface__,
        style_function=lambda _: {
            "color": "red", "weight": 2, "fillOpacity": 0.05, "dashArray": "4",
        },
        name="Fire Perimeter",
        tooltip="Historic fire perimeter",
    ).add_to(m)

    legend_html = (
        "<div style='position:fixed;bottom:40px;left:40px;z-index:1000;"
        "background:white;padding:12px;border-radius:6px;font-size:13px;"
        "box-shadow:2px 2px 6px rgba(0,0,0,0.3)'><b>Historic Validation</b><br>"
    )
    for (b, h), (color, label) in OUTCOME_STYLE.items():
        legend_html += (
            f"<span style='background:{color};display:inline-block;"
            f"width:14px;height:14px;margin-right:6px;border-radius:2px'></span>"
            f"{label}<br>"
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Map saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate risk model against historic CAL FIRE perimeters"
    )
    parser.add_argument("--fire", help="Fire name (e.g. WOOLSEY, EATON, CREEK)")
    parser.add_argument("--year", type=int, help="Fire year (e.g. 2018)")
    parser.add_argument(
        "--out",
        help="Output HTML map path (default: ../../../outputs/maps/04_validation_<fire>.html)",
    )
    args = parser.parse_args()

    rf, scaler = load_model()
    tracts = load_tracts()
    fires = load_fire_perimeters()

    if args.fire or args.year:
        fires = filter_fires(fires, args.fire, args.year)

    fire_label = args.fire or "all_la_fires"
    if args.year:
        fire_label = f"{fire_label}_{args.year}"

    tracts = tag_burned_tracts(tracts, fires)
    tracts = predict_risk(tracts, rf, scaler)
    evaluate(tracts, fire_label)

    map_path = Path(args.out) if args.out else MAP_DIR / f"04_validation_{fire_label.lower()}.html"
    build_map(tracts, fires, map_path)


if __name__ == "__main__":
    main()
