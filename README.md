# Wildfire Evacuation Route & Risk Optimization

A Machine Learning project designed to identify high-risk communities and optimize evacuation routes during wildfire events in Los Angeles County.
The problems divide in 3 sub-problems 
1. Risk Classifier for Vulnerable Neighborhoods
2. Traffic Congestion Prediction 
3. Route Optimization

The system produces an actionable, data-driven vulnerability assessment across all 2,493 LA County census tracts.

---

## ML Pipeline

### Features

**SVI Composite & Themes (from CDC)**
- `rpl_theme_1` — socioeconomic status
- `rpl_theme_2` — household characteristics
- `rpl_theme_3` — racial & ethnic minority status
- `rpl_theme_4` — housing type & transportation

**Engineered Spatial Features**
- `dist_fire_station_m` — distance from tract centroid to nearest fire station
- `dist_hospital_m` — distance from tract centroid to nearest hospital
- `hydrant_count` — DWP hydrants within tract polygon
- `hydrant_density` — hydrant count / tract area (km²)
- `road_length_m` — total drivable road length clipped to tract
- `road_density` — road length / tract area (km²)
- `tract_area_km2` — tract area

### Risk Classifier

- **Model:** Random Forest (`n_estimators=200`, `max_depth=10`, `class_weight=balanced`)
- **Target:** 3-class risk label — Low [0–0.33), Moderate [0.33–0.66), High [0.66–1.0]
- **Validation:** 5-fold cross-validation, 80/20 train-test split
- **Explainability:** SHAP feature importance

<!-- ### Evacuation Routing

- **Graph:** OSMnx drivable network filtered to motorway → tertiary roads (26,716 nodes, 61,375 edges), projected to EPSG:3310
- **Algorithm:** Dijkstra on `travel_time_s` edge weights
- **Risk penalty:** 5× weight multiplier applied to edges in high-risk tracts

--- -->

## Setup & Usage

### 1. Install dependencies

```bash
# Python >= 3.10 recommended
pip install -r requirements.txt
```

### 2. Process raw data (run once)

Run notebooks in `src/data_processing/` in any order — each is independent:

### 3. Run the ML pipeline

Run notebooks in `notebooks/` in order:
```
1. exploration.ipynb        — understand the data
2. feature_engineering.ipynb — build features.geojson
3. model_trainning.ipynb    — train model, save risk_classifier.pkl
4. implementation/visualizations.ipynb     — generate maps in outputs/maps/
5. implementation/predict.py            — compare model to historic wildfire data wjtb generate maps in outputs/maps/
6.  implementation/simulate_fire.py         — compare model to random spawn wildfire data wjtb generate maps in outputs/maps/
```

### 4. View outputs
Open any of the generated HTML maps directly in a browser:
```
outputs/maps/01_risk_choropleth.html
outputs/maps/02_risk_with_infrastructure.html
outputs/maps/03_evacuation_route.html
```