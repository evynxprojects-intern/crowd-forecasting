
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import xgboost as xgb
from math import radians, sin, cos, sqrt, atan2
import os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load Dataset ──────────────────────────────────
print("Loading dataset...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "india_crowd_forecast_final.csv"))
print(f"Dataset loaded: {df.shape}")

# ── Feature Setup ─────────────────────────────────
FEATURES = [
    "rating","num_reviews","entry_fee_inr","entry_free",
    "time_needed_hrs","age_years","dslr_allowed","airport_nearby",
    "busyness_avg","busyness_max","morning_avg","afternoon_avg",
    "evening_avg","weekday_avg","weekend_avg","weekend_premium",
    "peak_hour","peak_is_weekend",
    "mon_avg","tue_avg","wed_avg","thu_avg","fri_avg","sat_avg","sun_avg",
    "typical_open_hour","typical_close_hour",
    "is_museum","is_religious","is_heritage","is_park","is_beach",
    "is_nature","is_wildlife","is_hill_station","is_market",
    "is_cave","is_amusement_park","is_viewpoint","is_tourist_spot",
    "best_morning","best_afternoon","best_evening",
    "best_all_year","best_monsoon","best_winter","best_summer",
    "zone_encoded","same_type_in_city","total_places_in_city","popularity_score",
]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = "crowd_label_final"

X = df[FEATURES].copy()
y = df[TARGET].copy()
for col in X.columns:
    if X[col].dtype in ["float64","int64"]:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(0)
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# ── Train Model ───────────────────────────────────
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=2,
    min_samples_leaf=1, max_features="sqrt",
    class_weight="balanced", random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=42, multi_class="multinomial"))])
lr_pipe.fit(X_train, y_train)

model = VotingClassifier(
    estimators=[("xgb",xgb_model),("rf",rf_model),("lr",lr_pipe)],
    voting="soft", weights=[3,2,1])
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model ready! Accuracy: {acc*100:.2f}%")

# ── Recommendation Engine ─────────────────────────
SIM_FEATURES = [f for f in [
    "is_museum","is_religious","is_heritage","is_park","is_beach",
    "is_nature","is_wildlife","is_hill_station","is_market","is_cave",
    "is_amusement_park","is_viewpoint","is_tourist_spot",
    "entry_free","time_needed_hrs","dslr_allowed","airport_nearby",
    "best_morning","best_afternoon","best_evening","best_all_year",
    "best_monsoon","best_winter","best_summer","zone_encoded",
] if f in df.columns]

sim_sc     = StandardScaler()
sim_scaled = sim_sc.fit_transform(df[SIM_FEATURES].fillna(0))
sim_matrix = cosine_similarity(sim_scaled)
print("Recommendation engine ready!")

# ── Helper Functions ──────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    dlat=lat2-lat1; dlon=lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

def get_alternatives(place_idx, crowd_pred, top_n=3):
    place      = df.loc[place_idx]
    sim_scores = list(enumerate(sim_matrix[place_idx]))
    candidates = []
    for i, score in sim_scores:
        if i == place_idx: continue
        c        = df.loc[i]
        priority = 0
        if c["city"]  == place["city"]:    priority += 100
        elif c["state"] == place["state"]: priority += 50
        if c["crowd_label_final"] < crowd_pred:    priority += 80
        elif c["crowd_label_final"] == crowd_pred: priority += 20
        else: priority -= 50
        dist = None
        try:
            if all(pd.notna([place.get("latitude"), place.get("longitude"),
                             c.get("latitude"),     c.get("longitude")])):
                dist = haversine(
                    float(place["latitude"]),  float(place["longitude"]),
                    float(c["latitude"]),      float(c["longitude"]))
                if dist<=50:   priority+=40
                elif dist<=100: priority+=20
                elif dist<=200: priority+=10
        except: pass
        priority += score * 30
        candidates.append({
            "name"       : str(c["place_name"]),
            "type"       : str(c["place_type"]),
            "city"       : str(c["city"]),
            "state"      : str(c["state"]),
            "rating"     : float(c.get("rating",0) or 0),
            "crowd_level": str(c["crowd_level_final"]),
            "crowd_label": int(c["crowd_label_final"]),
            "entry_free" : int(c.get("entry_free",0) or 0),
            "entry_fee"  : float(c.get("entry_fee_inr",0) or 0),
            "best_time"  : str(c.get("best_time","")),
            "distance_km": round(dist,1) if dist else None,
            "priority"   : float(priority),
            "loc_tag"    : ("Same city"    if c["city"]==place["city"]   else
                            "Same state"   if c["state"]==place["state"] else
                            "Nearby region"),
        })
    candidates = sorted(candidates, key=lambda x: x["priority"], reverse=True)
    low = [c for c in candidates if c["crowd_label"] < crowd_pred]
    return (low if low else candidates)[:top_n]

# ── Routes ────────────────────────────────────────
@app.route("/")
def index():
    places     = sorted(df["place_name"].tolist())
    states     = sorted(df["state"].dropna().unique().tolist())
    types      = sorted(df["place_type"].dropna().unique().tolist())
    return render_template("index.html",
        places=places, states=states, types=types,
        total_places=len(df),
        total_states=int(df["state"].nunique()),
        total_cities=int(df["city"].nunique()),
        model_accuracy=round(acc*100,1),
        total_low=int((df["crowd_level_final"]=="Low").sum()),
        total_med=int((df["crowd_level_final"]=="Medium").sum()),
        total_high=int((df["crowd_level_final"]=="High").sum()))

@app.route("/api/predict", methods=["POST"])
def predict():
    data        = request.json
    place_name  = data.get("place_name","")
    day         = data.get("day","Saturday")
    time_of_day = data.get("time_of_day","Afternoon")

    matches = df[df["place_name"].str.lower().str.contains(
        place_name.lower(), na=False)]
    if len(matches) == 0:
        return jsonify({"error": f"Place not found: {place_name}"})

    place     = matches.iloc[0]
    place_idx = matches.index[0]
    feats     = X.loc[place_idx:place_idx]
    pred      = int(model.predict(feats)[0])
    proba     = model.predict_proba(feats)[0]

    adj_pred = pred
    if day in ["Saturday","Sunday"] and pred < 2: adj_pred = min(pred+1,2)
    if time_of_day == "Morning"     and pred > 0: adj_pred = max(pred-1,0)
    if time_of_day == "Evening"     and pred < 2: adj_pred = min(pred+1,2)

    day_col  = {"Monday":"mon_avg","Tuesday":"tue_avg","Wednesday":"wed_avg",
                "Thursday":"thu_avg","Friday":"fri_avg","Saturday":"sat_avg",
                "Sunday":"sun_avg"}.get(day,"weekday_avg")
    time_col = {"Morning":"morning_avg","Afternoon":"afternoon_avg",
                "Evening":"evening_avg"}.get(time_of_day,"busyness_avg")

    alts = get_alternatives(place_idx, adj_pred, top_n=3)

    return jsonify({
        "place": {
            "name"       : str(place["place_name"]),
            "type"       : str(place["place_type"]),
            "city"       : str(place["city"]),
            "state"      : str(place["state"]),
            "rating"     : float(place.get("rating",0) or 0),
            "entry_fee"  : "Free" if place.get("entry_free",0)
                           else f"₹{place.get('entry_fee_inr',0):.0f}",
            "best_time"  : str(place.get("best_time","N/A")),
            "time_needed": float(place.get("time_needed_hrs",0) or 0),
            "dslr"       : bool(place.get("dslr_allowed",0)),
            "airport"    : bool(place.get("airport_nearby",0)),
        },
        "prediction": {
            "crowd_level"  : {0:"Low",1:"Medium",2:"High"}[adj_pred],
            "base_level"   : {0:"Low",1:"Medium",2:"High"}[pred],
            "confidence"   : {
                "Low"   : round(float(proba[0])*100,1),
                "Medium": round(float(proba[1])*100,1),
                "High"  : round(float(proba[2])*100,1),
            },
            "busyness_avg" : float(place.get("busyness_avg",0) or 0),
            "day_busyness" : float(place.get(day_col,0) or 0),
            "time_busyness": float(place.get(time_col,0) or 0),
            "busiest_day"  : max(
                ["mon_avg","tue_avg","wed_avg","thu_avg",
                 "fri_avg","sat_avg","sun_avg"],
                key=lambda c: float(place.get(c,0) or 0)
            ).replace("_avg","").title(),
            "quietest_day" : min(
                ["mon_avg","tue_avg","wed_avg","thu_avg",
                 "fri_avg","sat_avg","sun_avg"],
                key=lambda c: float(place.get(c,0) or 0)
            ).replace("_avg","").title(),
        },
        "recommendations": alts,
        "day"           : day,
        "time_of_day"   : time_of_day,
    })

@app.route("/api/dashboard")
def dashboard():
    return jsonify({
        "type_dist" : df["place_type"].value_counts().to_dict(),
        "crowd_dist": df["crowd_level_final"].value_counts().to_dict(),
        "state_dist": df["state"].value_counts().head(10).to_dict(),
        "day_busy"  : {
            "Mon": float(df["mon_avg"].mean()),
            "Tue": float(df["tue_avg"].mean()),
            "Wed": float(df["wed_avg"].mean()),
            "Thu": float(df["thu_avg"].mean()),
            "Fri": float(df["fri_avg"].mean()),
            "Sat": float(df["sat_avg"].mean()),
            "Sun": float(df["sun_avg"].mean()),
        },
        "stats": {
            "total_places"  : int(len(df)),
            "total_states"  : int(df["state"].nunique()),
            "total_cities"  : int(df["city"].nunique()),
            "model_accuracy": round(acc*100,1),
        }
    })

@app.route("/api/search")
def search():
    q       = request.args.get("q","").lower()
    state   = request.args.get("state","")
    ptype   = request.args.get("type","")
    crowd   = request.args.get("crowd","")
    results = df.copy()
    if q:     results = results[results["place_name"].str.lower().str.contains(q,na=False)]
    if state: results = results[results["state"]==state]
    if ptype: results = results[results["place_type"]==ptype]
    if crowd: results = results[results["crowd_level_final"]==crowd]
    out = []
    for _, r in results.head(50).iterrows():
        out.append({
            "name"       : str(r["place_name"]),
            "type"       : str(r["place_type"]),
            "city"       : str(r["city"]),
            "state"      : str(r["state"]),
            "rating"     : float(r.get("rating",0) or 0),
            "crowd_level": str(r["crowd_level_final"]),
            "entry_free" : bool(r.get("entry_free",0)),
            "entry_fee"  : float(r.get("entry_fee_inr",0) or 0),
            "best_time"  : str(r.get("best_time","")),
        })
    return jsonify({"results": out, "count": len(out)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
