"""
Core prediction logic. Returns structured data for the GUI.
"""
import hashlib
import logging
import pickle
import sys
from pathlib import Path

# Base path: use app support when bundled (writable), else script directory
if getattr(sys, "frozen", False):
    _BASE = Path.home() / "Library" / "Application Support" / "F1 Winner Predictor"
    _BASE.mkdir(parents=True, exist_ok=True)
else:
    _BASE = Path(__file__).parent

import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Suppress fastf1 verbose logging
logging.getLogger("fastf1").setLevel(logging.WARNING)

CACHE_DIR = _BASE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Map historical team names to their 2026 successors for points inheritance
TEAM_LINEAGE = {
    "Kick Sauber": "Audi",
    "Alfa Romeo": "Audi",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
}

# 2026 F1 driver lineup (official FIA-confirmed numbers – formula1.com)
LINEUP_2026 = [
    {"DriverNumber": 3,  "Abbreviation": "VER", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 6,  "Abbreviation": "HAD", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 30, "Abbreviation": "LAW", "TeamName": "Racing Bulls"},
    {"DriverNumber": 41, "Abbreviation": "LIN", "TeamName": "Racing Bulls"},
    {"DriverNumber": 44, "Abbreviation": "HAM", "TeamName": "Ferrari"},
    {"DriverNumber": 16, "Abbreviation": "LEC", "TeamName": "Ferrari"},
    {"DriverNumber": 63, "Abbreviation": "RUS", "TeamName": "Mercedes"},
    {"DriverNumber": 12, "Abbreviation": "ANT", "TeamName": "Mercedes"},
    {"DriverNumber": 1,  "Abbreviation": "NOR", "TeamName": "McLaren"},
    {"DriverNumber": 81, "Abbreviation": "PIA", "TeamName": "McLaren"},
    {"DriverNumber": 14, "Abbreviation": "ALO", "TeamName": "Aston Martin"},
    {"DriverNumber": 18, "Abbreviation": "STR", "TeamName": "Aston Martin"},
    {"DriverNumber": 10, "Abbreviation": "GAS", "TeamName": "Alpine"},
    {"DriverNumber": 43, "Abbreviation": "COL", "TeamName": "Alpine"},
    {"DriverNumber": 31, "Abbreviation": "OCO", "TeamName": "Haas F1 Team"},
    {"DriverNumber": 87, "Abbreviation": "BEA", "TeamName": "Haas F1 Team"},
    {"DriverNumber": 23, "Abbreviation": "ALB", "TeamName": "Williams"},
    {"DriverNumber": 55, "Abbreviation": "SAI", "TeamName": "Williams"},
    {"DriverNumber": 27, "Abbreviation": "HUL", "TeamName": "Audi"},
    {"DriverNumber": 5,  "Abbreviation": "BOR", "TeamName": "Audi"},
    {"DriverNumber": 11, "Abbreviation": "PER", "TeamName": "Cadillac"},
    {"DriverNumber": 77, "Abbreviation": "BOT", "TeamName": "Cadillac"},
]


def load_data(years=(2022, 2023, 2024, 2025), progress_callback=None):
    # Check multiple locations for data.csv
    candidates = [
        _BASE / "data.csv",
        Path.cwd() / "data.csv",  # Current working directory
    ]
    if getattr(sys, "frozen", False):
        # .app/F1 Winner Predictor.app/executable -> check project root (parent of dist)
        app_dir = Path(sys.executable).resolve().parent
        candidates.append(app_dir.parent / "data.csv")   # dist/
        candidates.append(app_dir.parent.parent / "data.csv")  # project root
    for csv_path in candidates:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and "Year" in df.columns and len(df) >= 50:
                    return df
            except Exception:
                pass

    csv_path = _BASE / "data.csv"
    records = []
    for year in years:
        if progress_callback:
            progress_callback(f"Loading {year} season...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception:
            continue
        for rnd in schedule["RoundNumber"]:
            if progress_callback:
                progress_callback(f"Loading {year} round {int(rnd)}...")
            try:
                session = fastf1.get_session(year, int(rnd), "R")
                session.load(laps=False, telemetry=False)
            except Exception:
                continue

            res = session.results[
                [
                    "DriverNumber",
                    "Abbreviation",
                    "TeamName",
                    "GridPosition",
                    "Position",
                    "Points",
                ]
            ].copy()
            res["Year"] = year
            res["Round"] = int(rnd)
            records.append(res)

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)
    df.sort_values(["Year", "Round"], inplace=True)
    df["DriverPointsBefore"] = df.groupby("DriverNumber")["Points"].cumsum() - df["Points"]
    df["TeamPointsBefore"] = df.groupby("TeamName")["Points"].cumsum() - df["Points"]
    df = df.dropna(subset=["GridPosition", "DriverNumber", "Position"])
    df.to_csv(csv_path, index=False)
    return df


def _team_points_with_lineage(df: pd.DataFrame) -> dict:
    """
    Sum team points, merging historical names into their 2026 successors.
    e.g. Kick Sauber + Alfa Romeo points → Audi
    """
    raw = df.groupby("TeamName")["Points"].sum().to_dict()
    merged = {}
    for team, pts in raw.items():
        target = TEAM_LINEAGE.get(team, team)
        merged[target] = merged.get(target, 0) + pts
    # Keep originals too so non-2026 lookups still work
    for team, pts in raw.items():
        if team not in merged:
            merged[team] = pts
    return merged


def get_lineup_for_next_round(df: pd.DataFrame, next_year: int, features: list) -> pd.DataFrame:
    if next_year == 2026:
        lineup = pd.DataFrame(LINEUP_2026)
    else:
        prev = df[df["Year"] == next_year - 1]
        if prev.empty:
            lineup = pd.DataFrame(LINEUP_2026)
        else:
            lineup = (
                prev.groupby(["DriverNumber", "Abbreviation", "TeamName"])
                .tail(1)
                [["DriverNumber", "Abbreviation", "TeamName"]]
                .reset_index(drop=True)
            )

    driver_pts_by_abbr = df.groupby("Abbreviation")["Points"].sum()
    lineup["DriverPointsBefore"] = lineup["Abbreviation"].map(driver_pts_by_abbr).fillna(0)

    if next_year == 2026:
        team_totals = _team_points_with_lineage(df)
        lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_totals).fillna(0)
    else:
        team_totals = df.groupby("TeamName")["Points"].sum()
        lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_totals).fillna(0)
    lineup["GridPosition"] = 0

    return lineup


def build_model():
    categorical = ["Abbreviation", "TeamName"]
    numeric = ["GridPosition", "DriverNumber", "DriverPointsBefore", "TeamPointsBefore"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_dist = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=4,
        cv=3,
        n_jobs=1,
        scoring="accuracy",
        random_state=42,
    )

    return search


def _data_fingerprint(df) -> str:
    """Hash of data shape to detect changes."""
    key = f"{len(df)}_{int(df['Year'].max())}_{int(df['Round'].max())}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run_predictions(progress_callback=None, target_year=2026):
    """
    Run full prediction pipeline. Returns dict with results or error.
    target_year forces prediction for a specific season (default: 2026).
    progress_callback(status: str) is optional for GUI updates.
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        report("Loading race data...")
        df = load_data(progress_callback=progress_callback)
        if df is None or df.empty:
            return {"error": "No race data available. Check your internet connection."}

        df["Winner"] = (df["Position"] == 1).astype(int)
        last_year = int(df["Year"].max())
        last_round = int(df[df["Year"] == last_year]["Round"].max())
        fingerprint = _data_fingerprint(df)

        features = [
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "DriverNumber",
            "DriverPointsBefore",
            "TeamPointsBefore",
        ]

        train_df = df[~((df["Year"] == last_year) & (df["Round"] == last_round))]
        test_df = df[(df["Year"] == last_year) & (df["Round"] == last_round)]

        # Load cached model if data unchanged
        model_path = _BASE / "model_cache.pkl"
        best_model = None
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fingerprint:
                    best_model = cached["model"]
                    report("Using cached model...")
            except Exception:
                pass

        if best_model is None:
            report("Training model...")
            model = build_model()
            model.fit(train_df[features], train_df["Winner"])
            best_model = model.best_estimator_
            try:
                with open(model_path, "wb") as f:
                    pickle.dump({"fingerprint": fingerprint, "model": best_model}, f)
            except Exception:
                pass

        # Resolve race names from schedule
        try:
            last_schedule = fastf1.get_event_schedule(last_year, include_testing=False)
            last_race_event = last_schedule[last_schedule["RoundNumber"] == last_round]
            last_race_name = last_race_event.iloc[0]["EventName"] if not last_race_event.empty else f"Round {last_round}"
        except Exception:
            last_race_name = f"Round {last_round}"

        # Last race predictions
        probs = best_model.predict_proba(test_df[features])[:, 1]
        test_df = test_df.copy()
        test_df["WinProbability"] = probs
        last_race_preds = [
            {
                "abbreviation": row["Abbreviation"],
                "team": row["TeamName"],
                "probability": float(row["WinProbability"]),
            }
            for _, row in test_df.sort_values("WinProbability", ascending=False).iterrows()
        ]
        pred_winner = last_race_preds[0]
        actual_winner = test_df[test_df["Winner"] == 1]
        actual_abbr = actual_winner.iloc[0]["Abbreviation"] if not actual_winner.empty else "—"

        # Next round – find the actual next upcoming race by date
        import datetime
        report("Predicting next race...")
        next_race_name = None
        today = datetime.date.today()

        def _find_next_race(year):
            """Find the next race in `year` that hasn't happened yet."""
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            upcoming = schedule[schedule["EventDate"].dt.date >= today]
            if not upcoming.empty:
                evt = upcoming.iloc[0]
                return year, int(evt["RoundNumber"]), evt["EventName"]
            return None

        found = None
        if target_year and target_year > last_year:
            try:
                found = _find_next_race(target_year)
            except Exception:
                pass
        if not found:
            try:
                found = _find_next_race(last_year)
            except Exception:
                pass
        if not found:
            try:
                found = _find_next_race(last_year + 1)
            except Exception:
                pass

        if found:
            next_year, next_round, next_race_name = found
        else:
            next_year = target_year or last_year + 1
            next_round = 1
            next_race_name = f"Round {next_round}"

        lineup = get_lineup_for_next_round(df, next_year, features)
        next_probs = best_model.predict_proba(lineup[features])[:, 1]
        lineup["WinProbability"] = next_probs
        lineup = lineup.sort_values("WinProbability", ascending=False)

        next_race_preds = [
            {
                "abbreviation": row["Abbreviation"],
                "team": row["TeamName"],
                "probability": float(row["WinProbability"]),
            }
            for _, row in lineup.iterrows()
        ]

        # Accuracy
        X = df[features]
        y = df["Winner"]
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        accuracy = float(best_model.score(X_te, y_te))

        # Feature importance for algorithm viz
        try:
            clf = best_model.named_steps["classifier"]
            pre = best_model.named_steps["preprocess"]
            names = pre.get_feature_names_out()
            imp = clf.feature_importances_
            # Aggregate by base feature (e.g. TeamName_* -> TeamName)
            base_imp = {}
            for n, v in zip(names, imp):
                base = n.split("__")[-1].rsplit("_", 1)[0] if "_" in n.split("__")[-1] else n.split("__")[-1]
                base_imp[base] = base_imp.get(base, 0) + float(v)
            feature_importance = {k: float(v) for k, v in sorted(base_imp.items(), key=lambda x: -x[1])}
        except Exception:
            feature_importance = {}

        # Build full schedule for race cycling
        schedule_list = []
        try:
            sched = fastf1.get_event_schedule(next_year, include_testing=False)
            for _, row in sched.iterrows():
                schedule_list.append({
                    "round": int(row["RoundNumber"]),
                    "name": row["EventName"],
                    "date": str(row["EventDate"].date()),
                })
        except Exception:
            pass

        return {
            "feature_importance": feature_importance,
            "last_race": {
                "year": last_year,
                "round": last_round,
                "name": last_race_name,
                "predicted_winner": pred_winner["abbreviation"],
                "actual_winner": actual_abbr,
                "predictions": last_race_preds,
            },
            "next_race": {
                "year": next_year,
                "round": next_round,
                "name": next_race_name,
                "predicted_winner": next_race_preds[0]["abbreviation"],
                "top_probability": next_race_preds[0]["probability"],
                "predictions": next_race_preds,
            },
            "schedule": schedule_list,
            "accuracy": accuracy,
            "_model": best_model,
            "_features": features,
            "_base_lineup": lineup[["DriverNumber", "Abbreviation", "TeamName"]].to_dict("records"),
            "_base_driver_pts": lineup.set_index("Abbreviation")["DriverPointsBefore"].to_dict(),
            "_base_team_pts": lineup.drop_duplicates("TeamName").set_index("TeamName")["TeamPointsBefore"].to_dict(),
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}


F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def predict_with_standings(model, features, base_lineup, driver_pts, team_pts):
    """Re-run predictions with updated championship standings."""
    lineup = pd.DataFrame(base_lineup)
    lineup["DriverPointsBefore"] = lineup["Abbreviation"].map(driver_pts).fillna(0)
    lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_pts).fillna(0)
    lineup["GridPosition"] = 0

    probs = model.predict_proba(lineup[features])[:, 1]
    lineup["WinProbability"] = probs
    lineup = lineup.sort_values("WinProbability", ascending=False)

    return [
        {
            "abbreviation": row["Abbreviation"],
            "team": row["TeamName"],
            "probability": float(row["WinProbability"]),
        }
        for _, row in lineup.iterrows()
    ]


def run_predictions_all_races(progress_callback=None):
    """
    Run the algorithm for every race: train on all except target race, predict target.
    Returns list of {year, round, predicted, actual, correct} and overall accuracy.
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        report("Loading race data...")
        df = load_data(progress_callback=progress_callback)
        if df is None or df.empty:
            return {"error": "No race data available. Ensure data.csv exists or check your internet connection."}

        df["Winner"] = (df["Position"] == 1).astype(int)
        features = [
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "DriverNumber",
            "DriverPointsBefore",
            "TeamPointsBefore",
        ]

        races = df.groupby(["Year", "Round"])
        results = []
        total = len(races)
        correct = 0

        schedule_cache = {}

        for idx, ((year, rnd), race_df) in enumerate(races):
            report(f"Race {idx + 1}/{total}: {year} R{rnd}...")
            train_df = df[~((df["Year"] == year) & (df["Round"] == rnd))]
            test_df = race_df.copy()

            if len(train_df) < 100 or len(test_df) < 2:
                continue

            model = build_model()
            model.fit(train_df[features], train_df["Winner"])
            best = model.best_estimator_

            probs = best.predict_proba(test_df[features])[:, 1]
            test_df = test_df.copy()
            test_df["WinProbability"] = probs
            pred_row = test_df.sort_values("WinProbability", ascending=False).iloc[0]
            pred_abbr = pred_row["Abbreviation"]
            actual_row = test_df[test_df["Winner"] == 1]
            actual_abbr = actual_row.iloc[0]["Abbreviation"] if not actual_row.empty else "—"
            hit = pred_abbr == actual_abbr
            if hit:
                correct += 1

            race_name = f"Round {int(rnd)}"
            try:
                if year not in schedule_cache:
                    schedule_cache[year] = fastf1.get_event_schedule(int(year), include_testing=False)
                sched = schedule_cache[year]
                evt = sched[sched["RoundNumber"] == int(rnd)]
                if not evt.empty:
                    race_name = evt.iloc[0]["EventName"]
            except Exception:
                pass

            results.append({
                "year": int(year),
                "round": int(rnd),
                "name": race_name,
                "predicted": pred_abbr,
                "actual": actual_abbr,
                "correct": hit,
            })

        accuracy = correct / len(results) if results else 0
        return {
            "all_races": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}
