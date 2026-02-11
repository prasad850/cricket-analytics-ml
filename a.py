# app_advanced_single_file.py
# Features:
# - Structured logging (dictConfig) and app.logger usage
# - Env-based config via python-dotenv
# - Rate limiting (Flask-Limiter) with global and per-route limits
# - Robust HTTP requests with retry/backoff
# - Gemini API integration if API key present
# - Excel ingestion with pandas and safe cleaning
# - JSON APIs + embedded Jinja2 templates (DictLoader)
# - Team comparison and player prediction endpoints
# - Health and matches endpoints

import os
import json
import random
import logging
from logging.config import dictConfig
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from dotenv import load_dotenv

# Optional: Gemini SDK
try:
    import google.generativeai as genai
except Exception:  # SDK optional at runtime
    genai = None

# ML Models
try:
    from ml_models import ml_predictor
except ImportError:
    ml_predictor = None


# -----------------------------
# Environment & Configuration
# -----------------------------
load_dotenv()  # Loads .env if present; never commit secrets to source control

class Config:
    APP_NAME = os.getenv("APP_NAME", "CricketInsights")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # Files
    BATSMEN_XLSX = os.getenv("BATSMEN_XLSX", "IPLBAT4.xlsx")
    TEAM_XLSX = os.getenv("TEAM_XLSX", "ipl.xlsx")

    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

    # Cricket API (completed matches, etc.)
    CRICAPI_URL = os.getenv("CRICAPI_URL", "").strip()
    CRICAPI_KEY = os.getenv("CRICAPI_KEY", "").strip()

    # Rate limits
    # Examples: "200 per day;60 per hour"
    GLOBAL_LIMITS = os.getenv("GLOBAL_LIMITS", "200 per day;60 per hour")
    MATCHES_LIMIT = os.getenv("MATCHES_LIMIT", "10 per minute")
    PLAYER_LIMIT = os.getenv("PLAYER_LIMIT", "20 per minute")

    # HTTP timeouts/retries
    HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
    RETRY_TOTAL = int(os.getenv("RETRY_TOTAL", "3"))
    RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1"))
    RETRY_STATUS = os.getenv("RETRY_STATUS", "429,500,502,503,504")


# -----------------------------
# Logging
# -----------------------------
dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
})

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]


# -----------------------------
# Rate Limiter
# -----------------------------
global_limits = [lim.strip() for lim in app.config["GLOBAL_LIMITS"].split(";") if lim.strip()]
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=global_limits or None,  # falls back gracefully
)


# -----------------------------
# HTTP Session with Retries
# -----------------------------
def make_session() -> Session:
    statuses = [int(s.strip()) for s in app.config["RETRY_STATUS"].split(",") if s.strip().isdigit()]
    retry = Retry(
        total=app.config["RETRY_TOTAL"],
        backoff_factor=app.config["RETRY_BACKOFF"],
        status_forcelist=statuses,
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class ScoreLine:
    team: str
    runs: int
    wickets: int
    overs: float

@dataclass
class MatchCard:
    name: str
    teams: List[str]
    venue: str
    status: str
    scores: List[Dict[str, Any]]


# -----------------------------
# Data Loaders
# -----------------------------
def load_batsmen_data() -> pd.DataFrame:
    """
    Reads and tidies the batsmen/bowler sheet; resilient to minor formatting issues.
    """
    path = os.path.join(os.path.dirname(__file__), app.config["BATSMEN_XLSX"])
    df = pd.read_excel(path, sheet_name="Batsmen", skiprows=2, header=0)
    df.columns = [
        "Player", "Role", "Season_Performance", "Team", "Match_5_vs_MI",
        "Match_10_vs_RCB", "Match_14_vs_PBKS", "Match_18_vs_LSG",
        "Match_23_vs_MI", "Match_28_vs_SRH", "Match_31_vs_CSK",
        "Match_32_vs_RCB", "Unused", "Wickets"
    ]
    df = df.dropna(subset=["Player"]).drop(columns=["Unused"], errors="ignore")
    for col in ["Player", "Role", "Team"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    match_cols = [c for c in df.columns if c.startswith("Match")]
    tidy = pd.melt(
        df, id_vars=["Player", "Role", "Team", "Wickets"],
        value_vars=match_cols, var_name="Match", value_name="Performance"
    )
    tidy["Opponent"] = tidy["Match"].str.extract(r'vs_([A-Z]+)')[0]
    tidy["Performance"] = (
        tidy["Performance"].astype(str).str.replace("*", "", regex=False)
        .apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    tidy["Wickets"] = pd.to_numeric(tidy["Wickets"], errors="coerce").fillna(0)

    # Split metrics by role
    tidy["Runs"] = tidy.apply(lambda r: float(r["Performance"]) if "Bowler" not in str(r["Role"]) else 0, axis=1)
    tidy["WicketsMetric"] = tidy.apply(lambda r: float(r["Wickets"]) if "Bowler" in str(r["Role"]) else 0, axis=1)
    return tidy


def load_team_df() -> Optional[pd.DataFrame]:
    try:
        path = os.path.join(os.path.dirname(__file__), app.config["TEAM_XLSX"])
        tdf = pd.read_excel(path, sheet_name="Sheet1")
        cols = [c for c in tdf.columns]
        # Normalize expected columns if present
        if {"Team", "Matches Played", "Wins"}.issubset(set(cols)):
            tdf = tdf[["Team", "Matches Played", "Wins"]].dropna()
            tdf["Win Rate"] = tdf["Wins"] / tdf["Matches Played"]
            return tdf
        return None
    except FileNotFoundError:
        app.logger.warning("Team file not found; team comparison disabled")
        return None


TEAM_DF = load_team_df()


# -----------------------------
# Gemini & Cricket API Fetch
# -----------------------------
def fetch_gemini_matches() -> Dict[str, Any]:
    if not app.config["GEMINI_API_KEY"] or genai is None:
        return {"matches": []}
    try:
        genai.configure(api_key=app.config["GEMINI_API_KEY"])
        model = genai.GenerativeModel(app.config["GEMINI_MODEL_NAME"])
        prompt_text = """
Provide only JSON for the latest live and upcoming cricket matches in the format:
{
  "matches": [
    {
      "name": "Match Name",
      "teams": ["Team 1", "Team 2"],
      "venue": "Venue Name",
      "status": "Live / Upcoming",
      "scores": [
        {"team": "Team 1", "runs": 120, "wickets": 3, "overs": 15.2},
        {"team": "Team 2", "runs": 0, "wickets": 0, "overs": 0}
      ]
    }
  ]
}
Return only JSON with no extra text.
""".strip()
        resp = model.generate_content(prompt_text)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if isinstance(data, dict) and "matches" in data:
            return data
        return {"matches": []}
    except Exception as e:
        app.logger.warning(f"Gemini fetch failed: {e}")
        return {"matches": []}


def fetch_cricapi_matches(session: Session) -> Optional[Dict[str, Any]]:
    url = app.config["CRICAPI_URL"]
    if not url:
        return None
    headers = {}
    if app.config["CRICAPI_KEY"]:
        headers["X-API-KEY"] = app.config["CRICAPI_KEY"]
    try:
        r = session.get(url, headers=headers, timeout=app.config["HTTP_TIMEOUT"])
        if r.status_code == 200:
            return r.json()
        app.logger.warning(f"CricAPI non-200: {r.status_code}")
        return None
    except Exception as e:
        app.logger.warning(f"CricAPI request failed: {e}")
        return None


def process_matches(api_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    matches = {"live": [], "upcoming": [], "completed": []}

    # From Gemini
    gem = api_data.get("gemini", {})
    for m in gem.get("matches", []):
        status = str(m.get("status", "")).lower()
        card = {
            "name": m.get("name", "Match"),
            "teams": m.get("teams", []),
            "venue": m.get("venue", "Venue not available"),
            "scores": m.get("scores", []),
            "status": m.get("status", "Unknown"),
        }
        if "live" in status or "in progress" in status:
            matches["live"].append(card)
        elif "upcoming" in status or "scheduled" in status:
            matches["upcoming"].append(card)

    # From CricAPI
    cr = api_data.get("cricapi")
    if cr and isinstance(cr, dict) and "data" in cr:
        for m in cr["data"]:
            s = str(m.get("status", "")).lower()
            if any(x in s for x in ["completed", "won", "result"]):
                matches["completed"].append({
                    "id": m.get("id", ""),
                    "name": m.get("name", "Match"),
                    "status": m.get("status", "Status not available"),
                    "date": m.get("date", ""),
                    "venue": m.get("venue", "Venue not available"),
                    "teams": m.get("teams", []),
                    "scores": m.get("score", []),
                    "matchType": m.get("matchType", ""),
                    "series": m.get("series", "")
                })
    return matches


# -----------------------------
# Prediction Logic
# -----------------------------
def predict_performance(df: pd.DataFrame, player: str, next_opponent: str) -> str:
    try:
        pdata = df[df["Player"] == player]
        if pdata.empty:
            return "No data available for this player"

        role = str(pdata["Role"].iloc)
        if "Bowler" in role:
            # simple stochastic ranges
            roll = random.randint(1, 6)
            if roll <= 2:
                return "Likely to take 2 wickets (Prediction range: 2–3 wickets)"
            elif roll <= 4:
                return "In good form for 3 wickets (Prediction range: 3–4 wickets)"
            return "On fire! Might take 4 wickets (Prediction range: 4–5 wickets)"
        else:
            # Batsman: blend last-5 average with opponent history
            pdata["Runs"] = pd.to_numeric(pdata["Runs"], errors="coerce").fillna(0)
            last5 = pdata.tail(5)["Runs"].mean()
            vs_subset = pdata[pdata["Opponent"] == next_opponent]
            vs_avg = vs_subset["Runs"].mean() if not vs_subset.empty else last5
            pred = (0.6 * last5) + (0.4 * vs_avg)
            low, high = int(max(0, round(pred * 0.9))), int(round(pred * 1.1))
            if low == high:
                high += 5 if low < 50 else 10
            return f"{low}–{high} runs"
    except Exception as e:
        return f"Prediction error: {str(e)}"


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index2.html", app_name=app.config["APP_NAME"])


@app.route("/player", methods=["GET", "POST"])
@limiter.limit(lambda: app.config["PLAYER_LIMIT"])
def player_prediction():
    try:
        df = load_batsmen_data()

        form_team = request.form.get("team", "")
        form_role = request.form.get("role", "")
        form_player = request.form.get("player", "")
        form_opp = request.form.get("opponent", "")

        teams = sorted(df["Team"].dropna().unique().tolist())
        opponents = sorted(df["Opponent"].dropna().unique().tolist())

        scope = df[df["Team"] == form_team] if form_team else df
        if form_role == "batsman":
            players = sorted(scope[~scope["Role"].astype(str).str.contains("Bowler")]["Player"].unique())
        elif form_role == "bowler":
            players = sorted(scope[scope["Role"].astype(str).str.contains("Bowler")]["Player"].unique())
        else:
            players = sorted(scope["Player"].unique())

        prediction = None
        if form_player and form_opp:
            prediction = predict_performance(df, form_player, form_opp)

        return render_template(
            "player.html",
            teams=teams,
            players=players,
            opponents=opponents,
            team=form_team,
            role=form_role,
            player=form_player,
            opponent=form_opp,
            prediction=prediction
        )
    except Exception as e:
        app.logger.error(f"/player error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/team", methods=["GET", "POST"])
def team_comparison():
    if TEAM_DF is None:
        return jsonify({"error": "Team data not available"}), 503

    prediction = None
    selected_team1 = request.form.get("team1") if request.method == "POST" else None
    selected_team2 = request.form.get("team2") if request.method == "POST" else None
    teams = TEAM_DF["Team"].tolist()

    if request.method == "POST" and selected_team1 and selected_team2 and selected_team1 != selected_team2:
        # Try ML prediction first
        if ml_predictor and ml_predictor.is_trained:
            try:
                df_players = load_batsmen_data()  # Get player data for ML prediction
                ml_prediction = predict_match_outcome_ml(selected_team1, selected_team2, TEAM_DF, df_players)
                
                if "error" not in ml_prediction:
                    prediction = {
                        "winner": ml_prediction["predicted_winner"],
                        "team1_win_percentage": ml_prediction["team1_win_prob"],
                        "team2_win_percentage": ml_prediction["team2_win_prob"],
                        "confidence": f"{ml_prediction['confidence']:.2f}%",
                        "ml_prediction": True,
                        "algorithm_details": {
                            "logistic_regression": ml_prediction["logistic_pred"],
                            "random_forest": ml_prediction["rf_pred"]
                        }
                    }
                else:
                    # Fallback to rule-based
                    raise Exception("ML prediction failed")
                    
            except Exception:
                # Fallback to rule-based prediction
                w1 = TEAM_DF[TEAM_DF["Team"] == selected_team1]["Win Rate"].values[0]
                w2 = TEAM_DF[TEAM_DF["Team"] == selected_team2]["Win Rate"].values[0]
                total = w1 + w2 if (w1 + w2) > 0 else 1e-9
                t1p = round((w1 / total) * 100, 2)
                t2p = round((w2 / total) * 100, 2)
                
                if t1p > t2p:
                    winner = selected_team1
                elif t2p > t1p:
                    winner = selected_team2
                else:
                    winner = "Tie"
                    
                confidence = f"{abs(t1p - t2p):.2f}%"
                prediction = {
                    "winner": winner,
                    "team1_win_percentage": t1p,
                    "team2_win_percentage": t2p,
                    "confidence": confidence,
                    "team1_win_rate": round(w1 * 100, 2),
                    "team2_win_rate": round(w2 * 100, 2),
                    "ml_prediction": False
                }
        else:
            # Rule-based prediction
            w1 = TEAM_DF[TEAM_DF["Team"] == selected_team1]["Win Rate"].values[0]
            w2 = TEAM_DF[TEAM_DF["Team"] == selected_team2]["Win Rate"].values[0]
            total = w1 + w2 if (w1 + w2) > 0 else 1e-9
            t1p = round((w1 / total) * 100, 2)
            t2p = round((w2 / total) * 100, 2)
            
            if t1p > t2p:
                winner = selected_team1
            elif t2p > t1p:
                winner = selected_team2
            else:
                winner = "Tie"
                
            confidence = f"{abs(t1p - t2p):.2f}%"
            prediction = {
                "winner": winner,
                "team1_win_percentage": t1p,
                "team2_win_percentage": t2p,
                "confidence": confidence,
                "team1_win_rate": round(w1 * 100, 2),
                "team2_win_rate": round(w2 * 100, 2),
                "ml_prediction": False
            }

    return render_template(
        "team.html",
        teams=teams,
        prediction=prediction,
        team1=selected_team1,
        team2=selected_team2
    )


@app.route("/matches")
@limiter.limit(lambda: app.config["MATCHES_LIMIT"])
def show_matches():
    session = make_session()
    gemini_data = fetch_gemini_matches()
    cricapi_data = fetch_cricapi_matches(session)
    combined = {"gemini": gemini_data, "cricapi": cricapi_data}
    matches = process_matches(combined)
    return render_template(
        "matches.html",
        matches=matches,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.route("/api/matches")
@limiter.limit(lambda: app.config["MATCHES_LIMIT"])
def api_matches():
    session = make_session()
    gemini_data = fetch_gemini_matches()
    cricapi_data = fetch_cricapi_matches(session)
    combined = {"gemini": gemini_data, "cricapi": cricapi_data}
    matches = process_matches(combined)
    return jsonify(matches)


@app.route("/api/player-prediction", methods=["POST"])
@limiter.limit(lambda: app.config["PLAYER_LIMIT"])
def api_player_prediction():
    try:
        payload = request.get_json(force=True)
        player = payload.get("player")
        opponent = payload.get("opponent")
        if not player or not opponent:
            return jsonify({"error": "player and opponent are required"}), 400
        df = load_batsmen_data()
        pred = predict_performance(df, player, opponent)
        return jsonify({"player": player, "opponent": opponent, "prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "time": datetime.utcnow().isoformat(),
        "ml_models_trained": ml_predictor.is_trained if ml_predictor else False
    })
    # Try to load existing ML models on startup
    if ml_predictor:
        try:
            ml_predictor.load_models()
            if ml_predictor.is_trained:
                app.logger.info("✅ ML models loaded successfully!")
            else:
                app.logger.info("⚠️  No trained ML models found. Use /train-ml-models to train them.")
        except Exception as e:
            app.logger.warning(f"⚠️  Could not load ML models: {e}")
    else:
        app.logger.warning("⚠️  ML predictor not available")


@app.route("/train-ml-models", methods=["POST"])
@limiter.limit("1 per minute")  # Limit training requests
def train_ml_models():
    """
    Endpoint to train ML models
    """
    if not ml_predictor:
        return jsonify({"status": "error", "message": "ML predictor not available"}), 503
        
    try:
        batsmen_file = os.path.join(os.path.dirname(__file__), app.config["BATSMEN_XLSX"])
        team_file = os.path.join(os.path.dirname(__file__), app.config["TEAM_XLSX"])
        
        # Train models
        ml_predictor.train_all_models(batsmen_file, team_file)
        
        # Save models
        ml_predictor.save_models()
        
        return jsonify({
            "status": "success",
            "message": "ML models trained successfully!",
            "models_trained": list(ml_predictor.models.keys()),
            "algorithms_used": [
                "Linear Regression (for runs/wickets prediction)",
                "Random Forest (for runs/wickets prediction)", 
                "Logistic Regression (for match outcomes)",
                "Random Forest Classifier (for match outcomes)"
            ]
        })
        
    except Exception as e:
        app.logger.error(f"ML training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ml-status")


@app.route("/ml-status")
def ml_status():
    """
    Get ML models status and information
    """
    return jsonify({
        "models_trained": ml_predictor.is_trained,
        "available_models": list(ml_predictor.models.keys()) if ml_predictor.is_trained else [],
        "algorithms_info": {
            "player_performance": {
                "runs_prediction": ["Linear Regression", "Random Forest"],
                "wickets_prediction": ["Linear Regression", "Random Forest"],
                "features_used": [
                    "Player role encoding", "Team encoding", "Opponent encoding",
                    "Match number", "Historical averages", "Standard deviations", 
                    "Recent form (last 3 matches)", "Maximum/minimum performance"
                ]
            },
            "match_outcome": {
                "algorithms": ["Logistic Regression", "Random Forest Classifier"],
                "features_used": [
                    "Team win rates", "Net run rates", 
                    "Average team runs", "Average team wickets"
                ]
            }
        },
        "feature_engineering": [
            "Categorical encoding (LabelEncoder)",
            "Feature scaling (StandardScaler)", 
            "Historical statistics aggregation",
            "Recent form calculation (rolling windows)",
            "Cross-validation for model evaluation"
        ]
    })


if __name__ == "__main__":
    app.run(host=app.config["HOST"], port=app.config["PORT"], debug=app.config["DEBUG"])
