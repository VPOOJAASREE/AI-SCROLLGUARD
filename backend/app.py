from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Allows frontend to call Flask API

# ✅ Load trained ML model (MAKE SURE FILE EXISTS IN backend FOLDER)
risk_model = joblib.load("risk_model.pkl")

# -----------------------------
# Serve index.html
# -----------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# -----------------------------
# Storage (replace with DB if needed)
# -----------------------------
records = []
streak = 0
best_streak = 0
green_days = 0
red_days = 0

# -----------------------------
# Add daily entry
# -----------------------------
@app.route("/add", methods=["POST"])
def add_entry():
    global streak, best_streak, green_days, red_days

    data = request.get_json()
    usage = int(data.get("usage"))
    task = data.get("task")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status = "GREEN" if usage <= 60 else "RED"

    if status == "GREEN":
        streak += 1
        green_days += 1
    else:
        streak = 0
        red_days += 1

    best_streak = max(best_streak, streak)

    record = {
        "time": now,
        "usage_minutes": usage,
        "task": task,
        "status": status
    }
    records.append(record)

    return jsonify({"alert": status})

# -----------------------------
# Dashboard data
# -----------------------------
@app.route("/dashboard")
def dashboard():
    total_time = sum(r["usage_minutes"] for r in records)
    average_time = total_time / len(records) if records else 0

    data = {
        "total_time": total_time,
        "average_time": average_time,
        "records": records,
        "streak": streak,
        "best": best_streak,
        "green_days": green_days,
        "red_days": red_days
    }
    return jsonify(data)

# -----------------------------
# ✅ ML RISK PREDICTION API
# -----------------------------

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    data = request.get_json()

    instagram = int(data.get("instagram", 0))
    youtube = int(data.get("youtube", 0))
    whatsapp = int(data.get("whatsapp", 0))
    study = int(data.get("study", 0))
    night_usage = int(data.get("night_usage", 0))
    red_days_input = int(data.get("red_days", 0))

    total_social = instagram + youtube + whatsapp
    social_vs_study = total_social / (study + 1)

    X = np.array([[total_social, night_usage, social_vs_study, red_days_input]])

    prediction = risk_model.predict(X)[0]

    if prediction == 0:
        risk = "LOW"
    elif prediction == 1:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return jsonify({
        "risk": risk,
        "total_social": total_social,
        "night_usage": night_usage,
        "red_days": red_days_input
    })

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
