import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("risk_dataset.csv")

# 2. Feature engineering
df["total_social"] = df["instagram"] + df["youtube"] + df["whatsapp"]
df["social_vs_study"] = df["total_social"] / (df["study"] + 1)

# 3. Heuristic labeling (LOW / MEDIUM / HIGH)
def assign_label(row):
    score = (row.total_social / 120) + (row.night_usage / 60) + (row.red_days * 0.5) + row.social_vs_study
    if score < 1.5:
        return 0   # LOW
    elif score < 3:
        return 1   # MEDIUM
    else:
        return 2   # HIGH

df["label"] = df.apply(assign_label, axis=1)

# 4. Features and target
X = df[["total_social", "night_usage", "social_vs_study", "red_days"]]
y = df["label"]

# 5. Train ML model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# 6. Save model
joblib.dump(model, "risk_model.pkl")

print("✅ ML Model Trained & Saved as risk_model.pkl")
