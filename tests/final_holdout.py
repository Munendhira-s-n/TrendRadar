import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# ==========================================
# 1. Load feature data
# ==========================================

df = pd.read_csv("trend_features.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "topic"]
)


# ==========================================
# 2. Create future target
# ==========================================

LOOK_FORWARD = 4

df["future_interest"] = (
    df.groupby("topic")["interest"]
    .shift(-LOOK_FORWARD)
)

df["future_growth"] = (
    (df["future_interest"] - df["interest"])
    / df["interest"]
)

df["target"] = (
    df["future_growth"] >= 0.10
).astype(int)


# ==========================================
# 3. Features
# ==========================================

features = [
    "growth",
    "growth_rate",
    "acceleration",
    "momentum",
    "volatility_4w",
    "distance_from_peak",
    "upward_consistency",
    "slope_4w",
    "slope_8w"
]

df = df.dropna(
    subset=features + ["future_growth"]
).copy()


# ==========================================
# 4. Chronological holdout
# ==========================================

dates = sorted(
    df["date"].unique()
)

# Keep the final 20% completely untouched
split_index = int(
    len(dates) * 0.80
)

train_dates = dates[:split_index]

holdout_dates = dates[split_index:]


train = df[
    df["date"].isin(train_dates)
].copy()

holdout = df[
    df["date"].isin(holdout_dates)
].copy()


print("\n")
print("=" * 70)
print("             🔒 FINAL HOLDOUT TEST")
print("=" * 70)

print(
    f"\nTraining period : "
    f"{train['date'].min().date()} → "
    f"{train['date'].max().date()}"
)

print(
    f"Holdout period  : "
    f"{holdout['date'].min().date()} → "
    f"{holdout['date'].max().date()}"
)

print(
    f"\nTraining rows   : {len(train)}"
)

print(
    f"Holdout rows    : {len(holdout)}"
)


# ==========================================
# 5. Prepare data
# ==========================================

X_train = train[features]

y_train = train["target"]

X_holdout = holdout[features]

y_holdout = holdout["target"]


# ==========================================
# 6. Imputation
# ==========================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_holdout = imputer.transform(
    X_holdout
)


# ==========================================
# 7. Frozen Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. Generate probabilities
# ==========================================

probabilities = model.predict_proba(
    X_holdout
)[:, 1]


# ==========================================
# 9. Frozen threshold
# ==========================================

THRESHOLD = 0.45

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ==========================================
# 10. Metrics
# ==========================================

accuracy = accuracy_score(
    y_holdout,
    predictions
)

precision = precision_score(
    y_holdout,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_holdout,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_holdout,
    predictions,
    zero_division=0
)


# ==========================================
# 11. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_holdout,
    predictions
)


# ==========================================
# 12. Display
# ==========================================

print("\n")
print("=" * 70)
print("                 📊 FINAL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 13. Alert statistics
# ==========================================

alerts = (
    predictions == 1
).sum()

actual_trends = (
    y_holdout == 1
).sum()

correct_alerts = (
    (predictions == 1)
    &
    (y_holdout == 1)
).sum()


print("\n")
print("=" * 70)
print("                 🚨 ALERT ANALYSIS")
print("=" * 70)

print(
    f"\nAlerts generated : {alerts}"
)

print(
    f"Actual trends   : {actual_trends}"
)

print(
    f"Correct alerts  : {correct_alerts}"
)


# ==========================================
# 14. Save predictions
# ==========================================

output = holdout[
    [
        "date",
        "topic",
        "interest",
        "future_interest",
        "future_growth",
        "target"
    ]
].copy()

output["probability"] = probabilities

output["prediction"] = predictions

output["alert"] = np.where(
    predictions == 1,
    "🚀 Emerging Trend",
    "—"
)

output.to_csv(
    "final_holdout_predictions.csv",
    index=False
)


# ==========================================
# 15. Feature importance
# ==========================================

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "final_feature_importance.csv",
    index=False
)


print("\nSaved:")
print("→ final_holdout_predictions.csv")
print("→ final_feature_importance.csv")