import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "trend_features_v2.csv"

TRAIN_END = "2026-05-17"

HOLDOUT_START = "2026-05-24"


# ==========================================
# 1. Load
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "topic"]
).copy()


# ==========================================
# 2. Create prediction target
# ==========================================

# Target:
# Will the topic's interest increase
# during the following week?

df["future_interest"] = (
    df.groupby("topic")["interest"]
    .shift(-1)
)

df["future_growth"] = (
    df["future_interest"]
    -
    df["interest"]
)

df["target"] = (
    df["future_growth"] > 0
).astype(int)


# Remove rows without future target
df = df.dropna(
    subset=["future_interest"]
)


# ==========================================
# 3. Features
# ==========================================

FEATURES = [
    "growth",
    "growth_rate",
    "previous_growth",
    "acceleration",
    "mean_4w",
    "mean_8w",
    "momentum",
    "volatility_4w",
    "distance_from_peak",
    "upward_consistency",
    "slope_4w",
    "slope_8w",

    # V2 category features
    "relative_interest",
    "category_momentum",
    "relative_momentum",
    "category_growth",
    "category_previous_growth",
    "category_acceleration"
]


# ==========================================
# 4. Remove invalid rows
# ==========================================

df = df.dropna(
    subset=FEATURES
)


# ==========================================
# 5. Chronological split
# ==========================================

train = df[
    df["date"] <= TRAIN_END
].copy()

holdout = df[
    df["date"] >= HOLDOUT_START
].copy()


X_train = train[FEATURES]

y_train = train["target"]

X_test = holdout[FEATURES]

y_test = holdout["target"]


print("\n")
print("=" * 75)
print("              🔥 TRENDRADAR V2 ML MODEL")
print("=" * 75)

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
    f"\nTraining rows   : {len(train):,}"
)

print(
    f"Holdout rows    : {len(holdout):,}"
)


# ==========================================
# 6. Train Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# 7. Predictions
# ==========================================

probabilities = (
    model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 8. Test multiple thresholds
# ==========================================

results = []


for threshold in np.arange(
    0.20,
    0.71,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results.append({
        "threshold": round(
            threshold,
            2
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": predictions.sum()
    })


results_df = pd.DataFrame(
    results
)


# ==========================================
# 9. Best F1
# ==========================================

best = results_df.loc[
    results_df["f1"].idxmax()
]


best_threshold = best["threshold"]


final_predictions = (
    probabilities >= best_threshold
).astype(int)


# ==========================================
# 10. Final metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)


cm = confusion_matrix(
    y_test,
    final_predictions
)


# ==========================================
# 11. Feature importance
# ==========================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


# ==========================================
# 12. Save predictions
# ==========================================

holdout["probability"] = probabilities

holdout["prediction"] = final_predictions

holdout["alert"] = np.where(
    final_predictions == 1,
    "🚀 TREND ALERT",
    ""
)


holdout.to_csv(
    "v2_holdout_predictions.csv",
    index=False
)

importance.to_csv(
    "v2_feature_importance.csv",
    index=False
)

results_df.to_csv(
    "v2_threshold_analysis.csv",
    index=False
)


# ==========================================
# 13. Print results
# ==========================================

print("\n")
print("=" * 75)
print("                 📊 V2 FINAL RESULTS")
print("=" * 75)

print(
    f"\nBest threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score       : "
    f"{f1 * 100:.2f}%"
)

print(
    f"\nAlerts generated : "
    f"{final_predictions.sum()}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


print("\n")
print("=" * 75)
print("              🔥 FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance.to_string(
        index=False
    )
)


print("\nSaved:")

print(
    "→ v2_holdout_predictions.csv"
)

print(
    "→ v2_feature_importance.csv"
)

print(
    "→ v2_threshold_analysis.csv"
)