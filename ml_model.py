import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------
# 1. Load feature dataset
# ---------------------------------------

df = pd.read_csv("trend_features.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")


# ---------------------------------------
# 2. Create future target
# ---------------------------------------

LOOK_FORWARD = 4

df["future_interest"] = (
    df.groupby("topic")["interest"]
    .shift(-LOOK_FORWARD)
)

df["future_growth"] = (
    (df["future_interest"] - df["interest"])
    / df["interest"]
)

# Emerging trend = >= 10% future growth
df["target"] = (
    df["future_growth"] >= 0.10
).astype(int)


# ---------------------------------------
# 3. Select features
# ---------------------------------------

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


df_model = df.dropna(
    subset=features + ["future_growth"]
).copy()


# ---------------------------------------
# 4. Time-based split
# ---------------------------------------

dates = sorted(
    df_model["date"].unique()
)

split_index = int(
    len(dates) * 0.70
)

train_dates = dates[:split_index]

test_dates = dates[split_index:]


train = df_model[
    df_model["date"].isin(train_dates)
]

test = df_model[
    df_model["date"].isin(test_dates)
]


X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


# ---------------------------------------
# 5. Train Random Forest
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ---------------------------------------
# 6. Predict probabilities
# ---------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------
# 7. Default threshold
# ---------------------------------------

threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)


# ---------------------------------------
# 8. Metrics
# ---------------------------------------

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


# ---------------------------------------
# 9. Confusion matrix
# ---------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)


# ---------------------------------------
# 10. Feature importance
# ---------------------------------------

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


# ---------------------------------------
# 11. Display
# ---------------------------------------

print("\n")
print("=" * 65)
print("              🤖 TRENDRADAR ML MODEL")
print("=" * 65)

print(
    f"\nTraining rows : {len(train)}"
)

print(
    f"Testing rows  : {len(test)}"
)

print(
    f"\nPrecision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print("\nConfusion Matrix:")
print(cm)


print("\nFeature Importance:")
print(
    importance.to_string(
        index=False
    )
)


# ---------------------------------------
# 12. Save predictions
# ---------------------------------------

test_results = test[
    [
        "date",
        "topic",
        "interest",
        "future_interest",
        "future_growth",
        "target"
    ]
].copy()

test_results["probability"] = probabilities

test_results["prediction"] = predictions

test_results.to_csv(
    "ml_predictions.csv",
    index=False
)


importance.to_csv(
    "feature_importance.csv",
    index=False
)


print("\nSaved:")
print("→ ml_predictions.csv")
print("→ feature_importance.csv")