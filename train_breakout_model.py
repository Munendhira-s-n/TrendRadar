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

INPUT_FILE = "trend_features_v3.csv"

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
# 2. Features
# ==========================================

FEATURES = [

    # Topic behavior
    "interest",
    "growth",
    "growth_rate",
    "previous_growth",
    "acceleration",

    # Rolling behavior
    "mean_4w",
    "mean_8w",
    "momentum",
    "volatility_4w",

    # Peak behavior
    "peak_8w",
    "distance_from_peak",

    # Direction
    "upward_consistency",
    "slope_4w",
    "slope_8w",

    # Category intelligence
    "relative_interest",
    "category_momentum",
    "relative_momentum",
    "category_growth",
    "category_previous_growth",
    "category_acceleration"
]


# ==========================================
# 3. Validate columns
# ==========================================

missing = [
    x for x in FEATURES
    if x not in df.columns
]

if missing:

    print(
        "Missing features:",
        missing
    )

    raise SystemExit


# ==========================================
# 4. Handle zero-interest edge cases
# ==========================================

# Some topics can have zero Google Trends
# interest for a particular week.
#
# This can create mathematically undefined
# ratios such as:
#
# growth_rate = 0 / 0
#
# distance_from_peak = 0 / 0
#
# These mean "no measurable activity", not
# "missing observation".

ZERO_SAFE_FEATURES = [
    "growth_rate",
    "distance_from_peak"
]

for col in ZERO_SAFE_FEATURES:

    df[col] = (
        df[col]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# Verify remaining missing values

remaining_missing = (
    df[FEATURES + ["breakout"]]
    .isna()
    .sum()
)

remaining_missing = (
    remaining_missing[
        remaining_missing > 0
    ]
)


if len(remaining_missing) > 0:

    print(
        "\nRemaining missing values:"
    )

    print(
        remaining_missing
    )

    raise ValueError(
        "Unexpected missing values remain."
    )


print(
    "\n✓ Feature data validated"
)

print(
    f"Rows available : {len(df):,}"
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

y_train = train["breakout"]

X_test = holdout[FEATURES]

y_test = holdout["breakout"]


# ==========================================
# 6. Dataset information
# ==========================================

print("\n")
print("=" * 80)
print("           🚀 TRENDRADAR BREAKOUT MODEL")
print("=" * 80)

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
    f"\nTraining rows : {len(train):,}"
)

print(
    f"Holdout rows  : {len(holdout):,}"
)

print(
    f"\nTraining breakouts : "
    f"{y_train.sum():,}"
)

print(
    f"Holdout breakouts : "
    f"{y_test.sum():,}"
)


# ==========================================
# 7. Train model
# ==========================================

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=8,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


print(
    "\nTraining Random Forest..."
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. Probabilities
# ==========================================

probabilities = (
    model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 9. Threshold analysis
# ==========================================

results = []


for threshold in np.arange(
    0.10,
    0.81,
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

        "threshold":
            round(threshold, 2),

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "alerts":
            predictions.sum()
    })


results_df = pd.DataFrame(
    results
)


# ==========================================
# 10. Best F1 threshold
# ==========================================

best = results_df.loc[
    results_df["f1"].idxmax()
]


best_threshold = (
    best["threshold"]
)


# ==========================================
# 11. Final predictions
# ==========================================

predictions = (
    probabilities >= best_threshold
).astype(int)


# ==========================================
# 12. Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

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

cm = confusion_matrix(
    y_test,
    predictions
)


# ==========================================
# 13. Feature importance
# ==========================================

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


# ==========================================
# 14. Save predictions
# ==========================================

holdout["breakout_probability"] = (
    probabilities
)

holdout["predicted_breakout"] = (
    predictions
)

holdout["alert"] = np.where(
    predictions == 1,
    "🚀 BREAKOUT ALERT",
    ""
)


holdout.to_csv(
    "breakout_predictions.csv",
    index=False
)


results_df.to_csv(
    "breakout_threshold_analysis.csv",
    index=False
)


importance.to_csv(
    "breakout_feature_importance.csv",
    index=False
)


# ==========================================
# 15. Results
# ==========================================

print("\n")
print("=" * 80)
print("                  📊 FINAL RESULTS")
print("=" * 80)

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
    f"{predictions.sum()}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


print("\n")
print("=" * 80)
print("              🔥 FEATURE IMPORTANCE")
print("=" * 80)

print(
    importance.to_string(
        index=False
    )
)


print("\nSaved:")

print(
    "→ breakout_predictions.csv"
)

print(
    "→ breakout_threshold_analysis.csv"
)

print(
    "→ breakout_feature_importance.csv"
)