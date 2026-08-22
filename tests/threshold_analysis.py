import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Load data
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
# 4. Walk-forward settings
# ==========================================

dates = sorted(
    df["date"].unique()
)

initial_train_size = 24
fold_size = 8


# ==========================================
# 5. Probability thresholds
# ==========================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


# ==========================================
# 6. Store predictions
# ==========================================

all_predictions = []


# ==========================================
# 7. Walk-forward validation
# ==========================================

for split in range(
    initial_train_size,
    len(dates) - fold_size + 1,
    fold_size
):

    train_dates = dates[:split]

    test_dates = dates[
        split:
        split + fold_size
    ]

    train = df[
        df["date"].isin(train_dates)
    ]

    test = df[
        df["date"].isin(test_dates)
    ]

    if len(train) == 0 or len(test) == 0:
        continue


    X_train = train[features]

    y_train = train["target"]

    X_test = test[features]

    y_test = test["target"]


    # --------------------------------------
    # Impute missing values
    # --------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )


    # --------------------------------------
    # Random Forest
    # --------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_imp,
        y_train
    )


    probabilities = model.predict_proba(
        X_test_imp
    )[:, 1]


    for date, topic, actual, probability in zip(
        test["date"],
        test["topic"],
        y_test,
        probabilities
    ):

        all_predictions.append({

            "date": date,

            "topic": topic,

            "actual": actual,

            "probability": probability

        })


predictions = pd.DataFrame(
    all_predictions
)


# ==========================================
# 8. Test thresholds
# ==========================================

results = []


for threshold in thresholds:

    predicted = (
        predictions["probability"]
        >= threshold
    ).astype(int)


    precision = precision_score(
        predictions["actual"],
        predicted,
        zero_division=0
    )

    recall = recall_score(
        predictions["actual"],
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        predictions["actual"],
        predicted,
        zero_division=0
    )


    alerts = predicted.sum()


    results.append({

        "threshold":
            threshold,

        "precision":
            precision * 100,

        "recall":
            recall * 100,

        "f1":
            f1 * 100,

        "alerts":
            alerts

    })


results = pd.DataFrame(
    results
)


# ==========================================
# 9. Display
# ==========================================

print("\n")
print("=" * 80)
print("             🎯 THRESHOLD OPTIMIZATION")
print("=" * 80)

print(
    results.to_string(
        index=False,
        formatters={
            "threshold":
                "{:.2f}".format,
            "precision":
                "{:.2f}".format,
            "recall":
                "{:.2f}".format,
            "f1":
                "{:.2f}".format
        }
    )
)


# ==========================================
# 10. Best F1
# ==========================================

best_f1 = results.loc[
    results["f1"].idxmax()
]


print("\n")
print("=" * 80)
print("                 🏆 BEST F1")
print("=" * 80)

print(
    f"\nThreshold : "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Precision : "
    f"{best_f1['precision']:.2f}%"
)

print(
    f"Recall    : "
    f"{best_f1['recall']:.2f}%"
)

print(
    f"F1        : "
    f"{best_f1['f1']:.2f}%"
)

print(
    f"Alerts    : "
    f"{int(best_f1['alerts'])}"
)


# ==========================================
# 11. Best precision above 30%
# ==========================================

acceptable = results[
    results["precision"] >= 30
]

if len(acceptable) > 0:

    best_precision = acceptable.loc[
        acceptable["recall"].idxmax()
    ]

    print("\n")
    print("=" * 80)
    print("       🎯 BEST RECALL WITH ≥30% PRECISION")
    print("=" * 80)

    print(
        f"\nThreshold : "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_precision['precision']:.2f}%"
    )

    print(
        f"Recall    : "
        f"{best_precision['recall']:.2f}%"
    )

    print(
        f"F1        : "
        f"{best_precision['f1']:.2f}%"
    )


# ==========================================
# 12. Save
# ==========================================

results.to_csv(
    "threshold_analysis.csv",
    index=False
)

predictions.to_csv(
    "walk_forward_predictions.csv",
    index=False
)

print("\nSaved:")
print("→ threshold_analysis.csv")
print("→ walk_forward_predictions.csv")