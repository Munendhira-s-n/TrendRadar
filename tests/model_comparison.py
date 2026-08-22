import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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
# 4. Prepare chronological dates
# ==========================================

dates = sorted(
    df["date"].unique()
)

# We need enough history for training.
# Each fold predicts the next chunk.

fold_size = 8

initial_train_size = 24


# ==========================================
# 5. Models
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ])
}


# ==========================================
# 6. Probability thresholds
# ==========================================

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]


all_results = []


# ==========================================
# 7. Walk-forward validation
# ==========================================

for model_name, model in models.items():

    print(
        f"\nRunning: {model_name}"
    )

    for threshold in thresholds:

        predictions = []
        actuals = []

        # Start after initial training period
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
                df["date"].isin(
                    train_dates
                )
            ]

            test = df[
                df["date"].isin(
                    test_dates
                )
            ]

            if len(train) == 0:
                continue

            if len(test) == 0:
                continue

            X_train = train[features]
            y_train = train["target"]

            X_test = test[features]
            y_test = test["target"]


            # Train
            model.fit(
                X_train,
                y_train
            )


            # Probability
            probability = model.predict_proba(
                X_test
            )[:, 1]


            prediction = (
                probability >= threshold
            ).astype(int)


            predictions.extend(
                prediction
            )

            actuals.extend(
                y_test.values
            )


        # ==================================
        # Metrics
        # ==================================

        precision = precision_score(
            actuals,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actuals,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actuals,
            predictions,
            zero_division=0
        )


        all_results.append({

            "model": model_name,

            "threshold":
                threshold,

            "precision":
                precision * 100,

            "recall":
                recall * 100,

            "f1":
                f1 * 100

        })


# ==========================================
# 8. Results
# ==========================================

results = pd.DataFrame(
    all_results
)

results = results.sort_values(
    "f1",
    ascending=False
)


print("\n")
print("=" * 80)
print("              🔥 TRENDRADAR MODEL COMPARISON")
print("=" * 80)

print(
    results.to_string(
        index=False,
        formatters={
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
# 9. Best configuration
# ==========================================

best = results.iloc[0]

print("\n")
print("=" * 80)
print("                    🏆 BEST MODEL")
print("=" * 80)

print(
    f"\nModel     : {best['model']}"
)

print(
    f"Threshold : {best['threshold']}"
)

print(
    f"Precision : {best['precision']:.2f}%"
)

print(
    f"Recall    : {best['recall']:.2f}%"
)

print(
    f"F1        : {best['f1']:.2f}%"
)


# ==========================================
# 10. Save
# ==========================================

results.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nSaved → model_comparison.csv"
)