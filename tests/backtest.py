import pandas as pd
import numpy as np

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("trend_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["topic", "date"])


# -----------------------------
# Calculate historical metrics
# -----------------------------

df["growth"] = (
    df.groupby("topic")["interest"].diff()
)

df["previous_growth"] = (
    df.groupby("topic")["growth"].shift(1)
)

df["acceleration"] = (
    df["growth"] -
    df["previous_growth"]
)

df["rolling_mean"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(4).mean()
    )
)

df["previous_mean"] = (
    df.groupby("topic")["rolling_mean"]
    .shift(4)
)

df["momentum"] = (
    df["rolling_mean"] -
    df["previous_mean"]
)


# -----------------------------
# Backtest parameters
# -----------------------------

LOOK_FORWARD = 4

results = []


# -----------------------------
# Test every topic
# -----------------------------

for topic, group in df.groupby("topic"):

    group = group.reset_index(drop=True)

    for i in range(8, len(group) - LOOK_FORWARD):

        current_interest = group.loc[
            i, "interest"
        ]

        current_growth = group.loc[
            i, "growth"
        ]

        current_acceleration = group.loc[
            i, "acceleration"
        ]

        current_momentum = group.loc[
            i, "momentum"
        ]

        # Skip incomplete observations
        if pd.isna(current_growth):
            continue

        if pd.isna(current_acceleration):
            continue

        if pd.isna(current_momentum):
            continue


        # -----------------------------
        # Calculate future movement
        # -----------------------------

        future_interest = group.loc[
            i + LOOK_FORWARD,
            "interest"
        ]

        future_growth = (
            future_interest -
            current_interest
        )


        # -----------------------------
        # Define actual future trend
        # -----------------------------

        actual_trend = (
            future_growth >= 10
        )


        # -----------------------------
        # Prediction logic
        # -----------------------------

        predicted_trend = (

            current_growth > 0

            and current_acceleration > 0

            and current_momentum > 0

        )


        results.append({

            "topic": topic,

            "date": group.loc[
                i, "date"
            ],

            "interest": current_interest,

            "growth": current_growth,

            "acceleration":
                current_acceleration,

            "momentum":
                current_momentum,

            "future_interest":
                future_interest,

            "future_growth":
                future_growth,

            "predicted":
                predicted_trend,

            "actual":
                actual_trend

        })


# -----------------------------
# Create results dataframe
# -----------------------------

results = pd.DataFrame(results)


# -----------------------------
# Evaluate predictions
# -----------------------------

results["correct"] = (
    results["predicted"] ==
    results["actual"]
)


accuracy = (
    results["correct"].mean()
    * 100
)


# -----------------------------
# Precision
# -----------------------------

predicted_positive = (
    results["predicted"] == True
)

if predicted_positive.sum() > 0:

    precision = (
        results.loc[
            predicted_positive,
            "actual"
        ].mean()
        * 100
    )

else:

    precision = 0


# -----------------------------
# Recall
# -----------------------------

actual_positive = (
    results["actual"] == True
)

if actual_positive.sum() > 0:

    recall = (
        results.loc[
            actual_positive,
            "predicted"
        ].mean()
        * 100
    )

else:

    recall = 0


# -----------------------------
# Display
# -----------------------------

print("\n")
print("=" * 60)
print("             🔥 TRENDRADAR BACKTEST")
print("=" * 60)

print(
    f"\nTotal predictions : {len(results)}"
)

print(
    f"Accuracy          : {accuracy:.2f}%"
)

print(
    f"Precision         : {precision:.2f}%"
)

print(
    f"Recall            : {recall:.2f}%"
)


# -----------------------------
# Save
# -----------------------------

results.to_csv(
    "backtest_results.csv",
    index=False
)

print(
    "\nSaved → backtest_results.csv"
)