import pandas as pd
import numpy as np

# -----------------------------------
# Load data
# -----------------------------------

df = pd.read_csv("trend_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["topic", "date"])


# -----------------------------------
# Feature engineering
# -----------------------------------

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

df["rolling_4w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(4).mean()
    )
)

df["previous_4w"] = (
    df.groupby("topic")["rolling_4w"]
    .shift(4)
)

df["momentum"] = (
    df["rolling_4w"] -
    df["previous_4w"]
)


# -----------------------------------
# Backtest settings
# -----------------------------------

LOOK_FORWARD = 4

thresholds = [
    0.10,
    0.20,
    0.30
]

all_results = []


# -----------------------------------
# Backtest
# -----------------------------------

for topic, group in df.groupby("topic"):

    group = group.reset_index(drop=True)

    for i in range(8, len(group) - LOOK_FORWARD):

        current = group.loc[i, "interest"]

        growth = group.loc[i, "growth"]

        acceleration = group.loc[
            i, "acceleration"
        ]

        momentum = group.loc[
            i, "momentum"
        ]

        future = group.loc[
            i + LOOK_FORWARD,
            "interest"
        ]

        if current == 0:
            continue

        if pd.isna(growth):
            continue

        if pd.isna(acceleration):
            continue

        if pd.isna(momentum):
            continue


        # Relative future growth
        future_growth = (
            (future - current)
            / current
        )


        # Prediction signal
        predicted = (
            growth > 0
            and acceleration > 0
            and momentum > 0
        )


        for threshold in thresholds:

            actual = (
                future_growth >= threshold
            )

            all_results.append({

                "topic": topic,

                "date": group.loc[
                    i, "date"
                ],

                "current_interest": current,

                "future_interest": future,

                "future_growth":
                    future_growth,

                "predicted":
                    predicted,

                "actual":
                    actual,

                "threshold":
                    threshold

            })


# -----------------------------------
# Results
# -----------------------------------

results = pd.DataFrame(
    all_results
)


# -----------------------------------
# Evaluate
# -----------------------------------

summary = []

for threshold, group in results.groupby(
    "threshold"
):

    tp = (
        (group["predicted"] == True)
        & (group["actual"] == True)
    ).sum()

    fp = (
        (group["predicted"] == True)
        & (group["actual"] == False)
    ).sum()

    fn = (
        (group["predicted"] == False)
        & (group["actual"] == True)
    ).sum()

    tn = (
        (group["predicted"] == False)
        & (group["actual"] == False)
    ).sum()


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn)
        / len(group)
    )


    summary.append({

        "threshold":
            f"{int(threshold * 100)}%",

        "accuracy":
            accuracy * 100,

        "precision":
            precision * 100,

        "recall":
            recall * 100,

        "f1":
            f1 * 100,

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn

    })


summary = pd.DataFrame(summary)


# -----------------------------------
# Display
# -----------------------------------

print("\n")
print("=" * 75)
print("             🔥 BASELINE COMPARISON")
print("=" * 75)

print(
    summary.to_string(index=False)
)


# -----------------------------------
# Save
# -----------------------------------

summary.to_csv(
    "baseline_comparison.csv",
    index=False
)

results.to_csv(
    "baseline_predictions.csv",
    index=False
)

print("\nSaved:")
print("→ baseline_comparison.csv")
print("→ baseline_predictions.csv")