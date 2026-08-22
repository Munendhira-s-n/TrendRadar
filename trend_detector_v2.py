import pandas as pd
import numpy as np

# -----------------------------
# 1. Load data
# -----------------------------

df = pd.read_csv("trend_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["topic", "date"])


# -----------------------------
# 2. Calculate recent metrics
# -----------------------------

# Recent change
df["growth"] = (
    df.groupby("topic")["interest"]
    .diff()
)

# Previous growth
df["previous_growth"] = (
    df.groupby("topic")["growth"]
    .shift(1)
)

# Acceleration
df["acceleration"] = (
    df["growth"] - df["previous_growth"]
)

# 4-week average
df["rolling_mean"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(4).mean()
    )
)

# Previous 4-week average
df["previous_mean"] = (
    df.groupby("topic")["rolling_mean"]
    .shift(4)
)

# Momentum
df["momentum"] = (
    df["rolling_mean"] -
    df["previous_mean"]
)


# -----------------------------
# 3. Peak distance
# -----------------------------

# Historical maximum for each topic
df["historical_peak"] = (
    df.groupby("topic")["interest"]
    .transform("max")
)

# How far current interest is from its historical peak
df["peak_distance"] = (
    1 -
    df["interest"] / df["historical_peak"]
)

df["peak_distance"] = (
    df["peak_distance"].clip(0, 1)
)


# -----------------------------
# 4. Get latest observation
# -----------------------------

latest = (
    df.groupby("topic")
    .tail(1)
    .copy()
)


# -----------------------------
# 5. Normalize
# -----------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            50,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


latest["growth_score"] = normalize(
    latest["growth"]
)

latest["acceleration_score"] = normalize(
    latest["acceleration"]
)

latest["momentum_score"] = normalize(
    latest["momentum"]
)

latest["early_score"] = (
    latest["peak_distance"] * 100
)


# -----------------------------
# 6. Prevent declining topics
# -----------------------------

latest["direction_score"] = np.where(
    latest["growth"] > 0,
    100,
    20
)


# -----------------------------
# 7. Final Trend Score
# -----------------------------

latest["trend_score"] = (

    latest["growth_score"] * 0.30

    + latest["acceleration_score"] * 0.20

    + latest["momentum_score"] * 0.20

    + latest["early_score"] * 0.20

    + latest["direction_score"] * 0.10

)


# -----------------------------
# 8. Classification
# -----------------------------

def classify(row):

    if row["growth"] < 0:
        return "🔴 Declining"

    if row["trend_score"] >= 75:
        return "🚀 Breakout"

    if row["trend_score"] >= 60:
        return "🔥 Accelerating"

    if row["trend_score"] >= 45:
        return "🟢 Emerging"

    return "🟡 Stable"


latest["status"] = latest.apply(
    classify,
    axis=1
)


# -----------------------------
# 9. Sort results
# -----------------------------

results = latest.sort_values(
    "trend_score",
    ascending=False
)


# -----------------------------
# 10. Display
# -----------------------------

print("\n")
print("=" * 75)
print("                    🔥 TRENDRADAR V2")
print("=" * 75)

print(
    results[
        [
            "topic",
            "interest",
            "growth",
            "acceleration",
            "momentum",
            "peak_distance",
            "trend_score",
            "status"
        ]
    ].to_string(index=False)
)


# -----------------------------
# 11. Save
# -----------------------------

results.to_csv(
    "trend_scores_v2.csv",
    index=False
)

print("\n")
print("Saved → trend_scores_v2.csv")