import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("trend_data.csv")

# Make sure data is sorted
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["topic", "date"])

# Calculate changes for each topic
df["growth"] = df.groupby("topic")["interest"].diff()

df["acceleration"] = (
    df.groupby("topic")["growth"].diff()
)

# Rolling momentum
df["momentum"] = (
    df.groupby("topic")["interest"]
    .transform(lambda x: x.rolling(4).mean())
)

# Rolling volatility
df["volatility"] = (
    df.groupby("topic")["interest"]
    .transform(lambda x: x.rolling(4).std())
)

# Calculate latest metrics for every topic
latest = (
    df.groupby("topic")
    .tail(1)
    .copy()
)

# Normalize metrics
def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(50, index=series.index)

    return (series - minimum) / (maximum - minimum) * 100


latest["growth_score"] = normalize(latest["growth"])
latest["acceleration_score"] = normalize(latest["acceleration"])
latest["momentum_score"] = normalize(latest["momentum"])

# Lower volatility is better
latest["stability_score"] = 100 - normalize(latest["volatility"])

# Final Trend Score
latest["trend_score"] = (
    latest["growth_score"] * 0.35 +
    latest["acceleration_score"] * 0.30 +
    latest["momentum_score"] * 0.25 +
    latest["stability_score"] * 0.10
)

# Classify trends
def classify(score):
    if score >= 75:
        return "🚀 Breakout"
    elif score >= 60:
        return "🔥 Accelerating"
    elif score >= 45:
        return "🟢 Emerging"
    elif score >= 30:
        return "🟡 Stable"
    else:
        return "🔴 Declining"

latest["status"] = latest["trend_score"].apply(classify)

# Sort by score
results = latest.sort_values(
    "trend_score",
    ascending=False
)

# Display results
print("\n" + "=" * 60)
print("             🔥 TRENDRADAR")
print("=" * 60)

print(
    results[
        [
            "topic",
            "interest",
            "growth",
            "acceleration",
            "trend_score",
            "status"
        ]
    ].to_string(index=False)
)

# Save results
results.to_csv(
    "trend_scores.csv",
    index=False
)

print("\nTrend scores saved to trend_scores.csv")