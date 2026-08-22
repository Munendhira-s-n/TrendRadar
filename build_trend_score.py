import pandas as pd
import numpy as np

# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "breakout_predictions.csv"
OUTPUT_FILE = "trend_radar_final.csv"


# ==========================================
# 1. Load
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "topic"]
).copy()


# ==========================================
# 2. Point-in-time normalization
# ==========================================

def rolling_percentile(series, window=26):

    return (
        series
        .rolling(
            window,
            min_periods=8
        )
        .rank(
            pct=True
        )
    )


# ==========================================
# 3. Topic signals
# ==========================================

df["momentum_score"] = (
    df.groupby("topic")["momentum"]
    .transform(
        lambda x:
        rolling_percentile(x)
    )
)

df["acceleration_score"] = (
    df.groupby("topic")["acceleration"]
    .transform(
        lambda x:
        rolling_percentile(x)
    )
)

df["growth_score"] = (
    df.groupby("topic")["growth_rate"]
    .transform(
        lambda x:
        rolling_percentile(x)
    )
)


# ==========================================
# 4. Relative category signals
# ==========================================

df["relative_momentum_score"] = (
    df.groupby("topic")["relative_momentum"]
    .transform(
        lambda x:
        rolling_percentile(x)
    )
)

df["category_growth_score"] = (
    df.groupby("topic")["category_growth"]
    .transform(
        lambda x:
        rolling_percentile(x)
    )
)


# ==========================================
# 5. Peak signal
# ==========================================

# Being close to a recent peak can indicate
# sustained interest rather than random noise.

df["peak_score"] = (
    1 -
    df["distance_from_peak"].clip(
        0,
        1
    )
)


# ==========================================
# 6. Current regime
# ==========================================

# Rolling breakout frequency across
# the entire system.

df["system_breakout_rate"] = (
    df["breakout"]
    .shift(1)
    .rolling(
        26,
        min_periods=8
    )
    .mean()
)


# Avoid extreme regime effects.

df["regime_factor"] = (
    df["system_breakout_rate"]
    .clip(
        0.03,
        0.15
    )
)


# ==========================================
# 7. Normalize regime
# ==========================================

df["regime_score"] = (
    (
        df["regime_factor"] - 0.03
    )
    /
    (
        0.15 - 0.03
    )
)


# ==========================================
# 8. Model probability
# ==========================================

df["model_score"] = (
    df["breakout_probability"]
    .clip(0, 1)
)


# ==========================================
# 9. Fill early-period missing values
# ==========================================

score_columns = [

    "momentum_score",
    "acceleration_score",
    "growth_score",
    "relative_momentum_score",
    "category_growth_score",
    "peak_score",
    "regime_score"
]

for col in score_columns:

    df[col] = (
        df[col]
        .fillna(0.5)
    )


# ==========================================
# 10. TrendRadar Score
# ==========================================

df["trend_score"] = (

    0.35 *
    df["model_score"]

    +

    0.15 *
    df["momentum_score"]

    +

    0.10 *
    df["acceleration_score"]

    +

    0.15 *
    df["relative_momentum_score"]

    +

    0.10 *
    df["category_growth_score"]

    +

    0.10 *
    df["peak_score"]

    +

    0.05 *
    df["regime_score"]

) * 100


# ==========================================
# 11. Trend classification
# ==========================================

def classify(score):

    if score >= 75:
        return "🚀 BREAKOUT WATCH"

    elif score >= 60:
        return "🔥 RISING"

    elif score >= 45:
        return "🟢 EMERGING"

    elif score >= 30:
        return "🟡 STABLE"

    else:
        return "🔴 WEAKENING"


df["status"] = (
    df["trend_score"]
    .apply(classify)
)


# ==========================================
# 12. Latest snapshot
# ==========================================

latest_date = df["date"].max()

latest = df[
    df["date"] == latest_date
].copy()


latest = latest.sort_values(
    "trend_score",
    ascending=False
)


# ==========================================
# 13. Save
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

latest.to_csv(
    "latest_trends.csv",
    index=False
)


# ==========================================
# 14. Display
# ==========================================

print("\n")
print("=" * 90)
print("              🚀 FINAL TRENDRADAR")
print("=" * 90)

print(
    f"\nLatest date : "
    f"{latest_date.date()}"
)

print(
    f"Topics : "
    f"{latest['topic'].nunique()}"
)

print("\n")
print(
    latest[
        [
            "topic",
            "category",
            "interest",
            "breakout_probability",
            "trend_score",
            "status"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 90)
print("              🏆 TOP 10 TRENDS")
print("=" * 90)

print(
    latest[
        [
            "topic",
            "trend_score",
            "status"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print("\nSaved:")

print(
    "→ trend_radar_final.csv"
)

print(
    "→ latest_trends.csv"
)