import pandas as pd
import numpy as np

INPUT_FILE = "breakout_predictions.csv"
OUTPUT_FILE = "trend_radar_final.csv"

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "topic"]
).copy()


# ==========================================================
# 1. Calculate historical system breakout rate BY WEEK
# ==========================================================

weekly = (
    df.groupby("date")["breakout"]
    .mean()
    .reset_index()
)

weekly = weekly.sort_values("date")


# IMPORTANT:
# shift(1) means:
# "What was the breakout environment BEFORE this week?"

weekly["historical_breakout_rate"] = (
    weekly["breakout"]
    .shift(1)
    .rolling(
        8,
        min_periods=3
    )
    .mean()
)


# ==========================================================
# 2. Merge back
# ==========================================================

df = df.merge(
    weekly[
        [
            "date",
            "historical_breakout_rate"
        ]
    ],
    on="date",
    how="left"
)


# ==========================================================
# 3. Regime classification
# ==========================================================

def classify_regime(rate):

    if pd.isna(rate):
        return "UNKNOWN"

    if rate >= 0.10:
        return "🔥 HIGH ACTIVITY"

    elif rate >= 0.06:
        return "🟢 NORMAL"

    else:
        return "❄️ LOW ACTIVITY"


df["regime"] = (
    df["historical_breakout_rate"]
    .apply(classify_regime)
)


# ==========================================================
# 4. Regime adjustment
# ==========================================================

# We don't want the regime to dominate the score.
#
# High activity → slight boost
# Normal        → neutral
# Low activity  → slight penalty

def regime_multiplier(rate):

    if pd.isna(rate):
        return 1.0

    if rate >= 0.10:
        return 1.05

    elif rate >= 0.06:
        return 1.00

    else:
        return 0.92


df["regime_multiplier"] = (
    df["historical_breakout_rate"]
    .apply(regime_multiplier)
)


# ==========================================================
# 5. Point-in-time percentile signals
# ==========================================================

def percentile_rank(group):

    return group.expanding(
        min_periods=4
    ).rank(
        pct=True
    )


df["momentum_score"] = (
    df.groupby("topic")["momentum"]
    .transform(percentile_rank)
)

df["acceleration_score"] = (
    df.groupby("topic")["acceleration"]
    .transform(percentile_rank)
)

df["growth_score"] = (
    df.groupby("topic")["growth_rate"]
    .transform(percentile_rank)
)

df["relative_momentum_score"] = (
    df.groupby("topic")["relative_momentum"]
    .transform(percentile_rank)
)

df["category_growth_score"] = (
    df.groupby("topic")["category_growth"]
    .transform(percentile_rank)
)


# ==========================================================
# 6. Peak score
# ==========================================================

df["peak_score"] = (
    1 -
    df["distance_from_peak"].clip(
        0,
        1
    )
)


# ==========================================================
# 7. Fill early-period values
# ==========================================================

score_columns = [
    "momentum_score",
    "acceleration_score",
    "growth_score",
    "relative_momentum_score",
    "category_growth_score"
]

for col in score_columns:

    df[col] = (
        df[col]
        .fillna(0.5)
    )


# ==========================================================
# 8. Model score
# ==========================================================

df["model_score"] = (
    df["breakout_probability"]
    .clip(0, 1)
)


# ==========================================================
# 9. FINAL TREND SCORE
# ==========================================================

base_score = (

    0.35 * df["model_score"]

    +

    0.15 * df["momentum_score"]

    +

    0.10 * df["acceleration_score"]

    +

    0.15 * df["relative_momentum_score"]

    +

    0.10 * df["category_growth_score"]

    +

    0.10 * df["peak_score"]

)


df["trend_score"] = (
    base_score
    *
    df["regime_multiplier"]
    *
    100
)


# ==========================================================
# 10. Classification
# ==========================================================

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


# ==========================================================
# 11. Latest labeled snapshot
# ==========================================================

latest_date = df["date"].max()

latest = (
    df[
        df["date"] == latest_date
    ]
    .sort_values(
        "trend_score",
        ascending=False
    )
    .copy()
)


# ==========================================================
# 12. Save
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

latest.to_csv(
    "latest_trends.csv",
    index=False
)


# ==========================================================
# 13. Display
# ==========================================================

print("\n")
print("=" * 90)
print("          🚀 TRENDRADAR — POINT-IN-TIME SCORE")
print("=" * 90)

print(
    f"\nLatest labeled date : "
    f"{latest_date.date()}"
)

print(
    f"Topics : "
    f"{len(latest)}"
)

print("\nTOP 20")

print(
    latest[
        [
            "topic",
            "category",
            "interest",
            "breakout_probability",
            "historical_breakout_rate",
            "regime",
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
print("                  🏆 TOP 10")
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