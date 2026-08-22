import pandas as pd
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "trend_features_v2.csv"
OUTPUT_FILE = "trend_features_v3.csv"


# ==========================================
# 1. Load
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["topic", "date"]
).copy()


# ==========================================
# 2. Future movement
# ==========================================

g = df.groupby("topic")

df["future_interest"] = (
    g["interest"].shift(-1)
)

df["future_growth"] = (
    df["future_interest"]
    -
    df["interest"]
)


df["future_growth_rate"] = (
    df["future_growth"]
    /
    df["interest"].replace(
        0,
        np.nan
    )
)


# ==========================================
# 3. Recent movement baseline
# ==========================================

# Typical recent absolute movement

df["recent_growth_mean"] = (
    g["growth"]
    .transform(
        lambda x:
        x.abs()
        .rolling(
            8,
            min_periods=4
        )
        .mean()
    )
)


# Recent volatility

df["recent_growth_std"] = (
    g["growth"]
    .transform(
        lambda x:
        x.rolling(
            8,
            min_periods=4
        )
        .std()
    )
)


# ==========================================
# 4. Breakout threshold
# ==========================================

df["breakout_threshold"] = (
    df["recent_growth_mean"]
    +
    df["recent_growth_std"]
)


# ==========================================
# 5. Breakout target
# ==========================================

# A breakout requires:
#
# Future growth >
# recent typical movement
# + recent volatility

df["breakout"] = (
    (
        df["future_growth"]
        >
        df["breakout_threshold"]
    )
    &
    (
        df["future_growth"] > 0
    )
).astype(int)


# ==========================================
# 6. Strong breakout
# ==========================================

# Extra label for analysis:
# growth at least 2x recent movement

df["strong_breakout"] = (
    (
        df["future_growth"]
        >
        (
            2 *
            df["recent_growth_mean"]
        )
    )
    &
    (
        df["future_growth"] > 0
    )
).astype(int)


# ==========================================
# 7. Remove impossible rows
# ==========================================

df = df.dropna(
    subset=[
        "future_interest",
        "breakout_threshold"
    ]
)


# ==========================================
# 8. Save
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 9. Summary
# ==========================================

print("\n")
print("=" * 80)
print("             🚀 TRENDRADAR BREAKOUT TARGET")
print("=" * 80)

print(
    f"\nRows : {len(df):,}"
)

print(
    f"Topics : {df['topic'].nunique()}"
)

print(
    f"\nBreakouts : "
    f"{df['breakout'].sum():,}"
)

print(
    f"Breakout rate : "
    f"{df['breakout'].mean() * 100:.2f}%"
)

print(
    f"\nStrong breakouts : "
    f"{df['strong_breakout'].sum():,}"
)

print(
    f"Strong breakout rate : "
    f"{df['strong_breakout'].mean() * 100:.2f}%"
)


print("\nBreakouts by category:")

print(
    df.groupby("category")["breakout"]
    .agg(
        ["sum", "count", "mean"]
    )
    .assign(
        breakout_rate=lambda x:
        x["mean"] * 100
    )
    .sort_values(
        "breakout_rate",
        ascending=False
    )
    .to_string()
)


print(
    f"\nSaved → {OUTPUT_FILE}"
)