import pandas as pd
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "trend_data_v2.csv"
OUTPUT_FILE = "trend_features_v2.csv"


# ==========================================
# 1. Load data
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["topic", "date"]
).copy()


# ==========================================
# 2. Topic-level features
# ==========================================

group = df.groupby("topic")["interest"]


# Previous interest
df["previous_interest"] = (
    group.shift(1)
)


# Weekly growth
df["growth"] = (
    df["interest"]
    - df["previous_interest"]
)


# Percentage growth
df["growth_rate"] = (
    df["growth"]
    /
    df["previous_interest"].replace(
        0,
        np.nan
    )
)


# Previous growth
df["previous_growth"] = (
    df.groupby("topic")["growth"]
    .shift(1)
)


# Acceleration
df["acceleration"] = (
    df["growth"]
    -
    df["previous_growth"]
)


# ==========================================
# 3. Rolling statistics
# ==========================================

topic_group = df.groupby("topic")


df["mean_4w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(4, min_periods=2)
        .mean()
    )
)


df["mean_8w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(8, min_periods=3)
        .mean()
    )
)


# Momentum
df["momentum"] = (
    df["mean_4w"]
    -
    df["mean_8w"]
)


# 4-week volatility
df["volatility_4w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(4, min_periods=2)
        .std()
    )
)


# ==========================================
# 4. Peak features
# ==========================================

df["peak_8w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(8, min_periods=2)
        .max()
    )
)


df["distance_from_peak"] = (
    (
        df["peak_8w"]
        -
        df["interest"]
    )
    /
    df["peak_8w"].replace(
        0,
        np.nan
    )
)


# ==========================================
# 5. Upward consistency
# ==========================================

def upward_ratio(x):

    changes = x.diff()

    if len(changes) == 0:
        return np.nan

    return (
        (changes > 0).sum()
        /
        changes.notna().sum()
    )


df["upward_consistency"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=2
        ).apply(
            lambda y:
            upward_ratio(y),
            raw=False
        )
    )
)


# ==========================================
# 6. Trend slopes
# ==========================================

def calculate_slope(x):

    if len(x) < 2:
        return np.nan

    y = np.array(x)

    time = np.arange(
        len(y)
    )

    return np.polyfit(
        time,
        y,
        1
    )[0]


df["slope_4w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=3
        ).apply(
            calculate_slope,
            raw=False
        )
    )
)


df["slope_8w"] = (
    topic_group["interest"]
    .transform(
        lambda x:
        x.rolling(
            8,
            min_periods=4
        ).apply(
            calculate_slope,
            raw=False
        )
    )
)


# ==========================================
# 7. Category-level features
# ==========================================

category_group = (
    df.groupby(
        ["date", "category"]
    )
)


# Average category interest
category_mean = (
    category_group["interest"]
    .mean()
    .rename(
        "category_mean_interest"
    )
    .reset_index()
)


df = df.merge(
    category_mean,
    on=[
        "date",
        "category"
    ],
    how="left"
)


# ==========================================
# 8. Relative interest
# ==========================================

df["relative_interest"] = (
    df["interest"]
    /
    df["category_mean_interest"]
)


# ==========================================
# 9. Category momentum
# ==========================================

category_momentum = (
    df.groupby(
        ["category"]
    )["category_mean_interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=2
        ).mean()
        -
        x.rolling(
            8,
            min_periods=3
        ).mean()
    )
)


df["category_momentum"] = (
    category_momentum
)


# ==========================================
# 10. Relative momentum
# ==========================================

df["relative_momentum"] = (
    df["momentum"]
    -
    df["category_momentum"]
)


# ==========================================
# 11. Category growth
# ==========================================

df["category_growth"] = (
    df.groupby("category")[
        "category_mean_interest"
    ].diff()
)


# ==========================================
# 12. Category acceleration
# ==========================================

df["category_previous_growth"] = (
    df.groupby("category")[
        "category_growth"
    ].shift(1)
)


df["category_acceleration"] = (
    df["category_growth"]
    -
    df["category_previous_growth"]
)


# ==========================================
# 13. Save
# ==========================================

df = df.sort_values(
    ["date", "topic"]
).reset_index(
    drop=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 14. Summary
# ==========================================

print("\n")
print("=" * 80)
print("          🔥 TRENDRADAR FEATURE ENGINEERING V2")
print("=" * 80)

print(
    f"\nRows     : {len(df):,}"
)

print(
    f"Topics   : {df['topic'].nunique()}"
)

print(
    f"Categories : {df['category'].nunique()}"
)

print(
    f"Features : {len(df.columns)}"
)


print("\nColumns:")

for column in df.columns:

    print(
        f" - {column}"
    )


print("\nSample:")

print(
    df.tail(10).to_string(
        index=False
    )
)


print(
    f"\nSaved → {OUTPUT_FILE}"
)