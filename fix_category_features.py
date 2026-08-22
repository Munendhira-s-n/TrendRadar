import pandas as pd
import numpy as np

INPUT_FILE = "trend_data_v2.csv"
OUTPUT_FILE = "trend_features_v2.csv"


# ==========================================
# 1. Load
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "category", "topic"]
).copy()


# ==========================================
# 2. Topic-level features
# ==========================================

g = df.groupby("topic")

df["previous_interest"] = (
    g["interest"].shift(1)
)

df["growth"] = (
    df["interest"] -
    df["previous_interest"]
)

df["growth_rate"] = (
    df["growth"] /
    df["previous_interest"].replace(0, np.nan)
)

df["previous_growth"] = (
    g["growth"].shift(1)
)

df["acceleration"] = (
    df["growth"] -
    df["previous_growth"]
)


df["mean_4w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(4, min_periods=2).mean()
    )
)

df["mean_8w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(8, min_periods=3).mean()
    )
)

df["momentum"] = (
    df["mean_4w"] -
    df["mean_8w"]
)

df["volatility_4w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(4, min_periods=2).std()
    )
)

df["peak_8w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(8, min_periods=2).max()
    )
)

df["distance_from_peak"] = (
    (df["peak_8w"] - df["interest"])
    /
    df["peak_8w"].replace(0, np.nan)
)


def upward_ratio(x):
    changes = x.diff()

    if changes.notna().sum() == 0:
        return np.nan

    return (
        (changes > 0).sum()
        /
        changes.notna().sum()
    )


df["upward_consistency"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=2
        ).apply(
            upward_ratio,
            raw=False
        )
    )
)


def slope(x):
    if len(x) < 2:
        return np.nan

    return np.polyfit(
        np.arange(len(x)),
        np.array(x),
        1
    )[0]


df["slope_4w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=3
        ).apply(
            slope,
            raw=False
        )
    )
)

df["slope_8w"] = (
    g["interest"]
    .transform(
        lambda x:
        x.rolling(
            8,
            min_periods=4
        ).apply(
            slope,
            raw=False
        )
    )
)


# ==========================================
# 3. BUILD ONE ROW PER CATEGORY/WEEK
# ==========================================

category_weekly = (
    df.groupby(
        ["date", "category"],
        as_index=False
    )["interest"]
    .mean()
    .rename(
        columns={
            "interest":
                "category_mean_interest"
        }
    )
)


category_weekly = category_weekly.sort_values(
    ["category", "date"]
)


# ==========================================
# 4. Category features
# ==========================================

cg = category_weekly.groupby("category")


category_weekly["category_mean_4w"] = (
    cg["category_mean_interest"]
    .transform(
        lambda x:
        x.rolling(
            4,
            min_periods=2
        ).mean()
    )
)

category_weekly["category_mean_8w"] = (
    cg["category_mean_interest"]
    .transform(
        lambda x:
        x.rolling(
            8,
            min_periods=3
        ).mean()
    )
)

category_weekly["category_momentum"] = (
    category_weekly["category_mean_4w"]
    -
    category_weekly["category_mean_8w"]
)


category_weekly["category_growth"] = (
    cg["category_mean_interest"].diff()
)

category_weekly["category_previous_growth"] = (
    cg["category_growth"].shift(1)
)

category_weekly["category_acceleration"] = (
    category_weekly["category_growth"]
    -
    category_weekly["category_previous_growth"]
)


# ==========================================
# 5. Merge category features
# ==========================================

category_features = category_weekly[
    [
        "date",
        "category",
        "category_mean_interest",
        "category_momentum",
        "category_growth",
        "category_previous_growth",
        "category_acceleration"
    ]
]


df = df.drop(
    columns=[
        "category_mean_interest",
        "category_momentum",
        "category_growth",
        "category_previous_growth",
        "category_acceleration"
    ],
    errors="ignore"
)


df = df.merge(
    category_features,
    on=["date", "category"],
    how="left"
)


# ==========================================
# 6. Relative features
# ==========================================

df["relative_interest"] = (
    df["interest"] /
    df["category_mean_interest"].replace(
        0,
        np.nan
    )
)

df["relative_momentum"] = (
    df["momentum"] -
    df["category_momentum"]
)


# ==========================================
# 7. Save
# ==========================================

df = df.sort_values(
    ["date", "topic"]
).reset_index(drop=True)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 8. Summary
# ==========================================

print("\n")
print("=" * 80)
print("       🔥 TRENDRADAR V2 CATEGORY FIX")
print("=" * 80)

print(
    f"\nRows       : {len(df):,}"
)

print(
    f"Topics     : {df['topic'].nunique()}"
)

print(
    f"Categories : {df['category'].nunique()}"
)

print(
    f"Columns    : {len(df.columns)}"
)

print(
    "\nCategory features now calculated "
    "at weekly category level."
)

print(
    f"\nSaved → {OUTPUT_FILE}"
)