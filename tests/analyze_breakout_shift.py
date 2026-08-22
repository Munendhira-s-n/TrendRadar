import pandas as pd

df = pd.read_csv(
    "trend_features_v3.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)


# ==========================================
# Monthly breakout rate
# ==========================================

df["month"] = (
    df["date"]
    .dt.to_period("M")
    .astype(str)
)


monthly = (
    df.groupby("month")
    .agg(
        observations=("breakout", "count"),
        breakouts=("breakout", "sum")
    )
)

monthly["breakout_rate"] = (
    monthly["breakouts"]
    /
    monthly["observations"]
    *
    100
)


print("\n")
print("=" * 70)
print("          📊 BREAKOUT DISTRIBUTION ANALYSIS")
print("=" * 70)

print(
    monthly.to_string()
)


# ==========================================
# Training vs Holdout
# ==========================================

train = df[
    df["date"] <= "2026-05-17"
]

holdout = df[
    df["date"] >= "2026-05-24"
]


print("\n")
print("=" * 70)
print("              TRAIN vs HOLDOUT")
print("=" * 70)

print(
    f"\nTraining observations : {len(train):,}"
)

print(
    f"Training breakouts    : "
    f"{train['breakout'].sum():,}"
)

print(
    f"Training rate         : "
    f"{train['breakout'].mean() * 100:.2f}%"
)


print(
    f"\nHoldout observations  : {len(holdout):,}"
)

print(
    f"Holdout breakouts     : "
    f"{holdout['breakout'].sum():,}"
)

print(
    f"Holdout rate          : "
    f"{holdout['breakout'].mean() * 100:.2f}%"
)


# ==========================================
# Category shift
# ==========================================

print("\n")
print("=" * 70)
print("           CATEGORY BREAKOUT SHIFT")
print("=" * 70)


category = (
    df.groupby("category")
    .agg(
        observations=("breakout", "count"),
        breakouts=("breakout", "sum")
    )
)

category["rate"] = (
    category["breakouts"]
    /
    category["observations"]
    *
    100
)


print(
    category.sort_values(
        "rate",
        ascending=False
    ).to_string()
)


# ==========================================
# Save
# ==========================================

monthly.to_csv(
    "breakout_monthly_analysis.csv"
)

category.to_csv(
    "breakout_category_analysis.csv"
)

print(
    "\nSaved:"
)

print(
    "→ breakout_monthly_analysis.csv"
)

print(
    "→ breakout_category_analysis.csv"
)