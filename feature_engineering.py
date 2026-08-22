import pandas as pd
import numpy as np

# --------------------------------
# Load data
# --------------------------------

df = pd.read_csv("trend_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["topic", "date"])


# --------------------------------
# Basic growth
# --------------------------------

df["growth"] = (
    df.groupby("topic")["interest"].diff()
)

df["growth_rate"] = (
    df.groupby("topic")["interest"]
    .pct_change()
)


# --------------------------------
# Acceleration
# --------------------------------

df["previous_growth"] = (
    df.groupby("topic")["growth"]
    .shift(1)
)

df["acceleration"] = (
    df["growth"] -
    df["previous_growth"]
)


# --------------------------------
# Rolling averages
# --------------------------------

df["mean_4w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(4).mean()
    )
)

df["mean_8w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(8).mean()
    )
)


# --------------------------------
# Momentum
# --------------------------------

df["momentum"] = (
    df["mean_4w"] -
    df["mean_8w"]
)


# --------------------------------
# Volatility
# --------------------------------

df["volatility_4w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(4).std()
    )
)


# --------------------------------
# Local peak
# --------------------------------

df["peak_8w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x: x.rolling(8).max()
    )
)

df["distance_from_peak"] = (
    1 -
    df["interest"] /
    df["peak_8w"]
)


# --------------------------------
# Upward consistency
# --------------------------------

def upward_ratio(x):

    if len(x) < 2:
        return np.nan

    differences = np.diff(x)

    return np.mean(
        differences > 0
    )


df["upward_consistency"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x:
        x.rolling(4)
        .apply(
            upward_ratio,
            raw=True
        )
    )
)


# --------------------------------
# Short-term slope
# --------------------------------

def calculate_slope(x):

    if len(x) < 2:
        return np.nan

    return np.polyfit(
        np.arange(len(x)),
        x,
        1
    )[0]


df["slope_4w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x:
        x.rolling(4)
        .apply(
            calculate_slope,
            raw=True
        )
    )
)


# --------------------------------
# Medium-term slope
# --------------------------------

df["slope_8w"] = (
    df.groupby("topic")["interest"]
    .transform(
        lambda x:
        x.rolling(8)
        .apply(
            calculate_slope,
            raw=True
        )
    )
)


# --------------------------------
# Save
# --------------------------------

df.to_csv(
    "trend_features.csv",
    index=False
)


# --------------------------------
# Display
# --------------------------------

print("\nFeature dataset created!")

print(
    "\nColumns:"
)

for column in df.columns:
    print(" -", column)

print(
    "\nShape:",
    df.shape
)

print(
    "\nSample:"
)

print(
    df.tail(10).to_string(index=False)
)

print(
    "\nSaved → trend_features.csv"
)