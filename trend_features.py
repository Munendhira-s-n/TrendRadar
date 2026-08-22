# ============================================================
# TrendRadar — Trend Feature Engine
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

HISTORY_FILE = os.path.join(
    "data",
    "trend_history.csv"
)

FEATURE_FILE = os.path.join(
    "data",
    "trend_features.csv"
)


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):

        print("❌ History file not found:")
        print(HISTORY_FILE)

        return pd.DataFrame()

    df = pd.read_csv(HISTORY_FILE)

    if df.empty:

        print("❌ History file is empty.")

        return pd.DataFrame()

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Numeric traffic
    # --------------------------------------------------------

    df["traffic_number"] = pd.to_numeric(
        df["traffic_number"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["fetched_at"] = pd.to_datetime(
        df["fetched_at"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["fetched_at"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["topic", "fetched_at"]
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    grouped = df.groupby("topic")


    # ========================================================
    # 1. PREVIOUS TRAFFIC
    # ========================================================

    df["previous_traffic"] = (
        grouped["traffic_number"]
        .shift(1)
    )


    # ========================================================
    # 2. TRAFFIC CHANGE
    # ========================================================

    df["traffic_change"] = (
        df["traffic_number"]
        - df["previous_traffic"]
    )


    # ========================================================
    # 3. GROWTH %
    # ========================================================

    df["growth_percent"] = 0.0

    valid_previous = (
        df["previous_traffic"] > 0
    )

    df.loc[
        valid_previous,
        "growth_percent"
    ] = (
        (
            df.loc[
                valid_previous,
                "traffic_number"
            ]
            -
            df.loc[
                valid_previous,
                "previous_traffic"
            ]
        )
        /
        df.loc[
            valid_previous,
            "previous_traffic"
        ]
        * 100
    )


    # ========================================================
    # 4. OBSERVATION NUMBER
    # ========================================================

    df["observation_number"] = (
        grouped.cumcount() + 1
    )


    # ========================================================
    # 5. FIRST SEEN
    # ========================================================

    df["first_seen"] = (
        grouped["fetched_at"]
        .transform("min")
    )


    # ========================================================
    # 6. LAST SEEN
    # ========================================================

    df["last_seen"] = (
        grouped["fetched_at"]
        .transform("max")
    )


    # ========================================================
    # 7. TIME ACTIVE
    # ========================================================

    df["hours_active"] = (
        (
            df["last_seen"]
            - df["first_seen"]
        )
        .dt.total_seconds()
        / 3600
    )


    # ========================================================
    # 8. RECENT TRAFFIC AVERAGE
    # ========================================================

    df["recent_average"] = (
        grouped["traffic_number"]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )


    # ========================================================
    # 9. PREVIOUS AVERAGE
    # ========================================================

    df["previous_average"] = (
        grouped["traffic_number"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )


    # ========================================================
    # 10. ACCELERATION
    # ========================================================

    df["acceleration"] = (
        df["recent_average"]
        - df["previous_average"]
    )


    # ========================================================
    # 11. RECENT OBSERVATION COUNT
    # ========================================================

    df["recent_observations"] = (
        grouped["topic"]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1
            ).count()
        )
    )


    # ========================================================
    # 12. TRAFFIC VOLATILITY
    # ========================================================

    df["volatility"] = (
        grouped["traffic_number"]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2
            ).std()
        )
        .fillna(0)
    )


    # ========================================================
    # 13. PERSISTENCE SCORE
    # ========================================================

    df["persistence_score"] = (
        np.minimum(
            df["observation_number"] * 20,
            100
        )
    )


    # ========================================================
    # 14. MOMENTUM SCORE
    # ========================================================

    df["momentum_score"] = (
        df["growth_percent"]
        .clip(-100, 100)
        + 100
    ) / 2


    # ========================================================
    # 15. ACCELERATION SCORE
    # ========================================================

    df["acceleration_score"] = (
        df["acceleration"]
        .clip(-100, 100)
        + 100
    ) / 2


    # ========================================================
    # 16. ACTIVITY SCORE
    # ========================================================

    df["activity_score"] = (
        df["traffic_number"]
        .clip(0, 1000)
        / 1000
        * 100
    )


    # ========================================================
    # 17. TREND SCORE
    # ========================================================

    df["trend_score"] = (
        df["activity_score"] * 0.35
        +
        df["momentum_score"] * 0.25
        +
        df["acceleration_score"] * 0.20
        +
        df["persistence_score"] * 0.20
    )


    df["trend_score"] = (
        df["trend_score"]
        .clip(0, 100)
        .round(1)
    )


    return df


# ============================================================
# STATUS CLASSIFICATION
# ============================================================

def classify_status(row):

    score = row["trend_score"]
    traffic = row["traffic_number"]
    growth = row["growth_percent"]
    acceleration = row["acceleration"]
    observations = row["observation_number"]


    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if (
        score >= 75
        and traffic >= 500
        and (
            growth > 25
            or acceleration > 50
        )
    ):

        return "🚀 BREAKOUT"


    # --------------------------------------------------------
    # HOT
    # --------------------------------------------------------

    if (
        traffic >= 1000
        and observations >= 2
    ):

        return "🔥 HOT"


    # --------------------------------------------------------
    # RISING
    # --------------------------------------------------------

    if (
        growth > 10
        or acceleration > 20
    ):

        return "📈 RISING"


    # --------------------------------------------------------
    # EMERGING
    # --------------------------------------------------------

    if (
        observations >= 2
        and traffic >= 200
    ):

        return "🟢 EMERGING"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "⚪ LOW"


# ============================================================
# APPLY STATUS
# ============================================================

def add_status(df):

    df = df.copy()

    df["status"] = df.apply(
        classify_status,
        axis=1
    )

    return df


# ============================================================
# GET LATEST SNAPSHOT
# ============================================================

def get_latest_snapshot(df):

    latest = (
        df
        .sort_values("fetched_at")
        .groupby("topic")
        .tail(1)
        .copy()
    )

    latest = latest.sort_values(
        "trend_score",
        ascending=False
    ).reset_index(drop=True)

    latest["rank"] = (
        latest.index + 1
    )

    return latest


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(df):

    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        FEATURE_FILE,
        index=False
    )

    print()
    print(
        f"✅ Feature data saved:"
    )

    print(
        f"📁 {FEATURE_FILE}"
    )

    print(
        f"📊 Observations: {len(df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "🚀 TrendRadar Feature Engine"
    )

    print(
        "=" * 70
    )

    # Load
    history = load_history()

    if history.empty:
        return

    # Prepare
    history = prepare_data(
        history
    )

    # Create features
    features = create_features(
        history
    )

    # Status
    features = add_status(
        features
    )

    # Save
    save_features(
        features
    )

    # Latest snapshot
    latest = get_latest_snapshot(
        features
    )

    print()
    print(
        "=" * 100
    )

    print(
        "📊 CURRENT TREND SIGNALS"
    )

    print(
        "=" * 100
    )

    columns = [
        "rank",
        "topic",
        "traffic_number",
        "growth_percent",
        "acceleration",
        "persistence_score",
        "trend_score",
        "status"
    ]

    display = latest[
        columns
    ].copy()

    display["growth_percent"] = (
        display["growth_percent"]
        .round(1)
    )

    display["acceleration"] = (
        display["acceleration"]
        .round(1)
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "🎉 Trend feature engine is working!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()