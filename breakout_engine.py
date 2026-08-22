import os
import pandas as pd
import numpy as np


FEATURE_FILE = "data/trend_history.csv"
OUTPUT_FILE = "data/breakout_signals.csv"


# ============================================================
# LOAD MOMENTUM DATA
# ============================================================

def load_data():

    if not os.path.exists(FEATURE_FILE):

        print("❌ Momentum data not found.")
        print(f"Expected file: {FEATURE_FILE}")

        raise SystemExit

    df = pd.read_csv(FEATURE_FILE)

    if df.empty:

        print("❌ Momentum data is empty.")

        raise SystemExit

    return df


# ============================================================
# GET CURRENT SNAPSHOT
# ============================================================

def get_current_snapshot(df):
    
    df = df.copy()

    df["fetched_at"] = pd.to_datetime(
        df["fetched_at"],
        errors="coerce"
    )

    # Get the latest observation for EACH topic.
    current = (
        df.sort_values("fetched_at")
        .groupby("topic", as_index=False)
        .tail(1)
        .copy()
    )

    # Sort by latest fetch time for clean output.
    current = current.sort_values(
        "fetched_at"
    ).reset_index(drop=True)

    return current
# ============================================================
# CALCULATE BREAKOUT SCORE
# ============================================================

def calculate_breakout_score(df):

    df = df.copy()

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    momentum = (
        df["momentum_score"]
        .fillna(0)
        .clip(0, 100)
    )

    persistence = (
        df["persistence_score"]
        .fillna(0)
        .clip(0, 100)
    )

    traffic_level = (
        df["traffic_level_score"]
        .fillna(0)
        .clip(0, 100)
    )

    rank_change = (
        df["rank_change"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Rank movement score
    #
    # Positive rank movement means the topic is climbing.
    # --------------------------------------------------------

    rank_movement_score = (
        rank_change * 20
    ).clip(
        0,
        100
    )

    # --------------------------------------------------------
    # Breakout score
    #
    # Momentum       45%
    # Persistence    20%
    # Traffic        20%
    # Rank movement  15%
    # --------------------------------------------------------

    df["breakout_score"] = (
        momentum * 0.45
        + persistence * 0.20
        + traffic_level * 0.20
        + rank_movement_score * 0.15
    )

    df["breakout_score"] = (
        df["breakout_score"]
        .clip(0, 100)
        .round(1)
    )

    # --------------------------------------------------------
    # Probability
    #
    # This is a normalized intelligence score.
    # It is NOT a statistically calibrated probability.
    # --------------------------------------------------------

    df["breakout_probability"] = (
        df["breakout_score"]
        .clip(0, 100)
        .round(1)
    )

    return df


# ============================================================
# CLASSIFY BREAKOUT
# ============================================================

def classify_breakout(row):

    score = row["breakout_score"]

    rank_change = row["rank_change"]

    persistence = row["persistence_score"]

    # --------------------------------------------------------
    # HOT
    # --------------------------------------------------------

    if (
        score >= 70
        and (
            rank_change > 0
            or persistence >= 75
        )
    ):

        return "🔥 HOT"

    # --------------------------------------------------------
    # RISING
    # --------------------------------------------------------

    if (
        score >= 50
        and (
            rank_change > 0
            or persistence >= 50
        )
    ):

        return "📈 RISING"

    # --------------------------------------------------------
    # EMERGING
    # --------------------------------------------------------

    if score >= 35:

        return "🟢 EMERGING"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "⚪ LOW"


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("🚀 TrendRadar Breakout Engine")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

df = load_data()


print(
    f"\n📊 Total observations: {len(df)}"
)


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

current = get_current_snapshot(df)


if current.empty:

    print("\n❌ No current snapshot found.")

    raise SystemExit


# ============================================================
# CALCULATE
# ============================================================

current = calculate_breakout_score(
    current
)


current["breakout_status"] = (
    current.apply(
        classify_breakout,
        axis=1
    )
)


# ============================================================
# RANK
# ============================================================

current = current.sort_values(
    [
        "breakout_score",
        "momentum_score",
        "traffic_number"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(
    drop=True
)


current["breakout_rank"] = (
    current.index + 1
)


# ============================================================
# DISPLAY
# ============================================================

display_columns = [
    "breakout_rank",
    "topic",
    "traffic_number",
    "rank_change",
    "persistence_score",
    "momentum_score",
    "traffic_level_score",
    "breakout_score",
    "breakout_probability",
    "breakout_status",
]


display = current[
    display_columns
].copy()


print("\n")
print("=" * 125)
print("📊 CURRENT BREAKOUT SIGNALS")
print("=" * 125)


print(
    display.to_string(
        index=False,
        formatters={

            "rank_change":
                lambda x: f"{x:+.0f}",

            "persistence_score":
                lambda x: f"{x:.1f}",

            "momentum_score":
                lambda x: f"{x:.1f}",

            "traffic_level_score":
                lambda x: f"{x:.1f}",

            "breakout_score":
                lambda x: f"{x:.1f}",

            "breakout_probability":
                lambda x: f"{x:.1f}",
        }
    )
)


print("=" * 125)


# ============================================================
# SAVE
# ============================================================

current.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n✅ Breakout engine completed!"
)

print(
    f"\n📁 Saved to:"
)

print(
    f"   {OUTPUT_FILE}"
)

print(
    f"\n📊 Unique trends analyzed: "
    f"{len(current)}"
)


# ============================================================
# TOP BREAKOUT
# ============================================================

if not current.empty:

    top = current.iloc[0]

    print(
        "\n🎯 Highest breakout candidate:"
    )

    print(
        f"   {top['topic']} "
        f"→ {top['breakout_probability']:.1f} "
        f"({top['breakout_status']})"
    )


print(
    "\n🎉 TrendRadar breakout engine is working!"
)