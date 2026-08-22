import os
import pandas as pd


LIVE_FILE = "data/live_trends.csv"
HISTORY_FILE = "data/trend_history.csv"


# ============================================================
# LOAD LIVE DATA
# ============================================================

def load_live_data():

    if not os.path.exists(LIVE_FILE):
        print(f"❌ Live trends file not found: {LIVE_FILE}")
        raise SystemExit

    live = pd.read_csv(LIVE_FILE)

    if live.empty:
        print("❌ Live trends file is empty.")
        raise SystemExit

    required = [
        "topic",
        "traffic",
        "published"
    ]

    missing = [
        column
        for column in required
        if column not in live.columns
    ]

    if missing:
        print(f"❌ Missing columns: {missing}")
        raise SystemExit

    # --------------------------------------------------------
    # Clean topics
    # --------------------------------------------------------

    live["topic"] = (
        live["topic"]
        .astype(str)
        .str.strip()
    )

    live = live[
        live["topic"].ne("")
    ]

    # --------------------------------------------------------
    # Convert traffic bucket to number
    #
    # Examples:
    # 200+   → 200
    # 1,000+ → 1000
    # 5000+  → 5000
    # --------------------------------------------------------

    live["traffic_number"] = (
        live["traffic"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )

    live["traffic_number"] = pd.to_numeric(
        live["traffic_number"],
        errors="coerce"
    )

    live = live.dropna(
        subset=[
            "topic",
            "traffic_number"
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # One topic should appear only once in a snapshot.
    #
    # If duplicate topics exist, keep the highest traffic.
    # --------------------------------------------------------

    live = (
        live.sort_values(
            "traffic_number",
            ascending=False
        )
        .drop_duplicates(
            subset=["topic"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return live


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    history = pd.read_csv(
        HISTORY_FILE
    )

    if history.empty:
        return history

    history["fetched_at"] = pd.to_datetime(
        history["fetched_at"],
        errors="coerce"
    )

    history["traffic_number"] = pd.to_numeric(
        history["traffic_number"],
        errors="coerce"
    )

    history["topic"] = (
        history["topic"]
        .astype(str)
        .str.strip()
    )

    history = history.dropna(
        subset=[
            "topic",
            "fetched_at",
            "traffic_number"
        ]
    )

    return history


# ============================================================
# APPEND CURRENT SNAPSHOT
# ============================================================

def append_live_snapshot():

    live = load_live_data()

    history = load_history()

    # --------------------------------------------------------
    # Create ONE timestamp for the entire snapshot
    # --------------------------------------------------------

    fetched_at = pd.Timestamp.now()

    live["fetched_at"] = fetched_at

    # --------------------------------------------------------
    # Keep required history columns
    # --------------------------------------------------------

    snapshot = live[
        [
            "topic",
            "traffic",
            "traffic_number",
            "published",
            "fetched_at"
        ]
    ].copy()

    # --------------------------------------------------------
    # Prevent accidental duplicate snapshot
    #
    # Compare topics from the latest existing snapshot.
    # --------------------------------------------------------

    if not history.empty:

        latest_time = history["fetched_at"].max()

        latest_snapshot = history[
            history["fetched_at"] == latest_time
        ]

        old_topics = set(
            latest_snapshot["topic"]
            .astype(str)
            .str.strip()
        )

        new_topics = set(
            snapshot["topic"]
            .astype(str)
            .str.strip()
        )

        if old_topics == new_topics:

            print(
                "\n⚠️ Same topic set as previous snapshot."
            )

            print(
                "⚠️ Snapshot was not appended."
            )

            return history

    # --------------------------------------------------------
    # Append new snapshot
    # --------------------------------------------------------

    history = pd.concat(
        [
            history,
            snapshot
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final safety:
    # One topic per snapshot.
    # --------------------------------------------------------

    history = (
        history.sort_values(
            [
                "fetched_at",
                "traffic_number"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            subset=[
                "fetched_at",
                "topic"
            ],
            keep="first"
        )
        .reset_index(drop=True)
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    print(
        "\n✅ New live snapshot appended!"
    )

    print(
        f"📊 Snapshot topics: {len(snapshot)}"
    )

    print(
        f"📊 Total observations: {len(history)}"
    )

    print(
        f"📅 Total snapshots: "
        f"{history['fetched_at'].nunique()}"
    )

    print(
        f"📁 History: {HISTORY_FILE}"
    )

    return history


# ============================================================
# PREPARE SNAPSHOT RANKS
# ============================================================

def prepare_ranks(df):

    df = df.copy()

    df["rank"] = (
        df.groupby("fetched_at")[
            "traffic_number"
        ]
        .rank(
            method="min",
            ascending=False
        )
    )

    return df


# ============================================================
# CALCULATE MOMENTUM
# ============================================================

def calculate_momentum(df):

    df = df.copy()

    df = df.sort_values(
        [
            "topic",
            "fetched_at"
        ]
    )

    # --------------------------------------------------------
    # Previous traffic
    # --------------------------------------------------------

    df["previous_traffic"] = (
        df.groupby("topic")[
            "traffic_number"
        ]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous rank
    # --------------------------------------------------------

    df["previous_rank"] = (
        df.groupby("topic")[
            "rank"
        ]
        .shift(1)
    )

    # --------------------------------------------------------
    # Traffic change
    # --------------------------------------------------------

    df["traffic_change"] = (
        df["traffic_number"]
        - df["previous_traffic"]
    )

    # --------------------------------------------------------
    # Growth %
    # --------------------------------------------------------

    df["growth_percent"] = 0.0

    valid_growth = (
        df["previous_traffic"].notna()
        & (
            df["previous_traffic"] > 0
        )
    )

    df.loc[
        valid_growth,
        "growth_percent"
    ] = (
        (
            (
                df.loc[
                    valid_growth,
                    "traffic_number"
                ]
                -
                df.loc[
                    valid_growth,
                    "previous_traffic"
                ]
            )
            /
            df.loc[
                valid_growth,
                "previous_traffic"
            ]
        )
        * 100
    )

    # --------------------------------------------------------
    # Rank change
    #
    # Positive = moving UP
    # Negative = moving DOWN
    # --------------------------------------------------------

    df["rank_change"] = (
        df["previous_rank"]
        - df["rank"]
    )

    df["rank_change"] = (
        df["rank_change"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Appearance count
    # --------------------------------------------------------

    df["appearance_count"] = (
        df.groupby("topic")
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Total snapshots
    # --------------------------------------------------------

    total_snapshots = (
        df["fetched_at"]
        .nunique()
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    appearance_totals = (
        df.groupby("topic")[
            "fetched_at"
        ]
        .nunique()
    )

    df["persistence_score"] = (
        df["topic"]
        .map(appearance_totals)
        / total_snapshots
        * 100
    )

    # --------------------------------------------------------
    # Rank momentum
    # --------------------------------------------------------

    df["rank_momentum_score"] = (
        df["rank_change"] * 15
    ).clip(
        lower=-100,
        upper=100
    )

    # --------------------------------------------------------
    # Traffic level
    # --------------------------------------------------------

    max_traffic = (
        df["traffic_number"].max()
    )

    if max_traffic > 0:

        df["traffic_level_score"] = (
            df["traffic_number"]
            / max_traffic
            * 100
        )

    else:

        df["traffic_level_score"] = 0

    # --------------------------------------------------------
    # Momentum score
    # --------------------------------------------------------

    df["momentum_score"] = (
        df["rank_momentum_score"] * 0.40
        + df["persistence_score"] * 0.35
        + df["traffic_level_score"] * 0.25
    )

    df["momentum_score"] = (
        df["momentum_score"]
        .clip(0, 100)
        .round(1)
    )

    return df


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

def get_current_snapshot(df):

    latest_time = (
        df["fetched_at"].max()
    )

    current = df[
        df["fetched_at"] == latest_time
    ].copy()

    # --------------------------------------------------------
    # Safety guarantee:
    # ONE row per topic.
    # --------------------------------------------------------

    current = (
        current.sort_values(
            "traffic_number",
            ascending=False
        )
        .drop_duplicates(
            subset=["topic"],
            keep="first"
        )
    )

    return current


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("🚀 TrendRadar Momentum Engine")
print("=" * 80)


# ============================================================
# STEP 1 — APPEND LIVE SNAPSHOT
# ============================================================

history = append_live_snapshot()


# ============================================================
# STEP 2 — VALIDATE HISTORY
# ============================================================

if history.empty:

    print(
        "\n❌ No trend history found."
    )

    raise SystemExit


print(
    f"\n📊 Total observations: "
    f"{len(history)}"
)

print(
    f"📅 Snapshots: "
    f"{history['fetched_at'].nunique()}"
)


# ============================================================
# STEP 3 — PREPARE RANKS
# ============================================================

history = prepare_ranks(
    history
)


# ============================================================
# STEP 4 — CALCULATE MOMENTUM
# ============================================================

history = calculate_momentum(
    history
)


# ============================================================
# STEP 5 — CURRENT SNAPSHOT
# ============================================================

current = get_current_snapshot(
    history
)


# ============================================================
# DISPLAY
# ============================================================

display_columns = [
    "topic",
    "traffic_number",
    "previous_traffic",
    "traffic_change",
    "growth_percent",
    "rank",
    "previous_rank",
    "rank_change",
    "appearance_count",
    "persistence_score",
    "traffic_level_score",
    "momentum_score",
]


display = current[
    [
        column
        for column in display_columns
        if column in current.columns
    ]
].copy()


display = display.sort_values(
    [
        "momentum_score",
        "rank_change",
        "traffic_number"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


print("\n")

print(
    "=" * 125
)

print(
    "📈 CURRENT TREND MOMENTUM"
)

print(
    "=" * 125
)


print(
    display.to_string(
        index=False,
        formatters={

            "growth_percent":
                lambda x:
                f"{x:.1f}%",

            "persistence_score":
                lambda x:
                f"{x:.1f}",

            "traffic_level_score":
                lambda x:
                f"{x:.1f}",

            "momentum_score":
                lambda x:
                f"{x:.1f}",
        }
    )
)


print(
    "=" * 125
)


# ============================================================
# SAVE
# ============================================================

history.to_csv(
    HISTORY_FILE,
    index=False
)


print(
    "\n✅ Momentum data updated:"
)

print(
    f"📁 {HISTORY_FILE}"
)


# ============================================================
# VALIDATION
# ============================================================

unique_topics = (
    current["topic"]
    .nunique()
)

total_current_rows = (
    len(current)
)

print(
    "\n🔎 SNAPSHOT VALIDATION"
)

print(
    f"   Current snapshot rows → "
    f"{total_current_rows}"
)

print(
    f"   Unique current topics → "
    f"{unique_topics}"
)

if total_current_rows == unique_topics:

    print(
        "   ✅ One row per current topic."
    )

else:

    print(
        "   ❌ Duplicate current topics detected."
    )


# ============================================================
# TOP MOMENTUM
# ============================================================

if not current.empty:

    top = (
        current
        .sort_values(
            "momentum_score",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\n🔥 TOP MOMENTUM"
    )

    print(
        f"   {top['topic']} "
        f"→ {top['momentum_score']:.1f}"
    )


print(
    "\n🎉 Momentum engine completed!"
)