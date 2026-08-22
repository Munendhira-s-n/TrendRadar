import os
import pandas as pd


HISTORY_FILE = "data/trend_history.csv"
CATEGORY_FILE = "data/categorized_trends.csv"
BREAKOUT_FILE = "data/breakout_signals.csv"
OUTPUT_FILE = "data/trend_intelligence.csv"


# ============================================================
# LOAD FILE
# ============================================================

def load_file(path):

    if not os.path.exists(path):

        print(f"❌ File not found: {path}")
        raise SystemExit

    df = pd.read_csv(path)

    if df.empty:

        print(f"❌ File is empty: {path}")
        raise SystemExit

    return df


# ============================================================
# PREPARE HISTORY
# ============================================================

def prepare_history(history):

    history = history.copy()

    history["fetched_at"] = pd.to_datetime(
        history["fetched_at"],
        errors="coerce"
    )

    history["topic"] = (
        history["topic"]
        .astype(str)
        .str.strip()
    )

    history["traffic_number"] = pd.to_numeric(
        history["traffic_number"],
        errors="coerce"
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
# CURRENT SNAPSHOT
# ============================================================

def get_current_snapshot(history):
    
    latest_time = history["fetched_at"].max()

    current = history[
        history["fetched_at"] == latest_time
    ].copy()

    # Keep exactly one row per topic
    current = (
        current
        .sort_values(
            ["topic", "traffic_number"],
            ascending=[True, False]
        )
        .drop_duplicates(
            subset=["topic"],
            keep="first"
        )
    )

    return current

# ============================================================
# CATEGORY LOOKUP
# ============================================================

def build_category_lookup(
    current,
    categories
):

    if "topic" not in categories.columns:
        return {}

    if "category" not in categories.columns:
        return {}

    categories = categories.copy()

    categories["topic"] = (
        categories["topic"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # If category file contains fetched_at, use only the
    # latest category snapshot.
    # --------------------------------------------------------

    if "fetched_at" in categories.columns:

        categories["fetched_at"] = pd.to_datetime(
            categories["fetched_at"],
            errors="coerce"
        )

        latest_category_time = (
            categories["fetched_at"].max()
        )

        categories = categories[
            categories["fetched_at"]
            == latest_category_time
        ]

    # --------------------------------------------------------
    # Keep only topics that exist in current snapshot.
    # --------------------------------------------------------

    current_topics = set(
        current["topic"]
        .astype(str)
        .str.strip()
    )

    categories = categories[
        categories["topic"].isin(
            current_topics
        )
    ]

    categories = categories.drop_duplicates(
        subset=["topic"],
        keep="last"
    )

    lookup = dict(
        zip(
            categories["topic"],
            categories["category"]
        )
    )

    return lookup


# ============================================================
# BREAKOUT LOOKUP
# ============================================================

def prepare_breakout(
    current,
    breakout
):

    breakout = breakout.copy()

    if "topic" not in breakout.columns:

        return pd.DataFrame(
            columns=[
                "topic",
                "breakout_score",
                "breakout_probability",
                "breakout_status"
            ]
        )

    breakout["topic"] = (
        breakout["topic"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # If breakout file contains fetched_at, select the latest
    # snapshot.
    # --------------------------------------------------------

    if "fetched_at" in breakout.columns:

        breakout["fetched_at"] = pd.to_datetime(
            breakout["fetched_at"],
            errors="coerce"
        )

        latest_breakout_time = (
            breakout["fetched_at"].max()
        )

        breakout = breakout[
            breakout["fetched_at"]
            == latest_breakout_time
        ]

    # --------------------------------------------------------
    # Only keep current topics.
    # --------------------------------------------------------

    current_topics = set(
        current["topic"]
        .astype(str)
        .str.strip()
    )

    breakout = breakout[
        breakout["topic"].isin(
            current_topics
        )
    ]

    columns = [
        "topic",
        "breakout_score",
        "breakout_probability",
        "breakout_status",
    ]

    columns = [
        column
        for column in columns
        if column in breakout.columns
    ]

    breakout = breakout[
        columns
    ].drop_duplicates(
        subset=["topic"],
        keep="last"
    )

    return breakout


# ============================================================
# INTELLIGENCE CLASSIFICATION
# ============================================================

def classify_signal(row):

    traffic = row["traffic_number"]

    previous = row["previous_traffic"]

    growth = row["growth_percent"]

    rank_change = row["rank_change"]

    persistence = row["persistence_score"]

    # --------------------------------------------------------
    # FALLING
    # --------------------------------------------------------

    if (
        rank_change < 0
        and growth <= 0
    ):

        return "📉 FALLING"

    # --------------------------------------------------------
    # NEW TREND
    # --------------------------------------------------------

    if pd.isna(previous):

        if traffic >= 5000:
            return "🔥 NEW VIRAL"

        if traffic >= 1000:
            return "🔥 NEW HIGH TRAFFIC"

        if traffic >= 500:
            return "🟠 NEW TREND"

        return "🆕 NEW"

    # --------------------------------------------------------
    # RAPIDLY RISING
    # --------------------------------------------------------

    if (
        growth >= 100
        or rank_change >= 3
    ):

        return "🚀 RAPIDLY RISING"

    # --------------------------------------------------------
    # RISING
    # --------------------------------------------------------

    if (
        growth > 0
        or rank_change > 0
    ):

        return "📈 RISING"

    # --------------------------------------------------------
    # SUSTAINED
    # --------------------------------------------------------

    if (
        persistence >= 75
        and traffic >= 200
    ):

        return "🔵 SUSTAINED"

    # --------------------------------------------------------
    # LOW / STABLE
    # --------------------------------------------------------

    return "⚪ LOW / STABLE"


# ============================================================
# INTELLIGENCE SCORE
# ============================================================

def calculate_intelligence_score(row):

    traffic = row["traffic_number"]

    previous = row["previous_traffic"]

    growth = row["growth_percent"]

    rank_change = row["rank_change"]

    momentum = row["momentum_score"]

    breakout = row["breakout_score"]

    # --------------------------------------------------------
    # NEW TREND
    # --------------------------------------------------------

    if pd.isna(previous):

        if traffic >= 5000:
            return 90.0

        if traffic >= 1000:
            return 80.0

        if traffic >= 500:
            return 60.0

        if traffic >= 200:
            return 30.0

        return 15.0

    # --------------------------------------------------------
    # MOVEMENT BONUS
    # --------------------------------------------------------

    movement_bonus = 0

    if growth >= 100:

        movement_bonus += 20

    elif growth > 0:

        movement_bonus += 10

    if rank_change >= 3:

        movement_bonus += 15

    elif rank_change > 0:

        movement_bonus += 7

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (
        momentum * 0.40
        + breakout * 0.40
        + movement_bonus * 0.20
    )

    return min(
        round(score, 1),
        100
    )


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(score):

    if score >= 70:

        return "HIGH"

    if score >= 45:

        return "MEDIUM"

    return "LOW"


# ============================================================
# BUILD INTELLIGENCE
# ============================================================

def build_intelligence(
    history,
    categories,
    breakout
):

    current = get_current_snapshot(
        history
    )

    if current.empty:

        print(
            "\n❌ Current snapshot is empty."
        )

        raise SystemExit

    # --------------------------------------------------------
    # Required history columns
    # --------------------------------------------------------

    required_columns = [
        "topic",
        "traffic_number",
        "previous_traffic",
        "growth_percent",
        "rank_change",
        "persistence_score",
        "momentum_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in current.columns
    ]

    if missing:

        print(
            "\n❌ Missing required history columns:"
        )

        print(missing)

        raise SystemExit

    current = current[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category_lookup = build_category_lookup(
        current,
        categories
    )

    current["category"] = (
        current["topic"]
        .astype(str)
        .str.strip()
        .map(category_lookup)
        .fillna("Other")
    )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    breakout = prepare_breakout(
        current,
        breakout
    )

    current = current.merge(
        breakout,
        on="topic",
        how="left"
    )

    # --------------------------------------------------------
    # Missing breakout values
    # --------------------------------------------------------

    if "breakout_score" not in current.columns:

        current["breakout_score"] = 0.0

    if "breakout_probability" not in current.columns:

        current["breakout_probability"] = 0.0

    if "breakout_status" not in current.columns:

        current["breakout_status"] = "⚪ LOW"

    current["breakout_score"] = (
        pd.to_numeric(
            current["breakout_score"],
            errors="coerce"
        )
        .fillna(0)
    )

    current["breakout_probability"] = (
        pd.to_numeric(
            current["breakout_probability"],
            errors="coerce"
        )
        .fillna(0)
    )

    current["breakout_status"] = (
        current["breakout_status"]
        .fillna("⚪ LOW")
    )

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    numeric_columns = [
        "traffic_number",
        "previous_traffic",
        "growth_percent",
        "rank_change",
        "persistence_score",
        "momentum_score",
        "breakout_score",
    ]

    for column in numeric_columns:

        current[column] = pd.to_numeric(
            current[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Intelligence signal
    # --------------------------------------------------------

    current["intelligence_signal"] = (
        current.apply(
            classify_signal,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Intelligence score
    # --------------------------------------------------------

    current["intelligence_score"] = (
        current.apply(
            calculate_intelligence_score,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    current["priority"] = (
        current["intelligence_score"]
        .apply(
            calculate_priority
        )
    )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    current = current.sort_values(
        [
            "intelligence_score",
            "traffic_number",
        ],
        ascending=[
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    current["intelligence_rank"] = (
        current.index + 1
    )

    return current


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("🧠 TrendRadar Trend Intelligence Engine")
print("=" * 90)


# ============================================================
# LOAD
# ============================================================

history = load_file(
    HISTORY_FILE
)

categories = load_file(
    CATEGORY_FILE
)

breakout = load_file(
    BREAKOUT_FILE
)


history = prepare_history(
    history
)


print(
    f"\n📊 History observations: {len(history)}"
)

print(
    f"📅 History snapshots: "
    f"{history['fetched_at'].nunique()}"
)

print(
    f"🏷️ Category records: {len(categories)}"
)

print(
    f"🚀 Breakout records: {len(breakout)}"
)


# ============================================================
# CURRENT SNAPSHOT INFO
# ============================================================

current_snapshot = get_current_snapshot(
    history
)

print(
    f"📌 Current snapshot topics: "
    f"{len(current_snapshot)}"
)


# ============================================================
# BUILD
# ============================================================

intelligence = build_intelligence(
    history,
    categories,
    breakout
)


# ============================================================
# DISPLAY
# ============================================================

display_columns = [
    "intelligence_rank",
    "topic",
    "category",
    "traffic_number",
    "growth_percent",
    "rank_change",
    "momentum_score",
    "breakout_score",
    "intelligence_score",
    "intelligence_signal",
    "priority",
]


display = intelligence[
    display_columns
].copy()


print("\n")
print("=" * 125)
print("🧠 CURRENT TREND INTELLIGENCE")
print("=" * 125)


print(
    display.to_string(
        index=False,
        formatters={

            "growth_percent":
                lambda x: f"{x:.1f}%",

            "rank_change":
                lambda x: f"{x:+.0f}",

            "momentum_score":
                lambda x: f"{x:.1f}",

            "breakout_score":
                lambda x: f"{x:.1f}",

            "intelligence_score":
                lambda x: f"{x:.1f}",
        }
    )
)


print("=" * 125)


# ============================================================
# SAVE
# ============================================================

intelligence.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n✅ Trend intelligence generated!"
)

print(
    f"📁 Saved to: {OUTPUT_FILE}"
)

print(
    f"📊 Trends analyzed: {len(intelligence)}"
)


# ============================================================
# VALIDATION
# ============================================================

current_topics = set(
    current_snapshot["topic"]
    .astype(str)
    .str.strip()
)

intelligence_topics = set(
    intelligence["topic"]
    .astype(str)
    .str.strip()
)


missing_topics = (
    current_topics
    - intelligence_topics
)

extra_topics = (
    intelligence_topics
    - current_topics
)


print("\n🔎 SNAPSHOT VALIDATION")

print(
    f"   Current snapshot topics → "
    f"{len(current_topics)}"
)

print(
    f"   Intelligence topics → "
    f"{len(intelligence_topics)}"
)


if missing_topics:

    print(
        "\n⚠️ Missing from intelligence:"
    )

    for topic in sorted(missing_topics):

        print(
            f"   - {topic}"
        )

else:

    print(
        "   ✅ All current topics are present."
    )


if extra_topics:

    print(
        "\n⚠️ Unexpected extra topics:"
    )

    for topic in sorted(extra_topics):

        print(
            f"   - {topic}"
        )

else:

    print(
        "   ✅ No stale topics detected."
    )


# ============================================================
# TOP SIGNAL
# ============================================================

if not intelligence.empty:

    top = intelligence.iloc[0]

    print(
        "\n🎯 TOP INTELLIGENCE SIGNAL"
    )

    print(
        f"   {top['topic']}"
    )

    print(
        f"   Category → {top['category']}"
    )

    print(
        f"   Signal → {top['intelligence_signal']}"
    )

    print(
        f"   Score → {top['intelligence_score']:.1f}"
    )

    print(
        f"   Priority → {top['priority']}"
    )


print(
    "\n🎉 Trend Intelligence Engine completed!"
)