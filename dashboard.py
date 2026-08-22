import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrendRadar",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_DIR = "data"

HISTORY_FILE = os.path.join(DATA_DIR, "trend_history.csv")
INTELLIGENCE_FILE = os.path.join(DATA_DIR, "trend_intelligence.csv")
BREAKOUT_FILE = os.path.join(DATA_DIR, "breakout_signals.csv")
CATEGORY_FILE = os.path.join(DATA_DIR, "categorized_trends.csv")


# ============================================================
# LIGHT CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=30)
def load_data():

    def read_csv(path):
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return (
        read_csv(HISTORY_FILE),
        read_csv(INTELLIGENCE_FILE),
        read_csv(BREAKOUT_FILE),
        read_csv(CATEGORY_FILE)
    )


history, intelligence, breakout, categories = load_data()


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_topics(df):

    if df.empty:
        return df

    df = df.copy()

    if "topic" in df.columns:
        df["topic"] = (
            df["topic"]
            .astype(str)
            .str.strip()
        )

    return df


history = clean_topics(history)
intelligence = clean_topics(intelligence)
breakout = clean_topics(breakout)
categories = clean_topics(categories)


# ============================================================
# VALIDATION
# ============================================================

if intelligence.empty:

    st.error("❌ trend_intelligence.csv is missing or empty.")

    st.info(
        "Run your TrendRadar pipeline first."
    )

    st.stop()


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "traffic_number",
    "growth_percent",
    "rank_change",
    "momentum_score",
    "breakout_score",
    "breakout_probability",
    "intelligence_score",
    "intelligence_rank"
]

for column in numeric_columns:

    if column in intelligence.columns:

        intelligence[column] = pd.to_numeric(
            intelligence[column],
            errors="coerce"
        )


# ============================================================
# CATEGORY FIX
# ============================================================
#
# categorized_trends.csv is treated as the authoritative
# topic -> category mapping.
#
# This fixes the problem where only Business & Finance
# appeared correctly.
# ============================================================

if (
    not categories.empty
    and "topic" in categories.columns
    and "category" in categories.columns
):

    category_map = (
        categories[
            ["topic", "category"]
        ]
        .dropna(subset=["topic"])
        .drop_duplicates(
            subset=["topic"],
            keep="last"
        )
    )

    category_map["category"] = (
        category_map["category"]
        .astype(str)
        .str.strip()
    )

    # Remove blank categories
    category_map.loc[
        category_map["category"].isin(["", "nan", "None"]),
        "category"
    ] = pd.NA

    # Remove existing category before merge
    if "category" in intelligence.columns:

        intelligence = intelligence.drop(
            columns=["category"]
        )

    intelligence = intelligence.merge(
        category_map,
        on="topic",
        how="left"
    )

else:

    # If category file unavailable,
    # keep existing intelligence category.
    if "category" not in intelligence.columns:
        intelligence["category"] = "Other"


# ============================================================
# CATEGORY FALLBACK
# ============================================================

intelligence["category"] = (
    intelligence["category"]
    .fillna("Other")
    .astype(str)
    .str.strip()
)

intelligence.loc[
    intelligence["category"].isin(
        ["", "nan", "None", "NaN"]
    ),
    "category"
] = "Other"


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

current_topics = set()
latest_snapshot = None

if (
    not history.empty
    and "fetched_at" in history.columns
):

    history["fetched_at"] = pd.to_datetime(
        history["fetched_at"],
        errors="coerce"
    )

    valid_dates = history["fetched_at"].dropna()

    if not valid_dates.empty:

        latest_snapshot = valid_dates.max()

        current_history = history[
            history["fetched_at"] == latest_snapshot
        ].copy()

        if "topic" in current_history.columns:

            current_topics = set(
                current_history["topic"].dropna()
            )


# ============================================================
# KEEP CURRENT TRENDS ONLY
# ============================================================

if current_topics:

    intelligence = intelligence[
        intelligence["topic"].isin(current_topics)
    ].copy()


# ============================================================
# REMOVE DUPLICATES
# ============================================================

if "intelligence_score" in intelligence.columns:

    intelligence = (
        intelligence
        .sort_values(
            "intelligence_score",
            ascending=False
        )
        .drop_duplicates(
            "topic",
            keep="first"
        )
        .reset_index(drop=True)
    )


# ============================================================
# DISPLAY RANK
# ============================================================

intelligence["display_rank"] = (
    intelligence.index + 1
)


# ============================================================
# PREPARE BREAKOUT
# ============================================================

if not breakout.empty:

    for column in [
        "breakout_score",
        "breakout_probability",
        "traffic_number",
        "rank_change",
        "persistence_score",
        "momentum_score"
    ]:

        if column in breakout.columns:

            breakout[column] = pd.to_numeric(
                breakout[column],
                errors="coerce"
            )

    if current_topics:

        breakout = breakout[
            breakout["topic"].isin(current_topics)
        ]

    if "breakout_score" in breakout.columns:

        breakout = (
            breakout
            .sort_values(
                "breakout_score",
                ascending=False
            )
            .drop_duplicates(
                "topic",
                keep="first"
            )
        )


# ============================================================
# HELPER
# ============================================================

def number(value, default=0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def traffic(value):

    value = number(value)

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def signal_icon(signal):

    signal = str(signal).upper()

    if "RAPIDLY RISING" in signal:
        return "🚀"

    if "RISING" in signal:
        return "📈"

    if "FALLING" in signal:
        return "📉"

    if "SUSTAINED" in signal:
        return "🔵"

    if "NEW" in signal:
        return "🆕"

    return "⚪"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚀 TrendRadar")

    st.caption(
        "Real-Time Trend Intelligence"
    )

    st.divider()

    st.subheader("Filters")

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories_available = sorted(
        intelligence["category"]
        .dropna()
        .unique()
    )

    selected_categories = st.multiselect(
        "🏷️ Category",
        categories_available
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    if "intelligence_signal" in intelligence.columns:

        signals_available = sorted(
            intelligence["intelligence_signal"]
            .dropna()
            .unique()
        )

    else:

        signals_available = []

    selected_signals = st.multiselect(
        "🧠 Signal",
        signals_available
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if "priority" in intelligence.columns:

        priorities_available = sorted(
            intelligence["priority"]
            .dropna()
            .unique()
        )

    else:

        priorities_available = []

    selected_priorities = st.multiselect(
        "🎯 Priority",
        priorities_available
    )

    st.divider()

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.metric(
        "Current Trends",
        len(intelligence)
    )

    st.metric(
        "Categories",
        intelligence["category"].nunique()
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = intelligence.copy()


if selected_categories:

    filtered = filtered[
        filtered["category"].isin(
            selected_categories
        )
    ]


if selected_signals:

    filtered = filtered[
        filtered["intelligence_signal"].isin(
            selected_signals
        )
    ]


if selected_priorities:

    filtered = filtered[
        filtered["priority"].isin(
            selected_priorities
        )
    ]


# ============================================================
# HEADER
# ============================================================

st.title("🚀 TrendRadar")

st.caption(
    "Real-Time Trend Intelligence • "
    "Momentum • Breakout Detection • Market Signals"
)

if latest_snapshot is not None:

    st.caption(
        "Latest snapshot: "
        + latest_snapshot.strftime(
            "%d %b %Y, %H:%M:%S"
        )
    )

st.divider()


# ============================================================
# KPIs
# ============================================================

active_trends = len(filtered)


if "intelligence_signal" in filtered.columns:

    rapidly_rising = filtered[
        filtered["intelligence_signal"]
        .astype(str)
        .str.contains(
            "RAPIDLY RISING",
            case=False,
            na=False
        )
    ].shape[0]

else:

    rapidly_rising = 0


if (
    not breakout.empty
    and "breakout_status" in breakout.columns
):

    breakout_candidates = breakout[
        breakout["breakout_status"]
        .astype(str)
        .str.contains(
            "RISING|BREAKOUT|EMERGING",
            case=False,
            regex=True,
            na=False
        )
    ].shape[0]

else:

    breakout_candidates = 0


if "growth_percent" in filtered.columns:

    growth_values = pd.to_numeric(
        filtered["growth_percent"],
        errors="coerce"
    ).dropna()

    highest_growth = (
        growth_values.max()
        if not growth_values.empty
        else 0
    )

else:

    highest_growth = 0


k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "🔥 Active Trends",
        active_trends
    )

with k2:
    st.metric(
        "🚀 Rapidly Rising",
        rapidly_rising
    )

with k3:
    st.metric(
        "🎯 Breakout Candidates",
        breakout_candidates
    )

with k4:
    st.metric(
        "📈 Highest Growth",
        f"{highest_growth:.0f}%"
    )


# ============================================================
# TOP TREND
# ============================================================

if not filtered.empty:

    st.subheader("🎯 Top Intelligence Signal")

    top = filtered.iloc[0]

    left, right = st.columns([2, 1])

    with left:

        signal = top.get(
            "intelligence_signal",
            "UNKNOWN"
        )

        st.info(
            f"{signal_icon(signal)} {signal}"
        )

        st.header(
            str(
                top.get(
                    "topic",
                    "Unknown"
                )
            )
        )

        st.caption(
            f"Category: "
            f"{top.get('category', 'Other')}"
        )

        st.write(
            f"**Priority:** "
            f"{top.get('priority', 'LOW')}"
        )

    with right:

        st.metric(
            "Intelligence Score",
            f"{number(top.get('intelligence_score')):.1f}"
        )

        st.metric(
            "Traffic",
            traffic(
                top.get("traffic_number")
            )
        )


    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Growth",
            f"{number(top.get('growth_percent')):+.1f}%"
        )

    with d2:

        st.metric(
            "Rank Change",
            f"{number(top.get('rank_change')):+.0f}"
        )

    with d3:

        st.metric(
            "Momentum",
            f"{number(top.get('momentum_score')):.1f}"
        )

    with d4:

        st.metric(
            "Breakout",
            f"{number(top.get('breakout_score')):.1f}"
        )


# ============================================================
# TABS
# ============================================================

overview, analytics, radar, intelligence_tab = st.tabs(
    [
        "📊 Overview",
        "📈 Analytics",
        "🚀 Breakout Radar",
        "🧠 Intelligence"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview:

    st.subheader("📊 Trend Overview")

    c1, c2 = st.columns(2)

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    with c1:

        st.markdown("### 🔥 Momentum Leaders")

        if (
            not filtered.empty
            and "momentum_score" in filtered.columns
        ):

            df = (
                filtered
                .sort_values(
                    "momentum_score",
                    ascending=True
                )
                .tail(10)
            )

            fig = px.bar(
                df,
                x="momentum_score",
                y="topic",
                orientation="h",
                text="momentum_score",
                labels={
                    "momentum_score": "Momentum",
                    "topic": ""
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside"
            )

            fig.update_layout(
                height=430,
                margin=dict(
                    l=10,
                    r=30,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # INTELLIGENCE
    # --------------------------------------------------------

    with c2:

        st.markdown("### 🧠 Intelligence Leaders")

        if (
            not filtered.empty
            and "intelligence_score" in filtered.columns
        ):

            df = (
                filtered
                .sort_values(
                    "intelligence_score",
                    ascending=True
                )
                .tail(10)
            )

            fig = px.bar(
                df,
                x="intelligence_score",
                y="topic",
                orientation="h",
                text="intelligence_score",
                labels={
                    "intelligence_score":
                        "Intelligence Score",
                    "topic": ""
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside"
            )

            fig.update_layout(
                height=430,
                margin=dict(
                    l=10,
                    r=30,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # SIGNAL DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### 🧠 Signal Distribution")

    if "intelligence_signal" in filtered.columns:

        signal_counts = (
            filtered["intelligence_signal"]
            .value_counts()
            .reset_index()
        )

        signal_counts.columns = [
            "signal",
            "count"
        ]

        fig = px.pie(
            signal_counts,
            names="signal",
            values="count",
            hole=0.5
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ANALYTICS
# ============================================================

with analytics:

    st.subheader("📈 Trend Analytics")

    # --------------------------------------------------------
    # TRAFFIC VS INTELLIGENCE
    # --------------------------------------------------------

    if (
        "traffic_number" in filtered.columns
        and "intelligence_score" in filtered.columns
    ):

        st.markdown(
            "### Traffic vs Intelligence"
        )

        scatter = filtered.copy()

        fig = px.scatter(
            scatter,
            x="traffic_number",
            y="intelligence_score",
            size="momentum_score"
            if "momentum_score" in scatter.columns
            else None,
            hover_name="topic",
            color="category",
            labels={
                "traffic_number": "Traffic",
                "intelligence_score":
                    "Intelligence Score"
            }
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # GROWTH
    # --------------------------------------------------------

    st.markdown("### 📈 Growth Leaders")

    if "growth_percent" in filtered.columns:

        growth = (
            filtered
            .sort_values(
                "growth_percent",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            growth,
            x="topic",
            y="growth_percent",
            text="growth_percent",
            labels={
                "growth_percent": "Growth %",
                "topic": ""
            }
        )

        fig.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### 🏷️ Category Distribution")

    category_counts = (
        filtered["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "category",
        "count"
    ]

    fig = px.bar(
        category_counts,
        x="category",
        y="count",
        text="count",
        labels={
            "category": "",
            "count": "Trends"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# BREAKOUT RADAR
# ============================================================

with radar:

    st.subheader("🚀 Breakout Radar")

    if breakout.empty:

        st.info(
            "No breakout data available."
        )

    else:

        radar_df = breakout.copy()

        if "breakout_score" in radar_df.columns:

            radar_df = (
                radar_df
                .sort_values(
                    "breakout_score",
                    ascending=False
                )
            )

            chart_df = radar_df.head(10)

            fig = px.bar(
                chart_df.sort_values(
                    "breakout_score"
                ),
                x="breakout_score",
                y="topic",
                orientation="h",
                text="breakout_score",
                labels={
                    "breakout_score":
                        "Breakout Score",
                    "topic": ""
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside"
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        st.markdown(
            "### Breakout Candidates"
        )

        columns = [
            "topic",
            "breakout_score",
            "breakout_probability",
            "breakout_status",
            "rank_change",
            "persistence_score",
            "momentum_score"
        ]

        columns = [
            c for c in columns
            if c in radar_df.columns
        ]

        display = radar_df[
            columns
        ].head(15).copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INTELLIGENCE TABLE
# ============================================================

with intelligence_tab:

    st.subheader(
        "🧠 Current Trend Intelligence"
    )

    st.caption(
        f"{len(filtered)} current trends"
    )

    columns = [
        "display_rank",
        "topic",
        "category",
        "traffic_number",
        "growth_percent",
        "rank_change",
        "momentum_score",
        "breakout_score",
        "intelligence_score",
        "intelligence_signal",
        "priority"
    ]

    columns = [
        c for c in columns
        if c in filtered.columns
    ]

    table = filtered[
        columns
    ].copy()

    table = table.rename(
        columns={
            "display_rank": "Rank",
            "topic": "Trend",
            "category": "Category",
            "traffic_number": "Traffic",
            "growth_percent": "Growth %",
            "rank_change": "Rank Change",
            "momentum_score": "Momentum",
            "breakout_score": "Breakout",
            "intelligence_score": "Intelligence",
            "intelligence_signal": "Signal",
            "priority": "Priority"
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Current Intelligence",
        data=csv_data,
        file_name="trendradar_current_intelligence.csv",
        mime="text/csv"
    )


# ============================================================
# TREND TIMELINE
# ============================================================

st.divider()

st.subheader("⏱️ Trend Timeline")

if (
    not history.empty
    and "fetched_at" in history.columns
    and "topic" in history.columns
):

    timeline_topics = sorted(
        current_topics
        if current_topics
        else set(
            history["topic"].dropna()
        )
    )

    if timeline_topics:

        default_topic = timeline_topics[0]

        if (
            not filtered.empty
            and filtered.iloc[0]["topic"]
            in timeline_topics
        ):

            default_topic = filtered.iloc[0]["topic"]

        selected_topic = st.selectbox(
            "Select a trend",
            timeline_topics,
            index=timeline_topics.index(
                default_topic
            )
        )

        timeline = history[
            history["topic"] == selected_topic
        ].sort_values(
            "fetched_at"
        )

        if not timeline.empty:

            fig = px.line(
                timeline,
                x="fetched_at",
                y="traffic_number",
                markers=True,
                labels={
                    "fetched_at": "Snapshot",
                    "traffic_number": "Traffic"
                }
            )

            fig.update_layout(
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

else:

    st.info(
        "Historical trend data is not available."
    )


# ============================================================
# PIPELINE STATUS
# ============================================================

st.divider()

st.subheader("⚙️ Pipeline Status")

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "History Observations",
        len(history)
    )

with p2:

    snapshots = (
        history["fetched_at"].nunique()
        if (
            not history.empty
            and "fetched_at" in history.columns
        )
        else 0
    )

    st.metric(
        "Snapshots",
        snapshots
    )

with p3:

    st.metric(
        "Current Trends",
        len(intelligence)
    )

with p4:

    st.metric(
        "Breakout Records",
        len(breakout)
    )


# ============================================================
# DATA HEALTH
# ============================================================

st.subheader("🟢 Data Health")

h1, h2, h3, h4 = st.columns(4)

with h1:

    if not history.empty:
        st.success("✓ History loaded")
    else:
        st.warning("⚠ History unavailable")

with h2:

    if not intelligence.empty:
        st.success("✓ Intelligence loaded")
    else:
        st.error("✗ Intelligence unavailable")

with h3:

    if not categories.empty:
        st.success(
            f"✓ Categories loaded ({intelligence['category'].nunique()})"
        )
    else:
        st.warning("⚠ Category file unavailable")

with h4:

    if not breakout.empty:
        st.success("✓ Breakout loaded")
    else:
        st.warning("⚠ Breakout unavailable")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚀 TrendRadar | "
    "Live Trends → History → Categories → "
    "Momentum → Breakout → Intelligence"
)