import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="TrendRadar",
    page_icon="🚀",
    layout="wide"
)

DATA_DIR = "data"

HISTORY_FILE = os.path.join(DATA_DIR, "trend_history.csv")
MOMENTUM_FILE = os.path.join(DATA_DIR, "trend_features.csv")
BREAKOUT_FILE = os.path.join(DATA_DIR, "breakout_signals.csv")
CATEGORY_FILE = os.path.join(DATA_DIR, "categorized_trends.csv")


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return None


history_df = load_csv(HISTORY_FILE)
momentum_df = load_csv(MOMENTUM_FILE)
breakout_df = load_csv(BREAKOUT_FILE)
category_df = load_csv(CATEGORY_FILE)


# ============================================================
# CHECK DATA
# ============================================================

missing_files = []

for name, df in [
    ("trend_history.csv", history_df),
    ("trend_features.csv", momentum_df),
    ("breakout_signals.csv", breakout_df),
    ("categorized_trends.csv", category_df),
]:
    if df is None:
        missing_files.append(name)


if missing_files:

    st.error("⚠️ Some TrendRadar data files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Run your TrendRadar engines first, then refresh this page."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🚀 TrendRadar")

st.subheader(
    "Live Trend Intelligence Dashboard"
)

st.caption(
    "Google Trends → Momentum → Feature Engineering → "
    "Breakout Detection → Category Intelligence"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Explore")

categories = ["All"]

if "category" in category_df.columns:
    categories += sorted(
        category_df["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_category = st.sidebar.selectbox(
    "Category",
    categories
)


# ============================================================
# MERGE CURRENT DATA
# ============================================================

current = category_df.copy()

# Keep only the latest occurrence of each topic
current = current.drop_duplicates(
    subset=["topic"],
    keep="last"
)

if "topic" in breakout_df.columns:

    breakout_latest = breakout_df.drop_duplicates(
        subset=["topic"],
        keep="last"
    )

    breakout_columns = [
        col for col in [
            "topic",
            "breakout_score",
            "breakout_probability",
            "breakout_status"
        ]
        if col in breakout_latest.columns
    ]

    current = current.merge(
        breakout_latest[breakout_columns],
        on="topic",
        how="left"
    )


if "topic" in momentum_df.columns:

    momentum_latest = momentum_df.drop_duplicates(
        subset=["topic"],
        keep="last"
    )

    momentum_columns = [
        col for col in [
            "topic",
            "momentum_score",
            "growth_percent",
            "persistence_score"
        ]
        if col in momentum_latest.columns
    ]

    current = current.merge(
        momentum_latest[momentum_columns],
        on="topic",
        how="left"
    )


# ============================================================
# CATEGORY FILTER
# ============================================================

if selected_category != "All":

    current = current[
        current["category"]
        == selected_category
    ]


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.divider()

st.header("📊 Live Overview")

total_topics = len(current)

hot_count = 0
rising_count = 0
emerging_count = 0

if "breakout_status" in current.columns:

    status = (
        current["breakout_status"]
        .fillna("")
        .astype(str)
    )

    hot_count = status.str.contains(
        "HOT",
        case=False
    ).sum()

    rising_count = status.str.contains(
        "RISING",
        case=False
    ).sum()

    emerging_count = status.str.contains(
        "EMERGING",
        case=False
    ).sum()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📌 Live Trends",
        total_topics
    )

with col2:
    st.metric(
        "🔥 Hot",
        hot_count
    )

with col3:
    st.metric(
        "📈 Rising",
        rising_count
    )

with col4:
    st.metric(
        "🟢 Emerging",
        emerging_count
    )


# ============================================================
# TOP BREAKOUT TRENDS
# ============================================================

st.divider()

st.header("🚀 Top Breakout Candidates")

if "breakout_score" in current.columns:

    breakout_view = (
        current
        .sort_values(
            "breakout_score",
            ascending=False
        )
        .head(10)
    )

    columns = [
        col for col in [
            "topic",
            "category",
            "traffic_number",
            "momentum_score",
            "breakout_score",
            "breakout_probability",
            "breakout_status"
        ]
        if col in breakout_view.columns
    ]

    st.dataframe(
        breakout_view[columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Breakout data is not available."
    )


# ============================================================
# BREAKOUT CHART
# ============================================================

if "breakout_score" in current.columns:

    chart_df = (
        current
        .sort_values(
            "breakout_score",
            ascending=False
        )
        .head(10)
        .sort_values(
            "breakout_score"
        )
    )

    fig = px.bar(
        chart_df,
        x="breakout_score",
        y="topic",
        orientation="h",
        text="breakout_score",
        title="Breakout Score"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

st.divider()

st.header("🗂️ Trend Categories")

if "category" in current.columns:

    category_counts = (
        current["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "category",
        "count"
    ]

    fig_category = px.bar(
        category_counts,
        x="category",
        y="count",
        text="count",
        title="Live Trends by Category"
    )

    fig_category.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )


# ============================================================
# LIVE TREND TABLE
# ============================================================

st.divider()

st.header("🔴 Live Trend Feed")

display_columns = [
    col for col in [
        "topic",
        "traffic",
        "traffic_number",
        "category",
        "momentum_score",
        "breakout_score",
        "breakout_probability",
        "breakout_status"
    ]
    if col in current.columns
]

st.dataframe(
    current[display_columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOPIC EXPLORER
# ============================================================

st.divider()

st.header("🔍 Trend Explorer")

topics = sorted(
    current["topic"]
    .dropna()
    .astype(str)
    .unique()
)

if topics:

    selected_topic = st.selectbox(
        "Select a trend",
        topics
    )

    topic_row = current[
        current["topic"]
        == selected_topic
    ].iloc[0]

    st.subheader(
        f"🎯 {selected_topic}"
    )

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric(
            "Traffic",
            topic_row.get(
                "traffic_number",
                "N/A"
            )
        )

    with info2:
        st.metric(
            "Momentum",
            topic_row.get(
                "momentum_score",
                "N/A"
            )
        )

    with info3:
        st.metric(
            "Breakout",
            topic_row.get(
                "breakout_score",
                "N/A"
            )
        )

    # Historical chart

    topic_history = history_df[
        history_df["topic"]
        == selected_topic
    ].copy()

    if not topic_history.empty:

        if "fetched_at" in topic_history.columns:

            topic_history["fetched_at"] = pd.to_datetime(
                topic_history["fetched_at"],
                errors="coerce"
            )

            topic_history = topic_history.sort_values(
                "fetched_at"
            )

            fig_history = px.line(
                topic_history,
                x="fetched_at",
                y="traffic_number",
                markers=True,
                title=f"{selected_topic} — Traffic History"
            )

            st.plotly_chart(
                fig_history,
                use_container_width=True
            )

else:

    st.info(
        "No trends available for the selected category."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚀 TrendRadar — Live Trend Intelligence"
)