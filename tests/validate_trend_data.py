import pandas as pd


FILE = "trend_features_v3.csv"

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])


print("\n")
print("=" * 80)
print("             🔍 TRENDRADAR DATA VALIDATION")
print("=" * 80)


# ==========================================
# 1. Basic information
# ==========================================

print("\nDATASET")

print(
    f"Rows       : {len(df):,}"
)

print(
    f"Topics     : {df['topic'].nunique()}"
)

print(
    f"Categories : {df['category'].nunique()}"
)

print(
    f"Date range : "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ==========================================
# 2. Topics per date
# ==========================================

print("\nTOPICS PER DATE")

topics_per_date = (
    df.groupby("date")["topic"]
    .nunique()
)

print(
    topics_per_date.to_string()
)


# ==========================================
# 3. Find incomplete dates
# ==========================================

expected_topics = (
    df["topic"].nunique()
)

incomplete_dates = (
    topics_per_date[
        topics_per_date < expected_topics
    ]
)

print("\nINCOMPLETE DATES")

if len(incomplete_dates) == 0:

    print(
        "✓ Every date has all topics"
    )

else:

    print(
        incomplete_dates
        .to_string()
    )


# ==========================================
# 4. Missing topics
# ==========================================

print("\nMISSING TOPICS BY DATE")

all_topics = set(
    df["topic"].unique()
)

for date, group in df.groupby("date"):

    present = set(
        group["topic"]
    )

    missing = (
        all_topics - present
    )

    if missing:

        print(
            date.date(),
            "→",
            sorted(missing)
        )


# ==========================================
# 5. Duplicate records
# ==========================================

duplicates = df[
    df.duplicated(
        ["topic", "date"],
        keep=False
    )
]

print("\nDUPLICATES")

if duplicates.empty:

    print(
        "✓ No duplicate topic/date rows"
    )

else:

    print(
        duplicates[
            ["topic", "date"]
        ]
        .to_string(
            index=False
        )
    )


# ==========================================
# 6. Rows per topic
# ==========================================

topic_counts = (
    df.groupby("topic")
    .size()
)

print("\nROWS PER TOPIC")

print(
    topic_counts
    .value_counts()
    .sort_index()
    .to_string()
)


# ==========================================
# 7. Topics with unusual counts
# ==========================================

print(
    "\nTOPICS WITH UNUSUAL ROW COUNTS"
)

unusual = topic_counts[
    topic_counts != topic_counts.mode()[0]
]

if unusual.empty:

    print(
        "✓ All topics have equal observations"
    )

else:

    print(
        unusual.to_string()
    )


# ==========================================
# 8. Category consistency
# ==========================================

category_counts = (
    df.groupby("topic")["category"]
    .nunique()
)

inconsistent = (
    category_counts[
        category_counts > 1
    ]
)

print(
    "\nCATEGORY CONSISTENCY"
)

if inconsistent.empty:

    print(
        "✓ Every topic belongs to one category"
    )

else:

    print(
        inconsistent.to_string()
    )


# ==========================================
# 9. Latest records
# ==========================================

latest_date = df["date"].max()

latest = df[
    df["date"] == latest_date
]

print(
    "\nLATEST SNAPSHOT"
)

print(
    f"Date   : {latest_date.date()}"
)

print(
    f"Topics : {len(latest)}"
)

print(
    latest[
        [
            "topic",
            "category",
            "interest"
        ]
    ]
    .sort_values(
        "interest",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("                  VALIDATION COMPLETE")
print("=" * 80)