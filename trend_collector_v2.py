import time
import random
import pandas as pd

from pytrends.request import TrendReq
from topics import TOPICS


# ==========================================
# SETTINGS
# ==========================================

START = "2025-08-01"
END = "2026-08-21"

BATCH_SIZE = 5

OUTPUT_FILE = "trend_data_v2.csv"


# ==========================================
# Flatten topic dictionary
# ==========================================

topic_list = []

for category, topics in TOPICS.items():

    for topic in topics:

        topic_list.append({
            "topic": topic,
            "category": category
        })


topics_df = pd.DataFrame(topic_list)


print("\n")
print("=" * 70)
print("             🔥 TRENDRADAR DATA COLLECTOR V2")
print("=" * 70)

print(
    f"\nTotal topics : {len(topics_df)}"
)

print(
    f"Date range   : {START} → {END}"
)


# ==========================================
# Google Trends connection
# ==========================================

pytrends = TrendReq(
    hl="en-US",
    tz=330
)


# ==========================================
# Collect data
# ==========================================

all_data = []

total_batches = (
    len(topics_df) // BATCH_SIZE
    + (len(topics_df) % BATCH_SIZE > 0)
)


for batch_number, start_index in enumerate(
    range(
        0,
        len(topics_df),
        BATCH_SIZE
    ),
    start=1
):

    batch = topics_df.iloc[
        start_index:
        start_index + BATCH_SIZE
    ]

    keywords = batch["topic"].tolist()


    print(
        f"\n[{batch_number}/{total_batches}] "
        f"Collecting: {keywords}"
    )


    # --------------------------------------
    # Retry logic
    # --------------------------------------

    success = False

    for attempt in range(3):

        try:

            pytrends.build_payload(
                keywords,
                timeframe=f"{START} {END}",
                geo="",
                gprop=""
            )

            data = pytrends.interest_over_time()


            if data.empty:

                print(
                    "⚠️ Empty response"
                )

                continue


            # Remove partial flag
            if "isPartial" in data.columns:

                data = data.drop(
                    columns=["isPartial"]
                )


            # Convert wide → long
            data = (
                data
                .reset_index()
                .melt(
                    id_vars=["date"],
                    var_name="topic",
                    value_name="interest"
                )
            )


            # Add category
            data = data.merge(
                batch,
                on="topic",
                how="left"
            )


            all_data.append(data)

            print(
                f"✓ Collected "
                f"{len(data)} rows"
            )

            success = True

            break


        except Exception as e:

            print(
                f"⚠️ Attempt "
                f"{attempt + 1}/3 failed:"
            )

            print(e)

            wait_time = (
                10 * (attempt + 1)
                + random.uniform(1, 5)
            )

            print(
                f"Waiting "
                f"{wait_time:.1f}s..."
            )

            time.sleep(
                wait_time
            )


    if not success:

        print(
            f"❌ Failed batch: "
            f"{keywords}"
        )


    # --------------------------------------
    # Delay between batches
    # --------------------------------------

    delay = random.uniform(
        5,
        10
    )

    print(
        f"Waiting {delay:.1f}s..."
    )

    time.sleep(delay)


# ==========================================
# Combine
# ==========================================

if not all_data:

    print(
        "\n❌ No data collected."
    )

    raise SystemExit


df = pd.concat(
    all_data,
    ignore_index=True
)


# ==========================================
# Clean
# ==========================================

df["date"] = pd.to_datetime(
    df["date"]
)

df["interest"] = pd.to_numeric(
    df["interest"],
    errors="coerce"
)

df = df.dropna(
    subset=["interest"]
)

df = df.drop_duplicates(
    subset=[
        "date",
        "topic"
    ]
)

df = df.sort_values(
    ["topic", "date"]
)


# ==========================================
# Save
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)
print("                 ✅ COLLECTION COMPLETE")
print("=" * 70)

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
    f"Date range : "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print(
    "\nRows per category:"
)

print(
    df.groupby("category")
    .size()
    .sort_values(
        ascending=False
    )
    .to_string()
)

print(
    f"\nSaved → {OUTPUT_FILE}"
)