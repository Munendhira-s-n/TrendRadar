import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd


GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=IN"

RAW_FILE = "data/live_trends.csv"


def get_live_trends():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        GOOGLE_TRENDS_RSS,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    rows = []

    for item in root.findall(".//item"):

        topic = item.findtext(
            "title",
            default=""
        ).strip()

        traffic = item.findtext(
            "{https://trends.google.com/trending/rss}approx_traffic",
            default=""
        ).strip()

        published = item.findtext(
            "pubDate",
            default=""
        ).strip()

        if topic:

            rows.append(
                {
                    "topic": topic,
                    "traffic": traffic,
                    "published": published,
                }
            )

    df = pd.DataFrame(rows)

    return df


def save_live_trends(df):

    os.makedirs("data", exist_ok=True)

    if df.empty:
        print("❌ No live trends to save.")
        return

    # Numeric traffic value
    df["traffic_number"] = (
        df["traffic"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )

    df["traffic_number"] = pd.to_numeric(
        df["traffic_number"],
        errors="coerce"
    )

    # One timestamp for this fetch
    df["fetched_at"] = pd.Timestamp.now()

    # Keep the useful columns
    df = df[
        [
            "topic",
            "traffic",
            "traffic_number",
            "published",
            "fetched_at",
        ]
    ]

    # Append to raw live data
    if os.path.exists(RAW_FILE):

        old = pd.read_csv(RAW_FILE)

        combined = pd.concat(
            [old, df],
            ignore_index=True
        )

    else:

        combined = df

    combined.to_csv(
        RAW_FILE,
        index=False
    )

    print()
    print("✅ Live trends saved!")
    print(f"📁 File: {RAW_FILE}")
    print(f"📊 Total observations: {len(combined)}")


if __name__ == "__main__":

    print("🚀 TrendRadar Live Data")
    print("=" * 60)

    try:

        df = get_live_trends()

        print(
            f"✅ Live trends received: {len(df)}"
        )

        print()

        print(
            df.to_string(index=False)
        )

        save_live_trends(df)

        print()
        print(
            "🎉 Live data collection completed!"
        )

    except Exception as e:

        print("❌ Failed to fetch live trends")
        print()
        print(f"Error: {e}")