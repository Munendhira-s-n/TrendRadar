import pandas as pd
import re

from live_trends import get_live_trends


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {
    "Sports": [
        "cricket",
        "football",
        "fifa",
        "ipl",
        "test",
        "odi",
        "t20",
        "match",
        "vs",
        "player",
        "league",
        "cup",
        "tennis",
        "badminton",
        "wrestling",
        "formula",
        "f1",
    ],

    "Entertainment": [
        "movie",
        "film",
        "actor",
        "actress",
        "trailer",
        "song",
        "concert",
        "series",
        "netflix",
        "marvel",
        "bollywood",
        "kollywood",
        "tollywood",
        "anime",
        "celebrity",
        "music",
    ],

    "Technology": [
        "ai",
        "iphone",
        "android",
        "google",
        "apple",
        "chatgpt",
        "openai",
        "microsoft",
        "software",
        "technology",
        "tech",
        "samsung",
        "laptop",
        "phone",
        "robot",
    ],

    "Business": [
        "stock",
        "share",
        "market",
        "bank",
        "business",
        "company",
        "finance",
        "ipo",
        "rupee",
        "dollar",
        "economy",
        "investment",
    ],

    "Politics": [
        "election",
        "minister",
        "government",
        "bjp",
        "congress",
        "modi",
        "president",
        "prime minister",
        "parliament",
        "politics",
        "abvp",
    ],

    "Travel": [
        "flight",
        "train",
        "airport",
        "hotel",
        "travel",
        "tourism",
        "trip",
        "weather",
    ],
}


# ============================================================
# TRAFFIC CONVERSION
# ============================================================

def traffic_to_number(value):
    """
    Convert Google Trends traffic strings into numbers.

    Examples:
        100+     -> 100
        500+     -> 500
        1,000+   -> 1000
        10K+     -> 10000
        1M+      -> 1000000
    """

    if pd.isna(value):
        return 0

    value = str(value).strip().upper()

    value = value.replace(",", "")
    value = value.replace("+", "")

    multiplier = 1

    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]

    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]

    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value[:-1]

    # Extract numeric portion
    match = re.search(r"\d+(?:\.\d+)?", value)

    if not match:
        return 0

    number = float(match.group())

    return int(number * multiplier)


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(topic):
    """
    Assign a category using safer keyword matching.

    Exact words / phrases are checked instead of simple
    substring matching, preventing cases like:

        "modi" -> incorrectly matching "odi"
    """

    topic_text = str(topic).lower().strip()

    # Check multi-word phrases first
    phrase_categories = {
        "narendra modi": "Politics",
        "prime minister": "Politics",
        "is bank open": "Business",
        "bank open": "Business",
        "aus vs ban": "Sports",
        "jhoan duran": "Sports",
        "sergio gor": "Sports",
    }

    for phrase, category in phrase_categories.items():
        if phrase in topic_text:
            return category

    # Word-based matching
    words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            topic_text
        )
    )

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            keyword = keyword.lower().strip()

            # Multi-word keyword
            if " " in keyword:
                if keyword in topic_text:
                    return category

            # Single-word keyword
            elif keyword in words:
                return category

    return "General"
# ============================================================
# TREND SCORE
# ============================================================

def calculate_trend_score(df):
    """
    Calculate a score based on current search traffic.

    Since we don't have historical data yet,
    this is a LIVE popularity score rather than
    an ML breakout prediction.
    """

    df = df.copy()

    if df.empty:
        return df

    maximum = df["traffic_number"].max()

    if maximum <= 0:

        df["trend_score"] = 0.0

    else:

        df["trend_score"] = (
            df["traffic_number"]
            / maximum
            * 100
        )

    df["trend_score"] = (
        df["trend_score"]
        .round(1)
    )

    return df


# ============================================================
# STATUS
# ============================================================

def assign_status(score):

    if score >= 80:
        return "🔥 HOT"

    if score >= 50:
        return "📈 RISING"

    if score >= 25:
        return "🟢 EMERGING"

    return "⚪ LOW"


# ============================================================
# BUILD TREND DATASET
# ============================================================

def build_trend_dataset():

    df = get_live_trends()

    if df.empty:
        return df

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    df["traffic_number"] = (
        df["traffic"]
        .apply(traffic_to_number)
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    df["category"] = (
        df["topic"]
        .apply(detect_category)
    )

    # --------------------------------------------------------
    # Trend Score
    # --------------------------------------------------------

    df = calculate_trend_score(df)

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    df["status"] = (
        df["trend_score"]
        .apply(assign_status)
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    df = df.sort_values(
        "trend_score",
        ascending=False,
    ).reset_index(drop=True)

    df["rank"] = (
        df.index + 1
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    df = df[
        [
            "rank",
            "topic",
            "traffic",
            "traffic_number",
            "category",
            "trend_score",
            "status",
            "published",
        ]
    ]

    return df


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("🚀 TrendRadar Trend Engine")
    print("=" * 70)

    try:

        df = build_trend_dataset()

        if df.empty:

            print("❌ No live trends received.")

        else:

            print(
                f"✅ Processed {len(df)} live trends"
            )

            print()
            print(
                df.to_string(index=False)
            )

            print()
            print("=" * 70)
            print("🎉 Trend engine is working!")

    except Exception as e:

        print()
        print("❌ Trend engine failed")
        print()
        print("Error:", e)