# ============================================================
# TrendRadar — Improved Category Engine
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "trend_history.csv"
OUTPUT_FILE = DATA_DIR / "categorized_trends.csv"


# ============================================================
# EXACT ENTITY RULES
# ============================================================
# High-confidence entities are checked FIRST.
# This prevents generic keywords from producing wrong categories.
# ============================================================

EXACT_ENTITY_CATEGORIES = {

    # ---------------- SPORTS ----------------

    "aus vs ban test": "Sports",
    "jhoan duran": "Sports",
    "sergio gor": "Sports",

    # ---------------- POLITICS ----------------

    "narendra modi": "Politics",
    "abvp": "Politics",

    # ---------------- AUTOMOTIVE ----------------

    "टाटा मोटर्स": "Automotive",
    "tata motors": "Automotive",

    # ---------------- EDUCATION ----------------

    "शिक्षक": "Education",

    # ---------------- BUSINESS ----------------

    "is bank open today": "Business & Finance",

    # ---------------- OTHER / AMBIGUOUS ----------------

    "rai": "Other",
}


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    "Sports": [

        # English
        "cricket",
        "football",
        "soccer",
        "tennis",
        "badminton",
        "basketball",
        "baseball",
        "hockey",
        "volleyball",
        "golf",
        "boxing",
        "wrestling",
        "formula 1",
        "f1",

        # Cricket terminology
        "test match",
        "odi",
        "t20",
        "ipl",
        "world cup",
        "match",
        "player",
        "league",
        "championship",
        "wicket",
        "runs",
        "goal",
        "score",

        # Common trend patterns
        " vs ",
        " vs",
        "vs ",
    ],


    "Politics": [

        # English
        "politics",
        "election",
        "elections",
        "government",
        "minister",
        "prime minister",
        "president",
        "parliament",
        "congress",
        "bjp",
        "abvp",
        "mla",
        "mp",
        "lok sabha",
        "rajya sabha",
        "chief minister",

        # Indian political names / terms
        "narendra modi",
        "rahul gandhi",
        "amit shah",
        "arvind kejriwal",
    ],


    "Business & Finance": [

        "bank",
        "finance",
        "financial",
        "business",
        "company",
        "companies",
        "stock",
        "stocks",
        "share",
        "shares",
        "market",
        "ipo",
        "investment",
        "investor",
        "loan",
        "insurance",
        "tax",
        "salary",
        "profit",
        "revenue",
        "economy",
        "crypto",
        "bitcoin",
        "upi",
        "startup",
        "startups",
        "commerce",
        "ecommerce",
        "e-commerce",
    ],


    "Technology": [

        "artificial intelligence",
        "machine learning",
        "generative ai",
        "chatgpt",
        "openai",
        "claude",
        "gemini",
        "copilot",
        "robotics",
        "cybersecurity",
        "software",
        "technology",
        "technology",
        "tech",
        "coding",
        "programming",
        "python",
        "javascript",
        "android",
        "iphone",
        "google",
        "microsoft",
        "meta",
        "quantum computing",
        "cloud computing",
    ],


    "Entertainment": [

        "movie",
        "movies",
        "film",
        "films",
        "actor",
        "actress",
        "celebrity",
        "netflix",
        "youtube",
        "spotify",
        "music",
        "song",
        "songs",
        "concert",
        "anime",
        "kpop",
        "k-pop",
        "marvel",
        "disney",
        "series",
        "tv show",
        "television",
        "web series",
    ],


    "Automotive": [

        "tata motors",
        "mahindra",
        "toyota",
        "bmw",
        "mercedes",
        "tesla",
        "byd",
        "hyundai",
        "kia",

        "car",
        "cars",
        "vehicle",
        "vehicles",
        "automotive",
        "motorcycle",
        "motorcycles",
        "suv",
        "sedan",

        "electric vehicle",
        "electric vehicles",
        "electric car",
        "electric cars",
        "ev",
        "hybrid car",
        "hybrid cars",
    ],


    "Consumer & Shopping": [

        "amazon",
        "flipkart",
        "shopping",
        "discount",
        "sale",
        "product",
        "products",
        "airpods",
        "playstation",
        "xbox",
        "nintendo",
        "smartwatch",
    ],


    "Lifestyle": [

        "fitness",
        "gym",
        "yoga",
        "running",
        "makeup",
        "skincare",
        "fashion",
        "travel",
        "coffee",
        "cooking",
        "recipe",
        "recipes",
        "meditation",
        "photography",
        "home decor",
        "food",
    ],


    "Education": [

        # English
        "education",
        "school",
        "college",
        "university",
        "student",
        "teacher",
        "exam",
        "exams",
        "result",
        "results",
        "admission",
        "scholarship",
        "neet",
        "jee",
        "upsc",
        "ssc",

        # Hindi
        "शिक्षक",
        "शिक्षा",
        "परीक्षा",
        "छात्र",
        "स्कूल",
        "कॉलेज",
    ],


    "News & Events": [

        "breaking news",
        "news",
        "today",
        "incident",
        "accident",
        "earthquake",
        "weather",
        "rain",
        "storm",
        "fire",
        "festival",
        "event",
        "events",
    ],


    "Regional": [

        # Generic regional indicators
        "karnataka",
        "kerala",
        "tamil nadu",
        "andhra pradesh",
        "telangana",
        "maharashtra",
        "gujarat",
        "rajasthan",
        "punjab",
        "bengal",
        "bihar",
        "odisha",

        # Indian-language names
        "ಮಹಾರಾಷ್ಟ್ರ",
        "महाराष्ट्र",
        "తెలంగాణ",
        "కర్ణాటక",
        "ಕರ್ನಾಟಕ",
        "தமிழ்நாடு",
    ],
}


# ============================================================
# CATEGORY PRIORITY
# ============================================================
# Used when a topic matches more than one category.
# ============================================================

CATEGORY_PRIORITY = [

    "Politics",
    "Sports",
    "Automotive",
    "Business & Finance",
    "Technology",
    "Entertainment",
    "Education",
    "Consumer & Shopping",
    "Lifestyle",
    "News & Events",
    "Regional",
]


# ============================================================
# CATEGORY DETECTOR
# ============================================================

def detect_category(topic):

    text = str(topic).strip().lower()

    # --------------------------------------------------------
    # 1. Exact entity match
    # --------------------------------------------------------

    if text in EXACT_ENTITY_CATEGORIES:

        return EXACT_ENTITY_CATEGORIES[text]


    # --------------------------------------------------------
    # 2. Search category matches
    # --------------------------------------------------------

    matched_categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            keyword = keyword.lower()

            # Exact phrase containment.
            if keyword in text:

                matched_categories.append(category)

                break


    # --------------------------------------------------------
    # 3. No reliable match
    # --------------------------------------------------------

    if not matched_categories:

        return "Other"


    # --------------------------------------------------------
    # 4. Resolve multiple matches
    # --------------------------------------------------------

    for category in CATEGORY_PRIORITY:

        if category in matched_categories:

            return category


    return "Other"


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    print("❌ trend_history.csv not found.")

    print(
        f"Expected file: {INPUT_FILE}"
    )

    raise SystemExit(1)


df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# VALIDATE
# ============================================================

if "topic" not in df.columns:

    print(
        "❌ 'topic' column is missing."
    )

    raise SystemExit(1)


# ============================================================
# REMOVE EXISTING CATEGORY
# ============================================================

if "category" in df.columns:

    df = df.drop(
        columns=["category"]
    )


# ============================================================
# ASSIGN CATEGORY
# ============================================================

df["category"] = df["topic"].apply(
    detect_category
)


# ============================================================
# SAVE
# ============================================================

DATA_DIR.mkdir(
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

if "fetched_at" in df.columns:

    latest = (
        df.sort_values("fetched_at")
        .groupby("topic")
        .tail(1)
    )

else:

    latest = (
        df.groupby("topic")
        .tail(1)
    )


latest = latest.sort_values(
    "traffic_number",
    ascending=False
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 90)
print("🏷️ TrendRadar Improved Category Engine")
print("=" * 90)

print(
    f"\n✅ Processed {len(df)} observations"
)

print(
    f"📁 Saved to: {OUTPUT_FILE}"
)


print()
print("=" * 90)
print("📊 CURRENT TREND CATEGORIES")
print("=" * 90)

print(
    latest[
        [
            "topic",
            "traffic_number",
            "category",
        ]
    ].to_string(index=False)
)


print()
print("=" * 90)
print("📊 CATEGORY DISTRIBUTION")
print("=" * 90)

print(
    latest["category"]
    .value_counts()
    .to_string()
)


print()
print("=" * 90)
print("🎉 Category engine completed!")
print("=" * 90)