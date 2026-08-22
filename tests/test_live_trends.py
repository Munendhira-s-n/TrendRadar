import requests
import xml.etree.ElementTree as ET

URL = "https://trends.google.com/trending/rss?geo=IN"

print("Connecting to Google Trends...")
print("Fetching LIVE trends for India...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(
    URL,
    headers=headers,
    timeout=20
)

print("HTTP Status:", response.status_code)

response.raise_for_status()

root = ET.fromstring(response.content)

trends = []

for item in root.findall(".//item"):

    title = item.findtext("title", default="").strip()

    traffic = item.findtext(
        "{https://trends.google.com/trending/rss}approx_traffic",
        default=""
    ).strip()

    pub_date = item.findtext(
        "pubDate",
        default=""
    ).strip()

    if title:
        trends.append({
            "trend": title,
            "traffic": traffic,
            "published": pub_date
        })


print("\nLIVE INDIA TRENDS")
print("=" * 60)

for i, trend in enumerate(trends[:20], start=1):

    print(
        f"{i:02d}. "
        f"{trend['trend']} "
        f"| {trend['traffic']} "
        f"| {trend['published']}"
    )

print("\nTotal trends received:", len(trends))