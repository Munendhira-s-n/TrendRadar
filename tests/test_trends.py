from pytrends.request import TrendReq
import pandas as pd
import time

pytrends = TrendReq(hl="en-US", tz=330)

topics = [
    "AI",
    "ChatGPT",
    "Robotics",
    "Electric Vehicle",
    "Smartwatch",
    "Gaming",
    "Anime",
    "K-pop",
    "Netflix",
    "Skincare",
    "Fitness",
    "Yoga",
    "iPhone",
    "Android",
    "Tesla",
    "Bitcoin",
    "Travel",
    "Food Delivery",
    "Online Shopping",
    "Podcast"
]

all_data = []

for topic in topics:
    print(f"Collecting: {topic}")

    try:
        pytrends.build_payload(
            [topic],
            timeframe="today 12-m",
            geo="IN"
        )

        data = pytrends.interest_over_time()

        if not data.empty:
            data = data.reset_index()
            data = data[["date", topic]]
            data = data.rename(columns={topic: "interest"})
            data["topic"] = topic

            all_data.append(data)

        time.sleep(2)

    except Exception as e:
        print(f"Error with {topic}: {e}")

final_data = pd.concat(all_data, ignore_index=True)

final_data.to_csv("trend_data.csv", index=False)

print("\nDataset created!")
print(final_data.head())
print("\nShape:", final_data.shape)