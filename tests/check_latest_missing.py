import pandas as pd

features = pd.read_csv("trend_features_v3.csv")
predictions = pd.read_csv("breakout_predictions.csv")

features["date"] = pd.to_datetime(features["date"])
predictions["date"] = pd.to_datetime(predictions["date"])

latest = predictions["date"].max()

feature_topics = set(
    features[
        features["date"] == latest
    ]["topic"]
)

prediction_topics = set(
    predictions[
        predictions["date"] == latest
    ]["topic"]
)

missing = feature_topics - prediction_topics

print("=" * 70)
print("             🔎 LATEST MISSING TOPIC")
print("=" * 70)

print("\nLatest date:", latest.date())

print("\nMissing topic(s):")

for topic in sorted(missing):
    print("→", topic)

print("\n" + "=" * 70)

print("\nFEATURE ROW:")

print(
    features[
        (features["date"] == latest) &
        (features["topic"].isin(missing))
    ].to_string(index=False)
)

print("\n" + "=" * 70)