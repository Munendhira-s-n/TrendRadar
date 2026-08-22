import pandas as pd

features = pd.read_csv("trend_features_v3.csv")
predictions = pd.read_csv("breakout_predictions.csv")

feature_topics = set(features["topic"].unique())
prediction_topics = set(predictions["topic"].unique())

missing = feature_topics - prediction_topics

print("=" * 70)
print("             🔎 MISSING TOPIC CHECK")
print("=" * 70)

print("\nTopics in features    :", len(feature_topics))
print("Topics in predictions :", len(prediction_topics))

print("\nMissing topic(s):")

for topic in sorted(missing):
    print("→", topic)


print("\n" + "=" * 70)

# Also check latest date

features["date"] = pd.to_datetime(features["date"])
predictions["date"] = pd.to_datetime(predictions["date"])

latest_features = features["date"].max()
latest_predictions = predictions["date"].max()

print("\nLatest feature date     :", latest_features.date())
print("Latest prediction date  :", latest_predictions.date())

print("\nTopics on latest feature date:",
      features[
          features["date"] == latest_features
      ]["topic"].nunique())

print("Topics on latest prediction date:",
      predictions[
          predictions["date"] == latest_predictions
      ]["topic"].nunique())