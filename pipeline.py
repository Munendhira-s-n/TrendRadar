import subprocess
import sys


STEPS = [
    ("Live Trends", "live_trends.py"),
    ("Trend History / Momentum", "trend_history.py"),
    ("Category Engine", "category_engine.py"),
    ("Feature Engine", "trend_features.py"),
    ("Breakout Engine", "breakout_engine.py"),
    ("Trend Intelligence", "trend_intelligence.py"),
]


def run_step(name, filename):

    print("\n" + "=" * 80)
    print(f"🚀 Running: {name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, filename],
        capture_output=False
    )

    if result.returncode != 0:

        print("\n❌ Pipeline stopped.")
        print(f"Failed step: {name}")
        print(f"File: {filename}")

        sys.exit(result.returncode)

    print(f"\n✅ {name} completed")


def main():

    print("=" * 80)
    print("🚀 TrendRadar LIVE PIPELINE")
    print("=" * 80)

    for name, filename in STEPS:

        run_step(
            name,
            filename
        )

    print("\n" + "=" * 80)
    print("🎉 TREND RADAR PIPELINE COMPLETED!")
    print("=" * 80)

    print("\nAll live trend data has been processed.")
    print("Trend intelligence has been generated.")
    print("You can now open the dashboard.")


if __name__ == "__main__":
    main()