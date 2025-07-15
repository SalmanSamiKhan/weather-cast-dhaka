# scripts/download_data.py

from pathlib import Path
from meteostat import Point, Daily
from datetime import datetime
import pandas as pd

# ------------------------
# Configuration
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "dhaka_weather.csv"

# ------------------------
# Main logic
# ------------------------
def download_weather_data(
    lat=23.8103,
    lon=90.4125,
    start=datetime(1975, 1, 1),
    end=datetime(2025, 12, 31),
    output_path=OUTPUT_CSV
):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        location = Point(lat, lon)
        data = Daily(location, start, end)
        df = data.fetch()

        if df.empty:
            print("❌ No data fetched. Check date range or station availability.")
            return

        df.reset_index(inplace=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Data saved to {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Error downloading data: {e}")

# ------------------------
# Entry Point
# ------------------------
if __name__ == "__main__":
    download_weather_data()
