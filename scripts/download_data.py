# scripts/download_data.py
from meteostat import Point, Daily
from datetime import datetime
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dhaka = Point(23.8103, 90.4125)
start = datetime(1975, 1, 1)
end = datetime(2025, 12, 31)

data = Daily(dhaka, start, end)
df = data.fetch()
df.reset_index(inplace=True)
df.to_csv("data/dhaka_weather.csv", index=False)
print("✅ Data saved to data/dhaka_weather.csv")
