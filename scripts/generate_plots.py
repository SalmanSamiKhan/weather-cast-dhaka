# scripts/generate_plots.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------
# Constants
# --------------------
HOT_THRESHOLD = 38
COLD_THRESHOLD = 10
Z_SCORE_THRESHOLD = 2

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualizations"
CSV_FILE = DATA_DIR / "dhaka_weather_cleaned.csv"

# Create visualizations directory if not exists
VIS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Main plotting logic
# --------------------
def main():
    # Load and preprocess data
    df = pd.read_csv(CSV_FILE, parse_dates=["time"])
    if 'tavg' not in df.columns:
        raise ValueError("Missing 'tavg' column in dataset.")
    
    df['tavg'] = df['tavg'].interpolate()
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day

    # 1. Climate Trend
    yearly = df.groupby('year')['tavg'].agg(['mean', 'min', 'max']).reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(data=yearly, x='year', y='mean', label='Avg')
    sns.lineplot(data=yearly, x='year', y='max', label='Max')
    sns.lineplot(data=yearly, x='year', y='min', label='Min')
    plt.title("Climate Trend in Dhaka")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "climate_trend.png")
    plt.close()

    # 2. Extreme Events
    df['extreme'] = df['tavg'].apply(lambda x: 'hot' if x > HOT_THRESHOLD else 'cold' if x < COLD_THRESHOLD else 'normal')
    extreme_counts = df.groupby(['year', 'extreme']).size().unstack(fill_value=0)
    extreme_counts = extreme_counts.reindex(columns=['hot', 'cold'], fill_value=0)

    extreme_counts[['hot', 'cold']].plot(kind='bar', stacked=True, figsize=(14,6))
    plt.title("Extreme Weather Events Over Time")
    plt.ylabel("Days")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "extreme_events.png")
    plt.close()

    # 3. Monthly Heatmap
    monthly = df.groupby(['year', 'month'])['tavg'].mean().unstack()
    plt.figure(figsize=(12,6))
    sns.heatmap(monthly, cmap="coolwarm", linewidths=0.5)
    plt.title("Monthly Avg Temp Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "monthly_heatmap.png")
    plt.close()

    # 4. Z-score Anomalies
    df['month_avg'] = df.groupby('month')['tavg'].transform('mean')
    df['zscore'] = (df['tavg'] - df['month_avg']) / df.groupby('month')['tavg'].transform('std')
    df['anomaly'] = df['zscore'].apply(lambda z: 'hot-spike' if z > Z_SCORE_THRESHOLD else 'cold-spike' if z < -Z_SCORE_THRESHOLD else 'normal')
    anomalies = df.groupby(['year', 'anomaly']).size().unstack(fill_value=0)
    anomalies = anomalies.reindex(columns=['hot-spike', 'cold-spike'], fill_value=0)

    anomalies[['hot-spike', 'cold-spike']].plot(figsize=(12,6))
    plt.title("Anomaly Spikes (Z-score > ±2)")
    plt.ylabel("Anomaly Days")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "anomaly_trends.png")
    plt.close()

    print("✅ All plots saved to /visualizations/")

# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    main()
