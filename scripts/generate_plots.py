# scripts/generate_plots.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/dhaka_weather_cleaned.csv", parse_dates=["time"])
df['tavg'] = df['tavg'].interpolate()
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day

os.makedirs("visualizations", exist_ok=True)

# 1. Climate Trend
yearly = df.groupby('year')['tavg'].agg(['mean', 'min', 'max']).reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(data=yearly, x='year', y='mean', label='Avg')
sns.lineplot(data=yearly, x='year', y='max', label='Max')
sns.lineplot(data=yearly, x='year', y='min', label='Min')
plt.title("Climate Trend in Dhaka")
plt.ylabel("Temperature (°C)")
plt.savefig("visualizations/climate_trend.png")
plt.close()

# 2. Extreme Events
df['extreme'] = df['tavg'].apply(lambda x: 'hot' if x > 38 else 'cold' if x < 10 else 'normal')
extreme_counts = df.groupby(['year', 'extreme']).size().unstack(fill_value=0)

# Make sure 'hot' and 'cold' columns exist
for col in ['hot', 'cold']:
    if col not in extreme_counts.columns:
        extreme_counts[col] = 0
# Sort columns so hot and cold are first (optional)
extreme_counts = extreme_counts[['hot', 'cold'] + [c for c in extreme_counts.columns if c not in ['hot','cold']]]

extreme_counts[['hot', 'cold']].plot(kind='bar', stacked=True, figsize=(14,6))
plt.title("Extreme Weather Events Over Time")
plt.ylabel("Days")
plt.savefig("visualizations/extreme_events.png")
plt.close()

# 3. Monthly Heatmap
monthly = df.groupby(['year', 'month'])['tavg'].mean().unstack()
plt.figure(figsize=(12,6))
sns.heatmap(monthly, cmap="coolwarm", linewidths=0.5)
plt.title("Monthly Avg Temp Heatmap")
plt.xlabel("Month")
plt.ylabel("Year")
plt.savefig("visualizations/monthly_heatmap.png")
plt.close()

# 4. Z-score Anomalies
df['month_avg'] = df.groupby('month')['tavg'].transform('mean')
df['zscore'] = (df['tavg'] - df['month_avg']) / df.groupby('month')['tavg'].transform('std')
df['anomaly'] = df['zscore'].apply(lambda z: 'hot-spike' if z > 2 else 'cold-spike' if z < -2 else 'normal')
anomalies = df.groupby(['year', 'anomaly']).size().unstack(fill_value=0)

# Make sure anomaly columns exist
for col in ['hot-spike', 'cold-spike']:
    if col not in anomalies.columns:
        anomalies[col] = 0
anomalies = anomalies[['hot-spike', 'cold-spike'] + [c for c in anomalies.columns if c not in ['hot-spike', 'cold-spike']]]

anomalies[['hot-spike', 'cold-spike']].plot(figsize=(12,6))
plt.title("Anomaly Spikes (Z-score > ±2)")
plt.ylabel("Anomaly Days")
plt.savefig("visualizations/anomaly_trends.png")
plt.close()

print("✅ All plots saved to /visualizations/")
