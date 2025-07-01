# scripts/clean_data.py

import pandas as pd

def clean_weather_data(input_csv="data/dhaka_weather.csv", output_csv="data/dhaka_weather_cleaned.csv"):
    # Load raw data
    df = pd.read_csv(input_csv, parse_dates=["time"])
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Interpolate missing average temperature (tavg)
    df['tavg'] = df['tavg'].interpolate(limit_direction='both')

    # Fill any remaining missing tavg with mean of min and max temperature
    if 'tmin' in df.columns and 'tmax' in df.columns:
        df['tavg'] = df['tavg'].fillna((df['tmin'] + df['tmax']) / 2)

    # Drop rows where tavg is still missing
    missing_count = df['tavg'].isna().sum()
    if missing_count > 0:
        print(f"Dropping {missing_count} rows with missing average temperature")
    df = df.dropna(subset=['tavg'])

    # Remove outliers: unrealistic temperatures below -5 or above 45 Celsius
    before_rows = len(df)
    df = df[(df['tavg'] > -5) & (df['tavg'] < 45)]
    removed_outliers = before_rows - len(df)
    if removed_outliers > 0:
        print(f"Removed {removed_outliers} outlier rows outside temperature range [-5, 45] °C")

    # Add useful date features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day

    # Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    clean_weather_data()
