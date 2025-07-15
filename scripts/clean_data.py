# scripts/clean_data.py

import pandas as pd
from pathlib import Path

# ------------------------
# Configuration
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "dhaka_weather.csv"
OUTPUT_CSV = DATA_DIR / "dhaka_weather_cleaned.csv"

# ------------------------
# Main cleaning logic
# ------------------------
def clean_weather_data(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv, parse_dates=["time"], dayfirst=False)

    if 'tavg' not in df.columns:
        print("❌ Missing 'tavg' column.")
        return

    # Create fallback average temp column
    df['calc_tavg'] = df['tavg']
    missing_mask = df['calc_tavg'].isna()

    if 'tmin' in df.columns and 'tmax' in df.columns:
        df.loc[missing_mask, 'calc_tavg'] = (
            0.4 * df.loc[missing_mask, 'tmin'] + 0.6 * df.loc[missing_mask, 'tmax']
        )

    # Interpolate any remaining missing values
    df['calc_tavg'] = df['calc_tavg'].interpolate(limit_direction='both')

    # Outlier detection and correction
    roll_median = df['calc_tavg'].rolling(window=7, center=True).median()
    deviation = (df['calc_tavg'] - roll_median).abs()
    df['is_outlier'] = deviation > 5
    df.loc[df['is_outlier'], 'calc_tavg'] = roll_median[df['is_outlier']]

    # Clean up precipitation and wind speed if present
    if 'prcp' in df.columns:
        df['prcp'] = df['prcp'].replace(0.0001, 0).clip(upper=300)
    if 'wspd' in df.columns:
        df['wspd'] = df['wspd'].clip(upper=100)

    # Sanity check
    if df['calc_tavg'].isna().sum() > 0:
        raise ValueError("❌ Some 'calc_tavg' values are still missing.")

    # Enrichment
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Autumn"

    df['season'] = df['month'].apply(get_season)

    # Save cleaned data
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"✅ Cleaned data saved to {output_csv}")
    print(f"ℹ️ Outliers corrected: {df['is_outlier'].sum()}")

# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    try:
        clean_weather_data()
    except Exception as e:
        print(f"❌ Cleaning failed: {e}")
