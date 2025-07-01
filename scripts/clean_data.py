import pandas as pd

def clean_weather_data(input_csv="data/dhaka_weather.csv", output_csv="data/dhaka_weather_cleaned.csv"):
    df = pd.read_csv(input_csv, parse_dates=["time"], dayfirst=False)
    
    df['calc_tavg'] = df['tavg'].copy()
    mask = df['calc_tavg'].isna()
    
    if all(col in df.columns for col in ['tmin', 'tmax']):
        df.loc[mask, 'calc_tavg'] = (df['tmin']*0.4 + df['tmax']*0.6)[mask]
    
    df['calc_tavg'] = df['calc_tavg'].interpolate(limit_direction='both')
    
    roll_median = df['calc_tavg'].rolling(window=7, center=True).median()
    deviation = abs(df['calc_tavg'] - roll_median)
    df['is_outlier'] = deviation > 5
    df.loc[df['is_outlier'], 'calc_tavg'] = roll_median[df['is_outlier']]
    
    if 'prcp' in df.columns:
        df['prcp'] = df['prcp'].replace(0.0001, 0).clip(upper=300)
    
    if 'wspd' in df.columns:
        df['wspd'] = df['wspd'].clip(upper=100)
    
    assert df['calc_tavg'].isna().sum() == 0
    
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    
    seasons = [
        (0, 2, 'Winter'),
        (3, 5, 'Spring'), 
        (6, 9, 'Monsoon'),
        (10, 11, 'Autumn'),
        (12, 12, 'Winter')
    ]
    
    df['season'] = 'Unknown'
    for start, end, label in seasons:
        df.loc[df['month'].between(start, end), 'season'] = label
    
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
    print(f"Removed outliers: {df['is_outlier'].sum()}")

if __name__ == "__main__":
    clean_weather_data()