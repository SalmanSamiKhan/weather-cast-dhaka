# scripts/train_model.py

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
from math import sqrt
import joblib

# ------------------------
# Configuration
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "temperature_model.joblib"
CSV_FILE = DATA_DIR / "dhaka_weather_cleaned.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Main logic
# ------------------------
def train_model():
    # Load data
    df = pd.read_csv(CSV_FILE, parse_dates=["time"])
    if 'tavg' not in df.columns:
        raise ValueError("Missing 'tavg' in dataset.")

    # Handle missing values
    df['tavg'] = df['tavg'].interpolate(limit_direction='both').ffill().bfill()

    # Feature engineering
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    # Use 'season' as categorical feature for better winter prediction
    season_categories = ['Autumn', 'Monsoon', 'Spring', 'Winter', 'Summer']
    if 'season' in df.columns:
        X = df[['year', 'month', 'day', 'season']].copy()
        # One-hot encode 'season' and ensure all possible columns exist
        X = pd.get_dummies(X, columns=['season'])
        for cat in season_categories:
            col = f'season_{cat}'
            if col not in X.columns:
                X[col] = 0
        # Ensure column order is consistent
        season_cols = [f'season_{cat}' for cat in season_categories]
        X = X[['year', 'month', 'day'] + season_cols]
    else:
        X = df[['year', 'month', 'day']]
    y = df['tavg']

    # Time-based split
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Pipeline: Scaling + Ridge
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Fit and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# ------------------------
# Entry Point
# ------------------------
if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"❌ Training failed: {e}")
