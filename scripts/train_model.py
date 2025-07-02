import pandas as pd
from sklearn.linear_model import Ridge  # More stable than plain LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os
from math import sqrt

# Configuration
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/temperature_model.joblib"

def train_model():
    try:
        # Load data
        df = pd.read_csv("data/dhaka_weather_cleaned.csv", parse_dates=["time"])
        
        # Handle missing values - more robust than just interpolate
        df['tavg'] = df['tavg'].interpolate(limit_direction='both').ffill().bfill()
        
        # Feature engineering (keeping same features as original)
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        
        # Prepare features and target
        X = df[['year', 'month', 'day']]
        y = df['tavg']
        
        # Time-based split (better than random for time series)
        train_size = int(0.8 * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create pipeline with scaling and regularized regression
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)  # Regularization to prevent overfitting
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=5, scoring='neg_mean_absolute_error')
        print(f"Cross-validated MAE: {-cv_scores.mean():.2f} (+/- {-cv_scores.std():.2f})")
        
        # Final training
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")
        
        # Save model (now saves entire pipeline including scaler)
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()