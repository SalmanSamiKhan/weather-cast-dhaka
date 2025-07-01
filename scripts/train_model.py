# scripts/train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from math import sqrt

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/dhaka_weather_cleaned.csv", parse_dates=["time"])
df['tavg'] = df['tavg'].interpolate()
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day

X = df[['year', 'month', 'day']]
y = df['tavg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

joblib.dump(model, "models/temperature_model.joblib")
print("✅ Model saved to models/temperature_model.joblib")
