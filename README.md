# Dhaka Weather Patterns & Predictor

📈 A Streamlit web application that provides insightful analysis and temperature predictions based on 50 years of historical climate data for Dhaka, Bangladesh.

---

## Features

- **Historical Temperature Data Exploration**  
  Select any past date (from 1975 onwards) to view actual average temperature along with detailed climate insights such as temperature trends, extreme hot/cold days, and anomaly spikes.

- **Future Temperature Prediction**  
  Predict average temperature for future dates (up to 2075) using a trained machine learning model based on historical data trends.

- **Interactive Tabs for Data Visualizations**  
  - Climate change trends  
  - Extreme weather event counts  
  - Monthly temperature heatmaps  
  - Temperature anomaly spikes  

- **Stylish and Responsive UI**  
  Clear, color-coded insights with badges and charts to help you quickly understand the weather patterns and future projections.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- Required Python packages (listed in `requirements.txt`):
  - pandas
  - scikit-learn
  - joblib
  - matplotlib
  - seaborn
  - streamlit

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/dhaka-weather-predictor.git
   cd dhaka-weather-predictor
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Download weather data
   ```bash
   python scripts/download_data.py
5. Clean weather data
   ```bash
   python scripts/clean_data.py
6. Prepare data and model:
    ```bash
    python scripts/train_model.py
7. Run the app
   ```bash
   streamlit run app/streamlit_app.py

## Q&A

### 1. Why did you choose Meteostat and Streamlit?
Meteostat provides reliable, historical weather data with a simple Python API, making it easy to automate data collection for Dhaka over many years.
Streamlit is a modern, open-source framework for building interactive data apps quickly. It allows for rapid prototyping and sharing of data visualizations and models with users without complex web development.

### 2. How do you handle missing or anomalous data?
During data cleaning (handled in clean_data.py), missing values are identified and either filled using interpolation or removed, depending on the context and impact on analysis.
Anomalous data points (outliers) are detected using statistical methods (e.g., z-score, IQR) and are either flagged for analysis or handled appropriately to avoid skewing results.

### 3. What model did you use for forecasting and why?
The project uses a Ridge Regression model (a regularized linear regression) for temperature prediction. This is implemented using scikit-learn’s Ridge class, combined with feature scaling in a pipeline. Ridge regression is chosen for its stability and ability to prevent overfitting, making it suitable for time-dependent weather data. (saved as temperature_model.joblib) for temperature prediction. Likely candidates are linear regression, decision trees, or time series models (e.g., ARIMA), chosen for their interpretability and suitability for time-dependent weather data.
The model was selected based on performance (accuracy, RMSE) and its ability to generalize on unseen data.

### 4. How do your visualizations help in understanding climate trends?
Visualizations such as anomaly trends, climate trends, extreme events, and monthly heatmaps provide clear, intuitive insights into long-term weather patterns, frequency of extreme events, and seasonal variations.
These plots help both technical and non-technical audiences grasp complex climate data and identify significant changes or anomalies over time.

### 5. What are the limitations of your approach?
Data limitations: The accuracy depends on the quality and completeness of the Meteostat data.
Model limitations: The chosen model may not capture all nonlinearities or rare events in the data.
Scope: The analysis is limited to Dhaka and may not generalize to other regions.
External factors: The model does not account for all possible external influences (e.g., urbanization, climate change effects beyond historical trends).

## 6. Potential Improvements
Incorporating more advanced models (e.g., LSTM, Prophet).
Adding more granular weather features (humidity, rainfall).
Integrating real-time data updates.
Expanding to regional or national scale.