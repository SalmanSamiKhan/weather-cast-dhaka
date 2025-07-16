# Dhaka Weather Patterns & Predictor

📈 A Streamlit web application that provides insightful analysis and temperature predictions based on 50 years of historical climate data for Dhaka, Bangladesh.

---

## Features

- **Historical Temperature Data Exploration**: Select any past date (from 1975 onwards) to view the actual average temperature, along with detailed climate insights such as temperature trends, extreme hot/cold days, and anomaly spikes.
- **Future Temperature Prediction**: Predict the average temperature for future dates (up to 2075) using a trained Ridge Regression model based on historical data trends.
- **Interactive Data Visualizations**:
  - **Climate Trends**: Visualize long-term changes in average, maximum, and minimum temperatures.
  - **Extreme Weather Events**: Track the frequency of hot and cold days over the years.
  - **Monthly Temperature Heatmaps**: Identify seasonal patterns and temperature variations.
  - **Temperature Anomaly Spikes**: Detect unusual temperature spikes compared to monthly averages.
- **Stylish and Responsive UI**: A clear, color-coded interface with badges and charts to help you quickly understand weather patterns and future projections.

### Temperature Anomalies (Explained)
- "Anomalies" refer to how much a specific day's temperature deviates from what's typical for that time of year — in this case, based on monthly averages.
- We compute a z-score for each day's average temperature relative to its month
### Interpretation of Anomaly Report
- Hot spike days (z > 2): 4
→ There were 4 days where the temperature was more than 2 standard deviations above the average for that month — likely heatwaves or record highs.
- Cold spike days (z < -2): 16
→ There were 16 days where the temperature was more than 2 standard deviations below the monthly norm — possible cold waves or unusually chilly days.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- The required Python packages are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SalmanSamiKhan/weather-cast-dhaka.git
    cd weather-cast-dhaka
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Processing
4.  **Download the weather data:**
    ```bash
    python scripts/download_data.py
    ```

5.  **Clean the weather data:**
    ```bash
    python scripts/clean_data.py
    ```
### Training Model
6.  **Train the prediction model:**

    ```bash
    python scripts/train_model.py
    ```
### Running the application
7.  **Run the Streamlit application:**

    ```bash
    streamlit run app/streamlit_app.py
    ```
8. **Open in Browser**
- Local URL: http://localhost:8501
- Network URL: http://192.168.100.212:8501
---

## Q&A

### 1. Why did you choose Meteostat and Streamlit?

-   **Meteostat**: It provides a reliable, historical weather dataset with a simple Python API, making it easy to automate data collection for Dhaka over many years.
-   **Streamlit**: It is a modern, open-source framework for building interactive data applications quickly. It allows for rapid prototyping and sharing of data visualizations and models without requiring complex web development.

### 2. How do you handle missing or anomalous data?

-   **Missing Values**: During data cleaning (in `scripts/clean_data.py`), missing values are filled using interpolation to ensure data continuity.
-   **Anomalous Data**: Outliers are detected using statistical methods and are corrected to prevent them from skewing the analysis and model training.

### 3. What model did you use for forecasting and why?

This project uses a **Ridge Regression** model from scikit-learn, trained through a pipeline that includes standard scaling for feature normalization. Ridge Regression was selected due to its robustness in handling multicollinearity and its ability to prevent overfitting through L2 regularization — an important consideration when working with seasonal and time-based weather features. The model is saved as `temperature_model.joblib` after training.

#### The Model is Trained Using:
- Year, month, and day as temporal predictors
- 5-fold cross-validation on the training set to evaluate mean absolute error (MAE)
- Time-based data splitting to respect temporal dependencies
- One-hot encoded season categories (if available) to improve seasonal sensitivity, especially for capturing winter anomalies. If the dataset contains a season column, it's one-hot encoded and included in training. This helps improve seasonal temperature forecasting, particularly for months with high variance like January and April.

### 4. How do your visualizations help in understanding climate trends?

The visualizations provide clear, intuitive insights into long-term weather patterns, the frequency of extreme events, and seasonal variations. These plots help both technical and non-technical audiences grasp complex climate data and identify significant changes or anomalies over time.

### 5. What are the limitations of this approach?

-   **Data Limitations**: The accuracy of the predictions depends on the quality and completeness of the Meteostat data.
-   **Model Limitations**: The Ridge Regression model may not capture all non-linearities or rare events in the weather data.
-   **Scope**: The analysis is limited to Dhaka and may not be applicable to other regions.
-   **External Factors**: The model does not account for all possible external influences, such as urbanization or climate change effects that are not captured in the historical trends.

### 6. What are some potential improvements?

-   Incorporate more advanced models (e.g., LSTM, Prophet).
-   Add more granular weather features, such as humidity and rainfall.
-   Integrate real-time data updates.
-   Expand the analysis to a regional or national scale.
