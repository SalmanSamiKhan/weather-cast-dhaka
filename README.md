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