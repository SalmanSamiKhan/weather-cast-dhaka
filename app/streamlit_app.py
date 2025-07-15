# app/streamlit_app.py
import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import calendar
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import time

from styles import get_base_styles, get_date_input_styles
from components import styled_badge, show_loading_spinner, display_temperature_card

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "../data/dhaka_weather_cleaned.csv")
trained_model_path = os.path.join(BASE_DIR, "../models/temperature_model.joblib")

# Set page config and title
st.set_page_config("Dhaka Weather Patterns", layout="centered")
st.title("📈 Dhaka Weather Patterns & Predictor")

# Inject CSS styles
st.markdown(get_base_styles(), unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['tavg'] = df['tavg'].interpolate()
    return df

@st.cache_resource
def load_model():
    model_path = Path(trained_model_path)
    if model_path.exists():
        return joblib.load(model_path)
    return None

def plot_trends(df):
    yearly = df.groupby('year')['tavg'].agg(['mean', 'min', 'max']).reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(data=yearly, x='year', y='mean', label='Avg')
    sns.lineplot(data=yearly, x='year', y='max', label='Max')
    sns.lineplot(data=yearly, x='year', y='min', label='Min')
    plt.title("Climate Trend in Dhaka")
    plt.ylabel("Temperature (°C)")
    st.pyplot(plt.gcf())
    plt.close()

def plot_extremes(df):
    df['extreme'] = df['tavg'].apply(lambda x: 'hot' if x > 30 else 'cold' if x < 15 else 'normal')
    extreme_counts = df.groupby(['year', 'extreme']).size().unstack(fill_value=0)

    # Ensure 'hot' and 'cold' columns always exist
    for col in ['hot', 'cold']:
        if col not in extreme_counts.columns:
            extreme_counts[col] = 0

    # Select and reorder only 'hot' and 'cold'
    plot_data = extreme_counts[['hot', 'cold']]

    # Plot using correct color order: hot = orange, cold = blue
    plot_data.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 6),
        color=['orange', 'blue']  # this order matches ['hot', 'cold']
    )

    plt.title("Extreme Weather Events Over Time")
    plt.ylabel("Days")
    st.pyplot(plt.gcf())
    plt.close()


def plot_monthly_heatmap(df):
    monthly = df.groupby(['year', 'month'])['tavg'].mean().unstack()
    plt.figure(figsize=(12,6))
    sns.heatmap(monthly, cmap="coolwarm", linewidths=0.5)
    plt.title("Monthly Avg Temp Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    st.pyplot(plt.gcf())
    plt.close()

def plot_anomalies(df):
    df['month_avg'] = df.groupby('month')['tavg'].transform('mean')
    df['zscore'] = (df['tavg'] - df['month_avg']) / df.groupby('month')['tavg'].transform('std')
    df['anomaly'] = df['zscore'].apply(lambda z: 'hot-spike' if z > 2 else 'cold-spike' if z < -2 else 'normal')
    anomalies = df.groupby(['year', 'anomaly']).size().unstack(fill_value=0)
    for col in ['hot-spike', 'cold-spike']:
        if col not in anomalies.columns:
            anomalies[col] = 0
    anomalies = anomalies[['hot-spike', 'cold-spike'] + [c for c in anomalies.columns if c not in ['hot-spike', 'cold-spike']]]
    
    # Plot with specific colors
    ax = anomalies[['hot-spike', 'cold-spike']].plot(
        figsize=(12, 6),
        color={'hot-spike': 'gold', 'cold-spike': 'blue'}
    )
    plt.title("Anomaly Spikes (Z-score > ±2)")
    plt.ylabel("Anomaly Days")
    plt.legend(title="Anomaly Type", loc='upper right')
    st.pyplot(plt.gcf())
    plt.close()

# app/streamlit_app.py (updated section)

def display_historical_insights(df, selected_date):
    """Display historical data insights with detailed statistics"""
    day_data = df[(df['year'] == selected_date.year) & 
                  (df['month'] == selected_date.month) & 
                  (df['day'] == selected_date.day)]
    
    if not day_data.empty:
        display_temperature_card(selected_date, day_data.iloc[0]['tavg'])
    else:
        st.warning("No historical data for this date.")
        return
    
    st.markdown(f"""
    <h3 style="font-size:26px; font-weight:600; margin-top:20px; margin-bottom:20px;">
        📊 Climate Change & Insights for {selected_date.year}
    </h3>
    """, unsafe_allow_html=True)
    
    # Calculate all the required metrics
    yearly_avg = df[df['year'] == selected_date.year]['tavg'].mean()
    overall_avg = df['tavg'].mean()
    diff = yearly_avg - overall_avg
    
    # Extreme days calculation
    yearly_data = df[df['year'] == selected_date.year].copy()
    yearly_data['extreme'] = yearly_data['tavg'].apply(
        lambda x: 'extreme hot' if x > 35 else
                  'hot' if x > 30 else
                  'extreme cold' if x < 10 else
                  'cold' if x < 15 else
                  'normal'
    )
    extreme_counts = yearly_data['extreme'].value_counts()
    
    # Anomalies calculation
    yearly_data['month_avg'] = yearly_data.groupby('month')['tavg'].transform('mean')
    yearly_data['zscore'] = (yearly_data['tavg'] - yearly_data['month_avg']) / yearly_data.groupby('month')['tavg'].transform('std')
    hot_spikes = (yearly_data['zscore'] > 2).sum()
    cold_spikes = (yearly_data['zscore'] < -2).sum()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic temperature stats
        st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #2f3e46;
                border: 1.5px solid #52796f;
                box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-bottom: 20px;
            ">
                <h3 style="color: #84a98c; margin-top: 0;">Temperature Statistics</h3>
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Average temperature in {selected_date.year}:</b> {yearly_avg:.2f} °C
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Overall 50-year average temperature:</b> {overall_avg:.2f} °C
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Temperature difference:</b> 
                    <span style="color:{'lightgreen' if diff < 0 else '#ff6b6b'};">{diff:+.2f} °C</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Extreme days
        st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #2f3e46;
                border: 1.5px solid #52796f;
                box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-bottom: 20px;
            ">
                <h3 style="color: #84a98c; margin-top: 0;">Extreme Days Count</h3>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#ff9f1c;">Hot days (>30°C):</b> {extreme_counts.get('hot', 0)}
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#a8dadc;">Cold days (<15°C):</b> {extreme_counts.get('cold', 0)}
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#e63946;">Extreme hot days (>35°C):</b> {extreme_counts.get('extreme hot', 0)}
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#457b9d;">Extreme cold days (<10°C):</b> {extreme_counts.get('extreme cold', 0)}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Anomaly stats
        st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #2f3e46;
                border: 1.5px solid #52796f;
                box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-bottom: 20px;
            ">
                <h3 style="color: #84a98c; margin-top: 0;">Temperature Anomalies</h3>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#f4a261;">Hot spike days (z > 2):</b> {hot_spikes}
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style="color:#8ecae6;">Cold spike days (z < -2):</b> {cold_spikes}
                </p>
                <p style="font-size:14px; margin: 8px 0; color: #b7b7a4;">
                    <i>Anomalies are calculated relative to monthly averages</i>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visual badges for quick reference
        st.markdown("""
            <div style="margin-top: 20px;">
                <h4 style="color: #84a98c; margin-bottom: 10px;">Quick Stats</h4>
        """, unsafe_allow_html=True)
        
        # Create a grid of badges
        cols = st.columns(2)
        with cols[0]:
            st.markdown(styled_badge(f"Avg Temp: {yearly_avg:.1f}°C", "#2a9d8f"), unsafe_allow_html=True)
            st.markdown(styled_badge(f"Hot Days: {extreme_counts.get('hot', 0)}", "#e76f51"), unsafe_allow_html=True)
            st.markdown(styled_badge(f"Hot Spikes: {hot_spikes}", "#d62828"), unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(styled_badge(f"Diff: {diff:+.2f}°C", "#457b9d"), unsafe_allow_html=True)
            st.markdown(styled_badge(f"Cold Days: {extreme_counts.get('cold', 0)}", "#1d3557"), unsafe_allow_html=True)
            st.markdown(styled_badge(f"Cold Spikes: {cold_spikes}", "#003049"), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
def display_future_prediction(model, df, selected_date):
    """Display future prediction results with consistent styling"""
    if model is None:
        st.error("Model not found. Please train and save the model first.")
        return
    
    with st.spinner("Predicting future temperature..."):
        input_df = pd.DataFrame([[selected_date.year, selected_date.month, selected_date.day]],
                                columns=["year", "month", "day"])
        pred = model.predict(input_df)[0]
        
        display_temperature_card(selected_date, pred, is_predicted=True)
        
        st.markdown(f"""
        <h3 style="font-size:26px; font-weight:600; margin-top:20px; margin-bottom:20px;">
            📊 Projected Climate Insights for {selected_date.year}
        </h3>
        """, unsafe_allow_html=True)
        
        # Calculate all required metrics
        overall_avg = df['tavg'].mean()
        past_decade = df[df['year'] >= df['year'].max() - 9]
        decade_avg = past_decade['tavg'].mean()
        diff = pred - overall_avg
        
        # Historical extremes for projection
        hot_days = df[df['tavg'] > 30].groupby('year').size().mean()
        extreme_hot_days = df[df['tavg'] > 35].groupby('year').size().mean()
        cold_days = df[df['tavg'] < 15].groupby('year').size().mean()
        extreme_cold_days = df[df['tavg'] < 10].groupby('year').size().mean()
        
        # Anomalies projection
        df['month_avg'] = df.groupby('month')['tavg'].transform('mean')
        df['zscore'] = (df['tavg'] - df['month_avg']) / df.groupby('month')['tavg'].transform('std')
        hot_spikes = (df['zscore'] > 2).groupby(df['time'].dt.year).sum().mean()
        cold_spikes = (df['zscore'] < -2).groupby(df['time'].dt.year).sum().mean()
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature comparison stats
            st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 12px;
                    background-color: #2f3e46;
                    border: 1.5px solid #52796f;
                    box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #84a98c; margin-top: 0;">Projected Temperature</h3>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b>Predicted temperature:</b> <span style="color:#ff9f1c;">{pred:.2f} °C</span>
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b>Long-term average:</b> {overall_avg:.2f} °C
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b>Recent decade average:</b> {decade_avg:.2f} °C
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b>Projected difference:</b> 
                        <span style="color:{'lightgreen' if diff < 0 else '#ff6b6b'};">{diff:+.2f} °C</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Projected extremes
            st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 12px;
                    background-color: #2f3e46;
                    border: 1.5px solid #52796f;
                    box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #84a98c; margin-top: 0;">Projected Extremes</h3>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#ff9f1c;">Hot days (>30°C):</b> ~{hot_days:.0f}
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#e63946;">Extreme hot days (>35°C):</b> ~{extreme_hot_days:.0f}
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#a8dadc;">Cold days (<15°C):</b> ~{cold_days:.0f}
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#457b9d;">Extreme cold days (<10°C):</b> ~{extreme_cold_days:.0f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Projected anomalies
            st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 12px;
                    background-color: #2f3e46;
                    border: 1.5px solid #52796f;
                    box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #84a98c; margin-top: 0;">Projected Anomalies</h3>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#f4a261;">Hot spike days (z > 2):</b> ~{hot_spikes:.0f}
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#8ecae6;">Cold spike days (z < -2):</b> ~{cold_spikes:.0f}
                    </p>
                    <p style="font-size:14px; margin: 8px 0; color: #b7b7a4;">
                        <i>Based on historical anomaly patterns</i>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Quick stats badges
            st.markdown("""
                <div style="margin-top: 20px;">
                    <h4 style="color: #84a98c; margin-bottom: 10px;">Projected Quick Stats</h4>
            """, unsafe_allow_html=True)
            
            # Create a grid of badges
            cols = st.columns(2)
            with cols[0]:
                st.markdown(styled_badge(f"Predicted: {pred:.1f}°C", "#2a9d8f"), unsafe_allow_html=True)
                st.markdown(styled_badge(f"Hot Days: ~{hot_days:.0f}", "#e76f51"), unsafe_allow_html=True)
                st.markdown(styled_badge(f"Hot Spikes: ~{hot_spikes:.0f}", "#d62828"), unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(styled_badge(f"Diff: {diff:+.2f}°C", "#457b9d"), unsafe_allow_html=True)
                st.markdown(styled_badge(f"Cold Days: ~{cold_days:.0f}", "#1d3557"), unsafe_allow_html=True)
                st.markdown(styled_badge(f"Cold Spikes: ~{cold_spikes:.0f}", "#003049"), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
# Main app logic
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Temperature",
    "📊 Climate",
    "🔥 Extremes",
    "🌀 Patterns",
    "⛈️ Anomalies"
])

df = load_data()
model = load_model()
today = datetime.date.today()
max_future_year = 2075

with tab1:
    st.markdown(get_date_input_styles(), unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:24px;'>📅 Select a Date for Weather Info</h3>", unsafe_allow_html=True)
    
    selected_date = st.date_input(
        label="",
        value=today,
        min_value=datetime.date(1975, 1, 1),
        max_value=datetime.date(max_future_year, 12, 31)
    )

    if 'last_date' not in st.session_state:
        st.session_state.last_date = selected_date

    date_changed = st.session_state.last_date != selected_date
    st.session_state.last_date = selected_date

    if date_changed and selected_date > today:
        with st.spinner("Predicting future temperature..."):
            time.sleep(1)

    if selected_date <= today:
        display_historical_insights(df, selected_date)
    else:
        display_future_prediction(model, df, selected_date)

with tab2: plot_trends(df)
with tab3: plot_extremes(df)
with tab4: plot_monthly_heatmap(df)
with tab5: plot_anomalies(df)