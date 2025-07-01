# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import calendar
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config("Dhaka Weather Patterns", layout="centered")

st.title("📈 Dhaka Weather Patterns & Predictor")
# st.markdown("Machine learning-based analysis using 50 years of climate data.")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/dhaka_weather_cleaned.csv", parse_dates=["time"])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['tavg'] = df['tavg'].interpolate()
    return df

@st.cache_resource
def load_model():
    model_path = Path("../models/temperature_model.joblib")
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
    df['extreme'] = df['tavg'].apply(lambda x: 'hot' if x > 30 else 'cold' if x < 10 else 'normal')
    extreme_counts = df.groupby(['year', 'extreme']).size().unstack(fill_value=0)
    for col in ['hot', 'cold']:
        if col not in extreme_counts.columns:
            extreme_counts[col] = 0
    extreme_counts = extreme_counts[['hot', 'cold'] + [c for c in extreme_counts.columns if c not in ['hot', 'cold']]]
    extreme_counts[['hot', 'cold']].plot(kind='bar', stacked=True, figsize=(14,6))
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
    anomalies[['hot-spike', 'cold-spike']].plot(figsize=(12,6))
    plt.title("Anomaly Spikes (Z-score > ±2)")
    plt.ylabel("Anomaly Days")
    st.pyplot(plt.gcf())
    plt.close()

def styled_badge(text, color):
    return f'<span style="background-color:{color}; color:white; padding:4px 10px; border-radius:8px; font-weight:bold;">{text}</span>'

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Temperature",
    "📊 Climate Change",
    "🔥 Extremes",
    "🌀 Monthly Patterns",
    "⛈️ Anomalies"
])

df = load_data()
model = load_model()
today = datetime.date.today()
max_future_year = 2075

with tab1:
    st.markdown("<h3 style='font-size:24px; margin-bottom:10px;'>📅 Select a Date for Weather Info</h3>", unsafe_allow_html=True)

    selected_date = st.date_input(
        label="",
        value=today,
        min_value=datetime.date(1975, 1, 1),
        max_value=datetime.date(max_future_year, 12, 31)
    )


    if selected_date <= today:
        # Show actual temp
        day_data = df[(df['year'] == selected_date.year) & 
                      (df['month'] == selected_date.month) & 
                      (df['day'] == selected_date.day)]
        if not day_data.empty:
            actual_temp = day_data.iloc[0]['tavg']
            # st.success(f"Actual Avg Temperature on {selected_date}: {actual_temp:.2f} °C")
            st.markdown(f"""
            <div style="
                background-color: #d4edda;
                color: #155724;
                padding: 10px;
                border-left: 6px solid #28a745;
                border-radius: 8px;
                font-size: 20px;
                margin-bottom: 1rem;
            ">
                🌡️ <strong>Actual Avg Temperature on {selected_date}:</strong> 
                <span style="font-size:24px; color:#d62828; font-weight:bold;">{actual_temp:.2f} °C</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("No historical data for this date.")

        st.markdown(f"### Climate Change & Insights for {selected_date.year}")

        # Climate change: average temp trend for selected year
        yearly_avg = df[df['year'] == selected_date.year]['tavg'].mean()
        overall_avg = df['tavg'].mean()
        diff = yearly_avg - overall_avg

        
        st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #2f3e46;  /* dark slate blue-gray */
                border: 1.5px solid #52796f;  /* softer greenish-gray border */
                box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                max-width: 600px;
                margin-bottom: 20px;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Average temperature in {selected_date.year}:</b> {yearly_avg:.2f} °C
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Overall 50-year average temperature:</b> {overall_avg:.2f} °C
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b>Temperature difference from long-term average:</b> 
                    <span style="color:{'lightgreen' if diff < 0 else '#ff6b6b'};">{diff:+.2f} °C</span>
                </p>
            </div>
        """, unsafe_allow_html=True)


        # Extremes: count hot/cold/extreme hot/extreme cold days that year
        extremes = df[df['year'] == selected_date.year].copy()
        extremes['extreme'] = extremes['tavg'].apply(
            lambda x: 'extreme hot' if x > 35 else
                      'hot' if x > 30 else
                      'extreme cold' if x < 10 else
                      'cold' if x < 15 else
                      'normal'
        )
        counts = extremes['extreme'].value_counts()

        def get_count(label):
            return counts[label] if label in counts else 0

        st.markdown(f"""
            <div style="display:flex; gap:15px; flex-wrap:wrap; margin-bottom:20px;">
                <div>{styled_badge(f"Hot days (>30°C): {get_count('hot')}", '#ff8c00')}</div>
                <div>{styled_badge(f"Cold days (<15°C): {get_count('cold')}", '#1e90ff')}</div> 
            </div>
            <div style="display:flex; gap:15px; flex-wrap:wrap; margin-bottom:20px;">
                <div>{styled_badge(f"Extreme hot days (>35°C): {get_count('extreme hot')}", '#d62728')}</div>
                <div>{styled_badge(f"Extreme cold days (<10°C): {get_count('extreme cold')}", '#00509e')}</div>
            </div>
        """, unsafe_allow_html=True)


        # Anomalies: days with big temp spikes
        extremes['month_avg'] = extremes.groupby('month')['tavg'].transform('mean')
        extremes['zscore'] = (extremes['tavg'] - extremes['month_avg']) / extremes.groupby('month')['tavg'].transform('std')
        hot_spikes = (extremes['zscore'] > 2).sum()
        cold_spikes = (extremes['zscore'] < -2).sum()


        st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #2f3e46;  /* dark slate blue-gray */
                border: 1.5px solid #52796f;  /* softer greenish-gray border */
                box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                max-width: 400px;
                margin-top: 10px;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <p style="font-size:16px; margin: 8px 0;">
                    <b style='color:#f4a261;'>Hot spike days (z > 2):</b> <span style='color:#e76f51;'>{hot_spikes}</span>
                </p>
                <p style="font-size:16px; margin: 8px 0;">
                    <b style='color:#8ecae6;'>Cold spike days (z < -2):</b> <span style='color:#219ebc;'>{cold_spikes}</span>
                </p>
            </div>
        """, unsafe_allow_html=True)



    else:
        # Future prediction
        if model is None:
            st.error("Model not found. Please train and save the model first.")
        else:
            input_df = pd.DataFrame([[selected_date.year, selected_date.month, selected_date.day]], columns=["year", "month", "day"])
            pred = model.predict(input_df)[0]
            # st.success(f"Predicted Avg Temperature on {selected_date}: {pred:.2f} °C")
            st.markdown(f"""
                <div style="
                    background-color: #d4edda;
                    color: #155724;
                    padding: 10px 16px;
                    border-left: 6px solid #28a745;
                    border-radius: 8px;
                    font-size: 20px;
                    margin-bottom: 1rem;
                ">
                    🌡️ <strong>Predicted Avg Temperature on {selected_date}:</strong> 
                    <span style="font-size:24px; color:#d62828; font-weight:bold;">{pred:.2f} °C</span>
                </div>
            """, unsafe_allow_html=True)


            st.markdown("### Projected Climate Insights Based on Historical Patterns")
            overall_avg = df['tavg'].mean()
            st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 12px;
                    background-color: #2f3e46;
                    border: 1.5px solid #52796f;
                    box-shadow: 2px 2px 8px rgba(82, 121, 111, 0.5);
                    max-width: 600px;
                    margin-top: 15px;
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    <p style="font-size:16px; margin: 8px 0;">
                        <b style="color:#f4a261;">Long-term average temperature:</b> 
                        <span style="color:#ffd166;">{overall_avg:.2f} °C</span>
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        🔺 Expect <span style="color:#ef476f;"><b>increasing trend</b></span> in temperature consistent with past 50 years.
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        🔥 Extreme <span style="color:#f77f00;"><b>heat</b></span> and ❄️ <span style="color:#118ab2;"><b>cold</b></span> days projected based on historical patterns.
                    </p>
                    <p style="font-size:16px; margin: 8px 0;">
                        ⚠️ Anomalies expected to continue following historical <span style="color:#90e0ef;"><b>variability</b></span>.
                    </p>
                </div>
            """, unsafe_allow_html=True)


# Full charts in their tabs
with tab2: plot_trends(df)
with tab3: plot_extremes(df)
with tab4: plot_monthly_heatmap(df)
with tab5: plot_anomalies(df)
