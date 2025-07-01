# app/components.py
import streamlit as st

def styled_badge(text, color):
    return f'<span style="background-color:{color}; color:white; padding:4px 10px; border-radius:8px; font-weight:bold;">{text}</span>'

def show_loading_spinner(message="Processing..."):
    """Display a loading spinner with message"""
    st.markdown(f"""
    <div class="spinner-container">
        <div class="spinner"></div>
        <div style="font-size: 18px; color: #555;">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def display_temperature_card(date, temp, is_predicted=False):
    """Display temperature card with consistent styling"""
    if is_predicted:
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
            🌡️ <strong>Predicted Avg Temperature on {date}:</strong> 
            <span style="font-size:24px; color:#d62828; font-weight:bold;">{temp:.2f} °C</span>
        </div>
        """, unsafe_allow_html=True)
    else:
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
            🌡️ <strong>Actual Avg Temperature on {date}:</strong> 
            <span style="font-size:24px; color:#d62828; font-weight:bold;">{temp:.2f} °C</span>
        </div>
        """, unsafe_allow_html=True)