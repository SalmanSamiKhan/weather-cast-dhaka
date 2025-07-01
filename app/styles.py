# app/styles.py

def get_base_styles():
    return """
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.3rem;
        }
        /* Loading spinner styling */
        .spinner-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            margin: 20px 0;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """

def get_date_input_styles():
    return """
    <style>
        /* Style the label text */
        .custom-label {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Make the date input bigger */
        div[data-baseweb="input"] input {
            font-size: 1.3rem;
            padding: 0.6rem 0.8rem;
            height: 3rem;
        }

        div[data-baseweb="input"] svg {
            width: 1.5rem;
            height: 1.5rem;
        }

        .stDateInput {
            margin-bottom: 1.5rem;
        }
    </style>
    """