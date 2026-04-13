import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="centered")

# --- Load the Model ---
# We use @st.cache_resource so the model only loads once, making the app lightning fast
@st.cache_resource
def load_model():
    model_path = "model/churn_pipeline.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .risk-high {
        background-color: #ffcccc;
        color: #cc0000;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #cc0000;
    }
    .risk-low {
        background-color: #e6ffe6;
        color: #008000;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #008000;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🔮 Customer Churn Prediction Engine")
st.markdown("Enter the customer's details below to predict their likelihood of canceling their subscription.")

if model is None:
    st.error("⚠️ Model not found! Please run `train_model.py` first to generate the pipeline.")
    st.stop()

# --- Input Form ---
st.markdown("### 📋 Customer Profile")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
        
    with col2:
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        tech_support = st.selectbox("Tech Support Included?", ['Yes', 'No', 'No internet service'])
        
    submit_button = st.form_submit_button(label="Analyze Churn Risk")

# --- Prediction Logic ---
if submit_button:
    # 1. Package the inputs exactly how the model expects them
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'TechSupport': [tech_support]
    })
    
    # 2. Run the prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    churn_prob = probabilities[1] * 100  # Probability of class 1 (Churn)
    
    # 3. Display the Output
    st.divider()
    st.markdown("### 🎯 Analysis Results")
    
    if prediction == 1:
        st.markdown(f"""
            <div class="risk-high">
                <h2 style='margin:0;'>🚨 HIGH RISK OF CHURN</h2>
                <p style='font-size: 1.2rem; margin-top: 10px;'>There is a <b>{churn_prob:.1f}%</b> probability this customer will cancel.</p>
                <p><i>Recommendation: Reach out immediately with a retention offer or tech support upgrade.</i></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="risk-low">
                <h2 style='margin:0;'>✅ LOW RISK (SAFE)</h2>
                <p style='font-size: 1.2rem; margin-top: 10px;'>There is only a <b>{churn_prob:.1f}%</b> probability of churn.</p>
                <p><i>Recommendation: Standard monitoring. Excellent candidate for upselling.</i></p>
            </div>
        """, unsafe_allow_html=True)