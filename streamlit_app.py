
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Load trained model
with open("sales_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app configuration
st.set_page_config(page_title="Sales Predictor", layout="wide")

# --- Header / Branding ---
st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
        <h1 style='color: #4CAF50;'>📈 Sales Prediction Dashboard</h1>
        <h4 style='color: gray;'>AI-powered tool to estimate sales from advertising budgets</h4>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar inputs ---
st.sidebar.header("📊 Enter Advertising Budgets")
tv = st.sidebar.number_input("TV Budget ($)", min_value=0.0, step=1.0)
radio = st.sidebar.number_input("Radio Budget ($)", min_value=0.0, step=1.0)
newspaper = st.sidebar.number_input("Newspaper Budget ($)", min_value=0.0, step=1.0)

# --- Live Prediction ---
st.subheader("🧠 Prediction Result")
if st.button("Predict Sales"):
    input_df = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Estimated Sales: **{prediction:.2f} units**")

    # Log to CSV
    input_df["Predicted_Sales"] = prediction
    input_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_df.to_csv("sales_predictions_log.csv", mode="a", header=not pd.read_csv("sales_predictions_log.csv", nrows=1).empty if os.path.exists("sales_predictions_log.csv") else True, index=False)

# --- Batch Prediction via CSV Upload ---
st.markdown("### 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with columns: TV, Radio, Newspaper", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_df['Predicted_Sales'] = model.predict(batch_df)
    
    st.success("✅ Predictions completed:")
    st.dataframe(batch_df)

    # Download button
    st.download_button(
        label="📥 Download Predictions",
        data=batch_df.to_csv(index=False).encode('utf-8'),
        file_name='batch_sales_predictions.csv',
        mime='text/csv'
    )

# --- Visualization Section ---
if uploaded_file is not None:
    st.markdown("### 📉 Visualize Budget vs Predicted Sales")
    fig, ax = plt.subplots()
    ax.scatter(batch_df['TV'], batch_df['Predicted_Sales'], label='TV', color='blue')
    ax.scatter(batch_df['Radio'], batch_df['Predicted_Sales'], label='Radio', color='green')
    ax.scatter(batch_df['Newspaper'], batch_df['Predicted_Sales'], label='Newspaper', color='orange')
    ax.set_xlabel("Ad Budget ($)")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Ad Budget vs Predicted Sales")
    ax.legend()
    st.pyplot(fig)

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; color: gray;'>
        Developed by Aditya Vishwakarma | © 2025
    </div>
""", unsafe_allow_html=True)

