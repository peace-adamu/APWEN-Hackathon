
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and label encoder
model = joblib.load("tuned_random_forest_model.pkl")


st.title("🛠️ Milling Machine Failure Type Detection App")

st.markdown("""
Welcome to the **Milling Machine Failure Type Detection App**! 👋

This app helps you predict the type of failure a milling machine might experience based on key machine condition parameters. It uses a trained machine learning model (Random Forest) built from real-world data.

### 🔍 How to Use This App

You have two prediction options (choose from the sidebar):

- **Single Input**: Manually enter machine readings to get an instant prediction.
- **Batch Prediction via CSV**: Upload a `.csv` file containing multiple readings to get predictions for each.

### 📊 Inputs Required:
- **Torque within the range of 5Nm - 500Nm**
- **Air Temperature 20°C - 40°C**
- **Process Temp 	25°C -	700°C**
- **Rotational Speed 500 RPM - 25,000 RPM**
- **Tool Wear	0 sec -	300 sec**

You can also download your prediction results after uploading a file.

> This app was built using Python, Streamlit, and a trained Random Forest Classifier. It's intended for educational, monitoring, or experimental use with machine condition data.
""")

st.sidebar.title("Prediction Options")
mode = st.sidebar.radio("Choose prediction mode:", ["Single Input", "Batch Prediction via CSV"])
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

if mode == "Single Input":
    st.subheader("🔍 Enter Machine Condition Values")
    
    
    # Input fields with min and max values based on industry data
    torque = st.number_input("Torque (Nm)", min_value=5.0, max_value=500.0, value=50.0)
    air_temp = st.number_input("Air Temperature (°C)", min_value=20.0, max_value=40.0, value=25.0)
    process_temp = st.number_input("Process Temperature (°C)", min_value=25.0, max_value=700.0, value=150.0)
    rotational_speed = st.number_input("Rotational Speed (RPM)", min_value=500.0, max_value=25000.0, value=1500.0)
    tool_wear = st.number_input("Tool Wear (Seconds)", min_value=0.0, max_value=300.0, value=50.0)


    if st.button("Predict Failure Type"):
        input_data = np.array([[torque, air_temp, process_temp, rotational_speed, tool_wear]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        label = label_encoder.inverse_transform(prediction)
        st.success(f"Predicted Failure Type: **{label[0]}**")

elif mode == "Batch Prediction via CSV":
    st.subheader("📁 Upload CSV File")
    st.markdown("Expected columns: `Torque`, `Air Temperature`, `Rotational Speed`, `Tool Wear`")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            input_scaled = scaler.transform(input_df)
            predictions = model.predict(input_scaled)
            labels = label_encoder.inverse_transform(predictions)
            input_df["Predicted Failure Type"] = labels

            st.success("Batch prediction completed!")
            st.dataframe(input_df)

            # Download button
            csv = input_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
