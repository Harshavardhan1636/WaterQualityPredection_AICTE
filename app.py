# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model and model columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Set up the Streamlit app
st.set_page_config(page_title="Water Quality Predictor", layout="centered")
st.title("💧 Water Pollutants Prediction")
st.markdown("Predict pollutant levels for a monitoring station by year and forecast future trends.")

# Input fields
year_input = st.number_input("📅 Select Year", min_value=2000, max_value=2100, value=2025)
station_id = st.text_input("🏢 Enter Station ID", value='20')

pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Session state to track if initial prediction was made
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Main prediction for selected year
if st.button("🔍 Predict"):
    if not station_id:
        st.warning("⚠️ Please enter a valid Station ID.")
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        prediction = model.predict(input_encoded)[0]
        st.subheader(f"📌 Predicted Pollutant Levels for {year_input}")
        for p, val in zip(pollutants, prediction):
            st.write(f"**{p}**: {val:.2f} mg/L")

        # Mark prediction done
        st.session_state.predicted = True

# Show 5-year forecast button only after initial prediction
if st.session_state.predicted:
    if st.button("📅 Predict Next 5 Years"):
        future_results = []
        for future_year in range(year_input, year_input + 5):
            input_df = pd.DataFrame({'year': [future_year], 'id': [station_id]})
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_cols]

            preds = model.predict(input_encoded)[0]
            future_results.append([future_year] + list(np.round(preds, 2)))

        # Show table of predictions
        pred_df = pd.DataFrame(future_results, columns=['Year'] + pollutants)
        st.subheader("📊 Future 5-Year Pollutant Forecast")
        st.dataframe(pred_df.set_index('Year'), use_container_width=True)

        # Bar chart for last year
        st.subheader(f"📈 Year {year_input + 4} Prediction Overview")
        fig, ax = plt.subplots()
        ax.bar(pollutants, future_results[-1][1:], color='skyblue')
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(f"Pollutant Levels for {year_input + 4}")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔬 Feature Importance (for O₂ prediction)")
        try:
            importances = model.estimators_[0].feature_importances_
            fig2, ax2 = plt.subplots()
            ax2.bar(model_cols, importances, color='lightgreen')
            ax2.set_title(f'Feature Importance')
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Feature importance not available: {e}")
