# 💧 Water Quality Prediction System – AICTE Virtual Internship

This project predicts **multiple water pollutant levels** using machine learning, specifically `MultiOutputRegressor` with a `RandomForestRegressor`. It was developed as part of the **AICTE Virtual Internship by Edunet Foundation**, sponsored by **Shell**, during **June 2025**.

---

## 📌 Overview

Water pollution is a critical global issue. Accurately predicting pollutant concentrations can help in early warning systems, resource planning, and environmental protection.

This project uses historical station-wise water quality data to:
- Predict pollutants like **O₂, NO₃, NO₂, SO₄, PO₄, CL**
- Train a regression model to generalize across time and location
- Provide predictions for current and future years (up to 2100)
- Deploy a real-time, user-friendly **Streamlit web app**

---

## 🚀 Features

- Predict pollutants based on **Station ID** and **Year**
- Automatically forecasts for the **next 5 years**
- Displays predictions in clean table and chart formats
- Shows **feature importance** and pollutant **correlation**
- End-to-end system ready for deployment

---

## 🧪 Technologies Used

- **Python 3.12**
- **Pandas, NumPy** – Data preprocessing
- **Scikit-learn** – Machine learning modeling
- **Matplotlib, Seaborn** – Visualization
- **Streamlit** – Frontend web app
- **Jupyter Notebook** – ML pipeline development
- **Joblib** – Model persistence

---

## 📈 Pollutants Predicted

- **O₂** – Dissolved Oxygen  
- **NO₃** – Nitrate  
- **NO₂** – Nitrite  
- **SO₄** – Sulphate  
- **PO₄** – Phosphate  
- **CL** – Chloride  

---

## 📊 Model Evaluation

The model was evaluated using:
- **R² Score** – for prediction accuracy
- **Mean Squared Error (MSE)** – for error measurement

Overall, the model showed **strong predictive capability** across all pollutants.

---

## 🌐 Live Prediction App

The model is deployed as an interactive web application using **Streamlit**, where users can:
- Enter a year and station ID
- View pollutant levels
- Predict for future 5 years
- Visualize results with charts

---

## 📂 Project Structure

```
📦 WaterQualityPrediction/
├── Water_Quality_Prediction.ipynb     # ML development notebook
├── pollution_model.pkl                # Saved model
├── model_columns.pkl                  # Model input schema
├── WaterQualtiyPrediction_dataset.csv # Dataset
├── app.py                             # Streamlit app code
└── README.md                          # Project documentation
```

---

## 📎 Model Link

[https://drive.google.com/file/d/1cbPK-t4-2fM-dEszux2dtkEiTvha5usQ/view?usp=drive_link](https://drive.google.com/file/d/1cbPK-t4-2fM-dEszux2dtkEiTvha5usQ/view?usp=drive_link)

---

## 🎓 Internship Details

- **Program**: AICTE Virtual Internship  
- **Platform**: Edunet Foundation  
- **Sponsor**: Shell  
- **Duration**: June 2025 (1 month)  
- **Role**: Machine Learning Intern  
- **Focus**: Environmental Data Prediction and ML Deployment

---

## ✅ Summary

This project demonstrates the real-world application of machine learning in environmental analytics and monitoring. With a complete ML pipeline and an interactive UI, it is a deployable, data-driven system designed to support clean water initiatives.

---

> ✨ “Data for good starts with clean predictions.”
