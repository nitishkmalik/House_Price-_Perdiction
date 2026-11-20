"""Beginner-friendly Streamlit interface for the saved house-price model."""
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_model.pkl"

st.set_page_config(page_title="House Price Predictor", page_icon="⌂")
st.title("House Price Predictor")
st.write("Enter a few home details to get an example price estimate.")

if not MODEL_FILE.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

with MODEL_FILE.open("rb") as file:
    saved = pickle.load(file)
model = saved["model"]
feature_columns = saved["feature_columns"]

locations = sorted(column.replace("location_", "")
                   for column in feature_columns if column.startswith("location_"))

with st.form("prediction_form"):
    area = st.number_input("Area (square feet)", min_value=300, max_value=10000,
                           value=1500, step=50)
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10,
                               value=3, step=1)
    location = st.selectbox("Location", locations)
    submitted = st.form_submit_button("Predict price")

if submitted:
    # Create a row with the exact feature columns used during model training.
    input_row = {column: 0 for column in feature_columns}
    input_row["area_sqft"] = area
    input_row["bedrooms"] = bedrooms
    location_column = f"location_{location}"
    if location_column in input_row:
        input_row[location_column] = 1

    input_data = pd.DataFrame([input_row], columns=feature_columns)
    estimated_price = model.predict(input_data)[0]
    st.success(f"Estimated price: ${estimated_price:,.0f}")
    st.caption(f"Using the selected {saved['model_name']} model. This is an educational estimate, not a property appraisal.")
