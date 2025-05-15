# === app.py ===
import streamlit as st
import pandas as pd
import joblib
import sys

# Load models and encoders
crop_model = joblib.load('crop_model.pkl')
yield_model = joblib.load('yield_model.pkl')
fert_model = joblib.load('fert_model.pkl')
le_crop = joblib.load('le_crop.pkl')
le_fert = joblib.load('le_fert.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("🌾 Smart Agriculture Assistant")

# Session state for input persistence and chart toggle
if 'input_done' not in st.session_state:
    st.session_state.input_done = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False
if 'user_df' not in st.session_state:
    st.session_state.user_df = pd.DataFrame()

if not st.session_state.input_done:
    st.header("🔢 Enter Soil and Weather Parameters")
    user_input = {}
    for col in feature_names:
        user_input[col] = st.number_input(f"Enter value for {col}", value=0.0, key=col)
    user_df = pd.DataFrame([user_input])

    if st.button("🚀 Predict Crop, Yield & Fertilizer"):
        st.session_state.user_df = user_df
        st.session_state.input_done = True
        st.experimental_rerun()
else:
    user_df = st.session_state.user_df
    pred_crop = le_crop.inverse_transform(crop_model.predict(user_df))[0]
    pred_yield = yield_model.predict(user_df)[0]
    pred_fert = le_fert.inverse_transform(fert_model.predict(user_df))[0]

    st.success(f"🧾 **Recommended Crop**: {pred_crop}")
    st.info(f"📈 **Predicted Yield**: {pred_yield:.2f} quintals/hectare")
    st.warning(f"💡 **Recommended Fertilizer**: {pred_fert}")

    if st.button("🔁 Enter New Values"):
        st.session_state.input_done = False
        st.experimental_rerun()

