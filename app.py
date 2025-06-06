import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt

# App title
st.set_page_config(page_title="Energy Forecast", layout="centered")
st.title("🔋 Energy Consumption Forecast")


theme = st.selectbox("🌗 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #111111; color: white; }
        .stButton>button { background-color: #444444; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Number of periods
n_periods = st.slider("📅 Days to forecast:", 1, 30, 7)

st.subheader("📝 Enter Inputs for Each Day")
user_inputs = []

# Create manual input fields
for i in range(n_periods):
    with st.expander(f"➤ Day {i + 1} Input", expanded=False):
        voltage = st.number_input("🔌 Voltage (V)", key=f"voltage_{i}", value=240.0)
        reactive = st.number_input("⚡ Global Reactive Power (kW)", key=f"reactive_{i}", value=0.1)
        kitchen = st.number_input("🍽️ Kitchen (Watt-min)", key=f"kitchen_{i}", value=500.0)
        laundry = st.number_input("🧺 Laundry Room (Watt-min)", key=f"laundry_{i}", value=5000.0)
        heater = st.number_input("🔥 Water Heater & AC (Watt-min)", key=f"heater_{i}", value=12000.0)
        user_inputs.append([voltage, reactive, kitchen, laundry, heater])

# Convert to DataFrame
exog_df = pd.DataFrame(user_inputs, columns=[
    "Voltage", "Global_reactive_power", "Kitchen", "Laundry_Room", "Water_Heater_AC"
])

# Forecast
if st.button("🔮 Forecast Energy Usage"):
    with st.spinner("Running forecast..."):
        try:
            # Get forecast with confidence intervals
            forecast_result = model.predict(n_periods=n_periods, X=exog_df, return_conf_int=True)
            forecast_mean, conf_int = forecast_result

            # Build index of future dates (update if your last date is different)
            last_train_date = pd.to_datetime("2010-10-27")
            future_dates = pd.date_range(last_train_date + timedelta(days=1), periods=n_periods)

            forecast_series = pd.Series(forecast_mean, index=future_dates)
            lower_bounds = pd.Series(conf_int[:, 0], index=future_dates)
            upper_bounds = pd.Series(conf_int[:, 1], index=future_dates)

            # Plot
            st.success("✅ Forecast completed!")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_series, label="Forecast", color="dodgerblue")
            ax.fill_between(forecast_series.index, lower_bounds, upper_bounds, color='lightblue', alpha=0.4, label="Confidence Interval")
            ax.set_title("📈 Forecast with Confidence Intervals")
            ax.set_ylabel("Energy (kWh)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Display forecast data
            forecast_df = pd.DataFrame({
                "Forecast (kWh)": forecast_series,
                "Lower Bound": lower_bounds,
                "Upper Bound": upper_bounds
            })
            st.dataframe(forecast_df)

            # Download
            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast_df.to_csv().encode(),
                file_name="energy_forecast.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Forecast failed: {e}")
