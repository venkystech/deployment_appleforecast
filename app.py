import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Apple Stock Forecasting (LSTM)", layout="centered")
st.title("Apple Stock Price Forecasting (LSTM)")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "meta.pkl")

FORECAST_DAYS = 30


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    if "rolling_std_24" in df.columns:
        df.drop(columns=["rolling_std_24"], inplace=True)

    return df


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return model, scaler, meta


df = load_data()
model, scaler, meta = load_assets()

cols = meta["columns"]
WINDOW = meta["window"]

# ✅ FIX: force correct target column index (avoid wrong scaling)
target_index = cols.index("stock_price")

ts = df["stock_price"]


st.subheader("Historical Trend")
fig1, ax1 = plt.subplots()
ax1.plot(ts.index, ts.values)
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")
st.pyplot(fig1)


if st.button("Forecast Next 30 Days"):
    scaled_data = scaler.transform(df[cols].values)

    current_seq = scaled_data[-WINDOW:].copy()
    future_scaled_preds = []

    for _ in range(FORECAST_DAYS):
        pred = model.predict(
            current_seq.reshape(1, WINDOW, len(cols)),
            verbose=0
        )[0][0]

        future_scaled_preds.append(pred)

        next_row = current_seq[-1].copy()
        next_row[target_index] = pred
        current_seq = np.vstack([current_seq[1:], next_row])

    dummy = np.zeros((FORECAST_DAYS, len(cols)))
    dummy[:, target_index] = future_scaled_preds
    future_prices = scaler.inverse_transform(dummy)[:, target_index]

    future_index = pd.date_range(
        start=df.index[-1],
        periods=FORECAST_DAYS + 1,
        freq="B"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_index,
        "Predicted Stock Price": future_prices
    })

    st.subheader("Forecast Table")
    st.data routinely ?
    st.dataframe(forecast_df, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="AAPL_30day_forecast.csv",
        mime="text/csv"
    )

    st.subheader("Forecast Plot")
    fig2, ax2 = plt.subplots()
    last60 = ts.iloc[-60:]
    ax2.plot(last60.index, last60.values, label="Actual (Last 60 days)")
    ax2.plot(future_index, future_prices, label="Forecast (Next 30 days)")
    ax2.legend()
    st.pyplot(fig2)

st.caption("Educational project. Not financial advice.")
