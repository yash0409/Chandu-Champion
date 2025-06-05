import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import io
import base64
from openai import OpenAI

st.set_page_config(page_title="Personal Finance Tracker - Expense Forecast", layout="wide")
st.title("Personal Finance Tracker")
st.write("Upload your financial data to forecast expenses")

# File uploader
uploaded_file = st.file_uploader("Upload Financial Data (CSV)", type=["csv"])

metrics = None
forecast = None
error = None

if uploaded_file is not None:
    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Step 1: Load and preprocess the dataset
        df = pd.read_csv(tmp_path)
        df.columns = df.columns.str.strip().str.lower()
        required_columns = ['date', 'income', 'expenses', 'marketing_spend', 'inflation_rate']
        if not all(col in df.columns for col in required_columns):
            error = f"CSV must contain columns: {required_columns}"
        else:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Step 2: Feature scaling
            features = ['income', 'expenses', 'marketing_spend', 'inflation_rate']
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])
            scaled_df = pd.DataFrame(scaled, columns=features)
            scaled_df['date'] = df['date'].values

            # Step 3: Create sequences
            def create_sequences(data, seq_len):
                X, y = [], []
                for i in range(len(data) - seq_len):
                    X.append(data[i:i+seq_len, [0, 2, 3]])  # Use income, marketing_spend, inflation_rate as features
                    y.append(data[i+seq_len, 1])             # Predict expenses
                return np.array(X), np.array(y)

            seq_len = 6
            if len(scaled) <= seq_len:
                error = f"Not enough data. Need at least {seq_len + 1} rows."
            else:
                X, y = create_sequences(scaled, seq_len)
                split = int(len(X) * 0.8)
                if split < 1 or len(X) - split < 1:
                    error = "Dataset too small for train/test split."
                else:
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    # Step 5: Build and train LSTM model
                    model = Sequential([
                        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                        Dense(1),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

                    # Step 6: Evaluate
                    y_pred = model.predict(X_test, verbose=0)
                    dummy_features = np.zeros((len(y_test), len(features)))
                    dummy_features[:, 1] = y_test
                    dummy_features[:, [0, 2, 3]] = X_test[:, -1, :]
                    y_test_rescaled = scaler.inverse_transform(dummy_features)[:, 1]
                    dummy_features[:, 1] = y_pred.flatten()
                    y_pred_rescaled = scaler.inverse_transform(dummy_features)[:, 1]
                    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
                    metrics = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}

                    # Step 7: Forecast next 12 months
                    last_seq = scaled[-seq_len:, [0, 2, 3]]
                    future_preds = []
                    current_input = last_seq.copy()
                    for _ in range(12):
                        pred = model.predict(current_input[np.newaxis, :, :], verbose=0)
                        future_preds.append(pred[0, 0])
                        next_row = current_input[-1]
                        current_input = np.vstack([current_input[1:], next_row])
                    dummy_features = np.zeros((12, len(features)))
                    dummy_features[:, 1] = np.array(future_preds)
                    dummy_features[:, [0, 2, 3]] = scaled[-1, [0, 2, 3]]
                    final_preds_rescaled = scaler.inverse_transform(dummy_features)[:, 1]
                    forecast = [{'month': i + 1, 'expenses': float(val)} for i, val in enumerate(final_preds_rescaled)]
        os.remove(tmp_path)
    except Exception as e:
        error = f"Server error: {str(e)}"
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

if error:
    st.error(error)

if metrics:
    st.subheader("Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.2f}", "Mean Absolute Error")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}", "Root Mean Squared Error")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%", "Mean Absolute Percentage Error")
    st.markdown("<hr>", unsafe_allow_html=True)

def generate_advice(forecast_df):
    # Save the forecast plot as a temporary PNG file
    fig, ax = plt.subplots()
    ax.plot(forecast_df['Month'], forecast_df['Forecasted Expenses (₹)'], marker='o')
    ax.set_xlabel('Month')
    ax.set_ylabel('Expenses (₹)')
    ax.set_title('Forecasted Expenses for Next 12 Months')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Call OpenRouter (Llama 4 Maverick) multimodal LLM
    # (Replace <OPENROUTER_API_KEY> and <YOUR_SITE_URL> with your actual values)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="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")
    completion = client.chat.completions.create(
        extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "Personal Finance Tracker"},
        model="meta-llama/llama-4-maverick:free",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Based on the forecast graph (showing forecasted expenses for the next 12 months), please generate a short advice or insight." },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{img_b64}" } }
                ]
            }
        ]
    )
    advice = completion.choices[0].message.content
    return advice

if forecast:
    st.subheader("12-Month Expenses Forecast")
    forecast_df = pd.DataFrame(forecast)
    forecast_df['Month'] = forecast_df['month']
    forecast_df['Forecasted Expenses (₹)'] = forecast_df['expenses']
    col1, col2 = st.columns([2, 1])
    with col1:
         st.dataframe(forecast_df[['Month', 'Forecasted Expenses (₹)']], use_container_width=True)
         st.subheader("Expenses Forecast Visualization")
         fig, ax = plt.subplots(figsize=(6, 4))
         ax.plot(forecast_df['Month'], forecast_df['Forecasted Expenses (₹)'], marker='o')
         ax.set_xlabel('Month')
         ax.set_ylabel('Expenses (₹)')
         ax.set_title('Forecasted Expenses for Next 12 Months')
         st.pyplot(fig)
    with col2:
         st.subheader("AI Advice (Generated by Llama 4 Maverick)")
         advice = generate_advice(forecast_df)
         st.write(advice) 