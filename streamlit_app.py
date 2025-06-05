import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        df = pd.read_csv(tmp_path)
        df.columns = df.columns.str.strip().str.lower()
        required_columns = ['date', 'income', 'expenses', 'marketing_spend', 'inflation_rate']
        if not all(col in df.columns for col in required_columns):
            error = f"CSV must contain columns: {required_columns}"
        else:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Create lag features for Random Forest
            for lag in range(1, 7):  # 6 lag months
                df[f'expenses_lag{lag}'] = df['expenses'].shift(lag)

            df.dropna(inplace=True)

            # Scale features
            features = ['income', 'marketing_spend', 'inflation_rate'] + [f'expenses_lag{i}' for i in range(1, 7)]
            target = 'expenses'
            scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[features + [target]] = scaler.fit_transform(df[features + [target]])

            X = df_scaled[features].values
            y = df_scaled[target].values

            split = int(len(X) * 0.8)
            if split < 1 or len(X) - split < 1:
                error = "Dataset too small for train/test split."
            else:
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                dummy = np.zeros((len(y_test), len(features)+1))
                dummy[:, :-1] = X_test
                dummy[:, -1] = y_test
                y_test_rescaled = scaler.inverse_transform(dummy)[:, -1]

                dummy[:, -1] = y_pred
                y_pred_rescaled = scaler.inverse_transform(dummy)[:, -1]

                mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
                metrics = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}

                # Forecast next 12 months
                future_preds = []
                last_known = df.iloc[-6:].copy()
                last_lags = list(last_known['expenses'].values)

                for _ in range(12):
                    last_row = df.iloc[-1][['income', 'marketing_spend', 'inflation_rate']].values
                    lags_scaled = scaler.transform([last_row.tolist() + last_lags[-6:]])[0]
                    prediction_scaled = model.predict([lags_scaled])[0]

                    dummy = np.zeros(len(features)+1)
                    dummy[:-1] = lags_scaled
                    dummy[-1] = prediction_scaled
                    prediction = scaler.inverse_transform([dummy])[0][-1]

                    future_preds.append(prediction)
                    last_lags.append(prediction)
                    last_lags.pop(0)

                forecast = [{'month': i + 1, 'expenses': float(val)} for i, val in enumerate(future_preds)]

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
