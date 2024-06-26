import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def predict_stock_price(ticker_symbol, start_date, end_date):
    """
    Predicts stock price using a linear regression model and displays results.
    """
    try:
        # Download data
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the provided stock symbol and date range.")
            return

        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Feature engineering
        data['50d_MA'] = data['Close'].rolling(window=50).mean()
        data['200d_MA'] = data['Close'].rolling(window=200).mean()
        data['20d_MA'] = data['Close'].rolling(window=20).mean()
        data['10d_MA'] = data['Close'].rolling(window=10).mean()
        data['std_dev'] = data['Close'].rolling(window=20).std()

        # Define features and target
        features = ['Open', 'High', 'Low', 'Volume', '50d_MA', '200d_MA', '20d_MA', '10d_MA', 'std_dev']
        target = 'Close'

        # Split data
        X = data[features].dropna()
        y = data.loc[X.index, target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        predicted_next_day_close = predictions[-1]
        st.write(f"Predicted Next Day's Closing Price for {ticker_symbol}: {predicted_next_day_close:.2f}")

        # Model evaluation
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        st.write(f"Model Evaluation Metrics:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}")

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Actual Close Price')
        ax.plot(X_test.index, predictions, label='Predicted Close Price', linestyle='--')
        ax.plot(data.index, data['50d_MA'], label='50-day Moving Average')
        ax.plot(data.index, data['200d_MA'], label='200-day Moving Average')
        ax.plot(data.index, data['20d_MA'], label='20-day Moving Average')
        ax.plot(data.index, data['10d_MA'], label='10-day Moving Average')
        ax.set_title('Stock Price Prediction and Moving Averages')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit App
st.title("Stock Price Prediction")
st.write("""
    Enter the stock symbol and date range to predict the stock's closing price.
    The model uses linear regression based on historical data and moving averages.
""")

ticker_symbol = st.text_input("Enter Stock Symbol (e.g., TATASTEEL.NS)")
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365 * 2))
end_date = st.date_input("End Date", datetime.today())

if st.button("Predict"):
    predict_stock_price(ticker_symbol, start_date, end_date)
