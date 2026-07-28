import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_data(ticker="GC=F", period="5y"):
    """Fetches historical data for the given commodity ticker."""
    print(f"Fetching data for {ticker} over the last {period}...")
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def preprocess_data(data):
    """Engineers features and prepares the target variable (daily returns)."""
    
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
   
    data['Next_Day_Change'] = data['Close'].shift(-1) - data['Close']
    
    
    data = data.dropna()
    
    return data

def train_and_evaluate():
    # 1. Load and Preprocess
    df = load_data("GC=F", "5y") 
    df_clean = preprocess_data(df)
    
    # 2. Define Features (X) and Target (y)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50']
    X = df_clean[features]
    y = df_clean['Next_Day_Change']
    
    # Keep today's closing prices to reconstruct tomorrow's absolute prices
    close_prices = df_clean['Close'].squeeze()
    
    # 3. Train-Test Split (80% training, 20% testing)
 
    X_train, X_test, y_train, y_test, _, test_close_prices = train_test_split(
        X, y, close_prices, test_size=0.2, shuffle=False
    )
    
    # 4. Initialize and Train the Model
    print("Training Random Forest Regressor to predict daily price adjustments...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Make Predictions (This yields the predicted *change* for the next day)
    predicted_changes = model.predict(X_test)
    
    # 6. Reconstruct Absolute Prices
    actual_future_prices = test_close_prices.values + y_test.values
    
    # Predicted Absolute Price = Today's Close + Tomorrow's Predicted Change
    predicted_future_prices = test_close_prices.values + predicted_changes
    
    # 7. Evaluate the Model on Absolute Prices
    mae = mean_absolute_error(actual_future_prices, predicted_future_prices)
    rmse = np.sqrt(mean_squared_error(actual_future_prices, predicted_future_prices))
    
    print("\n--- Model Evaluation (Absolute Prices) ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    
    # 8. Visualize the Results
    plt.figure(figsize=(14, 7))
    plt.plot(test_close_prices.index, actual_future_prices, label="Actual Future Price", color="blue")
    plt.plot(test_close_prices.index, predicted_future_prices, label="Predicted Future Price", color="orange", alpha=0.7)
    plt.title("Gold Futures Price Prediction (Fixed Extrapolation) - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()