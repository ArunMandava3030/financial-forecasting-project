import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def prepare_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    data = data.dropna()
    print("Prepared data:", data.head())
    features = ['SMA_20', 'EMA_20', 'RSI', 'Volume']
    X = data[features]
    y = data['Close']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Ensure the data is correctly processed and saved
    X, y = prepare_data('AAPL_data_with_indicators.csv')
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    model = train_model(X, y)
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/random_forest_model.pkl')
