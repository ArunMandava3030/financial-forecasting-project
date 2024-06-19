# ratings.py

import joblib

def generate_rating(model, data):
    # Define the features used in the model
    features = ['SMA_20', 'EMA_20', 'RSI', 'Volume']  # Adjust this list based on the model's requirements
    
    # Ensure data has the necessary columns
    if not all(feature in data.columns for feature in features):
        raise ValueError("Data does not contain all required features for prediction.")
    
    latest_data = data[features].iloc[-1]
    
    # Debugging: Print the latest data to ensure it has the correct features
    print("Latest data for rating generation:", latest_data)

    prediction = model.predict([latest_data])[0]
    
    if prediction > data['Close'].iloc[-1]:
        return "Buy"
    else:
        return "Hold"

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_next_session(model, data):
    # Define the features used in the model
    features = ['SMA_20', 'EMA_20', 'RSI', 'Volume']  # Adjust this list based on the model's requirements
    
    # Ensure data has the necessary columns
    if not all(feature in data.columns for feature in features):
        raise ValueError("Data does not contain all required features for prediction.")
    
    latest_data = data[features].iloc[-1]
    
    # Debugging: Print the latest data to ensure it has the correct features
    print("Latest data for next session prediction:", latest_data)

    latest_data = latest_data.values.reshape(1, -1)
    
    # Predict using the model
    prediction = model.predict(latest_data)

    # Assuming the model returns a single prediction value (scalar)
    return {
        'Open': prediction,
        'High': prediction,
        'Low': prediction,
        'Close': prediction
    }
