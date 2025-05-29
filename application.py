import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os

# Load models with error handling
try:
    ridge_model = pickle.load(open("models/ridge_model.pkl", "rb"))
    print("✅ Ridge model loaded successfully")
except Exception as e:
    print(f"❌ Error loading ridge model: {e}")
    ridge_model = None

try:
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    print("✅ Scaler loaded successfully")
    if not hasattr(scaler, 'mean_'):
        print("❌ Scaler is not fitted!")
        scaler = None
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

try:
    feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
    print(f"✅ Feature names loaded: {feature_names}")
except Exception as e:
    print(f"❌ Error loading feature names: {e}")
    feature_names = None

try:
    correlated_features = pickle.load(open("models/correlated_features.pkl", "rb"))
    print(f"✅ Correlated features loaded: {correlated_features}")
except Exception as e:
    print(f"❌ Error loading correlated features: {e}")
    correlated_features = []

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Check if models are loaded
            if ridge_model is None or scaler is None or feature_names is None:
                return render_template("home.html", 
                                     result="Error: Models not loaded properly. Please retrain and save the models.")
            
            # Get form data
            temperature = float(request.form['temperature'])
            rh = float(request.form['RH'])
            ws = float(request.form['WS'])
            rain = float(request.form['Rain'])
            ffmc = float(request.form['FFMC'])
            dmc = float(request.form['DMC'])
            isi = float(request.form['ISI'])
            classes = float(request.form['Classes'])
            region = float(request.form['Region'])
            
            # Create DataFrame with all original features
            # Note: The order should match your training data columns
            input_data = pd.DataFrame({
                'Temperature': [temperature],
                'RH': [rh],
                'Ws': [ws], 
                'Rain': [rain],
                'FFMC': [ffmc],
                'DMC': [dmc],
                'ISI': [isi],
                'Classes': [classes],
                'Region': [region]
            })
            
            print(f"Input data columns: {list(input_data.columns)}")
            print(f"Input data shape: {input_data.shape}")
            
            # Drop correlated features (same as during training)
            input_data_cleaned = input_data.drop(columns=correlated_features, errors='ignore')
            
            print(f"After dropping correlated features: {input_data_cleaned.shape}")
            print(f"Expected features: {feature_names}")
            print(f"Actual features: {list(input_data_cleaned.columns)}")
            
            # Ensure features are in the same order as training
            input_data_cleaned = input_data_cleaned[feature_names]
            
            # Scale the data
            input_scaled = scaler.transform(input_data_cleaned)
            
            # Make prediction
            prediction = ridge_model.predict(input_scaled)
            
            print(f"Prediction successful: {prediction[0]}")
            
            return render_template("home.html", result=prediction[0])
            
        except KeyError as e:
            error_msg = f"Missing feature in input: {str(e)}"
            print(error_msg)
            return render_template("home.html", result=f"Error: {error_msg}")
            
        except ValueError as e:
            error_msg = "Invalid input values. Please check your data."
            print(f"ValueError: {e}")
            return render_template("home.html", result=f"Error: {error_msg}")
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            return render_template("home.html", result=f"Error: {error_msg}")
    else:
        return render_template("home.html")

if __name__ == "__main__":
    # Print model status on startup
    print("\n=== Model Status ===")
    print(f"Ridge model loaded: {ridge_model is not None}")
    print(f"Scaler loaded: {scaler is not None}")
    print(f"Feature names loaded: {feature_names is not None}")
    print(f"Correlated features: {correlated_features}")
    
    if all([ridge_model, scaler, feature_names]):
        print("✅ All models loaded successfully! Ready to make predictions.")
    else:
        print("❌ Some models failed to load. Please retrain and save your models.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)