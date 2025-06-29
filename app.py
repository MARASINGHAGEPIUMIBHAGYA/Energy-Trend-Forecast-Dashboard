from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

app = Flask(__name__)

MODEL_PATH = "model/lstm_model.h5"
SCALER_PATH = "model/scaler.pkl"
SAMPLE_VALUES_PATH = "model/sample_values.pkl"
METRICS_PATH = "model/metrics.pkl"
SEQ_LENGTH = 24
FEATURES = ['y', 'hour', 'dayofweek']

# Load model, scaler, sample values, and metrics
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    sample_values = joblib.load(SAMPLE_VALUES_PATH)
    metrics = joblib.load(METRICS_PATH)
except Exception as e:
    print(f"Error loading model, scaler, sample values, or metrics: {e}")
    raise

# Verify scaler feature count
if scaler.n_features_in_ != len(FEATURES):
    print(f"Warning: Scaler expects {scaler.n_features_in_} features, but using {len(FEATURES)} features.")

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    forecast = None
    input_values = None

    if request.method == "POST":
        input_str = request.form.get("input_values", "")
        try:
            # Validate input
            input_list = [float(x.strip()) for x in input_str.split(",") if x.strip()]
            if len(input_list) != SEQ_LENGTH:
                raise ValueError(f"Please enter exactly {SEQ_LENGTH} energy consumption values (one for each of the past 24 hours).")
            if any(x < 0 for x in input_list):
                raise ValueError("Energy consumption values cannot be negative.")
            
            # Generate time-based features
            current_time = datetime.now()
            time_steps = [current_time - timedelta(hours=i) for i in range(SEQ_LENGTH-1, -1, -1)]
            hours = [t.hour for t in time_steps]
            days = [t.weekday() for t in time_steps]

            # Prepare input array with all features
            arr = np.array([[y, h, d] for y, h, d in zip(input_list, hours, days)])
            scaled_arr = scaler.transform(arr)

            # Reshape for model input
            input_seq = scaled_arr.reshape(1, SEQ_LENGTH, len(FEATURES))

            # Predict
            pred_scaled = model.predict(input_seq, verbose=0)
            # Create a dummy array for inverse transform with same shape as scaler expects
            dummy_arr = np.zeros((1, len(FEATURES)))
            dummy_arr[0, 0] = pred_scaled[0, 0]  # Only 'y' is predicted
            pred_unscaled = scaler.inverse_transform(dummy_arr)[0, 0]

            forecast = round(pred_unscaled, 3)
            input_values = ", ".join(map(str, input_list))

        except ValueError as ve:
            error = str(ve)
            input_values = input_str
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            input_values = input_str

    # Prepare sample values for default display
    sample_input = ", ".join([f"{x:.3f}" for x in sample_values[:, 0]]) if sample_values is not None else ""

    return render_template(
        "dashboard.html",
        forecast=forecast,
        input_values=input_values,
        sample_values=sample_input,
        error=error,
        mae=round(metrics.get('mae', 0), 3),
        mse=round(metrics.get('mse', 0), 3),
        rmse=round(metrics.get('rmse', 0), 3),
        accuracy=round(metrics.get('accuracy', 0), 3),
        seq_length=SEQ_LENGTH
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001, host='0.0.0.0')