from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# UI prediction
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    air = float(request.form["air_temp"])
    process = float(request.form["process_temp"])
    speed = float(request.form["speed"])
    torque = float(request.form["torque"])
    wear = float(request.form["wear"])

    # Feature engineering (same as training)
    temp_diff = process - air
    power = torque * speed
    wear_per_speed = wear / (speed + 1)

    features = np.array([[air, process, speed, torque, wear,
                           0, 0, temp_diff, power, wear_per_speed]])

    features = scaler.transform(features)

    prob = model.predict_proba(features)[0][1]
    threshold = 0.15
    pred = 1 if prob >= threshold else 0
    result = "Machine Failure" if pred == 1 else "No Failure"

    return render_template(
        "index.html",
        result=result,
        prob=round(prob, 3)
    )

# API endpoint (keep for technical use)
@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"]).reshape(1, -1)
    data = scaler.transform(data)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    return jsonify({
        "Machine Failure": int(pred),
        "Failure Probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)
