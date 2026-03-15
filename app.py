
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("saved_model/linear_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        prediction = model.predict([[area]])

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: {prediction[0]:,.2f}"
        )
    except Exception:
        return render_template(
            "index.html",
            prediction_text="Invalid input. Please enter a number."
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
