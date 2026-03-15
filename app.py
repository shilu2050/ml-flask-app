from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("saved_model", "linear_model.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    
    if request.method == "POST":
        try:
            # Get input from form
            area = float(request.form.get("area", 0))
            
            # Make prediction
            prediction = model.predict([[area]])
            
            # Format prediction
            prediction_text = f"Predicted Price: {prediction[0]:,.2f}"
        except ValueError:
            prediction_text = "Invalid input. Please enter a number."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    # Set host=0.0.0.0 for Render deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)