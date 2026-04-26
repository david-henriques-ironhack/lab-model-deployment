import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

with open("ufo-model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    seconds = float(request.form["seconds"])
    latitude = float(request.form["latitude"])
    longitude = float(request.form["longitude"])

    features = np.array([[seconds, latitude, longitude]])
    prediction = model.predict(features)

    countries = ["au", "ca", "de", "gb", "us"]

    return render_template(
    "index.html",
    prediction_text=f"Predicted country: {countries[prediction[0]]}"
    )

if __name__ == "__main__":
    app.run(debug=True)
