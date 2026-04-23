import numpy as np
from flask import Flask, request, render_template
import pickle
from pathlib import Path

app = Flask(__name__)

# Load the model using a path relative to this file so the app can be
# started from any working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "ufo-model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH!s}")

model = pickle.load(open(MODEL_PATH, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)