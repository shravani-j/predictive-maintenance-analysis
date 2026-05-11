from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction_text=None, probability=None, recommendation=None)


@app.route("/predict", methods=["POST"])
def predict():

    metric1 = float(request.form["metric1"])
    metric2 = float(request.form["metric2"])
    metric3 = float(request.form["metric3"])
    metric4 = float(request.form["metric4"])
    metric5 = float(request.form["metric5"])
    metric6 = float(request.form["metric6"])
    metric7 = float(request.form["metric7"])
    metric8 = float(request.form["metric8"])
    metric9 = float(request.form["metric9"])

    features = np.array([[
        metric1,
        metric2,
        metric3,
        metric4,
        metric5,
        metric6,
        metric7,
        metric8,
        metric9
    ]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)[0][1]

    risk_percentage = round(probability * 100, 2)

    if prediction[0] == 1:
       result = "High Failure Risk"
       recommendation = "Immediate maintenance inspection recommended."
    else:
       result = "Low Failure Risk"
       recommendation = "Machine operating within normal conditions."

    return render_template(
    "index.html",
    prediction_text=result,
    probability=risk_percentage,
    recommendation=recommendation
)


if __name__ == "__main__":
    import os

    if __name__ == "__main__":
       port = int(os.environ.get("PORT", 5000))
       app.run(host="0.0.0.0", port=port)