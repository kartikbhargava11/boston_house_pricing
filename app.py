import pickle
import os
import sklearn
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template

print(sklearn.__version__)

app = Flask(__name__)

scaler = pickle.load(open("scaling.pkl", "rb"))

# Load the model
model = pickle.load(open("regmodel.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    s = np.array(list(data.values())).reshape(1, -1)
    print(s)
    new_data = scaler.transform(s)
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)


