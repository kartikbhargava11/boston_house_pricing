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

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json["data"]
    s = np.array(list(data.values())).reshape(1, -1)
    print(s)
    new_data = scaler.transform(s)
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)


