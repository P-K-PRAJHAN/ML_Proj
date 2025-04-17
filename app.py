from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("decision_tree_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([features])[0]
        species = ["Setosa", "Versicolor", "Virginica"][prediction]
        return render_template("index.html", prediction=species)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
