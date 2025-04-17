import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, "decision_tree_model.pkl")

if __name__ == "__main__":
    train_model()
