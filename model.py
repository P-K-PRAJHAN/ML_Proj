import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import joblib

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    joblib.dump(clf, "decision_tree_model.pkl")

    # Plot and save the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.savefig("static/tree.png")
    plt.close()

if __name__ == "__main__":
    train_model()
