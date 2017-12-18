import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import datasets

def main():
    clf = LogisticRegression()
    p = Pipeline([('clf', clf)])
    p.fit(X, y)
    filename_p = 'IrisClassifier.sav'
    joblib.dump(p, filename_p)

if __name__ == "__main__":
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    main()
