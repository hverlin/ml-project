import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def dist(a,b) -> float:
    return np.linalg.norm(a-b)


class KNNClassifier:
    """
    K-Nearest Neighbors classifier
    """
    def __init__(self, k:int=5):
        self.k = k
        self.model_features = None
        self.model_target = None

    def fit(self, X: np.matrix, y: np.matrix):
        self.model_features = X
        self.model_target = list(y)

    def predict(self, features: np.matrix) -> np.matrix:
        res = []

        for i in range(len(features)):
            dists = sorted([(dist(features[i], self.model_features[j]), j) for j in range(len(self.model_features))])
            classes = {}
            for (d, j) in dists[:self.k]:
                clazz = self.model_target[j]
                classes[clazz] = 1 + (classes[clazz] if clazz in classes else 0)

            res.append([max(classes.items(), key=lambda p: p[1])[0]])

            if i % 100 == 0:
                print(i)
        return np.matrix(res)

    def get_params(self, deep=True):
        return {'k': self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

def main():
    data = pd.read_csv("files/classification_dataset_training.csv")

    target_column = 'rating'

    features = data.ix[:, 1:51]
    target = data[target_column].map(lambda r: r > 0.5)

    features = SelectKBest(score_func=chi2, k=14).fit_transform(features, target)

    pipeline = Pipeline([
            ('classify', KNNClassifier())
        ])

    param_grid = [
        {
            'classify__k': [5]
        }
    ]

    grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid, verbose=2, scoring='f1')
    grid.fit(features, target)

    print(grid.best_estimator_)
    print(grid.best_score_)

    scores = pd.DataFrame(grid.cv_results_)
    scores.to_csv("results/cv_knn.csv")



if __name__ == '__main__':
    main()
