import numpy as np
import pandas as pd
import tensorflow.contrib.learn.python.learn as learn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FIG_FOLDER = 'figures/'


def display_features_scores(features, scores):
    col = {}
    for colname, score in zip(list(features.columns), scores):
        col[colname] = score

    fig = plt.figure()
    plot = fig.add_subplot(111)

    words = pd.Series(col).sort_values()
    words.plot(kind="barh", ax=plot)
    plot.set_xlabel("score")
    plot.set_ylabel("words")
    labels = plot.get_xticklabels()
    map(lambda l: l.set_fontsize(8), labels)
    plt.subplots_adjust(left=0.2)
    plt.show()

    fig.savefig(FIG_FOLDER + 'feature_selection', ftype='pdf')


def display_frequency(features: pd.DataFrame):
    f, ax = plt.subplots()
    features.sum().sort_values().plot(kind="barh", ax=ax)
    plt.show()
    f.savefig(FIG_FOLDER + 'frequency')


# def NN(X, Y):
#     feature_columns = learn.infer_real_valued_columns_from_input(X)
#
#     classifier = learn.DNNRegressor(hidden_units=[100], feature_columns=feature_columns)
#     classifier.fit(X, Y, steps=2000, batch_size=32)
#
#     predictions = classifier.predict(test)
#     score = metrics.accuracy_score(solutions, predictions)
#
#     print("Accuracy: %f" % score)


def sklearn_NN(features, target):
    svc = SVC(kernel='linear')

    # param_grid = [{
    #     'reduce_dim': [SelectKBest()],
    #     'reduce_dim__k': [10, 15],
    #     'classifier__hidden_layer_sizes': [(50,), (100,)],
    #     'classifier__activation': ['tanh', 'relu'],
    #     'classifier__solver': ['adam'],
    #     'classifier': [MLPClassifier()]
    # },
    # {
    #     'reduce_dim': [RFE(estimator=svc)],
    #     'reduce_dim__n_features_to_select': [10, 15],
    #     'classifier__hidden_layer_sizes': [(50,), (100,)],
    #     'classifier__activation': ['tanh', 'relu'],
    #     'classifier__solver': ['adam'],
    #     'classifier': [MLPClassifier()]
    # },
    # {
    #     'reduce_dim': [SelectKBest()],
    #     'reduce_dim__k': [10, 15],
    #     'classifier': [LinearRegression()]
    # }
    # ]

    # param_grid = [
    #     {
    #         'reduce_dim__k': [10, 15, 20],
            # 'classifier__activation': ['tanh'],
            # 'classifier__hidden_layer_sizes': [(100,)],
            # 'classifier__solver': ['adam'],
    #         'classifier__n_neighbors': [2,3,4,5,6],
    #         'classifier__weights': ['uniform', 'distance'],
    #     }
    # ]

    # pipeline = Pipeline([
    #     ('reduce_dim', SelectKBest()),
    #     ('classifier', KNeighborsClassifier())
    # ])

    # grid_cv = sklearn.model_selection.GridSearchCV(estimator=pipeline, param_grid=param_grid,  refit=True, cv=5)
    # grid_cv.fit(features, target)
    # print(grid_cv.score(features))

    # pd.DataFrame(grid_cv.cv_results_).to_csv("results/cv_nn2.csv")

    # scaler = StandardScaler()
    # scaler.fit(features)
    # features = scaler.transform(features)
    features = SelectKBest(score_func=chi2, k=13).fit_transform(features, target)

    pipeline = Pipeline([
            ('classify', MLPClassifier(alpha=0.1, learning_rate='adaptive', learning_rate_init='0.01', solver='sgd'))
        ])

    param_grid = [
        {
            'classify__alpha': 10.0 ** -np.arange(1, 7),
            'classify__solver': ["adam", "sgd"],
            'classify__learning_rate_init': 10.0 ** -np.arange(1, 6),
            'classify__learning_rate': ["adaptive"]
        }
    ]

    grid = GridSearchCV(pipeline, cv=5, n_jobs=4, param_grid=param_grid, verbose=2)
    grid.fit(features, target)
    print(grid.best_estimator_)
    print(grid.best_score_)

    scores = pd.DataFrame(grid.cv_results_)
    scores.to_csv("results/cv_search.csv")


def knn_cv(features, target):
    pass




def vectors_squared_distance(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum += (vec1 - vec2)**2

    return sum

def main():
    data = pd.read_csv("files/classification_dataset_training.csv")

    target_column = 'rating'

    features = data.ix[:, 1:51]
    target = data[target_column].map(lambda r: r > 0.5)
    # k = 10
    # s = SelectKBest(chi2, k=k).fit(features, target)
    # selected_features = s.transform(features)

    # display_features_scores(features, s.scores_)
    # display_frequency(features)
    sklearn_NN(features, target)


if __name__ == '__main__':
    main()
