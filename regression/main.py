import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.contrib.learn.python.learn as learn
from sklearn import feature_selection
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR

from regression.linear_regressor import LinearRegressor as LR

pp = pprint.PrettyPrinter(indent=2)

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

    fig.savefig(FIG_FOLDER + 'feature_selection.pdf', ftype='pdf')


def constraints(val):
    return 0 if val < 0 else val


def mlp_regressor_test(features, solutions, target, testing_data):
    mlp = MLPRegressor()
    mlp.fit(features, target)
    predictions = mlp.predict(testing_data)
    print(metrics.mean_squared_error(solutions, predictions))


def NN(X, Y, test, solutions):
    feature_columns = learn.infer_real_valued_columns_from_input(X)

    classifier = learn.DNNRegressor(hidden_units=[100], feature_columns=feature_columns)
    classifier.fit(X, Y, steps=2000, batch_size=32)

    predictions = list(classifier.predict(test, as_iterable=True))
    score = metrics.mean_squared_error(list(solutions), predictions)

    print("Accuracy: %f" % score)


def ml_pipeline(features, target, testing_data, solutions):
    pipe = Pipeline([
        ('reduce_dim', feature_selection.SelectKBest()),
        ('classify', MLPRegressor())
    ])

    N_FEATURES_OPTIONS = [5, 10, 20]
    svc = SVC(kernel="linear")
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    param_grid1 = [
        {
            'reduce_dim': [SelectKBest(f_regression)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify': [MLPRegressor()]
        },
        {
            'reduce_dim': [feature_selection.RFE(estimator=svc, n_features_to_select=7)],
            'classify': [MLPRegressor()]
        },
        {
            'reduce_dim': [feature_selection.RFE(estimator=svc, n_features_to_select=7)],
            'classify': [LinearRegression()]
        },
        {
            'reduce_dim': [SelectKBest(f_regression)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify': [MLPRegressor()]
        },
        {
            'reduce_dim': [SelectKBest(f_regression)],
            'classify': [SVR(kernel='rbf', C=1e3, gamma=0.1)]
        }
    ]

    param_grid2 = [
        {
            'reduce_dim': [feature_selection.RFE(estimator=svc, n_features_to_select=7)],
            'classify': [BaggingRegressor()]
        }
    ]

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    param_grid = [
        {
            'reduce_dim': [SelectKBest(f_regression)],
            'reduce_dim__k': [10],
            'classify': [MLPRegressor()],
            'classify__alpha': 10.0 ** -np.arange(1, 7),
            'classify__solver': ["adam", "sgd"],
            'classify__learning_rate_init': 10.0 ** -np.arange(1, 6),
            'classify__learning_rate': ["adaptive"]
        }
    ]

    grid = GridSearchCV(pipe, cv=5, n_jobs=2, param_grid=param_grid, verbose=2)
    grid.fit(features, target)
    print(grid.best_estimator_)
    print(grid.best_score_)

    scores = pd.DataFrame(grid.cv_results_)
    scores.to_csv("results/cv_search.csv")

    testing_data = scaler.transform(testing_data)
    predictions = grid.predict(testing_data)

    predictions = list(map(constraints, predictions))

    score = metrics.mean_squared_error(solutions, predictions)
    print("Accuracy: %f" % score)


def display_pairplot(features: pd.DataFrame):
    f, ax = plt.subplots()
    sns.pairplot(data=features)


def display_frequency(features: pd.DataFrame):
    f, ax = plt.subplots()
    features.sum().sort_values().plot(kind="barh", ax=ax)
    f.savefig(FIG_FOLDER + 'frequency')


def linear_regressor_test(features, target, testing_data, solutions):
    svc = SVC(kernel="linear")
    dim = feature_selection.RFE(estimator=svc, n_features_to_select=7)
    feat = dim.fit_transform(features, target)

    print(dim.n_features_to_select)

    lr = LR()
    lr.fit(np.matrix(feat), np.matrix(target))

    testing_data = dim.transform(testing_data)
    predictions = lr.predict(np.matrix(testing_data))

    predictions = [p[0] for p in predictions.tolist()]

    predictions = list(map(constraints, predictions))

    score = metrics.mean_squared_error(list(solutions), predictions)
    print("Accuracy: %f" % score)


def plot_features_weights():

    features = {
        'service': 4.22671911,
        'price': 2.97652506,
        'restaurant': 2.71838277,
        'menu': 3.85572435,
        'taste': 0.90751989,
        'food': 2.06325537,
        'staff': 1.0674791
    }

    fig = plt.figure()
    plot = fig.add_subplot(111)

    pairs = pd.Series(features).sort_values()
    pairs.plot(kind="barh", ax=plot)

    plot.set_xlabel("weights")
    plot.set_ylabel("words")

    plt.subplots_adjust(left=0.2)
    plt.show()

    fig.savefig(FIG_FOLDER + 'features_weights.pdf')


def generate_submission_regression(features, target, testing_data, solutions):
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    dim = SelectKBest(f_regression, k=10)
    features = dim.fit_transform(features, target)

    mlp = MLPRegressor(
        activation='relu', alpha=0.10000000000000001, batch_size='auto',
        beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
        hidden_layer_sizes=(100,), learning_rate='adaptive',
        learning_rate_init=0.10000000000000001, max_iter=200, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=None,
        shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,
        verbose=False, warm_start=False
    )

    mlp.fit(features, target)

    testing_data = scaler.transform(testing_data)
    testing_data = dim.transform(testing_data)

    predictions = mlp.predict(testing_data)
    predictions = list(map(constraints, predictions))
    score = metrics.mean_squared_error(list(solutions), predictions)
    print("Accuracy: %f" % score)


    # pd.DataFrame(predictions).to_csv('kaggle/results.csv')


def main():
    data = pd.read_csv("files/regression_dataset_training.csv")
    testing_data = pd.read_csv("files/regression_dataset_testing.csv").ix[:, 1:51]
    solutions = list(pd.read_csv("files/regression_dataset_testing_solution.csv").ix[:, 1])

    #  select features and target
    features = data.iloc[:, 1:51]
    target = data["vote"]

    s = SelectKBest(f_regression, k=10)
    s.fit(features, target)
    #  display_features_scores(features, s.scores_)
    # display_frequency(features)

    #  display_features_scores(features, s.scores_)
    #  NN(features, target, testing_data, solutions)
    #  mlp_regressor_test(features, solutions, target, testing_data)
    #  ml_pipeline(features, target, testing_data, solutions)
    # linear_regressor_test(features, target, testing_data, solutions)
    generate_submission_regression(features, target, testing_data, solutions)
    # plot_features_weights()

if __name__ == '__main__':
    main()
