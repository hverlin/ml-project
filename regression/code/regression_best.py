from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd


def constraints(val):
    return 0 if val < 0 else val


def generate_submission_regression(features, target, testing_data):
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
    pd.DataFrame(predictions).to_csv('kaggle/results.csv')
