import os
import time

import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from app.utilities import load_env, PROJECT_ROOT


def param_grid_search(X, y, estimator, param_grid, jobs=1, metric='neg_mean_absolute_error'):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        n_jobs=jobs,
        scoring=metric
    )
    grid_search.fit(X, y)

    return grid_search.best_score_, grid_search.best_params_


def cross_validation(X, y, random_state=42, splits=5, jobs=1, metric='neg_mean_absolute_error'):
    kf = KFold(
        n_splits=splits,
        shuffle=True,
        random_state=random_state
    )

    model = xgboost.XGBRegressor(
        max_depth=10,
        learning_rate=0.1,
        n_estimators=50,
        objective='reg:squarederror',
        n_jobs=jobs,
        reg_alpha=10,
        reg_lambda=10,
        seed=random_state
    )

    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        scoring=metric,
        cv=kf,
        n_jobs=jobs
    )

    model.fit(X, y)

    return model, scores


def train_model():
    print('Fitting XGBOOST')

    random_state = 42
    jobs = int(os.getenv('CPU_COUNT'))

    apartments = pd.read_csv(str(PROJECT_ROOT / 'data' / 'apartments.csv'))
    categorical = ['street_name', 'city_name', 'wall_type', 'heating', 'seller', 'water', 'building_condition']
    apartments = pd.get_dummies(apartments, columns=categorical)
    print(apartments.columns)
    X, y = apartments.drop('price_uah', axis=1), apartments['price_uah']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=random_state
    )
    start_time = time.time()
    model, score = cross_validation(
        X_train, y_train,
        random_state=random_state,
        jobs=jobs
    )
    print(f'Fit time {time.time() - start_time} s')

    param_grid = {
        'objective': ['reg:squarederror'],
        'learning_rate': [0.001, 0.01, 0.1, 0.3],
        'max_depth': [5, 8, 10, 20],
        'reg_alpha': [1, 10]
    }
    best_score, best_params = param_grid_search(
        X_train, y_train,
        estimator=model,
        param_grid=param_grid,
        jobs=jobs
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    info = f"""
XGBOOST:
    Grid best score: {best_score}, with params: {best_params}
    Model train score: {score}
    Train mse: {mean_squared_error(y_train, y_pred_train)}
    Test mse: {mean_squared_error(y_test, y_pred_test)}
    """

    print(info)

    print('Saving model')
    model.save_model(str(PROJECT_ROOT / 'data' / 'xgb.model'))


if __name__ == '__main__':
    load_env()

    train_model()
