import optuna
import mlflow
import pandas as pd
from sklearn.model_selection import cross_val_score,KFold
from ..utils.helpers import save_config_yaml
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


cv = KFold(n_splits=5, shuffle=True, random_state=12)

models = {
    'ridge': Ridge,
    'lasso': Lasso,
    'linear': LinearRegression,
    'gbr': GradientBoostingRegressor,
    'lgbm': LGBMRegressor,
    'xgb': XGBRegressor,
    'catboost': CatBoostRegressor
}

hyperparams = {
    'ridge': {
        'alpha': {'type': 'float', 'low': 1e-5, 'high': 500, 'log': True},
        'max_iter': {'type': 'int', 'low': 150, 'high': 200000}
    },
    'lasso': {
        'alpha': {'type': 'float', 'low': 1e-5, 'high': 500, 'log': True},
        'max_iter': {'type': 'int', 'low': 1000, 'high': 200000}
    },
    'linear': {},
    'gbr': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 2000},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 1},
        'max_depth': {'type': 'int', 'low': 2, 'high': 16},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 25},
        'max_features': {'type': 'int', 'low': 5, 'high': 60}
    },
    'lgbm': {
        'num_leaves': {'type': 'int', 'low': 20, 'high': 300},
        'max_depth': {'type': 'int', 'low': 2, 'high': 15},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.8},
        'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
        'verbose': {'type': 'fixed', 'value': -1}
    },
    'xgb': {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 1},
        'max_depth': {'type': 'int', 'low': 2, 'high': 20},

    },
    'catboost': {
        'iterations': {'type': 'int', 'low': 100, 'high': 5000},
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'log': True},
        'depth': {'type': 'int', 'low': 3, 'high': 10},
        'verbose': {'type': 'fixed', 'value': 0},
        'allow_writing_files': {'type': 'fixed', 'value': False}
    }
}

def evaluate_model(model, X, y):
    return cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv).mean()


def run_optimization_mlflow(model_name, X, y, n_trials=50):
    model_class = models[model_name]
    param_config = hyperparams.get(model_name, {})

    #mlflow.set_experiment(model_name)

    def objective(trial):
        trial_params = {}
        for param_name, info in param_config.items():
            if info['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, info['low'], info['high'], log=info.get('log', False)
                )
            elif info['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(param_name, info['low'], info['high'])
            elif info['type'] == 'categorical':
                trial_params[param_name] = trial.suggest_categorical(param_name, info['choices'])
            elif info['type'] == 'fixed':
                trial_params[param_name] = info['value']

        model = model_class(**trial_params)
        score = evaluate_model(model, X, y)

        with mlflow.start_run(nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric('neg_RMSE', score)
            mlflow.log_param("model_name", model_name)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with mlflow.start_run(run_name=model_name):
        best_model = model_class(**study.best_params)
        best_score = evaluate_model(best_model, X, y)
        mlflow.log_params(study.best_params)
        mlflow.log_metric('neg_RMSE_best', best_score)
        mlflow.sklearn.log_model(best_model, name='model')
        mlflow.log_param("model_name", model_name)

    return model_name, study.best_params, study.best_value


if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E_housing_price_prediction/tuned_models')

    best_params_dict = {}
    X_train = pd.read_csv('data/processed/processed_final_train_df.csv')
    y_train= pd.read_csv('data/processed/processed_target_train_feature.csv')
    for model_name in models:
        name, params, score = run_optimization_mlflow(model_name, X_train.drop('Id', axis=1), y_train.SalePrice, n_trials=50)
        best_params_dict[name+'_params'] = {"params": params, "score": score}

    save_config_yaml(best_params_dict, 'configs/tuned_models_params.yaml')