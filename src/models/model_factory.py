from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression,  Ridge, Lasso, RidgeCV
from sklearn.model_selection import KFold


def get_base_models(config):
    models = {
        'ridge': Ridge(**config['ridge_params']),
        'lasso': Lasso(**config['lasso_params']),
        'gbr': GradientBoostingRegressor(**config['gbr_params']),
        'linear': LinearRegression(),
        'lgbm': LGBMRegressor(**config['lgbm_params']),
        'xgb': XGBRegressor(**config['xgb_params']),
        'catboost': CatBoostRegressor(**config['catboost_params']),
    }
    return models


def get_final_models(config):
    models = {
        'ridge': Ridge(**config['ridge_params']['params']),
        'lasso': Lasso(**config['lasso_params']['params']),
        'gbr': GradientBoostingRegressor(**config['gbr_params']['params']),
        'linear': LinearRegression(**config['linear_params']['params']),
        'lgbm': LGBMRegressor(**config['lgbm_params']['params']),
        'xgb': XGBRegressor(**config['xgb_params']['params']),
        'catboost': CatBoostRegressor(**config['catboost_params']['params']),
    }
    return models


def get_stacked_model(config):
    cv_fold = KFold(n_splits=10, shuffle=True, random_state=12)
    models = get_final_models(config)
    stacked = StackingRegressor(
        estimators=[
            ('Ridge', models['ridge']),
            ('Lasso', models['lasso']),
            ('LinearRegression', models['linear']),
            ('GradientBoostingRegressor', models['gbr']),
            ('xgb', models['xgb']),
            ('lightgbm', models['lgbm']),
            ('catboost', models['catboost']),
        ],
        final_estimator=RidgeCV(),
        cv=cv_fold
    )
    return stacked