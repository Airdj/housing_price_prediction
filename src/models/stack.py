import yaml
import pandas as pd
import mlflow
from .model_factory import get_stacked_model

def train_final_model(X, y, config):
    with mlflow.start_run(run_name="full_stacking_regressor"):
        model = get_stacked_model(config)
        print(f'Training final(stacked) model')
        model.fit(X, y)
        mlflow.sklearn.log_model(model, name="final_model")
        mlflow.log_param("model_name", "final_model")
        return model

if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E/housing_price_prediction/stacked_models')

    try:
        with open('configs/tuned_models_params.yaml', 'r') as tuned_models_params:
            tmp = yaml.safe_load(tuned_models_params)
    except Exception as e:
        print(f'Error: {e}')

    X_train = pd.read_csv('data/processed/processed_final_train_df.csv')
    y_train = pd.read_csv('data/processed/processed_target_train_feature.csv')

    stacked_model = train_final_model(X_train.drop('Id', axis=1), y_train, tmp)
    print('done')



