import numpy as np
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.decomposition import PCA


def remodel_age_feature(df):
    year_feature = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    for feature in year_feature:
        df[feature] = df['YrSold'] - df[feature]

    df['MoSold_x'] = df['MoSold'].apply(lambda x: np.sin(np.pi * int(x) / 12))
    df['MoSold_y'] = df['MoSold'].apply(lambda x: np.cos(np.pi * int(x) / 12))
    return df


def add_new_features(df):
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + \
                           df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']
    df['QualityIndex'] = df['OverallQual'] * df['OverallCond']
    df['LivingAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['RoomsPerBedroom'] = np.where(df['BedroomAbvGr'] > 0, df['TotRmsAbvGrd'] / df['BedroomAbvGr'], 0)
    df['GarageEfficiency'] = np.where(df['GarageCars'] > 0, df['GarageArea'] / df['GarageCars'], 0)
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
    return df


class AutoOneHotEncoder:
    def __init__(self):
        self.encoder = None
        self.cat_cols = []
        self.num_cols = []

    def fit(self, df):
        df_copy = df.copy()

        self.has_target = 'SalePrice' in df_copy.columns
        if self.has_target:
            df_copy = df_copy.drop('SalePrice', axis=1)

        self.cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

        if self.cat_cols:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(df_copy[self.cat_cols])

        return self

    def transform(self, df):
        df_copy = df.copy()

        target = None
        if self.has_target and 'SalePrice' in df_copy.columns:
            target = df_copy['SalePrice']
            df_copy = df_copy.drop('SalePrice', axis=1)

        if not self.cat_cols:
            df_final = df_copy[self.num_cols].copy()
        else:
            encoded = self.encoder.transform(df_copy[self.cat_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_copy.index)

            df_final = pd.concat([df_copy[self.num_cols], df_encoded], axis=1)

        if target is not None:
            df_final['SalePrice'] = target

        return df_final

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def features_scaler(df_train=None, df_test=None, df_eval=None, scalers=None, df_inference=None):
    powertransform_cols = ['LotFrontage', 'LotArea', 'GarageYrBlt', 'RoomsPerBedroom',
                           'MasVnrArea', 'OpenPorchSF', 'TotalPorchSF', 'WoodDeckSF']

    cols_to_robust = ['YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                      'BsmtUnfSF', '2ndFlrSF','GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF',
                      'EnclosedPorch', 'ScreenPorch', 'TotalPorchSF']

    cols_to_standard = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
                        'GarageArea', 'TotalSF','QualityIndex', 'LivingAreaPerRoom',
                        'RoomsPerBedroom', 'GarageEfficiency', 'TotalArea']
    if scalers:
        for col in powertransform_cols:
            pt = scalers['powertransform'][col]
            df_inference[[col]] = pt.transform(df_inference[[col]])

        robust = scalers['robust']
        df_inference[cols_to_robust] = robust.transform(df_inference[cols_to_robust])

        standard = scalers['standard']
        df_inference[cols_to_standard] = standard.transform(df_inference[cols_to_standard])

        return df_inference
    else:
        target_feature_train = np.log1p(df_train['SalePrice'])
        target_feature_eval = np.log1p(df_eval['SalePrice'])
        df_train = df_train.drop('SalePrice', axis=1)
        df_eval = df_eval.drop('SalePrice', axis=1)
        pt_scalers = {}
        for col in powertransform_cols:
            pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
            df_train[[col]] = pt.fit_transform(df_train[[col]])
            df_test[[col]] = pt.transform(df_test[[col]])
            df_eval[[col]] = pt.transform(df_eval[[col]])
            pt_scalers[col] = pt

        robust = RobustScaler()
        robust.fit(df_train[cols_to_robust])
        df_train[cols_to_robust] = robust.transform(df_train[cols_to_robust])
        df_test[cols_to_robust] = robust.transform(df_test[cols_to_robust])
        df_eval[cols_to_robust] = robust.transform(df_eval[cols_to_robust])

        standard = StandardScaler()
        standard.fit(df_train[cols_to_standard])
        df_train[cols_to_standard] = standard.transform(df_train[cols_to_standard])
        df_test[cols_to_standard] = standard.transform(df_test[cols_to_standard])
        df_eval[cols_to_standard] = standard.transform(df_eval[cols_to_standard])

        scalers = {
            'powertransform': pt_scalers,
            'robust': robust,
            'standard': standard,
        }
        return df_train, df_test, df_eval, target_feature_train, target_feature_eval, scalers


def fit_pca_transformers(df_train):
    pca_1 = PCA(n_components=1)
    pca_1.fit(df_train[['GrLivArea', 'TotalSF']])

    pca_2 = PCA(n_components=1)
    pca_2.fit(df_train[['GrLivArea', 'TotalArea']])

    pca_3 = PCA(n_components=1)
    pca_3.fit(df_train[['TotalArea', 'TotalSF']])

    pca_transformers = {'pca_1': pca_1, 'pca_2': pca_2,'pca_3': pca_3}
    return pca_transformers

def apply_pca_transform(df, pca_dict):

    df = df.copy()

    df['multi_cols_1'] = (
        pca_dict['pca_1']
        .transform(df[['GrLivArea', 'TotalSF']])
        .ravel()
    )

    df['multi_cols_2'] = (
        pca_dict['pca_2']
        .transform(df[['GrLivArea', 'TotalArea']])
        .ravel()
    )

    df['multi_cols_3'] = (
        pca_dict['pca_3']
        .transform(df[['TotalArea', 'TotalSF']])
        .ravel()
    )

    df.drop(
        ['GrLivArea', 'TotalSF', 'TotalArea'],
        axis=1,
        inplace=True
    )

    return df


def num_to_cat_converter(df):
    num_to_convert = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath',
                      'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                      'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold',
                      'YrSold', 'MoSold_x', 'MoSold_y', 'TotalBathrooms']
    for col in num_to_convert:
        df[col] = df[col].astype('category')

    obj_to_cat = df.select_dtypes(include='object').columns
    df[obj_to_cat] = df[obj_to_cat].astype('category')

    return df


def outliers_remover(df):
    columns_to_check = list(df.select_dtypes(include='number').drop('Id', axis=1))
    iso = IsolationForest(contamination=0.02, random_state=12)
    mask = iso.fit_predict(df[columns_to_check]) != -1
    df = df[mask]
    return df


def build_pipeline(df):
    steps = [
        remodel_age_feature,
        add_new_features,
        num_to_cat_converter,
    ]

    for step in steps:
        df = step(df)

    return df


def build_features(df_train=None, df_test=None, df_eval=None, inference=False, df_inference=None):
    if inference:
        try:
            state = joblib.load('src/utils/build_features_pipeline_artifacts.pkl')
            df_inference = build_pipeline(df_inference)
            ohe = state['ohe']
            df_inference = ohe.transform(df_inference)
            df_inference = features_scaler(df_inference=df_inference, scalers=state['scalers'])
            pca_transformers = state['pca_transformers']
            final_df_inference = apply_pca_transform(df_inference, pca_transformers)
            return final_df_inference

        except Exception as e:
            print(f'Error: {e}')

    else:
        df_train = build_pipeline(df_train)
        df_test = build_pipeline(df_test)
        df_eval = build_pipeline(df_eval)

        ohe = AutoOneHotEncoder()
        df_train = ohe.fit_transform(df_train)
        df_test = ohe.transform(df_test)
        df_eval = ohe.transform(df_eval)

        df_train = outliers_remover(df_train)

        (df_train, df_test, df_eval, target_train_feature,
         target_eval_feature, scalers) = features_scaler(df_train, df_test, df_eval)

        pca_transformers = fit_pca_transformers(df_train)

        final_train_df = apply_pca_transform(df_train, pca_transformers)
        final_test_df = apply_pca_transform(df_test, pca_transformers)
        final_eval_df = apply_pca_transform(df_eval, pca_transformers)

        state = {'ohe': ohe, 'scalers': scalers, 'pca_transformers': pca_transformers}
        joblib.dump(state, 'src/utils/build_features_pipeline_artifacts.pkl')

        return final_train_df, final_test_df, final_eval_df, target_train_feature, target_eval_feature, scalers