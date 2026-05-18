from sklearn.linear_model import HuberRegressor
import warnings
import joblib
warnings.filterwarnings('ignore')


def drop_columns(df):
    cols_to_drop = ['Condition2','GarageQual','PoolQC', 'MasVnrType','Street','MiscFeature',
                    'HouseStyle','Alley','FireplaceQu','LotConfig','Utilities', 'Fence',
                    'Heating','LowQualFinSF','Functional','LandContour','BldgType','PoolArea',
                    'Exterior2nd','GarageCond','RoofMatl','KitchenAbvGr','Exterior1st',
                    '3SsnPorch','Condition1','PavedDrive','ExterCond','MiscVal']
    df = df.drop_duplicates()
    df = df.drop(columns=cols_to_drop)
    return df


def missing_lot_frontage(df, artifact=None):
        df_to_handle = df[['LotArea','LotFrontage']]
        missing = list(df_to_handle.loc[df_to_handle['LotFrontage'].isna()].index)
        if missing:
            df_to_train = df_to_handle.drop(missing, axis=0)
            df_to_predict = df_to_handle.loc[missing,:]
            if artifact:
                hub_reg = artifact
            else:
                features_train = df_to_train[['LotArea']]
                target_train = df_to_train[['LotFrontage']]
                hub_reg = HuberRegressor()
                hub_reg.fit(features_train, target_train)

            predictions = hub_reg.predict(df_to_predict[['LotArea']])
            predicted_data = df_to_predict.copy()
            predicted_data['LotFrontage'] = predictions
            df.loc[df['LotFrontage'].isna(),'LotFrontage'] = predictions
            return df, hub_reg

        else:
            return df, None


def missing_garage(df):
    garage_features_cat = ['GarageType', 'GarageFinish']
    garage_features_num = ['GarageYrBlt', 'GarageArea', 'GarageCars']

    for col in garage_features_cat:
        if col in df.columns:
            df[col] = df[col].fillna('NoGarage')

    for col in garage_features_num:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def missing_basement(df):
    basement_features = ['BsmtQual',
                         'BsmtCond',
                         'BsmtExposure',
                         'BsmtFinType1',
                         'BsmtFinType2']
    missing_basement_index = df.loc[df['BsmtQual'].isna()].index
    df.loc[missing_basement_index, basement_features] = 'NoBasement'

    return df


def missing_masvnrarea(df, ratio=None):
    non_zero_median = list(df.loc[df['MasVnrArea'] > 0, :].index)
    mas_vnr_area_median = df.loc[non_zero_median, 'MasVnrArea'].median()
    if ratio:
        ratio_median = ratio
    else:
        ratio_median = df['GrLivArea'].median() / mas_vnr_area_median
    df.loc[df['MasVnrArea'].isna(), 'MasVnrArea'] = df['GrLivArea'] / ratio_median

    return df, ratio_median


def missing_others(df):
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    num_with_na = df[num_cols].columns[df[num_cols].isna().any()]
    cat_with_na = df[cat_cols].columns[df[cat_cols].isna().any()]

    for feature in list(num_with_na):
        df[feature] = df[feature].fillna(df[feature].mean())

    for feature in list(cat_with_na):
        df[feature] = df[feature].fillna(df[feature].mode()[0])

    return df


def preprocess_pipeline(df, hub_reg=None, ratio_median=None):
    df = drop_columns(df)
    df, hubr = missing_lot_frontage(df, hub_reg)
    df = missing_garage(df)
    df = missing_basement(df)
    df, ratiom = missing_masvnrarea(df, ratio_median)
    df = missing_others(df)

    state = {'hub_reg': hubr, 'ratio_median': ratiom}


    return df, state


def preprocess_data(df_train=None, df_test=None, df_eval=None, inference=False, df_inference=None):
    if inference:
        try:
            state = joblib.load("src/utils/preprocess_pipeline_artifacts.pkl")
            df_inference, _ = preprocess_pipeline(df_inference, state['hub_reg'], state['ratio_median'])
            return df_inference

        except Exception as e:
            print(f'Error: {e}')

    else:
        df_train, state = preprocess_pipeline(df_train)
        df_test, _ = preprocess_pipeline(df_test)
        df_eval, _ = preprocess_pipeline(df_eval)
        joblib.dump(state, "src/utils/preprocess_pipeline_artifacts.pkl")
        return df_train, df_test, df_eval
