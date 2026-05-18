import pandas as pd
from pathlib import Path

cols_path = Path('src/utils/columns.csv')

def load_data(train_path='data/raw/train.csv',test_path='data/raw/test.csv'):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_eval = df_train[1000:]
    df_train = df_train[:1000]
    if not cols_path.exists():
        cols = pd.DataFrame(df_test.columns)
        cols.to_csv('src/utils/columns.csv', index=False, header=False)

    return df_train, df_test, df_eval