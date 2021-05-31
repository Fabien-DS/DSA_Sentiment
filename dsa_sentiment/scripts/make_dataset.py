
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path, y_col='sentiment', split=False, shuffle=False, test_size=0.2, random_state=42, dropNA=True):
    data = pd.read_csv(data_path, encoding='utf8' )

    if dropNA:
        data.dropna(inplace=True)
        data = data.reset_index(drop=True)

    # Split the data into training and test sets. test_size split.
    if split :
        train, test = train_test_split(data, shuffle=shuffle, test_size=test_size, random_state=random_state)
    else :
        train = data

    # The predicted column is "y_col" 
    train_x = train.drop([y_col], axis=1)
    train_y = train[[y_col]]
    if split :
        test_x = test.drop([y_col], axis=1)
        test_y = test[[y_col]]
        return train_x, train_y, test_x, test_y
    else :
        return train_x, train_y


def Preprocess_StrLower(df, columns_to_process=[]):
    for col in columns_to_process:
        df[col] = df[col].apply(lambda x : str(x).lower())
    return df

def Preprocess_transform_target(df, columns_to_process=[]):
    for col in columns_to_process:
        df[col] = df[col].apply(lambda x : -1 if x=="negative" else 1 if x=="positive" else 0)
    return df

