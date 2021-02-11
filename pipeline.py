import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def start_pipeline():
    print("pipeline started")
    dataset_url = 'http://data.insideairbnb.com/italy/lombardia/bergamo/2020-12-31/data/listings.csv.gz'
    data = pd.read_csv(dataset_url)
    print(data.head())


def train_validate_test_split(data, train_percent=.6, validate_percent=.2, seed=None):
    #Randomizziamo per rendere la divisione del dataset reale
    np.random.seed(seed)
    perm = np.random.permutation(data.index)
    m = len(data.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = data.iloc[perm[:train_end]]
    validate = data.iloc[perm[train_end:validate_end]]
    test = data.iloc[perm[validate_end:]]
    return train, validate, test

def save_file_split(data, path):
    train, validate, test = train_validate_test_split(data)
    train.to_csv(path + 'train.csv')
    validate.to_csv(path + 'validate.csv')
    test.to_csv(path + 'test.csv')



#    y = data.id
#    X = data.drop('id', axis=1)

#    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#    print("\nX_train:\n")
#   print(X_train.head())
#    print(X_train.shape)

#   print("\nX_test:\n")
#   print(X_test.head())
#   print(X_test.shape)
