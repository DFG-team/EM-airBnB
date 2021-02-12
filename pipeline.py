import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def start_pipeline(path):
    print("pipeline started")
    dataset_url = 'http://data.insideairbnb.com/italy/lombardia/bergamo/2020-12-31/data/listings.csv.gz'
    data = pd.read_csv(dataset_url)
    label_filecsv_truth(data)
    save_file_split(data, path)


def train_validate_test_split(data, train_percent=.6, validate_percent=.2, seed=None):
    # Randomizziamo per rendere la divisione del dataset reale
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
    if not os.path.exists('Dataset'):
        os.makedirs('Dataset')
        train.to_csv(path + 'train.csv')
        validate.to_csv(path + 'validate.csv')
        test.to_csv(path + 'test.csv')


def label_filecsv_truth(data):
    dataframe1 = pd.merge(data, data, left_on='id', right_on='id', suffixes=('_left', '_right'))

    dataframe1.insert(0, "Label", 1, True)
    print(dataframe1)


