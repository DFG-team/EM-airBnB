import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def start_pipeline2(path):
    path2 = "Data_csv\\"
    print("pipeline started")
    dataset_url = 'http://data.insideairbnb.com/italy/lombardia/bergamo/2020-12-31/data/listings.csv.gz'

    data2 = pd.read_csv(path2 + 'Match_airbnb_rome.csv')
    data2 = label_temp(data2)
    save_file_split(data2, path)
    print(data2)
    data = pd.read_csv(dataset_url)
    label_filecsv_truth(data)
    visualize_truth_csv(data, path)
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
        train.to_csv(path + 'train.csv', index=False, header=True)
        validate.to_csv(path + 'validate.csv', index=False, header=True)
        test.to_csv(path + 'test.csv', index=False, header=True)


def label_filecsv_truth(data):
    dataframe1 = pd.merge(data, data, left_on=['host_id', 'latitude', 'longitude', 'bathrooms', 'bedrooms', 'beds'],
                          right_on=['host_id', 'latitude', 'longitude', 'bathrooms', 'bedrooms', 'beds'],
                          suffixes=('_left', '_right'))
    data = dataframe1[dataframe1["id_left"] != dataframe1["id_right"]]
    data.insert(0, "label", 1, True)
    data.rename(columns={'id_left': 'id'}, inplace=True)

    print(data)
    return data


def visualize_truth_csv(data, path):
    df_groundtruth = label_filecsv_truth(data)
    df_groundtruth = df_groundtruth[['id', 'label', 'listing_url_left', 'listing_url_right']]
    df_groundtruth.to_csv(path + 'df_groundtruth.csv')


def label_temp(data2):
    data2.insert(1, "label", 1, True)

    data2.rename(columns={'ID_1': 'id'}, inplace=True)
    data2.rename(columns={"NAME_1": 'left_name', "HOSTNAME_1": 'left_hostname', "NEIGHB_1": 'left_neighb',
                          "LATITUDE_1": 'left_latitude',
                          "LONGITUDE_1": 'left_longitude', "ROOM_TYPE_1": 'left_room_type', "ID_2": 'right_id',
                          "NAME_2": 'right_name', "HOSTNAME_2": 'right_hostname',
                          "NEIGHB_2": 'right_neighb', "LATITUDE_2": 'right_latitude',
                          "LONGITUDE_2": 'right_longitude', "ROOM_TYPE_2": 'right_room_type'}, inplace=True)

    data2 = data2.reindex(columns=['id', 'label', 'left_name', 'left_hostname', 'left_neighb', 'left_latitude',
                                   'left_longitude', 'left_room_type', 'right_name', 'right_hostname',
                                   'right_neighb', 'right_latitude',
                                   'right_longitude', 'right_room_type'])

    print(data2.columns)

    return data2
