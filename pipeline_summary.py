import pandas as pd
import numpy as np
import os


def start_pipeline_summary(path):
    pd.options.mode.chained_assignment = None  # default='warn'
    print("pipeline started")
    dataset_url = 'http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2020-12-12/visualisations' \
                  '/listings.csv '
    data = pd.read_csv(dataset_url)
    realdata = merge_dataframe(data)
    # pd.set_option('display.max_columns', None)
    # for col in realdata.columns:
    # print(col)
    save_file_split(realdata, path)
    visualize_truth_csv(data, path)


def filecsv_label_with_1(data):
    dataframe1 = pd.merge(data, data, left_on=['host_id', 'latitude', 'longitude'],
                          right_on=['host_id', 'latitude', 'longitude'],
                          suffixes=('_left', '_right'))

    data = dataframe1[dataframe1["id_left"] != dataframe1["id_right"]]
    data.insert(0, "label", 1, True)

    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato
    datasupport = data['host_id'].copy()
    data.rename(columns={'host_id': 'host_id_left'}, inplace=True)
    data.insert(19, "host_id_right", datasupport, True)

    datasupport = data['latitude'].copy()
    data.rename(columns={'latitude': 'latitude_left'}, inplace=True)
    data.insert(23, "latitude_right", datasupport, True)

    datasupport = data['longitude'].copy()
    data.rename(columns={'longitude': 'longitude_left'}, inplace=True)
    data.insert(24, "longitude_right", datasupport, True)

    data = rename_columuns2(data)

    # print(data)

    return data


def rename_columuns2(data):
    datasupp1 = data[['label']]
    datasupp2 = data.loc[:, 'id_left':'reviews_per_month_left']
    datasupp3 = data.loc[:, 'id_right':'reviews_per_month_right']

    for col in datasupp2.columns:
        if "_left" in col:
            datasupp2 = datasupp2.add_prefix('left_')
            datasupp2.columns = datasupp2.columns.str.replace('_left', '')

    for col in datasupp3.columns:
        if "_right" in col:
            datasupp3 = datasupp3.add_prefix('right_')
            datasupp3.columns = datasupp3.columns.str.replace('_right', '')

    # print(datasupp2.columns)
    #  print(datasupp3.columns)

    concatenateFrames = [datasupp1, datasupp2, datasupp3]
    result = pd.concat(concatenateFrames, axis=1)

    return result


def filecsv_label_with_0(data):
    dataframe1 = pd.merge(data, data, left_on=['latitude', 'room_type', 'price'],
                          right_on=['latitude', 'room_type', 'price'],
                          suffixes=('_left', '_right'))

    data = dataframe1[dataframe1["id_left"] != dataframe1["id_right"]]
    data.insert(0, "label", 0, True)

    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato

    datasupport = data['latitude'].copy()
    data.rename(columns={'latitude': 'latitude_left'}, inplace=True)
    data.insert(23, "latitude_right", datasupport, True)

    datasupport = data['room_type'].copy()
    data.rename(columns={'room_type': 'room_type_left'}, inplace=True)
    data.insert(25, "room_type_right", datasupport, True)

    datasupport = data['price'].copy()
    data.rename(columns={'price': 'price_left'}, inplace=True)
    data.insert(26, "price_right", datasupport, True)

    data = data[(data["host_id_left"] != data["host_id_right"]) | (data["latitude_left"] != data["latitude_right"]) |
                (data["longitude_left"] != data["longitude_right"])]

    data = rename_columuns2(data)

    print(data)

    return data


def filecsv_label_with_0_type2(data):

    dataframe1 = pd.merge(data, data, left_on=['longitude', 'neighbourhood', 'minimum_nights'],
                          right_on=['longitude', 'neighbourhood', 'minimum_nights'],
                          suffixes=('_left', '_right'))

    data = dataframe1[dataframe1["id_left"] != dataframe1["id_right"]]
    data.insert(0, "label", 0, True)

    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato

    datasupport = data['latitude'].copy()
    data.rename(columns={'latitude': 'latitude_left'}, inplace=True)
    data.insert(23, "latitude_right", datasupport, True)

    datasupport = data['room_type'].copy()
    data.rename(columns={'room_type': 'room_type_left'}, inplace=True)
    data.insert(25, "room_type_right", datasupport, True)

    datasupport = data['price'].copy()
    data.rename(columns={'price': 'price_left'}, inplace=True)
    data.insert(26, "price_right", datasupport, True)

    data = data[(data["host_id_left"] != data["host_id_right"]) | (data["latitude_left"] != data["latitude_right"]) |
                (data["longitude_left"] != data["longitude_right"])]

    data = rename_columuns2(data)

    print(data)

    return data


def merge_dataframe(data):
    frame1 = filecsv_label_with_1(data)
    frame2 = filecsv_label_with_0(data)

    mainFrame = [frame1, frame2]
    result = pd.concat(mainFrame)

    result['id'] = range(1, len(result) + 1)
    datasupport = result['id'].copy()
    result.drop(result.columns[len(result.columns) - 1], axis=1, inplace=True)
    # result.drop('left_label', axis=1, inplace=True)
    result.insert(0, "id", datasupport, True)

    return result


def save_file_split(data, path):
    train, validate, test = train_validate_test_split(data)
    if not os.path.exists('DatasetAmsterdam'):
        os.makedirs('DatasetAmsterdam')
    train.to_csv(path + 'train.csv', index=False, header=True)
    validate.to_csv(path + 'validate.csv', index=False, header=True)
    test.to_csv(path + 'test.csv', index=False, header=True)


def train_validate_test_split(data, train_percent=.6, validate_percent=.2, seed=None):
    # Randomizziamo per rendere la divisione del dataset reale
    data = data.sample(frac=1).reset_index(drop=True)
    m = len(data.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = data.iloc[:train_end]
    validate = data.iloc[train_end:validate_end]
    test = data.iloc[validate_end:]
    return train, validate, test


def visualize_truth_csv(data, path):
    df_groundtruth = merge_dataframe(data)
    df_groundtruth = df_groundtruth[['id', 'label']]
    df_groundtruth.to_csv(path + 'df_groundtruth.csv', index=False, header=True)
