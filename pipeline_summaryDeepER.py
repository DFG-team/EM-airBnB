import pandas as pd
import numpy as np
import os
import gensim.downloader as api
import models.deepER as dp
from deepER_start import start_model

def start_pipeline_with_DeepER(path):

    pd.options.mode.chained_assignment = None  # default='warn'
    print("pipeline started")
    dataset_url_amsterdam = 'http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2020-12-12' \
                            '/visualisations/listings.csv '
    dataset_url_rome = 'http://data.insideairbnb.com/italy/lazio/rome/2021-01-13/visualisations/listings.csv'
    data = pd.read_csv(dataset_url_rome)
    realdata = merge_dataframe(data)

    save_file_split(realdata, path)
    start_model(path)

def filecsv_label_with_1(data):
    dataframe1 = pd.merge(data, data, left_on=['host_id', 'latitude', 'longitude'],
                              right_on=['host_id', 'latitude', 'longitude'],
                              suffixes=('_ltable', '_rtable'))

    data = dataframe1[dataframe1["id_ltable"] != dataframe1["id_rtable"]]
    data.insert(0, "label", 1, True)

    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato
    datasupport = data['host_id'].copy()
    data.rename(columns={'host_id': 'host_id_ltable'}, inplace=True)
    data.insert(19, "host_id_rtable", datasupport, True)

    datasupport = data['latitude'].copy()
    data.rename(columns={'latitude': 'latitude_ltable'}, inplace=True)
    data.insert(23, "latitude_rtable", datasupport, True)

    datasupport = data['longitude'].copy()
    data.rename(columns={'longitude': 'longitude_ltable'}, inplace=True)
    data.insert(24, "longitude_rtable", datasupport, True)

    data = rename_columuns(data)

    return data


def rename_columuns(data):
    datasupp1 = data[['label']]
    datasupp2 = data.loc[:, 'id_ltable':'reviews_per_month_ltable']
    datasupp3 = data.loc[:, 'id_rtable':'reviews_per_month_rtable']

    for col in datasupp2.columns:
        if "_ltable" in col:
            datasupp2 = datasupp2.add_prefix('ltable_')
            datasupp2.columns = datasupp2.columns.str.replace('_ltable', '')

    for col in datasupp3.columns:
        if "_rtable" in col:
            datasupp3 = datasupp3.add_prefix('rtable_')
            datasupp3.columns = datasupp3.columns.str.replace('_rtable', '')


    concatenateFrames = [datasupp1, datasupp2, datasupp3]
    result = pd.concat(concatenateFrames, axis=1)

    return result


def filecsv_label_with_0(data):
    dataframe1 = pd.merge(data, data, left_on=['latitude', 'room_type', 'price'],
                          right_on=['latitude', 'room_type', 'price'],
                          suffixes=('_ltable', '_rtable'))

    data = dataframe1[dataframe1["id_ltable"] != dataframe1["id_rtable"]]
    data.insert(0, "label", 0, True)

    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato

    datasupport = data['latitude'].copy()
    data.rename(columns={'latitude': 'latitude_ltable'}, inplace=True)
    data.insert(23, "latitude_rtable", datasupport, True)

    datasupport = data['room_type'].copy()
    data.rename(columns={'room_type': 'room_type_ltable'}, inplace=True)
    data.insert(25, "room_type_rtable", datasupport, True)

    datasupport = data['price'].copy()
    data.rename(columns={'price': 'price_ltable'}, inplace=True)
    data.insert(26, "price_rtable", datasupport, True)

    data = data[(data["host_id_ltable"] != data["host_id_rtable"]) | (data["latitude_ltable"] != data["latitude_rtable"]) |
                (data["longitude_ltable"] != data["longitude_rtable"])]

    data = rename_columuns(data)

    print(data)

    return data


def filecsv_label_with_0_type2(data):

    dataframe1 = pd.merge(data, data, left_on=['longitude', 'neighbourhood', 'minimum_nights'],
                          right_on=['longitude', 'neighbourhood', 'minimum_nights'],
                          suffixes=('_ltable', '_rtable'))

    data = dataframe1[dataframe1["id_ltable"] != dataframe1["id_rtable"]]
    data.insert(0, "label", 0, True)
    # Rinomino le variabili di join
    # Utilizzo il numero delle colonne per classificare come right i valori duplicati
    # poichè per i valori coinvolti nel merge il suffisso non è stato applicato

    datasupport = data['longitude'].copy()
    data.rename(columns={'longitude': 'longitude_ltable'}, inplace=True)
    data.insert(23, "longitude_rtable", datasupport, True)

    datasupport = data['neighbourhood'].copy()
    data.rename(columns={'neighbourhood': 'neighbourhood_ltable'}, inplace=True)
    data.insert(22, "neighbourhood_rtable", datasupport, True)

    datasupport = data['minimum_nights'].copy()
    data.rename(columns={'minimum_nights': 'minimum_nights_ltable'}, inplace=True)
    data.insert(27, "minimum_nights_rtable", datasupport, True)

    data = data[(data["host_id_ltable"] != data["host_id_rtable"]) | (data["latitude_ltable"] != data["latitude_rtable"]) |
            (data["longitude_ltable"] != data["longitude_rtable"])]

    data = rename_columuns(data)

    return data


def merge_dataframe(data):
    frame1 = filecsv_label_with_1(data)
    frame2 = filecsv_label_with_0(data)

    mainFrame = [frame1, frame2]
    result = pd.concat(mainFrame)

    result['id'] = range(1, len(result) + 1)
    datasupport = result['id'].copy()
    result.drop(result.columns[len(result.columns) - 1], axis=1, inplace=True)
    result.insert(0, "id", datasupport, True)

    return result


def save_file_split(data, path):
    train, validate, test = train_validate_test_split(data)
    if not os.path.exists('DatasetRomeDeepER'):
        os.makedirs('DatasetRomeDeepER')
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
