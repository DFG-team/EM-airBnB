import pandas as pd
import os


def start_pipeline_summary(path):
    pd.options.mode.chained_assignment = None  # default='warn'
    print("pipeline_summary deep matcher started")
    dataset_url_amsterdam = 'http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2020-12-12' \
                            '/visualisations/listings.csv '
    dataset_url_rome = 'http://data.insideairbnb.com/italy/lazio/rome/2021-01-13/visualisations/listings.csv'
    dataset_url_london = 'http://data.insideairbnb.com/united-kingdom/england/london/2021-01-11/visualisations' \
                         '/listings.csv '
    dataset_url_dublin = 'http://data.insideairbnb.com/ireland/leinster/dublin/2021-02-10/visualisations/listings.csv'
    dataset_url_brussels = 'http://data.insideairbnb.com/belgium/bru/brussels/2021-01-23/visualisations/listings.csv'
    # reading cities csv
    data_amsterdam = pd.read_csv(dataset_url_amsterdam)
    data_rome = pd.read_csv(dataset_url_rome)
    data_london = pd.read_csv(dataset_url_london)
    data_dublin = pd.read_csv(dataset_url_dublin)
    data_brussels = pd.read_csv(dataset_url_brussels)

    # merging cities data frames
    realdata_rome = merge_dataframe(data_rome)
    realdata_amsterdam = merge_dataframe(data_amsterdam)
    realdata_amsterdam = realdata_amsterdam.sample(frac=1).reset_index(drop=True)
    realdata_london = merge_dataframe(data_london)
    realdata_london = realdata_london.sample(frac=1).reset_index(drop=True)
    realdata_dublin = merge_dataframe(data_dublin)
    realdata_dublin = realdata_dublin.sample(frac=1).reset_index(drop=True)
    realdata_brussels = merge_dataframe(data_brussels)
    realdata_brussels = realdata_brussels.sample(frac=1).reset_index(drop=True)

    # split amsterdam data frame for training model
    save_file_split(realdata_rome, path)

    # saving test_city.csv
    realdata_amsterdam.to_csv(path + 'test_amsterdam.csv', index=False, header=True)
    realdata_london.to_csv(path + 'test_london.csv', index=False, header=True)
    realdata_dublin.to_csv(path + 'test_dublin.csv', index=False, header=True)
    realdata_brussels.to_csv(path + 'test_brussels.csv', index=False, header=True)

    visualize_truth_csv(data_rome, path)


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

    data = rename_columuns(data)

    return data


def rename_columuns(data):
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

    concatenateFrames = [datasupp1, datasupp2, datasupp3]
    result = pd.concat(concatenateFrames, axis=1)

    return result


def filecsv_label_with_0_type1(data):
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

    data = rename_columuns(data)

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

    datasupport = data['longitude'].copy()
    data.rename(columns={'longitude': 'longitude_left'}, inplace=True)
    data.insert(23, "longitude_right", datasupport, True)

    datasupport = data['neighbourhood'].copy()
    data.rename(columns={'neighbourhood': 'neighbourhood_left'}, inplace=True)
    data.insert(22, "neighbourhood_right", datasupport, True)

    datasupport = data['minimum_nights'].copy()
    data.rename(columns={'minimum_nights': 'minimum_nights_left'}, inplace=True)
    data.insert(27, "minimum_nights_right", datasupport, True)

    data = data[(data["host_id_left"] != data["host_id_right"]) | (data["latitude_left"] != data["latitude_right"]) |
                (data["longitude_left"] != data["longitude_right"])]

    data = rename_columuns(data)

    return data


def merge_dataframe(data):
    frame1 = filecsv_label_with_1(data)
    frame2 = filecsv_label_with_0_type1(data)
    frame3 = filecsv_label_with_0_type2(data)

    mainFrame = [frame1, frame2, frame3]
    result = pd.concat(mainFrame)

    result['id'] = range(1, len(result) + 1)
    datasupport = result['id'].copy()
    result.drop(result.columns[len(result.columns) - 1], axis=1, inplace=True)
    result.insert(0, "id", datasupport, True)

    return result


def save_file_split(data, path):
    train, validate, test = train_validate_test_split(data)
    if not os.path.exists('DatasetDeepMatcher'):
        os.makedirs('DatasetDeepMatcher')
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
