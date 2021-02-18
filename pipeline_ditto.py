import pandas as pd
import os

# deve ritornare un file .txt contenente COL nome_colonna VAL valore_colonna
# in modo tale da fornire il file serializzato utile per DITTO 
# dunque deve produrre test.txt, train.txt, valid.txt

def start_pipeline_ditto(path):
    print('Pipeline started')
    dataset_url = 'http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2020-12-12/visualisations' \
                  '/listings.csv '
    data = pd.read_csv(dataset_url)
    save_file_split(data, path)
    #serialize_train
    #serialize_test
    #serialize_valid


def save_file_split(data, path):
    train, validate, test = train_validate_test_split(data)
    if not os.path.exists('DatasetAmsterdamDitto'):
        os.makedirs('DatasetAmsterdamDitto')
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
