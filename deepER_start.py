from clean_dataset import preprocessing_pipeline
import pandas as pd
import os
import gensim.downloader as api
import models.deepER as dp


def to_deeper_data(df: pd.DataFrame):
    res = []
    for r in range(len(df)):
        row = df.iloc[r]
        lpd = row.filter(regex='^ltable_')
        rpd = row.filter(regex='^rtable_')
        if 'label' in row:
            label = row['label']
            res.append((lpd.values.astype('str'), rpd.values.astype('str'), label))
        else:
            res.append((lpd.values.astype('str'), rpd.values.astype('str')))
    return res


def start_model(path):
    train_df = pd.read_csv(path + 'train.csv')
    valid_df = pd.read_csv(path + 'validate.csv')
    test_df = pd.read_csv(path + 'test_dublin.csv')
    
    # train_df = preprocessing_pipeline(train_df)
    # valid_df = preprocessing_pipeline(valid_df)
    # test_df = preprocessing_pipeline(test_df)

    if not os.path.exists('models/glove.6B.50d.txt'):
        word_vectors = api.load("glove-wiki-gigaword-50")
        word_vectors.save_word2vec_format('models/glove.6B.50d.txt', binary=False)

    embeddings_index = dp.init_embeddings_index('models/glove.6B.50d.txt')
    emb_dim = len(embeddings_index['cat'])
    embeddings_model, tokenizer = dp.init_embeddings_model(embeddings_index)

    model = dp.init_DeepER_model(emb_dim)
    model = dp.train_model_ER(to_deeper_data(train_df), model, embeddings_model, tokenizer)

    dp.model_statistics(to_deeper_data(test_df), model, embeddings_model, tokenizer)
