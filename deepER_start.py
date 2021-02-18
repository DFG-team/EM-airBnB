import pandas as pd
import numpy as np
import os
import gensim.downloader as api
import models.DeepER as dp

def start_model(train_df, valid_df, test_df):
    if not os.path.exists('models/glove.6B.50d.txt'):
        word_vectors = api.load("glove-wiki-gigaword-50")
        word_vectors.save_word2vec_format('models/glove.6B.50d.txt', binary=False)
    
    
    
    return
    