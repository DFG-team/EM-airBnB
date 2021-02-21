import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def clean_dataset(dataset):
    cleaned = dataset
    cleaned = cleaned.drop(['ltable_neighbourhood_group', 'rtable_neighbourhood_group'], axis=1)
    cleaned['ltable_last_review'] = cleaned['ltable_last_review'].fillna(0)
    cleaned['ltable_reviews_per_month'] = cleaned['ltable_reviews_per_month'].fillna(0)
    cleaned['rtable_last_review'] = cleaned['rtable_last_review'].fillna(0)
    cleaned['rtable_reviews_per_month'] = cleaned['rtable_reviews_per_month'].fillna(0)

    return cleaned