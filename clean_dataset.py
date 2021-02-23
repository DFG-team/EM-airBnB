def clean_dataset(raw_dataset):
    cleaned = raw_dataset
    cleaned = cleaned.drop(['ltable_neighbourhood_group', 'rtable_neighbourhood_group'], axis=1)
    cleaned['ltable_last_review'] = cleaned['ltable_last_review'].fillna(0)
    cleaned['ltable_reviews_per_month'] = cleaned['ltable_reviews_per_month'].fillna(0)
    cleaned['rtable_last_review'] = cleaned['rtable_last_review'].fillna(0)
    cleaned['rtable_reviews_per_month'] = cleaned['rtable_reviews_per_month'].fillna(0)
    return cleaned


def preprocessing_pipeline(raw_dataset):
    dataset = clean_dataset(raw_dataset)
    return dataset
