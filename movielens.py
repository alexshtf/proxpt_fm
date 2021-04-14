import numpy as np
import pandas as pd
import torch


def load(path='data/ml-100k'):
    ratings = pd.read_csv(f'{path}/u.data', sep='\t', header=None, names=['user', 'movie', 'rating', 'timestamp'])
    users = pd.read_csv(f'{path}/u.user', sep='|', header=None,
                        names=['user', 'age', 'gender', 'occupation', 'zipcode'])
    users = pd.get_dummies(users, 'age', columns=['age'])
    users = pd.get_dummies(users, 'gender', columns=['gender'])
    users = pd.get_dummies(users, 'occupation', columns=['occupation'])
    items = pd.read_csv(f'{path}/u.item', sep='|', encoding='iso-8859-1', header=None, names=[
        'movie', 'title', 'release_date', 'video_release_data', 'imdb_url',
        'unknown', 'action', 'adventure', 'animation', 'chilren', 'comedy', 'crime', 'documentary',
        'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance',
        'sci_fi', 'thriller', 'war', 'western'
    ])

    dataset = ratings.join(users, on='user', rsuffix='_r').join(items, on='movie', rsuffix='_r')
    dataset = pd.get_dummies(dataset, columns=['user'])
    dataset = pd.get_dummies(dataset, columns=['movie'])
    dataset = dataset.drop([
        'user_r', 'movie_r', 'timestamp', 'zipcode',
        'title', 'release_date', 'video_release_data', 'imdb_url'], axis=1)
    dataset = dataset.dropna()
    labels = (dataset['rating'] >= 5).astype(np.float)
    features = dataset.drop(['rating'], axis=1).dropna()

    W_train = torch.from_numpy(features.to_numpy())
    y_train = torch.from_numpy(labels.to_numpy())
    y_train = y_train.reshape(-1, 1)

    return W_train, y_train
