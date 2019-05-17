import pandas as pd
import numpy as np
import os


def handle_user(path):
    '''
    处理user表
    :param path:
    :return:
    '''
    user_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

    file_path = os.path.join(path, 'users.dat')
    user_df = pd.read_csv(file_path, sep='::',
                          header=None, names=user_cols, engine='python')

    user_df.drop(['Zip-code'], axis=1, inplace=True)

    gender2id = {gender: idx
                 for idx, gender in enumerate(sorted(user_df.loc[:, 'Gender'].unique()))}
    age2id = {age: idx
              for idx, age in enumerate(sorted(user_df.loc[:, 'Age'].unique()))}
    occupation2id = {occu: idx
                     for idx, occu in enumerate(sorted(user_df.loc[:, 'Occupation'].unique()))}

    user_df.loc[:, 'Gender'] = user_df.loc[:, 'Gender'].map(gender2id)
    user_df.loc[:, 'Age'] = user_df.loc[:, 'Age'].map(age2id)
    user_df.loc[:, 'Occupation'] = user_df.loc[:, 'Occupation'].map(occupation2id)
    user_df.loc[:, 'UserID'] -= 1  # 从1开始变成从0开始

    # 合并年龄与性别
    user_df.loc[:, 'Age_Gender'] = user_df.loc[:, 'Age'].map(
        str) + user_df.loc[:, 'Gender'].map(str)
    agegen2id = {agegen: idx
                 for idx, agegen in enumerate(sorted(user_df.loc[:, 'Age_Gender'].unique()))}
    user_df.loc[:, 'Age_Gender'] = user_df.loc[:, 'Age_Gender'].map(agegen2id)

    user_df.drop(['Age', 'Gender'], axis=1, inplace=True)

    return user_df


def handle_movie(path):
    '''
    处理movie表
    :param path:
    :return:
    '''
    movie_cols = ['MovieID', 'Title', 'Genres']

    file_path = os.path.join(path, 'movies.dat')
    movie_df = pd.read_csv(file_path, sep='::',
                           header=None, names=movie_cols, engine='python')

    # year_pat='\([\d]{4}\)'
    # second_title_pat='\(\D+\)'
    movie_df.loc[:, 'Year'] = movie_df.loc[:, 'Title'].str.extract('({})'.format(
        '\([\d]{4}\)'), expand=False).str.replace('[\(\)]', '').astype('int32')  # 提取年份
    movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].str.replace(
        '\ \([\d]{4}\)', '')  # 删除年份
    movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].str.replace(
        '\ \(\D+\)', '')  # 删除第二title
    movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].str.replace(
        ', (The|An|A)$', '')  # 删掉末尾的冠词

    mid2id = {mid: idx
              for idx, mid in enumerate(sorted(movie_df.loc[:, 'MovieID'].unique()))}

    # 类别集
    genre_set = set()
    for genre in movie_df.loc[:, 'Genres'].str.split('|'):
        genre_set.update(genre)
    genre_set.add('<PAD>')

    # 类别转id
    genre2id = {genre: idx for idx, genre in enumerate(genre_set)}
    # 多类别转list
    genres2list = {genres: [genre2id[genre] for genre in genres.split('|')]
                   for idx, genres in enumerate(movie_df.loc[:, 'Genres'].unique())}

    # 规整化处理
    genres_max = 5  # 设定最大允许的类别数
    for genres in genres2list.keys():
        genres2list[genres] = genres2list[genres][:genres_max]  # 超出长度做截断
        for pad_num in range(genres_max - len(genres2list[genres])):  # 需要填充的数量
            genres2list[genres].append(genre2id['<PAD>'])

    # 单词集
    word_set = set()
    for word in movie_df.loc[:, 'Title'].str.split():
        word_set.update(word)
    word_set.add('<PAD>')
    word2id = {word: idx for idx, word in enumerate(word_set)}
    title2list = {title: [word2id[word] for word in title.split()]
                  for idx, title in enumerate(movie_df.loc[:, 'Title'].unique())}

    # 规整化处理
    title_max = 8
    for title in title2list.keys():
        title2list[title] = title2list[title][:title_max]
        for pad_num in range(title_max - len(title2list[title])):
            title2list[title].append(word2id['<PAD>'])

    year2id = {year: idx
               for idx, year in enumerate(sorted(movie_df.loc[:, 'Year'].unique()))}

    movie_df.loc[:, 'MovieID'] = movie_df.loc[:, 'MovieID'].map(mid2id)
    movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].map(title2list)
    movie_df.loc[:, 'Genres'] = movie_df.loc[:, 'Genres'].map(genres2list)
    movie_df.loc[:, 'Year'] = movie_df.loc[:, 'Year'].map(year2id)

    return movie_df, mid2id


def merge_and_save(user_df, movie_df, mid2id, rating_df):
    '''

    :param user_df:
    :param movie_df:
    :param mid2id:
    :param rating_df:
    :return:
    '''
    rating_df.loc[:, 'UserID'] -= 1
    rating_df.loc[:, 'MovieID'] = rating_df.loc[:, 'MovieID'].map(mid2id)
    rating_df.drop(['ts'], axis=1, inplace=True)    # 丢弃时间戳

    data = pd.merge(pd.merge(user_df, rating_df), movie_df)
    data = data[['UserID', 'Age_Gender', 'Occupation',
                 'MovieID', 'Title', 'Genres', 'Year', 'Rating']]
    data = data.sample(frac=1)  # shuffle

    n_samples = len(data)
    train_ratio = 0.8
    cut_idx = int(n_samples * train_ratio)
    train_df, test_df = data[:cut_idx], data[cut_idx:]
    np.save('train.npy', train_df.values)
    np.save('test.npy', test_df.values)


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'ml-1m')
    user_df = handle_user(path)
    movie_df, mid2id = handle_movie(path)

    rating_cols = ['UserID', 'MovieID', 'Rating', 'ts']
    rating_df = pd.read_csv(os.path.join(path, 'ratings.dat'), sep='::',
                            header=None, names=rating_cols, engine='python')

    merge_and_save(user_df, movie_df, mid2id, rating_df)
