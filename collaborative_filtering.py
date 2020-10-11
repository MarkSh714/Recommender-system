import pandas as pd
import numpy as np
import collections

# 读取动画评分数据，并减少数据量
anime = pd.read_csv('data/anime.csv')
rating = pd.read_csv('data/rating.csv')
rating = rating.iloc[:len(rating) // 20]
rating = rating.sort_values(by='anime_id')
rating = rating.iloc[:len(rating) // 5]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
anime_rating_mean = rating[rating['rating'] != -1].groupby('anime_id')['rating'].mean()
rating = rating[rating.anime_id.isin(list(anime_rating_mean.index))]
anime_id = rating['anime_id'].unique()
user_id = rating['user_id'].unique()
user_id_dict = dict(zip(user_id, [i for i in range(len(user_id))]))
anime_id_dict = dict(zip(anime_id, [i for i in range(len(anime_id))]))
rating['user_id_2'] = rating['user_id'].apply(lambda x: user_id_dict[x])
rating['anime_id_2'] = rating['anime_id'].apply(lambda x: anime_id_dict[x])
rating['rating'] = rating.apply(lambda x: x.rating if x.rating != -1 else anime_rating_mean[x.anime_id], axis=1)


def user_cf(K):
    item_users = collections.defaultdict(set)
    user_items = collections.defaultdict(set)
    for aid in anime_id:
        item_users[aid] = set(rating[rating['anime_id'] == aid]['user_id_2'])
    for uid in user_id:
        user_items[uid] = set(rating[rating['user_id'] == uid]['anime_id'])
    n = np.zeros(len(user_id))
    w = np.zeros((len(user_id), len(user_id)))

    for i, users in item_users.items():
        for u in users:
            n[u] += 1
            for v in users:
                if u != v:
                    w[u][v] += 1
    for i in range(len(user_id)):
        for j in range(len(user_id)):
            if i != j and n[i] * n[j] != 0:
                w[i][j] /= (n[i] * n[j]) ** 0.5
    w = pd.DataFrame(w, index=user_id, columns=user_id)
    similar_user = w[489].sort_values(ascending=False).iloc[:K]
    item_score = collections.defaultdict(int)
    for i in range(K):
        ind = similar_user.index[i]
        similar_pct = similar_user.iloc[i]
        for item in user_items[ind]:
            if item not in user_items[489]:
                item_score[item] += similar_pct
    recommender_items = pd.DataFrame(sorted(item_score.items(), key=lambda x: -1 * x[1])[:K],
                                     columns=['anime_id', 'score'])
    print(pd.merge(recommender_items, anime, how='left', on='anime_id'))


def item_cf(K):
    item_users = collections.defaultdict(set)
    user_items = collections.defaultdict(set)
    for aid in anime_id:
        item_users[aid] = set(rating[rating['anime_id'] == aid]['user_id'])
    for uid in user_id:
        user_items[uid] = set(rating[rating['user_id'] == uid]['anime_id_2'])
    n = np.zeros(len(anime_id))
    w = np.zeros((len(anime_id), len(anime_id)))

    for i, items in user_items.items():
        for u in items:
            n[u] += 1
            for v in items:
                if u != v:
                    w[u][v] += 1
    for i in range(len(anime_id)):
        for j in range(len(anime_id)):
            if i != j and n[i] * n[j] != 0:
                w[i][j] /= (n[i] * n[j]) ** 0.5
    w = pd.DataFrame(w, index=anime_id, columns=anime_id)
    item_score = collections.defaultdict(float)
    for item in user_items[489]:
        silimar_items = w[anime_id[item]].sort_values(ascending=False).iloc[:K]
        for i in range(K):
            if silimar_items.index[i] not in user_items[489]:
                item_score[silimar_items.index[i]] += silimar_items.iloc[i]

    recommender_items = pd.DataFrame(sorted(item_score.items(), key=lambda x: -1 * x[1])[:K],
                                     columns=['anime_id', 'score'])
    print(pd.merge(recommender_items, anime, how='left', on='anime_id'))


def loss(p, q, train, beta):
    res = 0
    for i in range(len(p)):
        res += beta * np.dot(p[i, :], p[i, :])
    for i in range(len(q)):
        res += beta * np.dot(q[i, :], q[i, :])
    for i in range(len(train)):
        r = train[i][0]
        pi = p[int(train[i][1]), :]
        qi = q[int(train[i][2]), :]
        res += (r - np.dot(pi, qi)) ** 2
    return res


def lfm(alpha, beta, k, epochs):
    p = np.random.random(size=(len(user_id), k))
    q = np.random.random(size=(len(anime_id), k))
    train = rating[['rating', 'user_id_2', 'anime_id_2']].values
    n = len(train)
    for epoch in range(epochs):
        print("epoch {0} loss: {1}".format(epoch, loss(p, q, train, beta)))
        for i in range(len(train)):
            r = train[i][0]
            pi = p[int(train[i][1]), :]
            qi = q[int(train[i][2]), :]
            p[int(train[i][1]), :] = pi - alpha * (-2 * (r - np.dot(pi, qi)) * qi + 2 * beta * pi) / n
            q[int(train[i][2]), :] = qi - alpha * (-2 * (r - np.dot(pi, qi)) * pi + 2 * beta * qi) / n


lfm(0.1, 0.1, 20, 1000)
