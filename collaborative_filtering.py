import pandas as pd
import numpy as np
import collections

anime = pd.read_csv('data/anime.csv')
rating = pd.read_csv('data/rating.csv')
rating = rating.iloc[:len(rating) // 40]
rating = rating.sort_values(by='anime_id')
rating = rating.iloc[:len(rating) // 5]
anime_id = rating['anime_id'].unique()
user_id = rating['user_id'].unique()
user_id_dict = dict(zip(user_id, [i for i in range(len(user_id))]))
rating['user_id_2'] = rating['user_id'].apply(lambda x: user_id_dict[x])


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


user_cf(10)
