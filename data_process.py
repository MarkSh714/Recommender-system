import pandas as pd
from datetime import datetime

def sample_data():
    data = pd.read_csv('data/train.csv', chunksize=100000)
    res = pd.DataFrame()
    for i in data:
        res = res.append(i.sample(n=5000, random_state=0))
    res.reset_index(drop=True).to_csv('data/sample_train.csv')

def process_data():
    data = pd.read_csv('data/sample_train.csv', index_col=0)
    del data['id']
    dummy_col = ['device_id', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_category',
                 'app_domain', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C18', 'C19',
                 'C21', 'C17', 'C20', 'hour', 'weekday']
    data['weekday'] = data['hour'].apply(lambda x: datetime.strptime('20' + str(x)[:-2], '%Y%m%d').weekday())
    data['hour'] = data['hour'].apply(lambda x: str(x)[-2:])
    data['device_id'] = data['device_id'].apply(lambda x: 1 if x == 'a99f214a' else 0)
    data['C1'] = data['C1'].apply(lambda x: x if x in [1005, 1002, 1010] else 0)
    data['banner_pos'] = data['banner_pos'].apply(lambda x: 1 if x == 0 else 0)
    site_id_top = list(data['site_id'].value_counts()[:3].index)
    data['site_id'] = data['site_id'].apply(lambda x: x if x in site_id_top else 0)
    site_domain_top = list(data['site_domain'].value_counts()[:7].index)
    data['site_domain'] = data['site_domain'].apply(lambda x: x if x in site_domain_top else 0)
    site_category_top = list(data['site_category'].value_counts()[:4].index)
    data['site_category'] = data['site_category'].apply(lambda x: x if x in site_category_top else 0)
    device_type = list(data['device_type'].value_counts()[:3].index)
    data['device_type'] = data['device_type'].apply(lambda x: x if x in device_type else 0)
    app_id_top = list(data['app_id'].value_counts()[:6].index)
    data['app_id'] = data['app_id'].apply(lambda x: x if x in app_id_top else 0)
    app_domain = list(data['app_domain'].value_counts()[:9].index)
    data['app_domain'] = data['app_domain'].apply(lambda x: x if x in app_domain else 0)
    app_category = list(data['app_category'].value_counts()[:5].index)
    data['app_category'] = data['app_category'].apply(lambda x: x if x in app_category else 0)
    del data['device_ip']
    device_model = list(data['device_model'].value_counts()[:30].index)
    data['device_model'] = data['device_model'].apply(lambda x: x if x in device_model else 0)
    C14 = list(data['C14'].value_counts()[:30].index)
    data['C14'] = data['C14'].apply(lambda x: x if x in C14 else 0)
    C17 = list(data['C17'].value_counts()[:30].index)
    data['C17'] = data['C17'].apply(lambda x: x if x in C17 else 0)
    C19 = list(data['C19'].value_counts()[:30].index)
    data['C19'] = data['C19'].apply(lambda x: x if x in C19 else 0)
    C20 = list(data['C20'].value_counts()[:30].index)
    data['C20'] = data['C20'].apply(lambda x: x if x in C20 else 0)
    C21 = list(data['C21'].value_counts()[:30].index)
    data['C21'] = data['C21'].apply(lambda x: x if x in C21 else 0)
    for col in dummy_col:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
        del data[col]
    data.to_csv('data/processed_sample_train.csv')


if __name__ == '__main__':
    print(1)
