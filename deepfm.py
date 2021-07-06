import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

def sample_data(saved_name):
    data = pd.read_csv('data/train.csv', chunksize=100000)
    res = pd.DataFrame()
    for i in data:
        res = res.append(i.sample(n=2000, random_state=0))
    del res['id']
    res.reset_index(drop=True).to_csv(f'data/{saved_name}.csv', index=False)


def preprocess_data():
    data = pd.read_csv('data/sample_data.csv')
    data['weekday'] = data['hour'].apply(lambda x: datetime.strptime('20' + str(x)[:-2], '%Y%m%d').weekday())
    data['hour'] = data['hour'].apply(lambda x: str(x)[-2:])


def fm(x):
    left = torch.square(torch.sum(x, dim=1, keepdim=True))
    right = torch.sum(torch.square(x), dim=1, keepdim=True)
    return 0.5 * torch.sum((left - right), dim=2)


class DeepFM(nn.Module):
    def __init__(self, label_cnts, embed_dim):
        super(DeepFM, self).__init__()
        self.embeds = nn.ModuleList()
        for cnt in label_cnts:
            self.embeds.append(nn.Embedding(cnt, embed_dim).cuda())
        self.label_cnt = len(label_cnts)
        self.deep_layer1 = nn.Sequential(nn.Linear(self.label_cnt * embed_dim, 128).cuda(),
                                         nn.ReLU().cuda(),
                                         nn.Dropout(0.3).cuda())
        self.deep_layer2 = nn.Sequential(nn.Linear(128, 16).cuda(),
                                         nn.ReLU().cuda(),
                                         nn.Dropout(0.3).cuda())
        self.deep_layer3 = nn.Sequential(nn.Linear(16, 2).cuda(),
                                         nn.ReLU().cuda())
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        fm_input = self.embeds[0](x[:, 0]).unsqueeze(dim=1)
        for i in range(1, self.label_cnt):
            fm_input = torch.cat((fm_input, self.embeds[i](x[:, i]).unsqueeze(dim=1)), dim=1)
        fm_output = fm(fm_input)
        out = self.embeds[0](x[:, 0])
        for i in range(1, self.label_cnt):
            out = torch.cat((out, self.embeds[i](x[:, i])), dim=1)
        out = self.deep_layer1(out)
        out = self.deep_layer2(out)
        out = self.deep_layer3(out)
        # final_out = self.sigmoid(fm_output + out).cuda()
        final_out = fm_output + out
        return final_out


if __name__ == '__main__':
    # sample_data('sample_data')
    epochs = 20
    batch_size = 128
    num_worker = 8
    data = pd.read_csv('data/sample_data.csv')
    device = torch.device('cuda')
    data['weekday'] = data['hour'].apply(lambda x: datetime.datetime.strptime('20' + str(x)[:-2], '%Y%m%d').weekday())
    data['hour'] = data['hour'].apply(lambda x: str(x)[-2:])
    y = data['click']
    del data['click']
    col_vc = []
    for col in data.columns:
        lb = LabelEncoder()
        data[col] = lb.fit_transform(data[col])
        col_vc.append(data[col].max() + 1)
    model = DeepFM(label_cnts=col_vc, embed_dim=128).cuda()
    celoss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.3, random_state=0)
    train_x, test_x, train_y, test_y = torch.tensor(train_x.values).cuda(), torch.tensor(
        test_x.values).cuda(), torch.tensor(train_y.values).cuda(), torch.tensor(test_y.values).cuda()
    train_data = torch.utils.data.TensorDataset(train_x, train_y)
    train_data = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, num_workers=0)
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_data):
            input, label = data
            output = model(input)
            opt.zero_grad()
            loss = celoss(output, label)
            if i % 100 == 0:
                print(f'epoch {epoch} i: {i} train_loss: {loss.item()}')
            loss.backward()
            opt.step()
        model.eval()
        #train_loss = loss(model(train_x), train_y)
        #test_loss = loss(model(test_x), test_y)
        # print(f'epoch:{epoch} train_loss: {train_loss} test_loss: {test_loss}')
    print(1)
