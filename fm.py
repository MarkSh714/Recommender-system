import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import metrics
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

epochs = 500
batch_size = 128
learningrate = 0.001
nadam_b1 = 0.9
nadam_b2 = 0.999
epsilon = 1e-07


# def lr_model(x_size):
#     t_model = Sequential()
#     t_model.add(Dense(1, activation="sigmoid", input_shape=(x_size,)))
#     t_model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(learning_rate=learningrate, beta_1=nadam_b1, beta_2=nadam_b2, epsilon=epsilon),
#         metrics=[metrics.mean_squared_logarithmic_error])
#     return (t_model)
#
#
# def lr_second_data(data):
#     columns = list(data.columns)
#     for i in range(len(columns) - 1):
#         for j in range(i + 1, len(columns)):
#             tmp = data[columns[i]] * data[columns[j]]
#             if sum(tmp) / len(data) > 0.01:
#                 data[str(i) + '_' + str(j)] = tmp
#     return data


class FM:
    def __init__(self, k, train, target, test, test_target):
        n = train.shape[1]
        self.w0 = np.random.normal(loc=0, scale=0.1, size=(1, 1))
        self.w1 = np.random.normal(loc=0, scale=0.1, size=(1, n))
        self.v = np.random.normal(loc=0, scale=0.1, size=(n, k))
        self.train = train
        self.target = target
        self.test = test
        self.test_target = test_target
        self.k = k

    def test_loss(self):
        y_pred = np.zeros((self.test.shape[0], 1))
        n = len(self.test)
        for i in range(n):
            y_pred[i] = self.clac(self.test[i, :])
        return log_loss(self.test_target, y_pred)

    def clac(self, x):
        res = self.w0[0][0]
        res += np.dot(self.w1, x)[0]
        for i in range(self.k):
            res += ((np.dot(self.v[:, i], x)) ** 2 - np.dot((self.v[:, 0] ** 2), (x ** 2))) / 2
        return 1 / (1 + np.exp(-1 * res))

    def fit(self, epochs):
        for epoch in range(epochs):
            y_pred = np.zeros((self.train.shape[0], 1))
            n = len(self.train)
            for i in range(n):
                y_pred[i] = self.clac(self.train[i, :])
                grad = self.target[i] - y_pred[i]
                self.w0[0][0] += grad / n
                self.w1 += grad * self.train[i, :].T / n
                for j in range(self.k):
                    tmp = np.dot(self.v[:, j], self.train[j, :])
                    self.v[:, j] += grad * (tmp * self.train[j, :] - self.v[:, j] * self.train[j, :] ** 2) / n
            print("epoch {0}  train loss: {1}; test loss: {2}".format(epoch, log_loss(self.target, y_pred),
                                                                      self.test_loss()))


if __name__ == '__main__':
    epochs = 100
    data = pd.read_csv('data/demo_train.csv', index_col=0)
    target = data['click']
    del data['click']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
    fm = FM(20, X_train.values, y_train.values, X_test.values, y_test.values)
    fm.fit(epochs)

    # model = lr_model(len(X_train.columns))
    # history = model.fit(X_train, y_train,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     shuffle=True,
    #                     verbose=2,
    #                     validation_data=(X_test, y_test))
    # print(1)
