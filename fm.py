import pandas as pd
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

epochs = 500
batch_size = 2048
learningrate = 0.001
nadam_b1 = 0.9
nadam_b2 = 0.999
epsilon = 1e-07


def lr_model(x_size):
    t_model = Sequential()
    t_model.add(Dense(1, activation="sigmoid", input_shape=(x_size,)))
    t_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learningrate, beta_1=nadam_b1, beta_2=nadam_b2, epsilon=epsilon),
        metrics=[metrics.mean_squared_logarithmic_error])
    return (t_model)


if __name__ == '__main__':
    data = pd.read_csv('data/processed_sample_train.csv', index_col=0)
    data = data.sample(200000)
    target = data['click']
    del data['click']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
    model = lr_model(len(X_train.columns))
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=2,
                        validation_data=(X_test, y_test))
    print(1)
