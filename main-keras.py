import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop

#max_features = 20000
maxlen = 32
batch_size = 128
nlangs = 2

def load_data(maxsize=None):
    with open("en.10000.shuffled.train", "rb") as f:
        x_en_train = [x for x in f.readlines()][:maxsize]
        y_en_train = [0 for _ in range(len(x_en_train))]
    with open("es.10000.shuffled.train", "rb") as f:
        x_es_train = [x for x in f.readlines()][:maxsize]
        y_es_train = [1 for _ in range(len(x_es_train))]
    with open("en.10000.shuffled.test", "rb") as f:
        x_en_test = [x for x in f.readlines()][:maxsize]
        y_en_test = [0 for _ in range(len(x_en_test))]
    with open("es.10000.shuffled.test", "rb") as f:
        x_es_test = [x for x in f.readlines()][:maxsize]
        y_es_test = [1 for _ in range(len(x_es_test))]

    x_train = x_en_train + x_es_train
    y_train = y_en_train + y_es_train
    x_test = x_en_test + x_es_test
    y_test = y_en_test + y_es_test

    x_train_sliced = []
    y_train_sliced = []
    for i, sentence in enumerate(x_train):
        for t in range(0, len(sentence) - maxlen, 3):
            x_train_sliced.append(sentence[t:t+maxlen])
            y_train_sliced.append(y_train[i])

    x_test_sliced = []
    y_test_sliced = []
    for i, sentence in enumerate(x_test):
        for t in range(0, len(sentence) - maxlen, 3):
            x_test_sliced.append(sentence[t:t+maxlen])
            y_test_sliced.append(y_test[i])

    return (x_train_sliced, y_train_sliced), (x_test_sliced, y_test_sliced)

print('Loading data...')
(X_train, y_train), (X_test, y_test) = load_data(1000)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Vectorization...')
X = np.zeros((len(X_train), maxlen, 256), dtype=np.bool)
y = np.zeros((len(y_train), nlangs), dtype=np.bool)
for i, sentence in enumerate(X_train):
    for t, char in enumerate(sentence):
        X[i, t, char] = 1
    y[i, y_train[i]] = 1
Xv = np.zeros((len(X_test), maxlen, 256), dtype=np.bool)
yv = np.zeros((len(y_test), nlangs), dtype=np.bool)
for i, sentence in enumerate(X_test):
    for t, char in enumerate(sentence):
        Xv[i, t, char] = 1
    yv[i, y_test[i]] = 1
print('X shape:', X.shape)
print('y shape:', y.shape)

print('Build model...')
model = Sequential()
model.add(GRU(256, dropout_W=0.2, dropout_U=0.2, input_shape=(maxlen, 256)))
model.add(Dense(nlangs))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X, y, batch_size=batch_size, nb_epoch=5,
          validation_data=(Xv, yv))
model.save('en.es.h5')
score, acc = model.evaluate(Xv, yv,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
