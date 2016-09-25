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
langs = ["en", "es", "pt"]
nlangs = len(langs)


def load_data(maxsize=None):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, lang in enumerate(langs):
        with open("%s.1000.shuffled.train" % lang, "rb") as f:
            lines = [x for x in f.readlines()][:maxsize]
            x_train += lines
            y_train += [i for _ in range(len(lines))]
        with open("%s.1000.shuffled.test" % lang, "rb") as f:
            lines = [x for x in f.readlines()][:maxsize]
            x_test += lines
            y_test += [i for _ in range(len(lines))]

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
(X_train, y_train), (X_test, y_test) = load_data()
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
