import numpy as np
from keras.models import load_model

model = load_model('en.es.h5')

maxlen = 32


def predict(sentence):
    sentence = bytes(sentence, 'utf8')
    subsentences = []
    subindices = []
    for t in range(0, max(1, len(sentence) - maxlen + 3), 3):
        subsentences.append(sentence[t:t+maxlen])
        subindices.append([t, t+maxlen])
    X = np.zeros((len(subsentences), maxlen, 256), dtype=np.bool)
    for i, subsentence in enumerate(subsentences):
        for t, char in enumerate(subsentence):
            X[i, t, char] = 1

    return list(zip(subindices, model.predict(X).tolist()))
