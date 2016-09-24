import numpy as np
from keras.models import load_model

print("Loading model...")
model = load_model('en.es.h5')

maxlen = 32

while True:
    sentence = bytes(input("Sentence: "), 'utf8')
    subsentences = []
    for t in range(0, max(1, len(sentence) - maxlen), 3):
        subsentences.append(sentence[t:t+maxlen])
    X = np.zeros((len(subsentences), maxlen, 256), dtype=np.bool)
    for i, subsentence in enumerate(subsentences):
        for t, char in enumerate(subsentence):
            X[i, t, char] = 1

    print(model.predict(X))
