import numpy as np
from sklearn.model_selection import train_test_split


def unpickle(file_name):
    mesra1_with_different_length, mesra2_with_different_length = [], []
    max_mesra1, max_mesra2 = 0, 0
    mesra1, mesra2 = [], []
    dic = ["_BOM_", "_PAD_", "_EOM_"]
    with open(file_name, 'r') as f:
        for b in f:
            m1, m2 = b.split(",")
            split = list(m1)
            mesra1_with_different_length.append(split)
            split = list(m2)
            split = split[:len(split[-1])-3]
            mesra2_with_different_length.append(split)
            if max_mesra1 < len(mesra1_with_different_length[-1]):
                max_mesra1 = len(mesra1_with_different_length[-1])
            if max_mesra2 < len(mesra2_with_different_length[-1]):
                max_mesra2 = len(mesra2_with_different_length[-1])
            dic += mesra1_with_different_length[-1] + mesra2_with_different_length[-1]
        for (m1, m2) in zip(mesra1_with_different_length, mesra2_with_different_length):
            mesra1.append(["_PAD_"] * (max_mesra1 - len(m1)) + ["_BOM_"] + m1 + ["_EOM_"])
            mesra2.append(["_BOM_"] + m2 + ["_EOM_"] + ["_PAD_"] * (max_mesra2 - len(m2)))
        return np.array(mesra1), np.array(mesra2), list(set(dic))


def load_ferdosi(dir):
    mesra1, mesra2, dictionary = unpickle(dir)
    mesra1_train, mesra1_test, mesra2_train, mesra2_test = train_test_split(mesra1, mesra2, test_size=0.1,
                                                                            random_state=42, shuffle=True)
    return mesra1_train, mesra1_test, mesra2_train, mesra2_test, dictionary


def load_ferdosi_for_wrod2vec(dir):
    sentences = []
    with open(dir, 'r') as f:
        for b in f:
            m1, m2 = b.split(",")
            split = m1.split(" ")
            sentences.append([elem for elem in split if elem!=''])
            split = m2.split(" ")
            split[-1] = split[-1][:len(split[-1])-1]
            sentences.append([elem for elem in split if elem!=''])
        return np.array(sentences)
