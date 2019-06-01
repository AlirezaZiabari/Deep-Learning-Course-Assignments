import numpy as np
import os
try:
    from six.moves import cPickle as pickle
except:
    import pickle


def unpickle(file_name):
    with open(file_name, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        return dict


def load_cifar10(dir):
    data_batches_names = ['data_batch_{}'.format(i) for i in range(1, 6)]
    X_train, y_train = [], []

    # loading training data and labels
    for batch_name in data_batches_names:
        data_dict = unpickle(os.path.join(dir, batch_name))
        X_train.append(data_dict['data'])
        y_train.append(data_dict['labels'])

    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_train = X_train.astype('float')
    
    # loading test data and labels
    data_dict = unpickle(os.path.join(dir, 'test_batch'))
    X_test, y_test = data_dict['data'], np.array(data_dict['labels'])
    X_test = X_test.astype('float')
    
    return X_train, y_train, X_test, y_test
