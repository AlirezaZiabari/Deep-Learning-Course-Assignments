from DatasetReader import read_hoda_dataset
import numpy as np
import tensorflow as tf


def load_data():
    # Set seed
    seed = 123
    np.random.seed(seed)

    # Loading dataset
    print("Loading the whole dataset...")

    x_train, y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                         images_height=32,
                                         images_width=32,
                                         one_hot=True,
                                         reshape=True)

    x_test, y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                       images_height=32,
                                       images_width=32,
                                       one_hot=True,
                                       reshape=True)

    # concat to a whole dataset
    x, y = np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])

    # shuffle images
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    # set train and test size
    train_size = int(0.7 * x.shape[0])
    test_size = int(0.1 * x.shape[0])

    # selecting indices of train, val, test set
    train_idx = idx[:train_size]
    val_idx = idx[train_size: -test_size]
    test_idx = idx[-test_size:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


def glorot_initializer(shape, name=None):
    #####################################################################################
    # TODO: Implement the uniform glorot initializer                                    #
    # First you have to define the range of initialization based on the shape of tensor #
    # Then use tf.random_uniform to define initial value                                #
    # Feed the initial value to tf.Variable                                             #
    # Return the defined tf.Variable                                                    #
    #####################################################################################
    initialization = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.random_uniform(shape, minval=-initialization, maxval=initialization),name=name)
    #####################################################################################
    #                                 END OF YOUR CODE                                  #
    #####################################################################################


def normal_initializer(shape, stddev, name=None):
    #####################################################
    # TODO: Implement the normal initializer            #
    # Set mean to zero and standard deviation to stddev #
    # Use tf.truncated_normal to define initial value   #
    # Feed the initial value to a tf.Variable           #
    # Return the defined tf.Variable                    #
    #####################################################
    return tf.Variable(tf.truncated_normal(shape, mean = 0.0, stddev = stddev),name=name)
    #####################################################
    #                 END OF YOUR CODE                  #
    #####################################################


def zero_initializer(shape, name=None):
    ###########################################
    # TODO: Implement the zero initializer    #
    # Use tf.zeros to define initial value    #
    # Feed the initial value to a tf.Variable #
    # Return the defined tf.Variable          #
    ###########################################
    return tf.Variable(tf.zeros(shape),name=name)
    ###########################################
    #           END OF YOUR CODE              #
    ###########################################
