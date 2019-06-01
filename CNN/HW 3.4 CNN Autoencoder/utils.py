import tensorflow as tf
import numpy as np
from DatasetReader import read_hoda_dataset
from scipy.ndimage.interpolation import rotate


flags = tf.app.flags
FLAGS = flags.FLAGS


def rotate_transform(images, min_deg, max_deg):
    num_images = images.shape[0]
    # rotate all input images by some degree in range of [min-deg, max-deg]
    # clockwise or counterclockwise which is set randomly
    rotated_images = [rotate(images[i, :, :, :], np.random.randint(min_deg, max_deg + 1) *
                             (np.random.randint(0, 2) * 2 - 1),
                             reshape=False) for i in range(num_images)]
    rotated_images = np.stack(rotated_images, axis=0)
    return rotated_images


def load_data():
    # Set seed
    seed = 123
    np.random.seed(seed)

    # Loading dataset
    print("Loading the whole dataset...")

    x_train, y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                         images_height=32,
                                         images_width=32,
                                         one_hot=False,
                                         reshape=False)

    x_test, y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                       images_height=32,
                                       images_width=32,
                                       one_hot=False,
                                       reshape=False)

    # concat to a whole dataset
    x, y = np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])

    # shuffle images
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    # set train and test size
    train_size = int(0.8 * x.shape[0])

    # selecting indices of train, test set
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    x_train_rotated = rotate_transform(x_train, 30, 45)
    x_test_rotated = rotate_transform(x_test, 30, 45)

    x_train_rotated[x_train_rotated >= 0.5] = 1.
    x_train_rotated[x_train_rotated < 0.5] = 0.

    x_test_rotated[x_test_rotated >= 0.5] = 1.
    x_test_rotated[x_test_rotated < 0.5] = 0.

    return x_train, x_train_rotated, y_train, x_test, x_test_rotated, y_test


def normal_initializer(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, mean = 0, stddev=0.01), name=name)



def zero_initializer(shape, name=None):
    return tf.Variable(tf.zeros(shape), name=name) 
