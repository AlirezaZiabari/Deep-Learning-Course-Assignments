import tensorflow as tf
from layers import DenseLayer


flags = tf.app.flags
FLAGS = flags.FLAGS


class Dense(object):
    def __init__(self, num_hidden, weight_initializer, bias_initializer,
                 act=tf.nn.sigmoid, logging=False, stddev=None):

        super(Dense, self).__init__()

        # saving batch info in placeholders
        self.placeholders = {'batch_images': tf.placeholder(shape=[None, 32 * 32], dtype=tf.float32),
                             'batch_labels': tf.placeholder(shape=[None, 10], dtype=tf.float32)}

        # storing input dimensions of all layers
        self.num_hidden = [self.placeholders['batch_images'].shape.as_list()[1]] + num_hidden

        # setting types of weight and bias initializer
        self.stddev = stddev
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # setting activation functions
        self.act = act

        # build layers
        self.layers = []
        self._build()

        # list of activations 
        self.activations = [self.placeholders['batch_images']]
        for layer in self.layers:
            self.activations.append(layer.__call__(self.activations[-1]))

        # output of last activations
        self.output = self.activations[-1]

        # defining loss and accuracy and optimizer
        self.loss = self._loss()
        self.acc = self._accuracy()

        # log vars
        if logging:
            self.log_vars()

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        self.training = self.optimizer.minimize(self.loss)


    def log_vars(self):
        for i, layer in enumerate(self.layers):
            tf.summary.histogram(name='bias_{}'.format(i + 1), values= layer.vars['bias'])
            tf.summary.histogram(name='weight_{}'.format(i + 1), values= layer.vars['weight'])
            

    def _build(self):
        for i in range(1, len(self.num_hidden)):
            # set last layer activation as linear function otherwise use self.act
            if i == len(self.num_hidden) - 1:
                act = (lambda x: x)
            else:
                act = self.act

            layer = DenseLayer(input_dim=self.num_hidden[i - 1], 
                               output_dim=self.num_hidden[i], 
                               act=act,
                               weight_initializer=self.weight_initializer, 
                               bias_initializer=self.bias_initializer, 
                               stddev=self.stddev)

            # add layer to layers list
            self.layers.append(layer)

    def _loss(self):
        # cross-entropy loss over logits and labels
        batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.placeholders['batch_labels'], logits=self.output)

        # L2 regularization on weights
        l2_loss = FLAGS.weight_decay * (tf.nn.l2_loss(self.layers[0].vars['weight']) +
                                        tf.nn.l2_loss(self.layers[1].vars['weight']))

        # compute average of batch loss plus l2 loss
        avg_loss = tf.add(l2_loss, tf.reduce_mean(batch_loss))

        # save summary scalar
        tf.summary.scalar('loss', avg_loss)

        return avg_loss

    def _accuracy(self):

        # prediction of output on batch
        batch_predictions = tf.argmax(self.output, axis=1)

        # true labels
        correct_predictions = tf.argmax(self.placeholders['batch_labels'],axis=1)

        # compute accuracy using reduce_mean
        avg_acc = tf.reduce_mean(tf.cast(tf.equal(batch_predictions, correct_predictions),tf.float64))

        # save summary scalar
        tf.summary.scalar('acc', avg_acc)

        return avg_acc
