import tensorflow as tf
from utils import normal_initializer, zero_initializer
from layers import ConvLayer, ConvPoolLayer, DeconvLayer
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class AutoEncoder(object):
    def __init__(self):
        # placeholder for storing rotated input images
        self.input_rotated_images = tf.placeholder(dtype=tf.float32,
                                                   shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))
        # placeholder for storing original images without rotation
        self.input_original_images = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))

        # self.output_images: images predicted by model
        # self.code_layer: latent code produced in the middle of network
        # self.reconstruct: images reconstructed by model
        self.code_layer, self.reconstruct, self.output_images = self.build()
        self.loss = self._loss()
        self.opt = self.optimization()

    def optimization(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        return optimizer.minimize(self.loss)

    def encoder(self, inputs):
        # convolutional layer
        conv1 = ConvLayer(input_filters=tf.cast(inputs.shape[3], tf.int32), output_filters=8, act=tf.nn.relu,
                          kernel_size=3, kernel_stride=1, kernel_padding="SAME")
        conv1_act = conv1.__call__(inputs)
        print(conv1_act.shape)
        # convolutional and pooling layer
        conv_pool1 = ConvPoolLayer(input_filters=8, output_filters=8, act=tf.nn.relu,
                                   kernel_size=3, kernel_stride=1, kernel_padding="SAME",
                                   pool_size=3, pool_stride=2, pool_padding="SAME")
        conv_pool1_act = conv_pool1.__call__(conv1_act)
        print(conv_pool1_act.shape)
        # convolutional layer
        conv2 = ConvLayer(input_filters=8, output_filters=16, act=tf.nn.relu,
                          kernel_size=3, kernel_stride=1, kernel_padding="SAME")
        conv2_act = conv2.__call__(conv_pool1_act)
        print(conv2_act.shape)
        # convolutional and pooling layer
        conv_pool2 = ConvPoolLayer(input_filters=16, output_filters=16, act=tf.nn.relu,
                                   kernel_size=3, kernel_stride=1, kernel_padding="SAME",
                                   pool_size=3, pool_stride=2, pool_padding="SAME")
        conv_pool2_act = conv_pool2.__call__(conv2_act)
        print(conv_pool2_act.shape)
        
        conv3 = ConvLayer(input_filters=16, output_filters=32, act=tf.nn.relu,
                          kernel_size=3, kernel_stride=1, kernel_padding="SAME")
        conv3_act = conv3.__call__(conv_pool2_act)
        print(conv3_act.shape)
        
        conv_pool3 = ConvPoolLayer(input_filters=32, output_filters=32, act=tf.nn.relu,
                                   kernel_size=3, kernel_stride=1, kernel_padding="SAME",
                                   pool_size=3, pool_stride=2, pool_padding="SAME")
        conv_pool3_act = conv_pool3.__call__(conv3_act)
        print(conv_pool3_act.shape)
        
        last_conv_dims = conv_pool3_act.shape[1:]
        # make output of pooling flatten

        flatten = tf.reshape(conv_pool3_act, [-1,last_conv_dims[0]*last_conv_dims[1]*last_conv_dims[2]])
        print(flatten.shape)
        weights_encoder = normal_initializer((tf.cast(flatten.shape[1], tf.int32), FLAGS.code_size))
        bias_encoder = zero_initializer((FLAGS.code_size))
        # apply fully connected layer

        dense = tf.matmul(flatten, weights_encoder) + bias_encoder
        print(dense.shape)

        return dense, last_conv_dims

    def decoder(self, inputs, last_conv_dims):

        # apply fully connected layer
        weights_decoder = normal_initializer((FLAGS.code_size, 
                                              tf.cast(last_conv_dims[0]*last_conv_dims[1]*last_conv_dims[2], tf.int32)))
        bias_decoder = zero_initializer((tf.cast(last_conv_dims[0]*last_conv_dims[1]*last_conv_dims[2], tf.int32)))
        decode_layer = tf.nn.relu(tf.matmul(inputs, weights_decoder) + bias_decoder)
        print(decode_layer.shape)
        
        
        
        # reshape to send as input to transposed convolutional layer
        
        deconv_input = tf.reshape(decode_layer, (-1,last_conv_dims[0],last_conv_dims[1],last_conv_dims[2]))

        print(deconv_input.shape)

        # transpose convolutional layer
        deconv1 = DeconvLayer (input_filters=tf.cast(deconv_input.shape[3], tf.int32), output_filters=16, act=tf.nn.relu,
                               kernel_size=3, kernel_stride=2, kernel_padding="SAME")
        deconv1_act = deconv1.__call__(deconv_input)
        print(deconv1_act.shape)
        # transpose convolutional layer
        deconv2 = DeconvLayer (input_filters=16, output_filters=8, act=tf.nn.relu,
                               kernel_size=3, kernel_stride=2, kernel_padding="SAME")
        deconv2_act = deconv2.__call__(deconv1_act)
        print(deconv2_act.shape)
        # transpose convolutional layer
        deconv3 = DeconvLayer (input_filters=8, output_filters=1, act=None,
                               kernel_size=3, kernel_stride=2, kernel_padding="SAME")
        deconv3_act = deconv3.__call__(deconv2_act)
        print(deconv3_act.shape)

        return deconv3_act

    def _loss(self):

        flatten_output = tf.reshape(self.reconstruct, 
                                            [-1,self.reconstruct.shape[1]*self.reconstruct.shape[2]*self.reconstruct.shape[3]])
        flatten_input = tf.reshape(self.input_original_images,[-1,
                                                               self.input_original_images.shape[1]*
                                                               self.input_original_images.shape[2]*
                                                               self.input_original_images.shape[3]])
        
        mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flatten_output, labels=flatten_input))

        return mean_loss

    def build(self):
        # evaluate encoding of images by self.encoder
        code_layer, last_conv_dims = self.encoder(self.input_rotated_images)

        # evaluate reconstructed images by self.decoder
        reconstruct = self.decoder(code_layer, last_conv_dims)

        # apply tf.nn.sigmoid to change pixel range to [0, 1]
        output_images = tf.nn.sigmoid(reconstruct)

        return code_layer, reconstruct, output_images
