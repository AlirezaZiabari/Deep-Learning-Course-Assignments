import tensorflow as tf
from utils import zero_initializer, normal_initializer


class ConvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(ConvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        
        self.conv_filter = normal_initializer((self.kernel_size, self.kernel_size, 
                                               self.input_filters, self.output_filters), name="conv_filter" )
        self.conv_bias = zero_initializer ((self.output_filters), name="conv_bias")
  
        self.conv_output = tf.nn.conv2d(inputs, self.conv_filter, [1, self.kernel_stride, self.kernel_stride, 1]
                                        , self.kernel_padding)
        # Add bias and apply activation
        self.total_output = self.act(self.conv_output + self.conv_bias)

        return self._call(self.total_output)


class ConvPoolLayer(ConvLayer):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding,
                 pool_size, pool_stride, pool_padding):

        # Calling ConvLayer constructor will store convolutional section config
        super(ConvPoolLayer, self).__init__(input_filters, output_filters, act,
                                            kernel_size, kernel_stride, kernel_padding)

        # size of kernel in pooling
        self.pool_size = pool_size

        # size of stride in pooling
        self.pool_stride = pool_stride

        # type of padding in pooling
        self.pool_padding = pool_padding

    def _call(self, inputs):

        self.pooling_output = tf.nn.max_pool(inputs, [1, self.pool_size, self.pool_size, 1]
                                             , [1, self.pool_stride, self.pool_stride, 1]
                                             , self.pool_padding)

        return self.pooling_output


class DeconvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(DeconvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # transposed convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of transposed convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def __call__(self, inputs):

        self.deconv_filter = normal_initializer((self.kernel_size, self.kernel_size,
                                                 self.output_filters, self.input_filters), 
                                                name="deconv_filter" )
        self.deconv_bias = zero_initializer ((self.output_filters), name="deconv_bias")

        # input height and width
        input_height = inputs.get_shape().as_list()[1]
        input_width = inputs.get_shape().as_list()[2]

        
        if self.kernel_padding == 'SAME':
            output_height = input_height * self.kernel_stride
            output_width = input_width * self.kernel_stride
        elif self.kernel_padding == 'VALID':
            output_height = (input_height - 1) * self.kernel_stride + self.kernel_size
            output_width = (input_width - 1) * self.kernel_stride + self.kernel_size
        else:
            raise Exception('No such padding')

        self.deconv_output = tf.nn.conv2d_transpose(inputs, self.deconv_filter, 
                                                    [tf.shape(inputs)[0], output_height, output_width, self.output_filters], 
                                                    [1, self.kernel_stride, self.kernel_stride, 1], self.kernel_padding)
        # Add bias and apply activation
        if self.act is None:
            self.total_output = self.deconv_output
        else:    
            self.total_output = self.act(self.deconv_output + self.deconv_bias)

        return self.total_output
