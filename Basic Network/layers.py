import numpy as np
from utils import Parameter, Layer, LossLayer


class Relu(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Relu, self).__init__()
    
    def forward(self, x, **kwargs):
        '''
        x: input to the forward pass which is a numpy 2darray
        kwargs: some extra inputs which are not used in Relu forward pass
        '''
        out = np.where(x > 0, x, 0)
        self.cache = np.where(x > 0, 1, 0)

        return out
        
    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = dout * self.cache

        return dx
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Relu'
    
    
class Sigmoid(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Sigmoid, self).__init__()
        
    def forward(self, x, **kwargs):
        '''
        x: input to the forward pass which is a numpy 2darray
        kwargs: some extra inputs which are not used in Sigmoid forward pass
        '''
        out = None
        e_x = np.exp(-1 * x)
        out = 1 / ( 1 + e_x)
        self.cache = e_x * np.power(out, 2)

        return out
    
    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = dout * self.cache

        return dx
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Sigmoid'

    
class Tanh(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Tanh, self).__init__()
    
    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: some extra inputs which are not used in Tanh forward pass
        '''
        out = None
        e_2x = np.exp(2 * x)
        out = (e_2x - 1) / (e_2x + 1)
        self.cache = 4 * e_2x / np.power((e_2x + 1),2) 

        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        
        '''
        dx = dout * self.cache

        return dx
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Tanh'

    
class SoftmaxCrossEntropy(LossLayer):
    def __init__(self):
        '''
        This layer is inherited from the LossLayer class in utils.py
        '''
        super(SoftmaxCrossEntropy, self).__init__()
        
    def forward(self, x, y):
        '''
        x is the input to the layer which is a numpy 2d-array with shape (N, D)
        y contains the ground truth labels for instances in x which has the shape (N,)
        This function should do two things:
        1) Apply softmax activation on the input
        2) Compute the loss function using cross entropy loss function and returns the loss. 
        '''
        loss = None
        N = y.shape[0]
        e = np.exp(x - np.max(x))
        entropy = (e.T / np.sum(e, axis=1)).T
        loss = np.sum(-np.log(entropy[range(N),y])) / N
        
        gradient_loss_x = entropy / N
        gradient_loss_x[range(N),y] = (entropy[range(N),y] - 1) / N
        self.cache = gradient_loss_x

        return loss
    
    def backward(self):
        dx = None
        dx = self.cache

        return dx
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Softmax Cross-Entropy'
    

class FullyConnected(Layer):
    def __init__(self, initial_value_w, initial_value_b, reg=0.):
        '''
        This layer is inherited from the class Layer in utils.py.
        initial_value_w: The inital value of weights
        initial_value_b: The initial value of biases
        reg: Regularization coefficient or strength used for L2-regularization
        Parameter class (in utils.py) is used for defining paramters
        '''
        super(FullyConnected, self).__init__()
        self.reg = reg
        self.params = {}
        self.params['w'] = Parameter('w', initial_value_w)
        self.params['b'] = Parameter('b', initial_value_b)
    
    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: some extra inputs which are not used in FullyConnected forward pass
        '''
        w, b = self.params['w'].data, self.params['b'].data
        self.cache = [x,w,b]
        out = x @ w + b

        return out
    
    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        reg, w, b = self.reg, self.params['w'].data, self.params['b'].data
        dx, dw, db = None, None, None
        dx = dout @ self.cache[1].T
        dw = self.cache[0].T @ dout + 2 * reg * self.cache[1]
        db = np.sum(dout, axis=0)

        # storing the gradients in grad attribute of parameters
        self.params['w'].grad = dw
        self.params['b'].grad = db
        return dx
    
    def get_params(self):
        '''
        This function overrides the get_params method of class Layer.
        '''
        return self.params
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'FullyConnected {}'.format(self.params['w'].data.shape)
    
    def get_reg(self):
        return self.reg
    
class BatchNormalization(Layer):
    def __init__(self, gamma_initial_value, beta_initial_value, eps=1e-5, momentum=0.9):
        '''
        This layer is inherited from the Layer class in utils.py.
        '''
        super(BatchNormalization, self).__init__()
        self.params = {}
        self.eps = eps
        self.momentum = momentum
        self.running_mean = self.running_var = np.zeros_like(gamma_initial_value)
        
        self.params['gamma'] = Parameter('gamma', gamma_initial_value)
        self.params['beta'] = Parameter('beta', beta_initial_value)
        
    
    def forward(self, x, **kwargs):
        mode = kwargs.pop('mode')
        N, D = x.shape
        running_mean, running_var = self.running_mean, self.running_var
        momentum, gamma, beta = self.momentum, self.params['gamma'].data, self.params['beta'].data
        eps = self.eps
        out =  None
        if mode == 'TRAIN':
            B = x.shape[0]
            x_mean = np.mean(x, axis=0)
            x_var = np.var(x, axis=0)
            running_mean = x_mean *(1 - momentum)  + momentum * running_mean
            running_var = x_var * (1 - momentum) + momentum * running_var
            u = (x - x_mean) / np.sqrt(x_var + eps)
            out = gamma * u + beta
            x_norm = (x - x_mean) / np.sqrt(x_var + eps)
            self.cache = [x, x_norm, x_mean, x_var, eps]

        elif mode == 'TEST':
            u = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * u + beta

        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        self.running_mean = running_mean
        self.running_var = running_var

        return out
    
    def backward(self, dout):
        gamma, beta = self.params['gamma'].data, self.params['beta'].data
        dx, dgamma, dbeta = None, None, None
        
        x, x_norm, mu, var, eps = self.cache
        B = x.shape[0]

        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mu), axis=0) * -.5 / np.power(var + eps,3/2)
        dmu = np.sum(-dx_norm / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2. * (x - mu), axis=0)

        dx = (dx_norm / np.sqrt(var + eps)) + (dvar * 2 * (x - mu) / B) + (dmu / B)
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        
        #saving the gradients in grad attribute of parameters
        self.params['gamma'].grad = dgamma
        self.params['beta'].grad = dbeta
        return dx
    
    def get_params(self):
        return self.params
    
    def reset(self):
        self.running_var = self.running_mean = np.zeros_like(self.running_mean)
        
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Batch Normalization eps={}, momentum={}'.format(self.eps, self.momentum)
    

     
class Dropout(Layer):
    def __init__(self, p):
        '''
        This layer is inherited from the class Layer in utils.py.
        p: probability of keeping a neuron active.
        '''
        super(Dropout, self).__init__()
        self.p = p
    
    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: Some extra input from which mode is used for dropout forward pass
        '''
        mask, out, p = None, None, self.p
        mode = kwargs.pop('mode')
        if mode == 'TRAIN':
            mask = (np.random.rand(*x.shape) < p) / p 
            out = mask * x
            self.cache = mask

        elif mode == 'TEST':
            out = x

        else:
            raise ValueError('Invalide mode')
            
        out = out.astype(x.dtype, copy=False)
        return out
    
    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = None
        mask = np.where(self.cache  != 0, 1,0)
        dx = dout * mask

        return dx
    
    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Dropout p={}'.format(self.p)
