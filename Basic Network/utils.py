import numpy as np
import matplotlib.pyplot as plt

class Parameter:
    def __init__(self, name, data):
        '''
        A wrapper for trainable parameters
        '''
        self.name = name
        self.data = data  # value that the parameter holds
        self.grad = None  # the gradient of a scalar value w.r.t. parameter
        
    def zero_grad(self):
        '''
        resets the grad attribute to zero.
        '''
        self.grad = np.zeros_like(self.value)
        
        
class Layer:
    def __init__(self):
        '''
        This class is fully implemented for you.
        Do not change anything in this class!
        Other layers will be inherited from this base class. 
        '''
        self.cache = None
    
    def forward(self, x, **kwargs):
        '''
        x: numpy 2darray with shape (N, D)
        forward pass of the layer
        **kwargs includes some extra parameters which will be useful for some layers (e.g. dropout)
        '''
        pass
    
    def backward(self):
        '''
        backward pass of the layer
        '''
        pass
    
    def get_params(self):
        '''
        gets the parameters of the layer (if there is any)
        '''
        return None
    
    def get_reg(self):
        '''
        gets the regularization strength of the layer (if there is any)
        '''
        return None
    
    
class LossLayer:
    '''
    This class is fully implemented for you.
    Do not change anything in this class!
    LossLayer is a base class for other loss layers which will be used as the last layer of a MLP.
    (In this assignment, only SoftmaxCrossEntropy will be inherited from this base class.)
    '''
    def __init__(self):
        pass
    
    def forward(self, x, y):
        '''
        x: numpy 2darray with shape (N, D)
        y: numpy 1darray with shape (N, )
        forward pass of a loss layer.
        (This method has an extra 'y' comparing to Layer which is used for computing loss.)
        '''
        pass
    
    def backward(self):
        '''
        backward pass of the layer.
        '''
        pass
    
    def get_params(self):
        '''
        get the parameters of the layer (if there is any)
        '''
        return None
    
    def get_reg(self):
        '''
        gets the regularization strength of the layer (if there is any)
        '''
        return None

    
def plot(title, xlabel, ylabel, plots, marker, labels=None):
    '''
    utility function for plotting the results
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    
    if labels is None:
        for plot in plots:
            plt.plot(plot, marker)
    else:
        for plot, label in zip(plots, labels):
            plt.plot(plot, marker, label=label)
            plt.legend(loc='lower center', ncol=len(plots)) 

    