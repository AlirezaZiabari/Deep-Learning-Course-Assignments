import numpy as np

class MLP:
    def __init__(self):
        self.layers = []
        self.mode = 'TRAIN'
        
    def add(self, layer):
        '''
        add a new layer to the layers of model.
        '''
        self.layers.append(layer)
    
    def set_mode(self, mode):
        if mode == 'TRAIN' or mode == 'TEST':
            self.mode = mode
        else:
            raise ValueError('Invalid Mode')
    
    def forward(self, x, y):
        loss, scores = 0, 0
        forward_ans = x
        for index in range(0,len(self.layers) - 1):
            forward_ans = self.layers[index].forward(forward_ans, mode=self.mode)
            if self.layers[index].get_reg() is not None:
                loss += self.layers[index].get_reg() * np.sum(np.power(self.layers[index].params['w'].data,2))
        scores = forward_ans
        loss += self.layers[len(self.layers) - 1].forward(forward_ans, y)

        return scores, loss
        
        
    def backward(self):
        back_ans = self.layers[len(self.layers) - 1].backward()
        for index in range(len(self.layers) - 2, -1, -1):
            back_ans = self.layers[index].backward(back_ans)

            
    def __str__(self):
        '''
        returns a nice representation of model
        '''
        splitter = '===================================================='
        return splitter + '\n' + '\n'.join('layer_{}: '.format(i) + 
                                           layer.__str__() for i, layer in enumerate(self.layers)) + '\n' + splitter
