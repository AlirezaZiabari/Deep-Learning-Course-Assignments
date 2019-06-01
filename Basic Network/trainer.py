import numpy as np
import optim

class Trainer:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.print_every = kwargs.pop('print_every', 100)
        
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)
        
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
            
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()
        
    def _reset(self):
        # Set up some variables for book-keeping
        self.epoch = 0  
        self.best_val_acc = 0
        
        # A dictionary for saving the best paramaters based on validation set accuracy during training.
        self.best_params = {}
        
        # During training, the losses and accuracies will be recorded in a list to be plotted after training is complete
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Making a deep copy of the optim_config for each parameter.
        # trainer keeps this deep copy in self.optim_configs and pass 
        # the optim config of each parameter to the optimizer function.
        # The key of the dictionary is the layer index along with the name of the parameter.
        # This kind of naming is inspired by tensorflow namescopes.
        self.optim_configs = {}
        for i, layer in enumerate(self.model.layers):
            if layer.get_params() is not None:
                for k, p in layer.params.items():
                    self.optim_configs['layer{}/{}'.format(i, k)] = {k: v for k, v in self.optim_config.items()}
        
    
    def _train_step(self):
        """
        Make a single gradient update. 
        In a single training step one forward and one backward pass is done.
        This function is called by train() and should not be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        # Compute loss and gradient
        self.model.set_mode('TRAIN')
        scores, loss = self.model.forward(X_batch, y_batch)  # forward pass
        self.loss_history.append(loss)  # Recording the loss to be plotted later
            
        # gradient of the computed loss w.r.t to parameters will be saved in the grad attribute of the parameters 
        # after calling model.backward()
        self.model.backward()
        
        # Perform a parameter update
        for i, layer in enumerate(self.model.layers):
            if layer.get_params() is not None:
                for k, param in layer.params.items():
                    config = self.optim_configs['layer{}/{}'.format(i, k)]
                    next_param, next_config = self.update_rule(param.data, param.grad, config)
                    self.model.layers[i].params[k].data = next_param
                    self.optim_configs['layer{}/{}'.format(i, k)] = next_config
    
    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        
        for t in range(num_iterations):
            self._train_step()
            
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %.2f%%; val_acc: %.2f%%' % (
                           self.epoch, self.num_epochs, 100*train_acc, 100*val_acc))

                # Keep track of the best model based on validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for i, layer in enumerate(self.model.layers):
                        if layer.get_params() is not None:
                            for k, param in layer.params.items():
                                self.best_params['layer{}/{}'.format(i, k)] = param.data.copy()

        # Replacing model's parameters with the best_params at the end of training
        for i, layer in enumerate(self.model.layers):
            if layer.get_params() is not None:
                for k in layer.params.keys():
                    self.model.layers[i].params[k].data = self.best_params['layer{}/{}'.format(i, k)]
                    
 
    def check_accuracy(self, X, y, num_samples=None):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        
        num_batches = N // self.batch_size
        
        if N % self.batch_size != 0:
            num_batches += 1
        y_pred = []
        
        self.model.set_mode('TEST')
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            scores, _ = self.model.forward(X[start:end], y[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc
        