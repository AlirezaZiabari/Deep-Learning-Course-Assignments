import numpy as np


class Perceptron:
    def __init__(self, feature_dim, num_classes):
        """
        in this constructor you have to initialize the weights of the model with zeros. Do not forget to put 
        the bias term! 
        """
        self.weights = np.zeros((num_classes, feature_dim + 1))
        pass
        
    def train(self, feature_vector, y):
        """
        this function gets a single training feature vector (feature_vector) with its label (y) and adjusts 
        the weights of the model with perceptron algorithm. 
        Hint: use self.predict() in your implementation.
        """
        prediction = self.predict(feature_vector)
        if prediction != y:
            self.weights[y] += feature_vector
            self.weights[prediction] -= feature_vector
        pass

    def predict(self, feature_vector):
        """
        returns the predicted class (y-hat) for a single instance (feature vector).
        Hint: use np.argmax().
        """
        return np.argmax(self.weights.dot(feature_vector))
        pass
                  
