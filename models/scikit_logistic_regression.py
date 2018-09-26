import numpy
import sys
import math
import utils
from sklearn.linear_model import LogisticRegression

class SKLogisticRegression:

    # Class constructor
    def __init__(self, features, target, learning_rate, threshold=0.5, epoch=100):
        self.features = features
        self.target = target
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.sklr = LogisticRegression()

    # Return computed final model
    def get_model(self):
        return self.sklr.get_params()

    # Return computed final model
    def get_cost(self):
        return self.sklr.score(self.features, self.target)

    # This will update the model using stochastic gradient descent
    # It won't return anything, meaning it only changes the model within the
    # class
    def update_model_sgd(self):
        self.sklr.fit(self.features, self.target)

    def get_predict(self):
        self.sklr.predict(self.features)