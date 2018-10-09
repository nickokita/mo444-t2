import numpy
import sys
import math
import utils
from random import random

class SoftmaxRegression:

    # Class constructor
    def __init__(self,
                 features,
                 target,
                 learning_rate,
                 classes,
                 epoch=100,
                 pctg=0.2,
                 model=[],
                 mini_batch_size=0):
        self.features = features
        self.target = target
        self.target = (numpy.arange(numpy.max(target) + 1) == target[:, None]).astype(float)
        self.classes = classes
        self.learning_rate = learning_rate
        self.learning_rate_static = learning_rate
        self.predict = 0

        self.model = model
        if model == []:
            self.set_model()

        self.sk = [] 

        self.epoch = epoch
        self.pctg = pctg

        self.update_sk()
        self.cost = self.cost_function()
        self.mini_batch_size = mini_batch_size

    def softmax(self, prob):
        e_x = (numpy.exp(prob.T) / numpy.sum(prob, axis=1)).T
        return e_x

    def set_model(self):
        for i in range(len(self.classes)):
            _class_param = []
            for i in self.features[0]:
                _class_param.append(1)
            self.model.append(_class_param)

    def get_pctg_right(self):
        right = 0
        wrong = 0

        predict = self.get_predict()
        for ii,ij in enumerate(predict):
            max = 0
            index = 0
            for jj,jk in enumerate(ij):
                if (jk > max):
                    max = jk
                    index = jj
            if (self.target[ii] == index):
                #print(str(ii) + ": Right")
                right += 1
            else:
                #print(str(ii) + ": Wrong")
                wrong += 1

        print("Right: " + str(right))
        print("Wrong: " + str(wrong))



    # Return computed final model
    def get_model(self):
        return self.model

    # Return computed cost
    def get_cost(self):
        return self.cost

    # Return list of predicted targets from features
    def get_predict(self):
        return self.sk.argmax(axis=1)

    def cross_entropy(self):
        epsilon_array = [[sys.float_info.epsilon] * self.sk.shape[1]] * len(self.sk)
        new_sk = self.sk + epsilon_array
        _ce = -numpy.sum(numpy.dot(numpy.log(new_sk).T, (self.target)), axis=1)
        return _ce

    def cost_function(self):
        return numpy.mean(self.cross_entropy())

    def update_sk(self):
        self.sk = self.softmax(numpy.dot(numpy.array(self.features), numpy.array(self.model).T))

    def derivative_cost(self):
        return numpy.dot(numpy.array(self.features).T, (self.sk - self.target)) / len(self.features)

    def update_model_sgd(self):
        for i in range(self.epoch):
            self.update_sk()
            self.model = self.model - self.learning_rate * self.derivative_cost().T
            print(self.cost_function())
