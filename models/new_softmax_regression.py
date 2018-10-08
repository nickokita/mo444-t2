import numpy
import sys
import math
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
        self.real_target = target
        self.target = (numpy.arange(numpy.max(target) + 1) == target[:, None]).astype(float)
        self.classes = classes
        if model == []:
            for i in range(len(self.classes)):
                _class_param = []
                for i in features[0]:
                    _class_param.append(1.0)
                model.append(_class_param)

        self.model = numpy.array(model)
        self.features = numpy.array(features)

        self.scores = self.softmax()
        self.learning_rate = learning_rate
        self.learning_rate_static = learning_rate
        self.epoch = epoch

        self.cost = self.compute_cost()

        return

    def net_input(self):
        return ((self.model).dot(self.features.T))

    def get_pctg_right(self):
        right = 0
        wrong = 0

        self.softmax()
        predict = self.to_classlabel()
        print(predict)
        print(self.real_target)

        for ii,ij in enumerate(numpy.nditer(predict)):
            if (int(self.real_target[ii]) == ij):
                #print(str(ii) + ": Right")
                right += 1
            else:
                #print(str(ii) + ": Wrong")
                wrong += 1

        print("Right: " + str(right))
        print("Wrong: " + str(wrong))

    def softmax(self):
        z = self.net_input()
        self.scores = numpy.array((numpy.exp(z - numpy.max(z)).T / numpy.sum(numpy.exp(z - numpy.max(z)), axis=1)))
        return self.scores

    def to_classlabel(self):
        return self.scores.argmax(axis=1)

    def cross_entropy(self):
        return - numpy.sum(numpy.log(self.scores) * (self.target), axis=1)

    def compute_cost(self):
        return numpy.mean(self.cross_entropy())

    def derivative_cost(self, j):
        sum = [0.0] * len(self.features[0])
        sum = numpy.array(sum)

        _features = (numpy.array(self.features))

        for i in range(len(self.features)):
            _feature = _features[i]
            _scores = numpy.array(self.scores[i])
            _target = numpy.array(self.target[i])

            val = _feature.T.dot(_scores[j] - _target[j])

            sum += val

        return sum / len(self.features)

    def update_model_sgd(self):
        for i in range(self.epoch):
            prev_model = self.model

            print("Epoch = " + str(i))
            print("Current cost = " + str(self.cost))
            print(self.get_pctg_right())
            #print("Current model = " + str(self.model))

            for j in range(0, len(self.classes)):
                self.model[j] -= self.learning_rate * self.derivative_cost(j)

            cur_cost = self.compute_cost()
            #if (cur_cost <= self.cost):
            self.cost = cur_cost
            self.learning_rate = self.learning_rate_static
            #else:
            #    self.model = prev_model
            #    self.learning_rate -= self.learning_rate_static * 0.01
            #    if (self.learning_rate <= 0):
            #        break