import numpy
import sys
import math
import utils

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
        self.classes = classes
        self.learning_rate = learning_rate
        self.predict = 0

        self.model = model
        if model == []:
            self.set_model()

        self.epoch = epoch
        self.pctg = pctg
        self.cost = self.cost_function()
        self.mini_batch_size = mini_batch_size

    def softmax(self, prob):
        e_x = numpy.exp(prob)
        e_x = e_x / e_x.sum(axis=0)
        return e_x

    def set_model(self):
        for i in range(len(self.classes)):
            _class_param = []
            for i in self.features[0]:
                _class_param.append(1)
            self.model.append(_class_param)

    # Return computed final model
    def get_model(self):
        return self.model

    # Return computed cost
    def get_cost(self):
        return self.cost

    def cost_function(self):
        return 10

    # Return list of predicted targets from features
    def get_predict(self):
        _list = []
        for i in self.features:
            _list.append(self.softmax_regression())

        return _list

    def compute_softmax_score(self, x, k):
        # _model is already the transpose of self.model
        _model = numpy.reshape(numpy.array(self.model[k], dtype="float64"), (1, len(self.model[k])))
        _features = numpy.reshape(numpy.array(self.features[x], dtype="float64"), (len(self.features[x]), 1))

        z = numpy.dot(_model, _features)
        print("z[" + str(x) + "," + str(k) + "] = " + str(z))
        return z


    def softmax_regression(self):
        sk = []
        for i in range(len(self.classes)):
            sk.append(float(self.compute_softmax_score(0, i)))

        print(self.softmax(sk))
        return sk