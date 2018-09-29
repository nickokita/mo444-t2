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

        self.sk = self.softmax_regression()

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
        m = len(self.features)
        sum = 0

        for index,f in enumerate(self.features):
            cur_prob = self.sk[index]
            for k in self.classes:
                if self.target[index] == k:
                    sum += numpy.log(cur_prob)

        return -sum / m

    # Return list of predicted targets from features
    def get_predict(self):
        _list = []
        for i in self.features:
            _list.append(self.softmax_regression())

        return _list

    def cross_entropy(self, k):
        m = len(self.features)
        sum = 0

        for index,f in enumerate(self.features):


    def compute_softmax_score(self, x, k):
        # _model is already the transpose of self.model
        _model = numpy.reshape(numpy.array(self.model[k], dtype="float64"), (1, len(self.model[k])))
        _features = numpy.reshape(numpy.array(self.features[x], dtype="float64"), (len(self.features[x]), 1))

        z = numpy.dot(_model, _features)
        return z


    def softmax_regression(self):
        sk = []
        for x in range(len(self.features)):
            feat_sk = []
            for i in range(len(self.classes)):
                feat_sk.append(float(self.compute_softmax_score(x, i)))
            sk.append(feat_sk)

        ret = []
        for _sk in sk:
            ret.append(self.softmax(_sk))

        self.sk = ret

        return ret