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
        self.learning_rate_static = learning_rate
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

    def get_pctg_right(self):
        right = 0
        wrong = 0

        predict = self.get_predict()
        for ii,ij in enumerate(predict[0]):
            max = 0
            index = 0
            for jj,jk in enumerate(ij):
                if (jk > max):
                    max = jk
                    index = jj
            if (self.target[ii] == index):
                print(str(ii) + ": Right")
                right += 1
            else:
                print(str(ii) + ": Wrong")
                wrong += 1

        print("Right: " + str(right))
        print("Wrong: " + str(wrong))


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
                    sum += numpy.log(cur_prob[int(k)])

        return -sum / m

    # Return list of predicted targets from features
    def get_predict(self):
        _list = []
        for i in self.features:
            _list.append(self.softmax_regression())

        return _list

    def cross_entropy(self, k):
        m = len(self.features)
        sum = [0] * len(self.features[0])

        for index,f in enumerate(self.features):
            if (self.target[index] == k):
                sum += numpy.dot((self.sk[index][k] - 1), f)
            else:
                sum += numpy.dot((self.sk[index][k]), f)

        return sum / m

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
            for i in self.classes:
                feat_sk.append(float(self.compute_softmax_score(x, int(i))))
            sk.append(feat_sk)

        ret = []
        for _sk in sk:
            ret.append(self.softmax(_sk))

        self.sk = ret

        return ret


    def update_model_sgd(self):
        prev_cost = self.cost

        for i in range(0, self.epoch):
            print("Epoch = " + str(i))
            print("Current cost = " + str(self.cost))
            #print("Current model = " + str(self.model))

            prev_model = self.model
            prev_cost = self.cost

            for index,m in enumerate(self.model):
                self.softmax_regression()
                _ce = self.cross_entropy(index)

                self.model[index] = m - self.learning_rate * _ce

            self.cost = self.cost_function()

            if (self.cost > prev_cost):
                self.cost = prev_cost
                self.model = prev_model

                self.learning_rate -= self.learning_rate_static*0.01
                if (self.learning_rate <= 0):
                    break
            else:
                self.learning_rate = self.learning_rate_static