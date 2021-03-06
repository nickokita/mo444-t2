import numpy
import sys
import math
import utils

class LogisticRegression:

    # Class constructor
    def __init__(self, features, target, learning_rate, threshold=0.5,
                 epoch=100, pctg=0.2, model=[], mini_batch_size=0):
        self.features = features
        self.target = target
        self.target_one_hot = (numpy.arange(numpy.max(target) + 1) == target[:, None]).astype(float)
        self.learning_rate = learning_rate
        self.learning_rate_static = learning_rate

        self.model = model
        if model == []:
            self.set_model()

        self.threshold = threshold
        self.epoch = epoch
        self.pctg = pctg
        self.cost = self.simplified_cost_function()
        self.mini_batch_size = mini_batch_size

    def normalized_accuracy(self):
        e = 0.2
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        _predict = self.get_predict()
        for i in range(0, len(self.target_one_hot)):
            for j in range(0, len(self.target_one_hot[i])):
                diff = label[i][j] - _predict[i][j]
                if (label[i][j] == 1):
                    if (diff <= e):
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if (diff <= e - 1):
                        fp = fp + 1
                    else:
                        tn = tn + 1

        return (((tp/(tp+fn))+(tn/(tn+fp)))/len(self.classes))


    # Creates a new model
    def set_model(self):
        for i in self.features[0]:
            self.model.append(1.0)

    # Return computed final model
    def get_model(self):
        return self.model

    # Return computed cost
    def get_cost(self):
        return self.cost

    # Return list of predicted targets from features
    def get_predict(self):
        _list = []
        for i in self.features:
            _list.append(utils.logistic_regression(i, self.model, self.threshold))

        return _list

    # This will compute the cost function for the logistic regression given a
    # model
    # Return:
    #   J: cost function
    def cost_function(self):
        m = len(self.features)
        sum = 0

        for i in range(0, len(self.target)):
            cur_target = self.target[i]
            cur_features = self.features[i]

            _arg = (utils.logistic_regression(cur_features, self.model) - cur_target)
            _arg = _arg ** 2
            _arg = _arg / 2
            sum += _arg

        return sum / m

    def get_pctg_right(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for index,i in enumerate(self.get_predict()):
            if (i == self.target[index] and i == 1):
                tp += 1
            elif (i == self.target[index] and i == 0):
                tn += 1
            elif (i != self.target[index] and i == 1):
                fp += 1
            else:
                fn += 1
        print("True Positive: " + str(tp))
        print("False Positive: " + str(fp))
        print("True Negative: " + str(tn))
        print("False Negative: " + str(fn))

    # This will compute the simplified cost function for the logistic regression given a
    # model
    # Return:
    #   J: cost function
    def simplified_cost_function(self):
        m = len(self.features)
        sum = 0

        for i in range(0, len(self.target)):
            cur_target = self.target[i]
            cur_features = self.features[i]
            _lr = utils.logistic_regression(cur_features, self.model)

            arg0 = cur_target * numpy.log(sys.float_info.epsilon + _lr)
            arg1 = (1 - cur_target) * numpy.log(sys.float_info.epsilon + 1 - _lr)

            sum += arg0 + arg1

        return -sum / m

    # This will return the derivative of cost function for the logistic regression given a
    # model
    # Return:
    #   J: derivative of the cost function
    def derivative_cost_function(self, j, batch_init=0, batch_final=-1):
        m = len(self.features)
        if (batch_final == -1) or (batch_final > m):
            batch_final = m

        sum = 0

        for i in range(batch_init, batch_final):
            cur_target = self.target[i]
            cur_features = self.features[i]

            predict = utils.logistic_regression(cur_features, self.model)

            arg0 = predict - cur_target
            arg0 = arg0 * cur_features[j]

            sum += arg0

        return sum / (batch_final-batch_init)

    # This will update the model using stochastic gradient descent
    # It won't return anything, meaning it only changes the model within the
    # class
    def update_model_sgd(self):
        counter = 0
        for i in range(0, self.epoch):
            prev_model = self.model

            print("Epoch = " + str(i))
            print("Current cost = " + str(self.cost))
            print("Current model = " + str(self.model))

            for j in range(len(self.features)):
                for index, m in enumerate(self.model):
                    self.model[index] = m - self.learning_rate*self.derivative_cost_function(index, j, j+1)

            cur_cost = self.simplified_cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
                self.learning_rate = self.learning_rate_static
            else:
                self.model = prev_model
                self.learning_rate -= self.learning_rate_static * 0.01
                if (self.learning_rate <= 0):
                    break

    # This will update the model using batch gradient descent
    # It won't return anything, meaning it only changes the model within the
    # class
    def update_model_batch(self):
        counter = 0
        for i in range(0, self.epoch):
            prev_model = self.model

            print("Epoch = " + str(i))
            print("Current cost = " + str(self.cost))
            print("Current model = " + str(self.model))

            for index, m in enumerate(self.model):
                self.model[index] = m - self.learning_rate * self.derivative_cost_function(index)

            cur_cost = self.simplified_cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
                self.learning_rate = self.learning_rate_static
            else:
                self.model = prev_model
                self.learning_rate -= self.learning_rate_static * 0.01
                if (self.learning_rate <= 0):
                    break

    # This will update the model using mini batch gradient descent
    # It won't return anything, meaning it only changes the model within the
    # class
    def update_model_mini_batch(self):
        counter = 0
        for i in range(0, self.epoch):
            prev_model = self.model

            if (i % 100 == 0 or i == self.epoch):
                print("Epoch = " + str(i))
                print("Current cost = " + str(self.cost))
                #print("Current model = " + str(self.model))
                self.get_pctg_right()

            for j in range(0, len(self.features), self.mini_batch_size):
                for index, m in enumerate(self.model):
                    self.model[index] = m - self.learning_rate * self.derivative_cost_function(index, j, j + self.mini_batch_size)

            cur_cost = self.simplified_cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
                self.learning_rate = self.learning_rate_static
            else:
                self.model = prev_model
                self.learning_rate -= self.learning_rate_static * 0.01
                if (self.learning_rate <= 0):
                    break
