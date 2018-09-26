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
        self.learning_rate = learning_rate

        self.model = model
        if model == []:
            self.set_model()

        self.threshold = threshold
        self.epoch = epoch
        self.pctg = pctg
        self.cost = self.cost_function()
        self.mini_batch_size = mini_batch_size

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

            _arg = (utils.logistic_regression(cur_features, self.model, threshold=self.threshold) - cur_target)
            _arg = _arg ** 2
            _arg = _arg / 2
            sum += _arg

        return sum / m

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

            arg0 = cur_target * math.log(utils.logistic_regression(cur_features, self.model, threshold=self.threshold))
            arg1 = (1 - cur_target) * math.log(1 - utils.logistic_regression(cur_features, self.model, threshold=self.threshold))

            sum += arg0 + arg1

        return - sum / m

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

            predict = utils.logistic_regression(cur_features, self.model, threshold=self.threshold)

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

            for index, m in enumerate(self.model):
                self.model[index] = m - self.learning_rate*self.derivative_cost_function(index, index, index+1)

            cur_cost = self.cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
            elif (cur_cost == self.cost):
                counter += 1
                if (counter == self.epoch*(self.pctg)):
                    i = self.epoch + 1
            else:
                self.model = prev_model

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

            cur_cost = self.cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
            elif (cur_cost == self.cost):
                counter += 1
                if (counter == self.epoch * (self.pctg)):
                    i = self.epoch + 1
            else:
                self.model = prev_model

    # This will update the model using batch gradient descent
    # It won't return anything, meaning it only changes the model within the
    # class
    def update_model_mini_batch(self):
        counter = 0
        for i in range(0, self.epoch):
            prev_model = self.model

            print("Epoch = " + str(i))
            print("Current cost = " + str(self.cost))
            print("Current model = " + str(self.model))

            for index, m in enumerate(self.model):
                self.model[index] = m - self.learning_rate * self.derivative_cost_function(index, index, index + self.mini_batch_size + 1)

            cur_cost = self.cost_function()
            if (cur_cost < self.cost):
                self.cost = cur_cost
            elif (cur_cost == self.cost):
                counter += 1
                if (counter == self.epoch * (self.pctg)):
                    i = self.epoch + 1
            else:
                self.model = prev_model