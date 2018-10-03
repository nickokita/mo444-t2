import numpy
import csv
import sys
import math

def load_data(data_fp, _target):
    print("Loading data")

    # Opens file and loads CSV into a list then convert it to a numpy array
    file = open(data_fp)
    reader = csv.reader(file, delimiter=",")
    _list = list(reader)
    _list.pop(0)
    _list = numpy.array(_list, dtype=float)

    target = _list[:, _target]
    features = numpy.delete(_list, _target, 1)

    ret = [features, target]

    # Normalize data - pixel max color is 255
    #for index,r in enumerate(ret[0]):
    #    ret[0][index] = r/255

    return ret

def logistic_load(data_fp, _targetcol, _target):
    print("Loading data")

    # Opens file and loads CSV into a list then convert it to a numpy array
    file = open(data_fp)
    reader = csv.reader(file, delimiter=",")
    _list = list(reader)
    _list.pop(0)
    _list = numpy.array(_list, dtype=float)

    target = _list[:, _targetcol]
    features = numpy.delete(_list, _targetcol, 1)

    for index, f in enumerate(target):
        if (f == _target):
            target[index] = 1
        else:
            target[index] = 0

    ret = [features, target]
    # Normalize data - pixel max color is 255
    #for index, r in enumerate(ret[0]):
    #    ret[0][index] = r / 255.0

    return ret

# Allows the user to select a polynomial as a model
# Input:
#   x: features
#   power: the power of the polynomial
# Return:
#   x: features to the power of power input
#
# For example if the input is x: [ x0, x1, x2 ], power: 3
# It will return:
#   [ 1, x0, x1, x2, x0^2, x1^2, x2^2, x0^3, x1^3, x2^3 ]
def decision_boundaries_pol(x, power):
    ret_x = []

    _list = [ 1 ]
    for i in x:
        for k in range(1, power + 1):
            for j in i:
                _list.append(j ** k)
        ret_x.append(_list)
        _list = [ 1 ]

    return ret_x


# Calculates logistic regression function h_{theta}(x)
# Input:
#   Features
#   Model
# Return:
#   Logistic regression
def logistic_regression(features, model, threshold=-1):
    # Theta^t * x
    # _model already transposes the model array
    _model = numpy.reshape(numpy.array(model, dtype="float64"), (1, len(model)))
    _features = numpy.reshape(numpy.array(features, dtype="float64"), (len(features), 1))

    z = numpy.dot(_model, _features)
    try:
        arg0 = 1 + math.exp(-z)
        arg0 = 1 / arg0
    except OverflowError:
        arg0 = 0


    if (threshold != -1):
        if (arg0 > threshold):
            arg0 = 1
        else:
            arg0 = 0

    return arg0

# Return the list of unique classes
def unique_classes(target):
    _list = []
    for i in target:
        if (not i in _list):
            _list.append(i)

    return _list