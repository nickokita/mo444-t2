from utils import *
from models.logistic_regression import LogisticRegression
from models.new_softmax_regression import SoftmaxRegression
import sys

def run_softmax():
    ret = load_data(sys.argv[1], int(sys.argv[2]))
    features = decision_boundaries_pol(ret[0], 1)
    target = ret[1]

    if len(sys.argv) == 6:
        _model = sys.argv[5]
    else:
        _model = []

    LR = SoftmaxRegression(features, target, 0.01, classes=unique_classes(target),
                           epoch=int(sys.argv[4]), model=_model, mini_batch_size=128)

    print(LR.model)
    LR.update_model_sgd()
    print(LR.model)
    LR.softmax()
    print(LR.to_classlabel())
    print(LR.get_pctg_right())

def run_logistic():
    if (len(sys.argv) < 7):
        print("Usage: python3 main.py <data> <target_col> <numofclasses> <epoch> <validation_data> <test_data>")
        return

    models = []
    for i in range(0,int(sys.argv[3])):
        print("Running class " + str(i))
        ret = logistic_load(sys.argv[1], int(sys.argv[2]), i)
        features = decision_boundaries_pol(ret[0], 1)
        target = ret[1]
        _model = []

        LR = LogisticRegression(features, target, 0.01, epoch=int(sys.argv[4]), model=_model, mini_batch_size=128,
                                threshold=0.2)

        LR.update_model_mini_batch()
        #LR.update_model_sgd()
        #LR.update_model_batch()

        models.append(LR.get_model())
        #print(LR.get_model())
        #print(LR.simplified_cost_function())
        #print(LR.get_predict())
        LR.get_pctg_right()

    data = load_data(sys.argv[5], int(sys.argv[2]))
    features = decision_boundaries_pol(data[0], 1)
    print("\n\n\n\n\n\n")
    print("Running validation")
    print("Pctg right:")
    print(logistic_validation(models, features, data[1]))
    print("Target:")
    print(data[1])

    data = load_data(sys.argv[6], int(sys.argv[2]))
    features = decision_boundaries_pol(data[0], 1)
    print("\n\n\n\n\n\n")
    print("Running Test")
    print("Pctg right:")
    print(logistic_validation(models, features, data[1]))
    print("Target:")
    print(data[1])


if __name__ == "__main__":
    run_logistic()
#    run_softmax()
