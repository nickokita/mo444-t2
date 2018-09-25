from utils import logistic_load, decision_boundaries_pol, logistic_regression
from logistic_regression import LogisticRegression
from scikit_logistic_regression import SKLogisticRegression
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: python main.py <data> <targetcol> <target> <epoch>")
        exit(1)

    ret = logistic_load(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    features = decision_boundaries_pol(ret[0], 1)
    target = ret[1]
    print(target)

    if len(sys.argv) == 5:
        _model = sys.argv[4]
    else:
        _model = []

    LR = LogisticRegression(features, target, 0.01, 0.5, epoch=int(sys.argv[4]), model=_model)
    if len(sys.argv) < 5:
        LR.set_model()
    
    LR.update_model_sgd()

    print(LR.get_model())
    print(LR.get_cost())
    print(LR.get_predict())

    #SKLR = SKLogisticRegression(features, target, 0.01, 0.5, 100)
    #SKLR.update_model_sgd()

    #print(SKLR.get_predict())
    print(target)

if __name__ == "__main__":
    main()