import numpy
from Printer import Printer
from utilityML.Functions.dimred import *


def split_leave_one_out(D, L, index):
    D_train = numpy.delete(D, index, 1)
    L_train = numpy.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)


def evaluate_by_parameter(classifier, DTR, LTR, DTE, LTE, priors):
    m = DTR.shape[0]
    results = {}
    D = numpy.concatenate((DTR, DTE), axis=1)
    L = numpy.concatenate((LTR, LTE), axis=0)

    while m > 0:
        DP = pca(D, L, m)
        model = classifier(DP[:, 0:DTR.shape[1]], L[:DTR.shape[1]], DP[:, DTR.shape[1]:-1], L[DTR.shape[1]:-1], priors)
        #model = classifier(DTR, LTR, DTE, LTE, priors)
        model.train()
        model.test()
        results[m] = model.accuracy
        m -= 1

    sortedRes = sorted(results.items(), key=lambda x: x[1], reverse=True);
    for (k,v) in sortedRes:
        print(f"m = {k} -> {v * 100:.2f}%")
    return sorted(results.items(), key=lambda x: x[1], reverse=True)