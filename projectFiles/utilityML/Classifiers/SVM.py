import numpy
import scipy

from ..Functions.genpurpose import mcol, mrow
from ..Functions.bayes import compute_confusion_matrix_binary, compute_normalized_dcf_binary


def calculate_lbgf(H, DTR, C):
    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = [(0, C)]*DTR.shape[1],
        factr = 1,
        maxiter = 100000,
        maxfun=100000,
    )

    return alphaStar, JDual(alphaStar), LDual(alphaStar)

def train_SVM_linear(DTR, LTR, C, K=1):
    DTREXT = numpy.vstack([DTR, K*numpy.ones((1, DTR.shape[1]))])
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(DTREXT.T, DTREXT)
    #Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    #H = numpy.exp(-Dist)
    H = mcol(Z) * mrow(Z) * H

    def JPrimal(w):
        S = numpy.dot(mrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * numpy.linalg.norm(w)**2 + C * loss

    
    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)
    #print(_x),
    #print(_y)

    wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))
    #print (JPrimal(wStar))
    #print (JDual(alphaStar)[0])

    def get_duality_gap(alpha, w):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        return JPrimal(w) - (- 0.5 * aHa.ravel() + numpy.dot(mrow(alpha), numpy.ones(alpha.size)))

    return wStar, JPrimal(wStar), JDual[0], get_duality_gap(alphaStar, wStar);
	

class SVM_linear:
    def __init__(self, DTR, LTR, DTE, LTE, priors, C=0.1, K=1):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = priors
        self.C = C
        self.K = K

        self.accuracy = 0.
        self.error = 0.
        self.predicted = []

        self.dcf = 0.

    def train(self):
        self.wStar, self.JPrimal, self.JDual, self.dualityGap = train_SVM_linear(self.DTR, self.LTR, self.C, self.K)
    
    def test(self):
        DTEEXT = numpy.vstack([self.DTE, self.K*numpy.ones((1, self.DTE.shape[1]))])
        self.score = numpy.dot(self.wStar.T, DTEEXT)
        self.accuracy = numpy.sum( (self.score > 0) == self.LTE) / len(self.LTE)
        self.error = 1 - self.accuracy
        self.predicted = 1 * (self.score > 0)
        confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.score[0], self.priors[1], 1, 1)
        self.dcf = compute_normalized_dcf_binary(confusion_matrix, self.priors[1], 1, 1)
class SVM_poly:
    def __init__(self, DTR, LTR, DTE, LTE, priors, C=0.1, K=1, degree=2, costant = 0):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = priors
        self.C = C
        self.K = K

        self.degree = degree
        self.costant = costant
        self.predicted = []
        self.accuracy = 0.
        self.error = 0.
        self.dcf = 0.

    def train(self):
        Z = numpy.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1

        H = (numpy.dot(self.DTR.T, self.DTR)+self.costant)**self.degree + self.K**2
        H = mcol(Z) * mrow(Z) * H

        self.alphaStar, self.JDual, self.LDual = calculate_lbgf(H, self.DTR, self.C)

    def test(self):
        Z = numpy.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1
        kernel = (numpy.dot(self.DTR.T, self.DTE)+self.costant)**self.degree + self.K*self.K
        self.score = numpy.sum( numpy.dot(self.alphaStar * mrow(Z), kernel), axis=0 )
        self.predicted = 1*(self.score > 0)
        self.accuracy = numpy.sum( (self.score > 0) == self.LTE) / len(self.LTE)
        self.error = 1 - self.accuracy
        confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.score, self.priors[1], 1, 1)
        self.dcf = compute_normalized_dcf_binary(confusion_matrix, self.priors[1], 1, 1)
class SVM_RBF:
    def __init__(self, DTR, LTR, DTE, LTE, priors, C=0.1, K=1, gamma=1):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = priors

        self.C = C
        self.K = K
        self.gamma = gamma
        self.predicted = []
        self.accuracy = 0.
        self.error = 0.
        self.dcf = 0.

    def train(self):
        Z = numpy.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1

        # kernel function
        kernel = numpy.zeros((self.DTR.shape[1], self.DTR.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTR.shape[1]):
                kernel[i,j] = numpy.exp(-self.gamma*(numpy.linalg.norm(self.DTR[:,i]-self.DTR[:,j])**2)) +self.K**2
        H = mcol(Z) * mrow(Z) * kernel

        self.alphaStar, self.JDual, self.LDual = calculate_lbgf(H, self.DTR, self.C)

    def test(self):
        Z = numpy.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1
        kern = numpy.zeros((self.DTR.shape[1], self.DTE.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTE.shape[1]):
                kern[i,j] = numpy.exp(-self.gamma*(numpy.linalg.norm(self.DTR[:,i]-self.DTE[:,j])**2)) + self.K**2
        
        self.score = numpy.sum( numpy.dot(self.alphaStar * mrow(Z), kern), axis=0 )
        self.predicted = 1*(self.score > 0)
        self.accuracy = numpy.sum( (self.score > 0) == self.LTE) / len(self.LTE)
        self.error = 1 - self.accuracy
        confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.score, self.priors[1], 1, 1)
        self.dcf = compute_normalized_dcf_binary(confusion_matrix, self.priors[1], 1, 1)