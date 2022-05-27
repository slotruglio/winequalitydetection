from utility import *
import sys
sys.path.append("/Users/slotruglio/Documents/polito/ml/labs/")


# compute multivariate gaussian classifier given
# D     = data
# L     = labels
# K     = number of classes
# prior = prior probability
# log   = True if logpdf is used
# kfold = True if k-fold cross-validation is used
# returns:
# S     = class conditional probability (or logpdf)
# SJoint= matrix of joint density (or log-density)
# SMarginal= matrix of marginal density (or log-density)
# SPost= matrix of posterior density (or log-density)
# accuracy = accuracy of model
# error = error of model

def compute_multi_gaussian_classifier(DTRs, DTE, LTE, prior, log=False):
    means = compute_mean_of_DTRs(*DTRs)
    covs = compute_covariance_of_DTRs(DTRs, means)
    loglikes = compute_logpdf_of_DTE(DTE, means, covs)

    S, SJoint, SMarginal, SPost = compute_probabilities(np.exp(
        loglikes), prior) if not log else compute_logprobabilities(loglikes, prior)
    accuracy = compute_accuracy(LTE, SPost)
    error = 1 - accuracy
    return S, SJoint, SMarginal, SPost, accuracy, error

# compute naive bayes gaussian classifier given
# D     = data
# L     = labels
# K     = number of classes
# prior = prior probability
# log   = True if logpdf is used
# kfold = True if k-fold cross-validation is used
# returns:
# S     = class conditional probability (or logpdf)
# SJoint= matrix of joint density (or log-density)
# SMarginal= matrix of marginal density (or log-density)
# SPost= matrix of posterior density (or log-density)
# accuracy = accuracy of model
# error = error of model


def compute_naive_bayes_classifier(DTRs, DTE, LTE, prior, log=False):
    means = compute_mean_of_DTRs(*DTRs)
    covs = compute_diagcovariance_of_DTRs(DTRs, means)
    loglikes = compute_logpdf_of_DTE(DTE, means, covs)
    S, SJoint, SMarginal, SPost = compute_probabilities(np.exp(
        loglikes), prior) if not log else compute_logprobabilities(loglikes, prior)
    accuracy = compute_accuracy(LTE, SPost)
    error = 1-accuracy
    return S, SJoint, SMarginal, SPost, accuracy, error


# compute Tied Covariance Gaussian classifier given
# D     = data
# L     = labels
# K     = number of classes
# prior = prior probability
# log   = True if logpdf is used
# kfold = True if k-fold cross-validation is used
# returns:
# S     = class conditional probability (or logpdf)
# SJoint= matrix of joint density (or log-density)
# SMarginal= matrix of marginal density (or log-density)
# SPost= matrix of posterior density (or log-density)
# accuracy = accuracy of model
# error = error of model
def compute_tied_gaussian_classifier(DTRs, DTE, LTE, K, prior, shape, log=False):
    means = compute_mean_of_DTRs(*DTRs)
    cov = compute_tied_covariance(DTRs, means, shape)
    loglikes = [logpdf_GAU_ND(DTE, means[i], cov) for i in range(K)]
    S, SJoint, SMarginal, SPost = compute_probabilities(np.exp(
        loglikes), prior) if not log else compute_logprobabilities(loglikes, prior)
    accuracy = compute_accuracy(LTE, SPost)
    error = 1-accuracy
    return S, SJoint, SMarginal, SPost, accuracy, error

# compute tied naive bayes classifier given
# D     = data
# L     = labels
# K     = number of classes
# prior = prior probability
# log   = True if logpdf is used
# kfold = True if k-fold cross-validation is used
# returns:
# S     = class conditional probability (or logpdf)
# SJoint= matrix of joint density (or log-density)
# SMarginal= matrix of marginal density (or log-density)
# SPost= matrix of posterior density (or log-density)
# accuracy = accuracy of model
# error = error of model


def compute_tied_naive_classifier(DTRs, DTE, LTE, K, prior, shape, log=False):
    means = compute_mean_of_DTRs(*DTRs)
    cov = compute_tied_covariance(DTRs, means, shape, naive=True)
    loglikes = [logpdf_GAU_ND(DTE, means[i], cov) for i in range(K)]
    S, SJoint, SMarginal, SPost = compute_probabilities(np.exp(
        loglikes), prior) if not log else compute_logprobabilities(loglikes, prior)
    accuracy = compute_accuracy(LTE, SPost)
    error = 1-accuracy
    return S, SJoint, SMarginal, SPost, accuracy, error

# perform k-fold leave one out cross-validation given:
# model = classifier function to use (ex : compute_naive_bayes_classifier)
# D     = data
# L     = labels
# K     = number of classes
# prior = prior probability
# log   = if True will be used only log functions
# tied  = if True will be used tied covariance
# return error in decimalspe


def leave_one_out_cross_validation(model, D, L, K, prior, log=False, tied=False):
    finalError = 0.
    for i in range(D.shape[1]):
        (DTR, LTR), (DTE, LTE) = split_leave_one_out(D, L, i)
        DTRs = get_DTRs(DTR, LTR, K)
        error = model(DTRs, DTE, [LTE], prior, log=log)[-1] if not tied else model(
            DTRs, DTE, [LTE], K, prior, D.shape[1], log=log)[-1]
        finalError += error
    return finalError/D.shape[1]


# SVM Linear classifier given:
# DTR = data
# LTR = labels
# C = hyperparameter
# K = regularization parameter
# return wStar, primal loss, dual loss, duality gap

def train_SVM_linear(DTR, LTR, C, K=1):
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))])
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = np.dot(DTREXT.T, DTREXT)
    #Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*np.dot(DTR.T, DTR)
    #H = np.exp(-Dist)
    H = mcol(Z) * mrow(Z) * H

    def JDual(alpha):
        Ha = np.dot(H, mcol(alpha))
        aHa = np.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = np.dot(mrow(w), DTREXT)
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * np.linalg.norm(w)**2 + C * loss
    
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds = [(0, C)]*DTR.shape[1],
        factr = 1.0,
        maxiter = 100000,
        maxfun=100000,
    )

    print(_x),
    print(_y)

    wStar = np.dot(DTREXT, mcol(alphaStar) * mcol(Z))
    #print (JPrimal(wStar))
    #print (JDual(alphaStar)[0])
    gap = JPrimal(wStar) - JDual(alphaStar)[0]
    return wStar, JPrimal(wStar), JDual(alphaStar)[0], gap;


if __name__ == "__main__":
    print("This module is not supposed to be run as main")
    exit(1)
