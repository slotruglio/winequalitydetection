import numpy as numpy  # import numpy
import scipy



def split_leave_one_out(D, L, index):
    D_train = numpy.delete(D, index, 1)
    L_train = numpy.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)

# !!! to check
def mrow(v):
    return v.reshape((1, v.size))


# get loglikelihood of matrix X with mean mu and covariance C
def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

# get likelihood of matrix X with mean mu and covariance C


def likelihood(X, mu, C):
    return numpy.exp(loglikelihood(X, mu, C))

# separate DTR by class


def get_DTRs(DTR, LTR, number_of_classes):
    DTRs = []
    for i in range(number_of_classes):
        DTRs.append(DTR[:, LTR == i])
    return DTRs

# compute mean of each class
# return a list of means


def compute_mean_of_DTRs(*DTRs):
    means = []
    for DTR in DTRs:
        means.append(compute_mean(DTR))
    return means
# compute covariance of each class
# return a list of covariances


def compute_covariance_of_DTRs(DTRs, means):
    covs = []
    for i in range(len(DTRs)):
        covs.append(compute_covariance(DTRs[i], means[i]))
    return covs
# compute diagonal covariance of each class
# return a list of diagonal covariances


def compute_diagcovariance_of_DTRs(DTRs, means):
    covs = []
    for i in range(len(DTRs)):
        covs.append(compute_diag_covariance(DTRs[i], means[i]))
    return covs

# compute tied covariance
# return a list of tied covariances


def compute_tied_covariance(DTRs, means, shape, naive=False):
    covs = compute_covariance_of_DTRs(
        DTRs, means) if not naive else compute_diagcovariance_of_DTRs(DTRs, means)
    cov = numpy.zeros(covs[0].shape)
    for i in range(len(covs)):
        cov += DTRs[i].shape[1]*covs[i]
    return cov/shape

# compute logpdf of each class
# return a list of logpdfs


def compute_logpdf_of_DTE(DTE, means, covs):
    logpdfs = []
    for i in range(len(means)):
        logpdfs.append(logpdf_GAU_ND(DTE, means[i], covs[i]))
    return logpdfs
# compute probabilities given likelihood and prior
# S = class conditional probability
# SJoint = matrix of joint density
# SMarginal = matrix of marginal density
# SPost = matrix of posterior density


def compute_probabilities(likes, prior):
    S = numpy.array(likes)
    SJoint = prior * S
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return S, SJoint, SMarginal, SPost

# compute probabilites given logpdf and prior
# logS = class conditional logpdf
# logSJoint = matrix of joint log-density
# logSMarginal = matrix of marginal log-density
# logSPost = matrix of posterior log-density


def compute_logprobabilities(loglikes, prior):
    logS = numpy.array(loglikes)
    logSJoint = prior * logS
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return logS, logSJoint, logSMarginal, logSPost

# BAYES STUFF

def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -numpy.log(pi*Cfn)+numpy.log((1-pi)*Cfp)
    P = scores > th
    return numpy.int32(P)

# calculate empirical bayes decision
# inumpyut : confusion matrix, pi, Cfn, Cfp
# output : empirical bayes decision


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi*Cfn*fnr + (1-pi)*Cfp*fpr

# calculate DCF for a given p
# inumpyut: confusion matrix, prior, Cfn, Cfp
# output: DCF


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi*Cfn, (1-pi)*Cfp)

# calculate Normalized DCF for a given prior pi
# inumpyut: scores, labels, prior, Cfn, Cfp and threshold (optional)
# output: Normalized DCF


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)

# calculate min DCF for a given prior pi
# inumpyut: scores, labels, prior, Cfn, Cfp and threshold (optional)
# output: min DCF


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()

# calculate bayes error data to plot
# inumpyut: array of samples, scores, labels
# if minCost is True, compute minDCF and append to y
# if minCost is False, compute DCF and append to y
# output: array of y values to plot


def bayes_error_plot(pArray, scores, labels, minCost=False):
    y = []
    for p in pArray:
        pi = 1./(1.+numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return numpy.array(y)

if __name__ == "__main__":
    print("This module is not supposed to be run as main")
    exit(1)
