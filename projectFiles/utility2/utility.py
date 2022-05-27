import numpy as np  # import numpy
import scipy

#Function to load the input file into two data structures:
#1. Matrix of attributes
#2. Array of labels
def load(filename, featuresCols, labelCol):
    floats_array = np.loadtxt(filename, delimiter=",", usecols=featuresCols)
    label_array = np.loadtxt(filename, delimiter=",", usecols=labelCol)

    return [floats_array.T, label_array]



# get iris dataset


def load_iris():
    from sklearn.datasets import load_iris
    D, L = load_iris()['data'].T, load_iris()['target']
    return D, L

# split the dataset D into two parts, one for training and one for validation
# return two matrices, one for training and one for validation


def split_db_2to1(D, L, percTraining = 2.0/3.0,seed=0):
    nTrain = int(D.shape[1]*percTraining)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def split_leave_one_out(D, L, index):
    D_train = np.delete(D, index, 1)
    L_train = np.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)



# get vertical row vector


def vrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))



# get mean of matrix


def compute_mean(X):
    return X.mean(1).reshape((X.shape[0], 1))
# get covariance of matrix X with mean mu


def compute_covariance(X, mu):
    X_centered = X - mu
    return np.dot(X_centered, X_centered.T) / float(X.shape[1])


def computeCovariance(X, mu):
    X_centered = X - mu
    return np.dot(X_centered, X_centered.T) / float(X.shape[1])

# get diagonal covariance of matrix X with mean mu


def compute_diag_covariance(X, mu):
    return compute_covariance(X, mu)*np.identity(X.shape[0])

# get tied covariance matrix of matrix X with mean mu

# get logpdf of sample x with mean mu and covariance C


def logpdf_sample(x, mu, C):
    logpdf = -0.5 * (x.shape[0] * np.log(2*np.pi) + np.linalg.slogdet(C)
                     [1] + np.dot((x-mu).T, np.dot(np.linalg.inv(C), (x-mu))))
    return logpdf.ravel()

# get logpdf of matrix X of samples with mean mu and covariance C


def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * X.shape[0] * np.log(2*np.pi)
    const += -0.5 * np.linalg.slogdet(C)[1]

    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    return np.array(Y).ravel()

# get loglikelihood of matrix X with mean mu and covariance C


def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

# get likelihood of matrix X with mean mu and covariance C


def likelihood(X, mu, C):
    return np.exp(loglikelihood(X, mu, C))

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
    cov = np.zeros(covs[0].shape)
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
    S = np.array(likes)
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
    logS = np.array(loglikes)
    logSJoint = prior * logS
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return logS, logSJoint, logSMarginal, logSPost

# compute accuracy of model
# given LTE = labels of test data
# given S = class conditional probability
# given shape = number of test data


def compute_accuracy(LTE, S):
    accuracy = np.sum(LTE == np.argmax(S, axis=0)) / len(LTE)
    return accuracy

# BAYES STUFF

def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -np.log(pi*Cfn)+np.log((1-pi)*Cfp)
    P = scores > th
    return np.int32(P)

# calculate Binary Confusion Matrix
# inputs: Pred, Labels
# outputs: Confusion Matrix


def compute_conf_matrix_binary(Pred, Labels):
    C = np.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (Labels == 1)).sum()
    return C

# calculate empirical bayes decision
# input : confusion matrix, pi, Cfn, Cfp
# output : empirical bayes decision


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi*Cfn*fnr + (1-pi)*Cfp*fpr

# calculate DCF for a given p
# input: confusion matrix, prior, Cfn, Cfp
# output: DCF


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi*Cfn, (1-pi)*Cfp)

# calculate Normalized DCF for a given prior pi
# input: scores, labels, prior, Cfn, Cfp and threshold (optional)
# output: Normalized DCF


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)

# calculate min DCF for a given prior pi
# input: scores, labels, prior, Cfn, Cfp and threshold (optional)
# output: min DCF


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()

# calculate bayes error data to plot
# input: array of samples, scores, labels
# if minCost is True, compute minDCF and append to y
# if minCost is False, compute DCF and append to y
# output: array of y values to plot


def bayes_error_plot(pArray, scores, labels, minCost=False):
    y = []
    for p in pArray:
        pi = 1./(1.+np.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return np.array(y)

# compute accuracy for SVM
def compute_svm_accuracy(DTE, LTE, wStar):
    DTEEXT = np.vstack([DTE, np.ones((1, DTE.shape[1]))])
    score = np.dot(wStar.T, DTEEXT)
    return np.sum( (score > 0) == LTE) / len(LTE)

if __name__ == "__main__":
    print("This module is not supposed to be run as main")
    exit(1)
