from matplotlib import pyplot as plt
import numpy

#region Loading Data

# get data and labels from file
# INPUT: filename - the name of the file to be loaded
#        featuresCols - # of consecutive cols (0-n) to be considered
#        labelCol - the col number of the label (tipically the last col)
# OUTPUT: X - the data matrix (n x m)
#         L - the label vector (1 x m)

def load(filename, featuresCols, labelCol):
    floats_array = numpy.loadtxt(filename, delimiter=",", usecols=featuresCols)
    label_array = numpy.loadtxt(filename, delimiter=",", usecols=labelCol).astype(int)

    return [floats_array.T, label_array]

#endregion

#region particular lines from a matrix

def vrow(v):
    return numpy.array([v])
    
def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

#endregion

#region matrix operations

# get logpdf of matrix X of samples with mean mu and covariance C
def logpdf_GAU_ND(x, mu, C):
    """
    logpdf of Gaussian distribution in N-dimensions
    """

    x = numpy.array(x)
    mu = numpy.array(mu)
    C = numpy.array(C)
    n = len(x)

    C_inverse = numpy.linalg.inv(C)
    C_det_log = numpy.linalg.slogdet(C)[1]
    
    partial_result = -0.5 * n * numpy.log(2 * numpy.pi) - 0.5 * C_det_log

    Y = []

    for i in range(x.shape[1]):
        sub_x = x[:,i:i+1]

        res = partial_result -0.5 * numpy.dot(numpy.dot((sub_x - mu).T, C_inverse), (sub_x - mu))
        Y.append(res)

    return numpy.array(Y).ravel()

# get logpdf of sample x with mean mu and covariance C
def logpdf_sample(x, mu, C):
    logpdf = -0.5 * (x.shape[0] * numpy.log(2*numpy.pi) + numpy.linalg.slogdet(C)
                     [1] + numpy.dot((x-mu).T, numpy.dot(numpy.linalg.inv(C), (x-mu))))
    return logpdf.ravel()


# get mean of matrix
def compute_mean(X):
    return X.mean(1).reshape((X.shape[0], 1))

# get covariance of matrix X with mean mu
def compute_covariance(X, mu):
    X_centered = X - mu
    return numpy.dot(X_centered, X_centered.T) / float(X.shape[1])

# get diagonal covariance of matrix X with mean mu
def compute_diag_covariance(X, mu):
    return compute_covariance(X, mu)*numpy.identity(X.shape[0])


#endregion


#region accuracy computation

# compute accuracy of model
# given LTE = labels of test data
# given S = class conditional probability
def compute_accuracy(LTE, S):
    accuracy = numpy.sum(LTE == numpy.argmax(S, axis=0)) / len(LTE)
    return accuracy

# compute accuracy for SVM
def compute_svm_accuracy(DTE, LTE, wStar):
    DTEEXT = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1]))])
    score = numpy.dot(wStar.T, DTEEXT)
    return numpy.sum( (score > 0) == LTE) / len(LTE)

#endregion

def get_DTRs(DTR, LTR, number_of_classes):

    DTRs = []
    for i in range(number_of_classes):
        DTRs.append(DTR[:, LTR == i])
    return DTRs

def split_db_2to1(D, L, percTraining = 2.0/3.0, seed=0):

    #Per selezionare i sample in maniera randomica,
    #Prendiamo un array di indici dei sample
    #Poi mischiamo questo array tramite una permutazione
    #Infine usiamo tale array per accedere ai sample 
    #(METODO SIMILE AD ANALISI DI IMMAGINI DIGITALI)

    nTrain = int(D.shape[1] * percTraining)
    numpy.random.seed(seed)

    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


if __name__ == "__main__":
    print("This module is not supposed to be run as main")
    exit(1)
