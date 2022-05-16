import numpy

def vrow(v):
    return numpy.array([v])
    
def mcol(v):
    return v.reshape((v.size, 1))

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

def leave_one_out(D, L, classifier_train, classifier_test, prior_prob_array):
    acc = 0
    for i in range(D.shape[1]):
        D_ = D[:, [i]]
        L_ = L[i]

        DTR = numpy.delete(D, i, 1)
        LTR = numpy.delete(L, i)

        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        DTR2 = DTR[:, LTR == 2]

        (mu_array, cov_array) = classifier_train([DTR0, DTR1, DTR2])

        predicted_labels = classifier_test(D_, L_, mu_array, cov_array, prior_prob_array)[0]

        if predicted_labels[0] == L_:
            acc += 1
    return acc / D.shape[1]


def split_db_2to1(D, L, seed=0):

    #Per selezionare i sample in maniera randomica,
    #Prendiamo un array di indici dei sample
    #Poi mischiamo questo array tramite una permutazione
    #Infine usiamo tale array per accedere ai sample 
    #(METODO SIMILE AD ANALISI DI IMMAGINI DIGITALI)

    nTrain = int(D.shape[1] * 2.0/3.0)
    numpy.random.seed(seed)

    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)