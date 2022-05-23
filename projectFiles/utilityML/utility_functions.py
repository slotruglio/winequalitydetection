from matplotlib import pyplot as plt
import numpy
from scipy import special
from scipy import linalg

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

def compute_bayes_decision_binary(labels, llrs, pi, Cfn, Cfp):

	#creo la confusion matrix
	confusion_matrix = numpy.zeros((2,2))

	indexes_label_0 = (labels == 0)
	indexes_label_1 = (labels == 1)

	confusion_matrix[0][0] = (llrs[indexes_label_0] <= -numpy.log((pi * Cfn) / ((1-pi) * Cfp))).sum()
	confusion_matrix[0][1] = (llrs[indexes_label_1] <= -numpy.log((pi * Cfn) / ((1-pi) * Cfp))).sum()	
	
	confusion_matrix[1][1] = (llrs[indexes_label_1] > -numpy.log((pi * Cfn) / ((1-pi) * Cfp))).sum()
	confusion_matrix[1][0] = (llrs[indexes_label_0] > -numpy.log((pi * Cfn) / ((1-pi) * Cfp))).sum()

	return confusion_matrix

def compute_bayes_decision(labels, cond_ll, pi_array, c_matrix):

	logSJoint = cond_ll + numpy.log(pi_array)
	
	logSMarginal = vrow(special.logsumexp(logSJoint, axis=0))

	logSPost = logSJoint - logSMarginal

	SPost = numpy.exp(logSPost)

	optimal_classes = numpy.argmin(c_matrix @ SPost, axis=0)

	numpy.savetxt("data/commedia_bayes_decision.txt", optimal_classes, fmt='%d')

	confusion_matrix = numpy.zeros((len(cond_ll), len(cond_ll)))

	for i in range(len(confusion_matrix)):

		for j in range(len(confusion_matrix)):


			confusion_matrix[i][j] = (optimal_classes[(labels == j)] == i).sum()


	return confusion_matrix

def pca(D, L):

    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))

    DC = D - mu.reshape((mu.size, 1))

    C = 1/DC.shape[1] * DC.dot(DC.T)

    #autovalori ed autovettori
    s, U = numpy.linalg.eigh(C)

    m = 2
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)


    # ***** PLOT GENERATION ***** #

    plt.figure()

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    #non c'è bisogno di classi (detto a lezione)
    plt.scatter(DP[0, :], DP[1, :])

    plt.legend()

    plt.show()

def lda(D, L):

    #+++ WITHIN COVARIANCE +++

    #Divide the matrix for each class
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    #Compute the mean for each class
    mu0 = D0.mean(1)
    mu1 = D1.mean(1)
    mu2 = D2.mean(1)

    mu0 = mu0.reshape((mu0.size,1))
    mu1 = mu1.reshape((mu1.size,1))
    mu2 = mu2.reshape((mu2.size,1))

    #Subtract the mean from each sub matrix
    DC0 = D0 - mu0
    DC1 = D1 - mu1
    DC2 = D2 - mu2 

    #Calculate C
    C0 = 1/DC0.shape[1] * DC0.dot(DC0.T)
    C1 = 1/DC1.shape[1] * DC1.dot(DC1.T)
    C2 = 1/DC2.shape[1] * DC2.dot(DC2.T)

    #Calcolare il numero totale di samples N
    N = D.shape[1]

    #Calcolare il numero di samples per ogni classe nc
    n0 = D0.shape[1]
    n1 = D1.shape[1]
    n2 = D2.shape[1]

    #Formula: 1/N * [n0 * C0 + n1 * C1 + n2 * C2]
    Sw = 1/N * (n0 * C0 + n1 * C1 + n2 * C2)

    print(Sw)

    #+++ BETWEEN COVARIANCE +++
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    Sb_0 = n0 * (mu0 - mu) * (mu0-mu).T
    Sb_1 = n1 * (mu1 - mu) * (mu1-mu).T
    Sb_2 = n2 * (mu2 - mu) * (mu2-mu).T
    Sb = 1/N * (Sb_0 + Sb_1 + Sb_2)

    print(Sb)

    #+++ SOLVING THE GENERALIZED EIGENVALUE PROBLEM TO FIND THE DIRECTIONS (Columns of W) +++
    s, U = linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:9]