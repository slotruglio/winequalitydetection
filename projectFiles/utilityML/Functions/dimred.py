import numpy
from scipy import linalg
from genpurpose import *
from plot import *

def pca(D, L, m=2):

    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))

    DC = D - mu.reshape((mu.size, 1))

    C = 1/DC.shape[1] * DC.dot(DC.T)

    #autovalori ed autovettori
    s, U = numpy.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)

    # plot_scatter(DP, "sepal length", "sepal width", "scatter_title")
    return DP

def lda(D, L, m=9):

    #+++ WITHIN COVARIANCE +++

    #Divide the matrix for each class
    Ds = get_DTRs(D, L, range(L.max()+1))
    # D0 = D[:, L==0]
    # D1 = D[:, L==1]
    # D2 = D[:, L==2]

    #Compute the mean for each class
    MUs = [compute_mean(Ds[i]) for i in range(len(Ds))]
    # mu0 = D0.mean(1)
    # mu1 = D1.mean(1)
    # mu2 = D2.mean(1)

    # mu0 = mu0.reshape((mu0.size,1))
    # mu1 = mu1.reshape((mu1.size,1))
    # mu2 = mu2.reshape((mu2.size,1))

    #Subtract the mean from each sub matrix
    DCs = [Ds[i] - MUs[i] for i in range(len(Ds))]
    # DC0 = D0 - mu0
    # DC1 = D1 - mu1
    # DC2 = D2 - mu2 

    #Calculate C
    Cs = [1/DCs[i].shape[1] * DCs[i].dot(DCs[i].T) for i in range(len(Ds))]
    # C0 = 1/DC0.shape[1] * DC0.dot(DC0.T)
    # C1 = 1/DC1.shape[1] * DC1.dot(DC1.T)
    # C2 = 1/DC2.shape[1] * DC2.dot(DC2.T)

    #Calcolare il numero totale di samples N
    N = D.shape[1]

    #Calcolare il numero di samples per ogni classe nc
    ns = [Ds[i].shape[1] for i in range(len(Ds))]
    # n0 = D0.shape[1]
    # n1 = D1.shape[1]
    # n2 = D2.shape[1]

    #Formula: 1/N * [n0 * C0 + n1 * C1 + n2 * C2]
    Sw = 0.
    for i in range(len(Ds)):
        Sw += ns[i] * Cs[i]
    Sw = 1/N * Sw
    #Sw = 1/N * (n0 * C0 + n1 * C1 + n2 * C2)


    print(Sw)

    #+++ BETWEEN COVARIANCE +++
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    Sb_s = [ns[i]*(MUs[i]-mu).dot((MUs[i]-mu).T) for i in range(len(Ds))]

    # Sb_0 = n0 * (mu0 - mu) * (mu0-mu).T
    # Sb_1 = n1 * (mu1 - mu) * (mu1-mu).T
    # Sb_2 = n2 * (mu2 - mu) * (mu2-mu).T
    Sb = 0.
    for i in range(len(Ds)):
        Sb += Sb_s[i]
    Sb = 1/N * Sb
    # Sb = 1/N * (Sb_0 + Sb_1 + Sb_2)

    print(Sb)

    #+++ SOLVING THE GENERALIZED EIGENVALUE PROBLEM TO FIND THE DIRECTIONS (Columns of W) +++
    s, U = linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    return W