import matplotlib.pyplot as plt
import numpy as np
from utility import *

def plotHist(D, L, hFea, hLab, title, save=False):
    nLabels = len(set(L))
    for i in range(len(hFea)):
        plt.figure()
        plt.title(title)
        plt.xlabel(hFea[i])
        for j in range(nLabels):
            plt.hist(D[:, L == j][i, :],
                     density=True, label=hLab[j])
        plt.legend()
        plt.tight_layout()
    if save:
        save_plot('hist_%s.pdf' % title)
    plt.show()

def plot_scatter(D, L, hFea, hLab, title, save=False):

    D_p = []
    for i in range(len(L)):
        D_p.append(D[:, L == i])    

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.title(title)
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            for i in range(len(hLab)):
                plt.scatter(D_p[i][dIdx1, :], D_p[i][dIdx2, :], label=hLab[i])
            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            if save:
                save_plot('scatter_%s_%d_%d.pdf' % (title,dIdx1, dIdx2))
        plt.show()


def plot_density(save=False):
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000) 
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.title('Gaussian')
    if save:
        save_plot('gaussian.pdf')
    plt.show()

def plot_hist_and_density(X1D, save=False):
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000) 
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.hist(X1D.ravel(), bins=50, density=True)
    m_ML = compute_mean(X1D)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, compute_covariance(X1D, m_ML))))
    if save: 
        save_plot('hist_and_density.pdf')
    plt.show()

def plot_ROC(FPR, TPR, save=False):
    plt.figure()
    plt.plot(FPR, TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if save: 
        save_plot('roc.pdf')
    plt.show()

def plot_bayes_error(p, Y, NAMES, colors=None, save=False, legend=None):
    if colors is None:
        colors = ['r', 'b', 'y', 'c', 'm', 'g', 'k']
    for idx, name in enumerate(NAMES):
        plt.plot(p, Y[idx], color=colors[idx], label=name)
    plt.ylim(0, 1.1)
    plt.xlim(-3, 3)
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    if legend is not None:
        plt.legend(legend)
    if save: 
        save_plot('bayes_errors.pdf')
    plt.figure()
    


def save_plot(name):
    plt.savefig(name)