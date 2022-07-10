from turtle import color
import matplotlib.pyplot as plt
import numpy
from utilityML.Functions.genpurpose import *

def plotHist(D, L, hFea, hLab, title, save=False):
    nLabels = len(set(L))
    for i in range(len(hFea)):
        plt.figure()
        plt.title(title+": Feature {}".format(i+1))
        plt.xlabel(hFea[i])
        for j in range(nLabels):
            plt.hist(D[:, L == j][i, :],
                     density=True, label=hLab[j], alpha=0.5)
        plt.legend()
        plt.tight_layout()
        if save:
            save_plot('first_analysis/histograms/hist_{}_f{}.png'.format(title, i+1))
    if not save: plt.show()

def plotLabels(L, labels, save=False):
    class_0 = 0
    class_1 = 0

    for i in range(len(L)):
        if L[i] == 0: class_0+=1
        else: class_1+=1 

    plt.figure()
    plt.title("Labels")
    plt.bar(labels, [class_0, class_1], color= ["r", "b"], width=0.3)
    plt.legend()
    plt.tight_layout()
    if save: save_plot("first_analysis/labels.png")
    else: plt.show()

def plot_scatter(DP, L, title, save=False):

    class_0 = DP[:, L < 1]
    class_1 = DP[:, L > 0]

    plt.figure()
    plt.title(title)
    plt.scatter(class_0[0], class_0[1], marker='x')
    plt.scatter(class_1[0], class_1[1], marker='x')
    plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
    
    if save:
        save_plot('first_analysis/scatter_plots/scatter_%s.png' % (title))
    if not save: plt.show()


def plot_ROC(FPR, TPR, save=False):
    plt.figure()
    plt.plot(FPR, TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if save:
        save_plot('roc.png')
    plt.show()


def save_plot(name):
    plt.savefig('./img/'+name)
