# Generic import
import numpy
from sklearn import naive_bayes

# Functions import
from utilityML.Functions.genpurpose import load

# Plot import
from utilityML.Functions.plot import *

# Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.NaiveBayes import NaiveBayes
from utilityML.Classifiers.TiedCovariance import TiedCovariance
from utilityML.Classifiers.TiedNaive import TiedNaive
from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.SVM import SVM_linear, SVM_poly, SVM_RBF

from Printer import Printer

#Parameters tuning import (this actually executes the file)
from parameters_tuning import *


#LOAD THE DATA
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)

# Show the data
show = False
# do or not do svm
do_svm = True


#region plotting data
# features as array
features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
# labels as array
labels = [
    "bad quality",
    "good quality"
]

if show :

    # histogram
    plotHist(DTR, LTR, features, labels, "Dataset's histogram")

    # scatter
    plot_scatter_dual(DTR, LTR, features, labels, "Dataset's scatter")
#endregion


#COMPUTE CLASS PRIORS: label_i / total_labels
prior_0 = (LTR == 0).sum() / LTR.shape[0]
prior_1 = (LTR == 1).sum() / LTR.shape[0]

#TRAIN AND TEST FOR MVG
mvg = MVG(DTR, LTR, DTE, LTE, [prior_0, prior_1])
mvg.train()
mvg.test()

#TRAIN AND TEST FOR NAIVEBAYES
naive_bayes = NaiveBayes(DTR, LTR, DTE, LTE, [prior_0, prior_1])
naive_bayes.train()
naive_bayes.test()

#TRAIN AND TEST FOR TIEDCOVARIANCE
tied_covariance = TiedCovariance(DTR, LTR, DTE, LTE, [prior_0, prior_1])
tied_covariance.train()
tied_covariance.test()

#TRAIN AND TEST FOR TIEDNAIVE
tied_naive = TiedNaive(DTR, LTR, DTE, LTE, [prior_0, prior_1])
tied_naive.train()
tied_naive.test()

#TRAIN AND TEST FOR LOGREG
log_reg = LogReg(DTR, LTR, DTE, LTE, 1)
log_reg.estimate_model_parameters()
log_reg.logreg_test()

#TRAIN AND TEST FOR SVG
if do_svm:
    svm_l = SVM_linear(DTR, LTR, DTE, LTE)
    svm_l.train()
    svm_l.test()

    svm_p = SVM_poly(DTR, LTR, DTE, LTE)
    svm_p.train()
    svm_p.test()

    svm_rbf = SVM_RBF(DTR, LTR, DTE, LTE)
    svm_rbf.train()
    svm_rbf.test()


#PRINT ALL CLASSIFIERS RESULTS
Printer.print_title("MVG data")
Printer.print_line(f"Accuracy: {mvg.accuracy * 100:.2f}%")
Printer.print_line(f"Error: {mvg.error * 100:.2f}%")
Printer.print_empty_lines(1)

Printer.print_title("Naive Bayes data")
Printer.print_line(f"Accuracy: {naive_bayes.accuracy * 100:.2f}%")
Printer.print_line(f"Error: {naive_bayes.error * 100:.2f}%")
Printer.print_empty_lines(1)

Printer.print_title("Tied Covariance data")
Printer.print_line(f"Accuracy: {tied_covariance.accuracy * 100:.2f}%")
Printer.print_line(f"Error: {tied_covariance.error * 100:.2f}%")
Printer.print_empty_lines(1)

Printer.print_title("Tied Naive data")
Printer.print_line(f"Accuracy: {tied_naive.accuracy * 100:.2f}%")
Printer.print_line(f"Error: {tied_naive.error * 100:.2f}%")
Printer.print_empty_lines(1)

Printer.print_title("Logistic Regression data")
Printer.print_line(f"Accuracy: {log_reg.accuracy * 100:.2f}%")
Printer.print_line(f"Error: {log_reg.error * 100:.2f}%")
Printer.print_empty_lines(1)

if do_svm:
    Printer.print_title("SVM linear data")
    Printer.print_line(f"Accuracy: {svm_l.accuracy * 100:.2f}%")
    Printer.print_line(f"Error: {svm_l.error * 100:.2f}%")
    Printer.print_empty_lines(1)

    Printer.print_title("SVM poly data")
    Printer.print_line(f"Accuracy: {svm_p.accuracy * 100:.2f}%")
    Printer.print_line(f"Error: {svm_p.error * 100:.2f}%")
    Printer.print_empty_lines(1)

    Printer.print_title("SVM RBF data")
    Printer.print_line(f"Accuracy: {svm_rbf.accuracy * 100:.2f}%")
    Printer.print_line(f"Error: {svm_rbf.error * 100:.2f}%")
    Printer.print_empty_lines(1)



