#In this file, we look for the best parameters through tools like cross validation
#TODO: add the bayesian matrix calculation

# Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.NaiveBayes import NaiveBayes
from utilityML.Classifiers.TiedCovariance import TiedCovariance
from utilityML.Classifiers.TiedNaive import TiedNaive
from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.GMM import GMM

#Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.crossvalid import *
from Printer import Printer

#Statistics import
import time


#LOAD THE DATA
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


#COMPUTE CLASS PRIORS: label_i / total_labels
prior_0 = (LTR == 0).sum() / LTR.shape[0]
prior_1 = (LTR == 1).sum() / LTR.shape[0]
 

#CROSSVAL FOR MVG
mvg_xval_accuracies = gaussian_pca_crossvalidation(MVG, DTR, LTR, [prior_0, prior_1], 10)
Printer.print_title("XVal Data for MVG")
Printer.print_line(f"m: (Accuracies, dcf): {mvg_xval_accuracies}")
Printer.print_line(f"Best accuracy (max): {max(mvg_xval_accuracies.items(), key=lambda x: x[1][0])}")
Printer.print_line(f"Best dcf (min): {min(mvg_xval_accuracies.items(), key=lambda x: x[1][1][0])}")
Printer.print_empty_lines(1)

#CROSSVAL FOR NAIVEBAYES
naivebayes_xval_accuracies = gaussian_pca_crossvalidation(NaiveBayes, DTR, LTR, [prior_0, prior_1], 10)
Printer.print_title("XVal Data for Naive Bayes")
Printer.print_line(f"m: (Accuracies, dcf): {naivebayes_xval_accuracies}")
Printer.print_line(f"Best accuracy (max): {max(naivebayes_xval_accuracies.items(), key=lambda x: x[1][0])}")
Printer.print_line(f"Best dcf (min): {min(naivebayes_xval_accuracies.items(), key=lambda x: x[1][1][0])}")
Printer.print_empty_lines(1)

#CROSSVAL FOR TIED COVARIANCE
tiedcov_xval_accuracies = gaussian_pca_crossvalidation(TiedCovariance, DTR, LTR, [prior_0, prior_1], 10)
Printer.print_title("XVal Data for Tied Covariance")
Printer.print_line(f"m: (Accuracies, dcf): {tiedcov_xval_accuracies}")
Printer.print_line(f"Best accuracy (max): {max(tiedcov_xval_accuracies.items(), key=lambda x: x[1][0])}")
Printer.print_line(f"Best dcf (min): {min(tiedcov_xval_accuracies.items(), key=lambda x: x[1][1][0])}")
Printer.print_empty_lines(1)

#CROSSVAL FOR TIED NAIVE
tiednaive_xval_accuracies = gaussian_pca_crossvalidation(TiedNaive, DTR, LTR, [prior_0, prior_1], 10)
Printer.print_title("XVal Data for Tied Naive")
Printer.print_line(f"m: (Accuracies, dcf): {tiednaive_xval_accuracies}")
Printer.print_line(f"Best accuracy (max): {max(tiednaive_xval_accuracies.items(), key=lambda x: x[1][0])}")
Printer.print_line(f"Best dcf (min): {min(tiednaive_xval_accuracies.items(), key=lambda x: x[1][1][0])}")
Printer.print_empty_lines(1)

#CROSSVAL FOR LOGREG
logreg_xval_accuracies = logreg_pca_crossvalidation(DTR, LTR, [prior_0, prior_1], 10)
Printer.print_title("XVal Data for Logistic Regression")
Printer.print_line(f"Accuracies: {logreg_xval_accuracies}")
Printer.print_line(f"Best accuracy: {max(logreg_xval_accuracies.items(), key=lambda x: x[1][0])}")
Printer.print_line(f"Best dcf (min): {min(logreg_xval_accuracies.items(), key=lambda x: x[1][1][0])}")
Printer.print_empty_lines(1)


#******* SVM CROSS VALIDATION *******
# Inside svmPreprocessing.py and svm_gmmOptimization.py

#******* GMM CROSS VALIDATION *******
# Inside gmmPreprocessing.py and svm_gmmOptimization.py


#SEPARATOR FOR THE PRINTER
Printer.print_separator()