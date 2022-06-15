#In this file, we look for the best parameters through tools like cross validation
#TODO: add the bayesian matrix calculation

# Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.NaiveBayes import NaiveBayes
from utilityML.Classifiers.TiedCovariance import TiedCovariance
from utilityML.Classifiers.TiedNaive import TiedNaive
from utilityML.Classifiers.LogReg import LogReg

#Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.crossvalid import gaussian_pca_crossvalidation, logreg_pca_crossvalidation, svm_linear_cross_valid_C, svm_poly_cross_valid, svm_RBF_cross_valid
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
#Separated from the rest of the file because of its complexity
#SVM CROSS VALIDATION SECTION
# do or not do svm
do_svm = False

if do_svm :
    Printer.print_title("SVM linear cross validation of C")

    start = time.time()
    
    xval = svm_linear_cross_valid_C(DTR, LTR, [0.1, 1, 10], [prior_0, prior_1], 1, percentage=2./3.)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of 70/30: {end - start:.2f}s")
    Printer.print_empty_lines(1)

    start = time.time()
    
    xval = svm_linear_cross_valid_C(DTR, LTR, [0.1, 1, 10], [prior_0, prior_1], 1, 10)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of kfold: {end - start:.2f}s")
    Printer.print_empty_lines(1)

    Printer.print_title("SVM poly cross validation")

    start = time.time()
    
    xval = svm_poly_cross_valid(DTR, LTR, [0.1, 1, 10], [0,1], [prior_0, prior_1], [0,1], percentage=2./3.)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of 70/30: {end - start:.2f}s")
    Printer.print_empty_lines(1)

    start = time.time()

    xval = svm_poly_cross_valid(DTR, LTR, [0.1, 1, 10], [0,1], [prior_0, prior_1], [0,1], 10)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of kfold: {end - start:.2f}s")
    Printer.print_empty_lines(1)

    Printer.print_title("SVM RBF cross validation")

    start = time.time()

    xval = svm_RBF_cross_valid(DTR, LTR, [0.1, 1, 10], [1.,10.], [prior_0, prior_1], [0,1], percentage=2./3.)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of 70/30: {end - start:.2f}s")
    Printer.print_empty_lines(1)

    start = time.time()

    xval = svm_RBF_cross_valid(DTR, LTR, [0.1, 1, 10], [1.,10.], [prior_0, prior_1], [0,1], folds=10)
    end = time.time()
    Printer.print_line(f"Accuracies: {xval}")
    Printer.print_empty_lines(1)
    Printer.print_line(f"Time of kfold: {end - start:.2f}s")
    Printer.print_empty_lines(1)

#SEPARATOR FOR THE PRINTER
Printer.print_separator()