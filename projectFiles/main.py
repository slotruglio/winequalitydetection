#Generic import
import numpy
from sklearn import naive_bayes

#Functions import
from utilityML.Functions.genpurpose import load

#Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.NaiveBayes import NaiveBayes
from utilityML.Classifiers.TiedCovariance import TiedCovariance
from utilityML.Classifiers.TiedNaive import TiedNaive
from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.Multinomial import Multinomial

from Printer import Printer

#Step 1 - Trovare il classificatore migliore
#Bisogna fare parameters tuning tramite cross validation
#La cross validation verifica ogni volta o la accuracy o la confusion matrix (misura più accurata)

#Step 2 - Valutare dimensionality reduction
#Una volta trovato il classificatore migliore con i parametri migliori, si fa
#dimensionality reduction con la PCA, valutando vari valori per m, sempre con la cross validation

#Questi due step vanno accompagnati da eventuali plot e commenti, utili per il report finale

#Load the data
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3,4,5,6,7,8,9,10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3,4,5,6,7,8,9,10], 11)

#Compute class priors: label_i / total_labels
prior_0 = (LTR == 0).sum() / LTR.shape[0]
prior_1 = (LTR == 1).sum() / LTR.shape[0]

mvg = MVG(DTR, LTR, DTE, LTE, [prior_0, prior_1])
mvg.train()
mvg.test()

naive_bayes = NaiveBayes(DTR, LTR, DTE, LTE, [prior_0, prior_1])
naive_bayes.train()
naive_bayes.test()

tied_covariance = TiedCovariance(DTR, LTR, DTE, LTE, [prior_0, prior_1])
tied_covariance.train()
tied_covariance.test()

tied_naive = TiedNaive(DTR, LTR, DTE, LTE, [prior_0, prior_1])
tied_naive.train()
tied_naive.test()

log_reg = LogReg(DTR, LTR, DTE, LTE, 1)
log_reg.estimate_model_parameters()
log_reg.logreg_test()

#print all accuracies and errors in percentual form and table form
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




