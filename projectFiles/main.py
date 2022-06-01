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

#print all accuracies and errors in percentual form and table form
print("MVG data")
print("Accuracy: ", mvg.accuracy * 100, "%")
print("Error: ", mvg.error * 100, "%")

print("Naive Bayes data")
print("Accuracy: ", naive_bayes.accuracy * 100, "%")
print("Error: ", naive_bayes.error * 100, "%")

print("Tied Covariance data")
print("Accuracy: ", tied_covariance.accuracy * 100, "%")
print("Error: ", tied_covariance.error * 100, "%")

print("Tied Naive data")
print("Accuracy: ", tied_naive.accuracy * 100, "%")
print("Error: ", tied_naive.error * 100, "%")






