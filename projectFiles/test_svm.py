# Generic import
import numpy

from utilityML.Functions.bayes import bayes_error_plots, compute_confusion_matrix_binary, compute_min_dcf, compute_normalized_dcf_binary

# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.dimred import pca


from utilityML.Functions.normalization import normalize
from utilityML.Functions.calibration import calibration

# Classifiers import
from utilityML.Classifiers.SVM import SVM_linear, SVM_poly, SVM_RBF

#Printer import
from Printer import Printer

#LOAD THE DATA
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
normalized_dtr, mu, sigma = normalize(DTR)
normalized_dte, _, _ = normalize(DTE, mu, sigma)

#CLASS PRIORS: WE CONSIDER A BALANCED APPLICATION
prior_0 = 0.5
prior_1 = 0.5


#SVM MODELS

optimal_m = 10
optimal_K = 1
optimal_C = 0.1
reduced_normalized_dtr, P = pca(normalized_dtr, optimal_m)
reduced_normalized_dte = numpy.dot(P.T, normalized_dte)

svm_l = SVM_linear(reduced_normalized_dtr, LTR, reduced_normalized_dte, LTE, [prior_0, prior_1], optimal_C, optimal_K)
svm_l.train()
svm_l.test()


#DCF CALIBRATO
calibrated_scores, calibrated_labels = calibration(svm_l.score[0], svm_l.LTE)
confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), prior_1, 1, 1)
calibrated_dcf = compute_normalized_dcf_binary(confusion_matrix, prior_1, 1, 1)

#DCF CALIBRATO MIN
calibrated_min_dcf = compute_min_dcf(calibrated_labels,calibrated_scores, prior_1, 1, 1)[0]

#------------------------------------------------

optimal_m = 10
optimal_K = 1
optimal_C = 1
optimal_c = 1
optimal_d = 2
reduced_normalized_dtr, P = pca(normalized_dtr, optimal_m)
reduced_normalized_dte = numpy.dot(P.T, normalized_dte)

svm_p = SVM_poly(reduced_normalized_dtr, LTR, reduced_normalized_dte, LTE, [prior_0, prior_1], C=optimal_C, K=optimal_K, degree=optimal_d, costant=optimal_c)
svm_p.train()
svm_p.test()

#DCF EMPIRICO
empiric_dcf = svm_p.compute_dcf()
#DCF XVAL THRESHOLD
xvalthreshold_dcf = svm_p.compute_dcf(-0.30825297694737075)
#DCF MINIMO
min_dcf = svm_p.compute_min_dcf()[0]

#------------------------------------------------

optimal_m = 10
optimal_K = 0
optimal_C = 1
optimal_gamma = 1
reduced_normalized_dtr, P = pca(normalized_dtr, optimal_m)
reduced_normalized_dte = numpy.dot(P.T, normalized_dte)

svm_rbf = SVM_RBF(reduced_normalized_dtr, LTR, reduced_normalized_dte, LTE, [prior_0, prior_1], C=optimal_C, K=optimal_K, gamma=optimal_gamma)
svm_rbf.train()
svm_rbf.test()

#DCF EMPIRICO
empiric_dcf = svm_rbf.compute_dcf()
#DCF XVAL THRESHOLD
xvalthreshold_dcf = svm_rbf.compute_dcf(6.985426439970719e-07)
#DCF MINIMO
min_dcf = svm_rbf.compute_min_dcf()[0]



Printer.print_title("SVM linear data")
Printer.print_line(f"DCF calibrated: {calibrated_dcf}")
Printer.print_line(f"DCF calibrated min: {calibrated_min_dcf}")
Printer.print_empty_lines(1)

Printer.print_title("SVM polynomial data")
Printer.print_line(f"DCF: {empiric_dcf}")
Printer.print_line(f"DCF xval: {xvalthreshold_dcf}")
Printer.print_line(f"DCF min: {min_dcf}")
Printer.print_empty_lines(1)

Printer.print_title("SVM RBF data")
Printer.print_line(f"DCF: {empiric_dcf}")
Printer.print_line(f"DCF xval: {xvalthreshold_dcf}")
Printer.print_line(f"DCF min: {min_dcf}")
