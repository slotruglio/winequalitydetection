# Generic import
import numpy

from utilityML.Functions.bayes import bayes_error_plots, compute_confusion_matrix_binary, compute_min_dcf, compute_normalized_dcf_binary

# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.dimred import pca


from utilityML.Functions.normalization import normalize
from utilityML.Functions.calibration import calibration

# Classifiers import
from utilityML.Classifiers.LogReg import LogReg

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

#LOGISTIC REGRESSION MODELS
#The linear logistic regression performed worse than MVG: however, we try it anyway
#The quadratic logistic regression, however, brought very good results

optimal_m = 11
optimal_lambda = 10**-2
reduced_normalized_dtr, P = pca(normalized_dtr, optimal_m)
reduced_normalized_dte = numpy.dot(P.T, normalized_dte)

log_reg = LogReg(reduced_normalized_dtr, LTR, reduced_normalized_dte, LTE, optimal_lambda)
log_reg.estimate_model_parameters()
log_reg.logreg_test()


#DCF EMPIRICO
empiric_dcf = log_reg.compute_dcf(prior_1)
#DCF XVAL THRESHOLD
xvalthreshold_dcf = log_reg.compute_dcf(prior_1, -0.5865054923862975)
#DCF MINIMO
min_dcf = log_reg.compute_min_dcf(prior_1)[0]

#DCF CALIBRATO
calibrated_scores, calibrated_labels = calibration(log_reg.S, log_reg.LTE)
confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), prior_1, 1, 1)
calibrated_dcf = compute_normalized_dcf_binary(confusion_matrix, prior_1, 1, 1)

#DCF CALIBRATO CON XVAL THRESHOLD
confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), prior_1, 1, 1,-0.5865054923862975)
calibrated_xvalthreshold_dcf = compute_normalized_dcf_binary(confusion_matrix, prior_1, 1, 1)

#DCF CALIBRATO MIN
calibrated_min_dcf = compute_min_dcf(calibrated_labels,calibrated_scores, prior_1, 1, 1)[0]

#Plottare: DCF empirico, DCF minimo, DCF validation, DCF calibrated (empirico + threshold)
#bayes_error_plots("DCF error plots for LogReg", LTE, log_reg.S, -0.5865054923862975, calibrated_scores, calibrated_labels)




Printer.print_title("Logistic Regression data")
Printer.print_line(f"DCF: {empiric_dcf}")
Printer.print_line(f"DCF xval: {xvalthreshold_dcf}")
Printer.print_line(f"DCF min: {min_dcf}")
Printer.print_line(f"DCF calibrated: {calibrated_dcf}")
Printer.print_line(f"DCF calibrated xval: {calibrated_xvalthreshold_dcf}")
Printer.print_line(f"DCF calibrated min: {calibrated_min_dcf}")
Printer.print_empty_lines(1)