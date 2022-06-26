# Generic import
import numpy

from utilityML.Functions.bayes import bayes_error_plots, compute_confusion_matrix_binary, compute_min_dcf, compute_normalized_dcf_binary

# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.dimred import pca


from utilityML.Functions.normalization import normalize

# Classifiers import
from utilityML.Classifiers.QuadLogReg import QuadLogReg

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


optimal_m = 11
optimal_lambda = 10**-6
reduced_normalized_dtr, P = pca(normalized_dtr, optimal_m)
reduced_normalized_dte = numpy.dot(P.T, normalized_dte)


quad_log_reg = QuadLogReg(reduced_normalized_dtr, LTR, reduced_normalized_dte, LTE, optimal_lambda)
quad_log_reg.estimate_model_parameters()
quad_log_reg.logreg_test()

#DCF EMPIRICO
empiric_dcf = quad_log_reg.compute_dcf(prior_1)
#DCF XVAL THRESHOLD
xvalthreshold_dcf = quad_log_reg.compute_dcf(prior_1, -0.43632266055187996)
#DCF MINIMO
min_dcf = quad_log_reg.compute_min_dcf(prior_1)[0]


Printer.print_title("Quadratic Logistic Regression data")
Printer.print_line(f"DCF: {empiric_dcf}")
Printer.print_line(f"DCF xval: {xvalthreshold_dcf}")
Printer.print_line(f"DCF min: {min_dcf}")
Printer.print_empty_lines(1)

with open("results/experimental/quadlogreg.txt", "w") as f:
    f.write(f"empiric DCF: {empiric_dcf}\n")
    f.write(f"X-validation Threshold DCF: {xvalthreshold_dcf}\n")
    f.write(f"minDCF Threshold DCF: {min_dcf}\n")

bayes_error_plots("DCF for Quadratic LogReg", quad_log_reg.LTE, quad_log_reg.S, validation_threshold = -0.43632266055187996)