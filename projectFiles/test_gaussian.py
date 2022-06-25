# Generic import
import numpy

from utilityML.Functions.bayes import bayes_error_plots


# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.dimred import pca


from utilityML.Functions.normalization import normalize

# Classifiers import
from utilityML.Classifiers.MVG import MVG

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

#GAUSSIAN MODELS - MVG (the others have been discared due to poor performance)
optimal_m = 9
reduced_dtr, P = pca(normalized_dtr, optimal_m)
reduced_dte = numpy.dot(P.T, normalized_dte)

mvg = MVG(reduced_dtr, LTR, reduced_dte, LTE, [prior_0, prior_1])
mvg.train()
mvg.test()

empiric_dcf = mvg.compute_dcf()
xvalthreshold_dcf = mvg.compute_dcf(0.42050259957894554)
min_dcf = mvg.compute_min_dcf()[0]

#PRINT ALL CLASSIFIERS RESULTS
Printer.print_title("MVG data")
Printer.print_line(f"Empiric DCF: {empiric_dcf}")
Printer.print_line(f"X-validation Threshold DCF: {xvalthreshold_dcf}")
Printer.print_line(f"Min DCF: {min_dcf}")
Printer.print_empty_lines(1)


bayes_error_plots("DCF For MVG", mvg.LTE, mvg.llrs, validation_threshold = 0.42050259957894554)
