# Generic import
import numpy

from utilityML.Functions.bayes import bayes_error_plots, compute_confusion_matrix_binary, compute_normalized_dcf_binary

# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.dimred import pca


from utilityML.Functions.normalization import normalize
from utilityML.Functions.calibration import calibration

# Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.QuadLogReg import QuadLogReg
from utilityML.Classifiers.SVM import SVM_linear, SVM_poly, SVM_RBF
from utilityML.Classifiers.GMM import GMM

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

#GMM MODELS

optimal_m = 11
optimal_comp = 16
optimal_cov = 'full'
optimal_alpha = 0.1
optimal_psi = 0.01
reduced_dtr, P = pca(DTR, optimal_m)
reduced_dte = numpy.dot(P.T, DTE)

gmm = GMM(reduced_dtr, LTR, reduced_dte, LTE, [prior_0, prior_1], iterations=int(numpy.log2(optimal_comp)), alpha=optimal_alpha, psi=optimal_psi, typeOfGmm=optimal_cov)
gmm.train()
gmm.test()

#DCF EMPIRICO
empiric_dcf = gmm.compute_dcf()
#DCF XVAL THRESHOLD
xvalthreshold_dcf = gmm.compute_dcf(-0.09985079274315645)
#DCF MINIMO
min_dcf = gmm.compute_min_dcf()[0]


Printer.print_title("GMM data")
Printer.print_line(f"DCF: {empiric_dcf}")
Printer.print_line(f"DCF xval: {xvalthreshold_dcf}")
Printer.print_line(f"DCF min: {min_dcf}")
Printer.print_empty_lines(1)


bayes_error_plots("DCF For GMM", gmm.LTE, gmm.llrs, validation_threshold = -0.09985079274315645)