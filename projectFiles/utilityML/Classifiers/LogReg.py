import numpy
from scipy import optimize
import sklearn
from ..Functions.genpurpose import mcol
from ..Functions.bayes import compute_confusion_matrix_binary, compute_normalized_dcf_binary

#BINARY LOGISTIC REGRESSION
class LogReg:
	def __init__(self, DTR, LTR, DTE, LTE, l):
		self.DTR = DTR
		self.LTR = LTR
		self.DTE = DTE
		self.LTE = LTE

		self.l = l
		self.Z = 2 * LTR -1

		self.estimated_w = None 
		self.estimated_b = None

		self.accuracy = 0.
		self.error = 0.
		self.predicted_label = []

		self.S = []
		self.dcf = 0.
	
	def logreg_obj(self, v):

		w = mcol(v[0:self.DTR.shape[0]])
		b = v[-1]

		S = numpy.dot(w.T, self.DTR) + b 

		first_part = 0.5 * self.l * numpy.linalg.norm(w)**2

		second_part = numpy.logaddexp(0, -self.Z * S).mean()

		return first_part + second_part

	def estimate_model_parameters(self):
		model_parameters = optimize.fmin_l_bfgs_b(self.logreg_obj, numpy.zeros(self.DTR.shape[0]+1), approx_grad=True)
		self.estimated_w = model_parameters[0][0:-1]
		self.estimated_b = model_parameters[0][-1]

	def logreg_test(self, prior):
		
		self.S = numpy.dot(self.estimated_w.T, self.DTE) + self.estimated_b 

		self.predicted_labels = numpy.zeros(self.LTE.shape)
		self.predicted_labels[self.S > 0] = 1

		self.accuracy = (self.predicted_labels == self.LTE).sum() / len(self.LTE)
		self.error = 1 - self.accuracy

		confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.S, prior,1,1)
		self.dcf = compute_normalized_dcf_binary(confusion_matrix, prior, 1, 1)
