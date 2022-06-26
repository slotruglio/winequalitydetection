import numpy
from scipy import optimize
from ..Functions.genpurpose import mcol

#WEIGHTED LOGISTIC REGRESSION
class WeighLogReg:
	def __init__(self, DTR, LTR, DTE, LTE, l):
		self.DTR = DTR
		self.LTR = LTR
		self.DTE = DTE
		self.LTE = LTE

		self.l = l
		self.Z = 2 * LTR -1

		self.estimated_w = None 
		self.estimated_b = None

		self.calibration = 0

		self.S = []
		self.dcf = 0.
	
	def logreg_obj(self, v):

		w = mcol(v[0:self.DTR.shape[0]])
		b = v[-1]

		reg = 0.5 * self.l * numpy.linalg.norm(w) ** 2
		s = (numpy.dot(w.T, self.DTR) + b).ravel()
		avg_risk_0 = (numpy.logaddexp(0, -s[self.LTR == 0]*self.Z[self.LTR == 0])).mean()
		avg_risk_1 = (numpy.logaddexp(0, -s[self.LTR == 1]*self.Z[self.LTR == 1])).mean()
		return reg + avg_risk_1 + avg_risk_0


	def estimate_model_parameters(self, p = None):
		model_parameters = optimize.fmin_l_bfgs_b(self.logreg_obj, numpy.zeros(self.DTR.shape[0]+1), approx_grad=True, factr=1.0)
		self.estimated_w = model_parameters[0][0:-1]
		self.estimated_b = model_parameters[0][-1]

		self.calibration = 0 if p == None else numpy.log(p/(1-p))

	def logreg_test(self):
		
		self.S = numpy.dot(self.estimated_w.T, self.DTE) + self.estimated_b - self.calibration




