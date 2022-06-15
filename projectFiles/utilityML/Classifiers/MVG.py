import numpy
from scipy import special

from ..Functions.genpurpose import vrow, logpdf_GAU_ND, get_DTRs

from ..Functions.bayes import compute_confusion_matrix_binary, compute_dcf_binary, compute_normalized_dcf_binary, compute_min_dcf, generate_roc_curve

class MVG:
	def __init__(self, DTR, LTR, DTE, LTE, prior_prob_array):
		# initialization of the attributes
		self.DTR = DTR
		self.LTR = LTR
		self.DTE = DTE
		self.LTE = LTE

		self.mu_array = []
		self.cov_array = []
		self.prior_prob_array = prior_prob_array
		self.SPost = []
		self.llrs = 0.
		self.predicted_labels = []
		self.accuracy = 0.
		self.error = 0.

		self.dcf = 0.


	def train(self):
		

		DTR_array = get_DTRs(self.DTR, self.LTR, self.LTR.max() +1)
		
		
		for DTRi in DTR_array:
			mu_i = numpy.mean(DTRi, axis=1)
			mu_i = mu_i.reshape((mu_i.shape[0], 1))
			cov_i = 1/DTRi.shape[1] * numpy.dot(DTRi-mu_i, (DTRi-mu_i).T)

			self.mu_array.append(mu_i)
			self.cov_array.append(cov_i)
        
    
	def test(self):

		log_density_array = []

		for mu_i, cov_i, prior_prob_i in zip(self.mu_array, self.cov_array, self.prior_prob_array):

			log_density_array.append(logpdf_GAU_ND(self.DTE, mu_i, cov_i) + numpy.log(prior_prob_i))

		logSJoint = numpy.vstack((log_density_array))
		logSMarginal = vrow(special.logsumexp(logSJoint, axis=0))
		logSPost = logSJoint - logSMarginal
		self.SPost = numpy.exp(logSPost)

		self.predicted_labels = numpy.argmax(self.SPost, axis=0)

		#To include the leave-one-out special case
		denominator = 1 if numpy.isscalar(self.LTE) else self.LTE.shape[0]
		
		self.accuracy = (self.predicted_labels == self.LTE).sum() / denominator
		self.error = 1 - self.accuracy

		self.llrs = logSPost[1,:] - logSPost[0,:]
		confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.llrs, self.prior_prob_array[1],1,1)
		self.dcf = compute_normalized_dcf_binary(confusion_matrix, self.prior_prob_array[1], 1, 1)
		