import numpy
from ..Functions.genpurpose import vrow, logpdf_GAU_ND, get_DTRs, split_db_2to1
from ..Functions.bayes import compute_confusion_matrix_binary, compute_normalized_dcf_binary


class TiedCovariance:

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

		self.cov = 0
		sample_size = 0

		for DTRi in DTR_array:
			mu_i = numpy.mean(DTRi, axis=1)
			mu_i = mu_i.reshape((mu_i.shape[0], 1))

			self.mu_array.append(mu_i)

			sample_size += DTRi.shape[1]

			self.cov += (DTRi - mu_i) @ (DTRi - mu_i).T
		
		self.cov *= 1/sample_size

	def test(self):
			
		density_array = []

		for mu_i, prior_prob_i in zip(self.mu_array, self.prior_prob_array):

			density_array.append(numpy.exp(logpdf_GAU_ND(self.DTE, mu_i, self.cov)) * prior_prob_i)

		SJoint = numpy.vstack((density_array))

		self.SPost = SJoint / vrow(SJoint.sum(0))

		self.predicted_labels = numpy.argmax(self.SPost, axis=0)
		
		#To include the leave-one-out special case
		denominator = 1 if numpy.isscalar(self.LTE) else self.LTE.shape[0]
		
		self.accuracy = (self.predicted_labels == self.LTE).sum() / denominator

		self.error = 1 - self.accuracy

		self.llrs = numpy.log(self.SPost[1,:]) - numpy.log(self.SPost[0,:])
		confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.llrs, self.prior_prob_array[1],1,1)
		self.dcf = compute_normalized_dcf_binary(confusion_matrix, self.prior_prob_array[1], 1, 1)