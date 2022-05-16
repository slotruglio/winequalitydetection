import numpy
import scipy
from ..utility_functions import vrow, logpdf_GAU_ND

class MVG:
    def train(DTR_array):

        mu_array = []
        cov_array = []

        for DTRi in DTR_array:
            mu_i = numpy.mean(DTRi, axis=1)
            mu_i = mu_i.reshape((mu_i.shape[0], 1))
            cov_i = 1/DTRi.shape[1] * numpy.dot(DTRi-mu_i, (DTRi-mu_i).T)

            mu_array.append(mu_i)
            cov_array.append(cov_i)
        
        return (mu_array, cov_array)
    
    def test(DTE, LTE, mu_array, cov_array, prior_prob_array):

        density_array = []

        for mu_i, cov_i, prior_prob_i in zip(mu_array, cov_array, prior_prob_array):

            density_array.append(numpy.exp(logpdf_GAU_ND(DTE, mu_i, cov_i)) * prior_prob_i)

        SJoint = numpy.vstack((density_array))
        logSJoint = numpy.log(SJoint)

        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

        logSPost = logSJoint - logSMarginal

        SPost = numpy.exp(logSPost)

        predicted_labels = numpy.argmax(SPost, axis=0)

        #To include the leave-one-out special case
        denominator = 1 if numpy.isscalar(LTE) else LTE.shape[0]
        
        acc = (predicted_labels == LTE).sum() / denominator

        return [predicted_labels, acc]