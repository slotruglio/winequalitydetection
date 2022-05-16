import numpy
from ..utility_functions import vrow, logpdf_GAU_ND


class Tied_covariance:
    def train(DTR_array):
            
        mu_array = []

        cov = 0
        sample_size = 0

        for DTRi in DTR_array:
            mu_i = numpy.mean(DTRi, axis=1)
            mu_i = mu_i.reshape((mu_i.shape[0], 1))

            mu_array.append(mu_i)

            sample_size += DTRi.shape[1]

            cov += (DTRi - mu_i) @ (DTRi - mu_i).T
        
        cov *= 1/sample_size
        
        return (mu_array, cov)
    
    def test(DTE, LTE, mu_array, cov, prior_prob_array):
            
        density_array = []

        for mu_i, prior_prob_i in zip(mu_array, prior_prob_array):

            density_array.append(numpy.exp(logpdf_GAU_ND(DTE, mu_i, cov)) * prior_prob_i)

        SJoint = numpy.vstack((density_array))

        SPost = SJoint / vrow(SJoint.sum(0))

        predicted_labels = numpy.argmax(SPost, axis=0)
        
        #To include the leave-one-out special case
        denominator = 1 if numpy.isscalar(LTE) else LTE.shape[0]
        
        acc = (predicted_labels == LTE).sum() / denominator

        return [predicted_labels, acc]