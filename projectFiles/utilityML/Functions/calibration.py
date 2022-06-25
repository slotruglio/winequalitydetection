import numpy
from ..Classifiers.WeighLogReg import WeighLogReg

#This method uses the previously computed scores as a dataset for a new logistic regression model
#This will allow us to generate calibrated scores
def calibration(scores, labels):

	#split scores in two groups of 70% and 30%
	scores_70 = scores[:int(len(scores)*0.7)]
	scores_30 = scores[int(len(scores)*0.7):]
	labels_70 = labels[:int(len(labels)*0.7)]
	labels_30 = labels[int(len(labels)*0.7):]

	logreg = WeighLogReg(numpy.array([scores_70]), labels_70, numpy.array([scores_30]), labels_30, 10**-3)

	logreg.estimate_model_parameters()

	logreg.logreg_test()


	return numpy.array(logreg.S), labels_30, logreg.estimated_w, logreg.estimated_b
