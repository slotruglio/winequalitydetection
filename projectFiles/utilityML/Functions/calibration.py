import numpy

from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.SVM import SVM_linear
from utilityML.Functions.crossvalid import fold_data
from utilityML.Functions.dimred import pca
from utilityML.Functions.bayes import compute_confusion_matrix_binary, compute_normalized_dcf_binary
from utilityML.Functions.normalization import normalize
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


#This method calibrates the score using k-fold cross validations only on selected hyper parameters
def calibrate_scores_logreg(DTR, LTR, best_pca, best_lambda, priors, k, data_type):

	#split the data in k-folds
	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	calibrated_dcfs = {}

	#kfold cross validation 
		
	scores = []
	labels = []

	for j in range(k):
	
		cv_dtr = cv_dtr_array[j]
		cv_ltr = cv_ltr_array[j]
		
		reduced_cv_dtr, P = pca(cv_dtr, best_pca)

		#get the test data
		cv_dte = cv_dte_array[j]
		cv_lte = cv_lte_array[j]

		# get projected samples of test data
		reduced_cv_dte = numpy.dot(P.T, cv_dte)

		#train the model
		
		model = LogReg(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, best_lambda)
		model.estimate_model_parameters()
		model.logreg_test()

		#concatenate model.LTE and model.S
		labels.extend(model.LTE)
		scores.extend(model.S)

	#calibrate the scores

	scores = numpy.array(scores)
	labels = numpy.array(labels)
	calibrated_scores, calibrated_labels, w, b = calibration(scores, labels)


	confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), priors[1], 1, 1)

	calibratedDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)
	calibrated_dcfs[(data_type, best_pca, best_lambda)] = (calibratedDcf, w, b)
	
	return calibrated_dcfs


#This method calibrates the score using k-fold cross validations only on selected hyper parameters
def calibrate_scores_svm_linear(DTR, LTR, best_pca, best_C, priors, k, data_type):

	#split the data in k-folds
	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	calibrated_dcfs = {}
		
	scores = []
	labels = []

	for j in range(k):

		#delete i-th component from indices
		cv_dtr = cv_dtr_array[j]
		cv_ltr = cv_ltr_array[j]
		
		reduced_cv_dtr, P = pca(cv_dtr, best_pca)

		#get the test data
		cv_dte = cv_dte_array[j]
		cv_lte = cv_lte_array[j]

		# get projected samples of test data
		reduced_cv_dte = numpy.dot(P.T, cv_dte)

		#train the model
		
		model = SVM_linear(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, best_C, 1)
		model.train()
		model.test()

		#concatenate model.LTE and model.S
		labels.extend(model.LTE)
		scores.extend(model.score[0])

	#calibrate the scores

	scores = numpy.array(scores)
	labels = numpy.array(labels)
	calibrated_scores, calibrated_labels, w, b = calibration(scores, labels)


	confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), priors[1], 1, 1)

	calibratedDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)
	calibrated_dcfs[(data_type, best_pca, best_C)] = (calibratedDcf, w, b)
	
	return calibrated_dcfs