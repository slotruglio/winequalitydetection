import numpy
from utilityML.Functions.bayes import compute_min_dcf, compute_confusion_matrix_binary, compute_normalized_dcf_binary, compute_dcf_binary
from utilityML.Functions.calibration import calibration
from utilityML.Functions.dimred import *
from utilityML.Functions.genpurpose import split_db_2to1
from utilityML.Classifiers.SVM import SVM_linear, SVM_poly, SVM_RBF
from utilityML.Classifiers.GMM import GMM
from utilityML.Classifiers.LogReg import LogReg
from utilityML.Classifiers.QuadLogReg import QuadLogReg

def split_leave_one_out(D, L, index):
    D_train = numpy.delete(D, index, 1)
    L_train = numpy.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)

#region gaussian pca

def gaussian_pca_k_fold_crossvalidation(classifier, DTR, LTR, priors, k):


	#for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	for m in range(1,12):

		accuracies = []

		labels = []
		llrs = []

		for i in range(k):

			#delete i-th component from indices
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]
			
			reduced_cv_dtr, P = pca(cv_dtr, m)

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)

			#train the model
			model = classifier(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors)
			model.train()
			model.test()

			accuracies.append(model.accuracy)

			#concatenate model.LTE and model.llrs
			labels.extend(model.LTE)
			llrs.extend(model.llrs)
		
		#compute the mindcf on ALL the folds permutations' LLRS and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(llrs), priors[1], 1, 1)
		
		#compute the dcf
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(llrs), priors[1], 1, 1)
		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)
		

		#append the mean of accuracies to the global_accuracies dictionary, with m as key
		global_accuracies[m] = (numpy.mean(accuracies), mindcf, normDcf)


	
	return global_accuracies
	
	
#endregion


#region logreg pca

def logreg_pca_k_fold_crossvalidation(DTR, LTR, priors, k, quadratic = False):

	#for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	for m in range(1,12):

		for l in [10**-6, 10**-3, 10**-2, 10**-1, 1.0]:

			accuracies = []

			labels = []
			scores = []

			for i in range(k):

				#delete i-th component from indices
				cv_dtr = cv_dtr_array[i]
				cv_ltr = cv_ltr_array[i]
				
				reduced_cv_dtr, P = pca(cv_dtr, m)

				#get the test data
				cv_dte = cv_dte_array[i]
				cv_lte = cv_lte_array[i]

				# get projected samples of test data
				reduced_cv_dte = numpy.dot(P.T, cv_dte)

				#train the model
				if quadratic:
					model = QuadLogReg(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, l)
				else:
					model = LogReg(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, l)
				model.estimate_model_parameters()
				model.logreg_test()

				accuracies.append(model.accuracy)

				#concatenate model.LTE and model.S
				labels.extend(model.LTE)
				scores.extend(model.S)


			#compute the mindcf on ALL the folds permutations' SCORES and LABELS
			mindcf = compute_min_dcf(numpy.array(labels), numpy.array(scores), priors[1], 1, 1)

			confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(scores), priors[1], 1, 1)
			#compute norm dcf
			normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)

			if(quadratic):
				global_accuracies[(m,l)] = (numpy.mean(accuracies), mindcf, normDcf)
			else:
				llrs_array = numpy.array(scores)
				labels_array = numpy.array(labels)
				calibrated_scores, calibrated_labels, w, b = calibration(llrs_array, labels_array)

				#compute the dcf
				confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), priors[1], 1, 1)
				#compute norm dcf
				calibratedDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)


				#append the mean of accuracies to the global_accuracies dictionary, with m as key
				global_accuracies[(m,l)] = (numpy.mean(accuracies), mindcf, normDcf, calibratedDcf, w, b)
	
	return global_accuracies
#endregion


def fold_data(DTR, LTR, k):

	#create k groups of random indices
	indices = numpy.random.permutation(DTR.shape[1])
	indices = numpy.array_split(indices, k, axis=0)

	#for each group, compute the accuracy
	
	cv_dtr_array = []
	cv_ltr_array = []
	cv_dte_array = []
	cv_lte_array = []

	for i in range(k):

		#delete i-th component from indices
		indices_to_keep = numpy.delete(indices, i, 0)
		indices_to_keep = numpy.concatenate(indices_to_keep, axis=0)

		#get the training set
		cv_dtr = DTR[:, indices_to_keep]
		cv_ltr = LTR[indices_to_keep]
		

		#get the validation set
		cv_dte = DTR[:, indices[i]]
		cv_lte = LTR[indices[i]]

		cv_dtr_array.append(cv_dtr)
		cv_ltr_array.append(cv_ltr)
		cv_dte_array.append(cv_dte)
		cv_lte_array.append(cv_lte)
	
	return cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array


#region svm

def svm_linear_pca_k_cross_valid(DTR, LTR, priors, folds, K=1):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for m in range(5,12):
		accuracies = []

		labels = []
		score = []


		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			reduced_cv_dtr, P = pca(cv_dtr, m)

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)

			#train the model
			# K = 1 and C = 0.1
			svm = SVM_linear(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

			#concatenate model.LTE and model.llrs
			labels.extend(svm.LTE)
			score.extend(svm.score[0])

		#compute the mindcf on ALL the folds permutations' SCORES and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)
		
		#append the mean of accuracies to the global_accuracies dictionary, with C as key
		global_accuracies[m] = (numpy.mean(accuracies), mindcf, normDcf)

	return global_accuracies


def svm_linear_k_cross_valid_C(DTR, LTR, folds, C_array, priors, K=1, pcaVal=-1):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for C in C_array:
		accuracies = []

		labels = []
		score = []


		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			if(pcaVal > 0):
				cv_dtr, P = pca(cv_dtr, pcaVal)
				cv_dte = numpy.dot(P.T, cv_dte)

			#train the model
			svm = SVM_linear(cv_dtr, cv_ltr, cv_dte, cv_lte, priors, C, K)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

			#concatenate model.LTE and model.llrs
			labels.extend(svm.LTE)
			score.extend(svm.score[0])

		#compute the mindcf on ALL the folds permutations' SCORES and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)

		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)

		llrs_array = numpy.array(score)
		labels_array = numpy.array(labels)
		calibrated_scores, calibrated_labels, w, b = calibration(llrs_array, labels_array)

		#compute the dcf
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(calibrated_labels), numpy.array(calibrated_scores), priors[1], 1, 1)
		#compute norm dcf
		calibratedDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)


		#append the mean of accuracies to the global_accuracies dictionary, with C as key
		global_accuracies[C] = (numpy.mean(accuracies), mindcf, normDcf, calibratedDcf, w, b)

	return global_accuracies
	
		
def svm_poly_pca_k_cross_valid(DTR, LTR, priors, folds):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for m in range(5,12):
		accuracies = []

		labels = []
		score = []
		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			reduced_cv_dtr, P = pca(cv_dtr, m)

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)


			#train the model
			# C = 0.1, K = 1, degree = 2, costant = 0
			svm = SVM_poly(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

			#concatenate model.LTE and model.score
			labels.extend(svm.LTE)
			score.extend(svm.score)

		#compute the mindcf on ALL the folds permutations' SCORES and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)

		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)
	
		global_accuracies[m] = (numpy.mean(accuracies), mindcf, normDcf)

	return global_accuracies


def svm_poly_k_cross_valid(DTR, LTR, folds, C_array, costant_array, priors, K_array=[1], degree=2, pcaVal=-1):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for K in K_array:

		for C in C_array:

			for c in costant_array:
				accuracies = []

				labels = []
				score = []

				for i in range(folds):

					#get the training data
					cv_dtr = cv_dtr_array[i]
					cv_ltr = cv_ltr_array[i]

					#get the test data
					cv_dte = cv_dte_array[i]
					cv_lte = cv_lte_array[i]

					if(pcaVal > 0):
						cv_dtr, P = pca(cv_dtr, pcaVal)
						cv_dte = numpy.dot(P.T, cv_dte)

					#train the model
					svm = SVM_poly(cv_dtr, cv_ltr, cv_dte, cv_lte, priors, C, K, degree=degree, costant=c)
					svm.train()
					svm.test()

					accuracies.append(svm.accuracy)

					#concatenate model.LTE and model.score
					labels.extend(svm.LTE)
					score.extend(svm.score)

				#compute the mindcf on ALL the folds permutations' SCORES and LABELS
				mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)

				confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
				#compute norm dcf
				normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)


				global_accuracies[(K,C,c)] = (numpy.mean(accuracies), mindcf, normDcf)

	return global_accuracies


def svm_rbf_pca_k_cross_valid(DTR, LTR, priors, folds):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for m in range(5,12):
		accuracies = []

		labels = []
		score = []
		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			reduced_cv_dtr, P = pca(cv_dtr, m)

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)


			#train the model
			# C = 0.1, K = 1, gamma = 1
			svm = SVM_RBF(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

			#concatenate model.LTE and model.score
			labels.extend(svm.LTE)
			score.extend(svm.score)

		#compute the mindcf on ALL the folds permutations' SCORES and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)

		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)

		
		global_accuracies[m] = (numpy.mean(accuracies), mindcf, normDcf)

	return global_accuracies


def svm_RBF_k_cross_valid(DTR, LTR, folds, C_array, gamma_array, priors, K_array=[1], pcaVal = -1):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for K in K_array:

		for C in C_array:

			for gamma in gamma_array:
				accuracies = []

				labels = []
				score = []

				for i in range(folds):

					#get the training data
					cv_dtr = cv_dtr_array[i]
					cv_ltr = cv_ltr_array[i]

					#get the test data
					cv_dte = cv_dte_array[i]
					cv_lte = cv_lte_array[i]

					if(pcaVal > 0):
						cv_dtr, P = pca(cv_dtr, pcaVal)
						cv_dte = numpy.dot(P.T, cv_dte)

					#train the model
					svm = SVM_RBF(cv_dtr, cv_ltr, cv_dte, cv_lte, priors, C, K, gamma=gamma)
					svm.train()
					svm.test()

					accuracies.append(svm.accuracy)

					#concatenate model.LTE and model.score
					labels.extend(svm.LTE)
					score.extend(svm.score)

				#compute the mindcf on ALL the folds permutations' SCORES and LABELS
				mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)

				confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
				#compute norm dcf
				normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)


				global_accuracies[(K,C,gamma)] = (numpy.mean(accuracies), mindcf, normDcf)


	return global_accuracies

#endregion

#region GMM
def gmm_pca_k_cross_valid(DTR, LTR, priors, folds):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for m in range(5,12):
		accuracies = []

		labels = []
		score = []
		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			reduced_cv_dtr, P = pca(cv_dtr, m)

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)


			#train the model
			# alpha = 0.1, psi=0.01, type=full covariance, n_components=4
			svm = GMM(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors, 2)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

			#concatenate model.LTE and model.score
			labels.extend(svm.LTE)
			score.extend(svm.llrs)

		#compute the mindcf on ALL the folds permutations' SCORES and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(score), priors[1], 1, 1)
		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)

		
		global_accuracies[m] = (numpy.mean(accuracies), mindcf, normDcf)

	return global_accuracies

	
def gmm_k_fold_cross_valid_components(DTR, LTR, folds, priors, alpha, psi, type="full", pcaVal=-1):
	#for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	# 2**iteration = number of components
	for iteration in range(5):
		accuracies = []
		labels = []
		llrs = []
		
		for i in range(folds):
			#delete i-th component from indices
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]


			if(pcaVal > 0):
				cv_dtr, P = pca(cv_dtr, pcaVal)
				cv_dte = numpy.dot(P.T, cv_dte)

			#train the model
			gmm = GMM(cv_dtr, cv_ltr, cv_dte, cv_lte, priors, iterations=iteration, alpha=alpha, psi=psi, typeOfGmm=type)
			gmm.train()
			gmm.test()
			accuracies.append(gmm.accuracy)

			#concatenate model.LTE and model.llrs
			labels.extend(gmm.LTE)
			llrs.extend(gmm.llrs)

		#compute the mindcf on ALL the folds permutations' LLRS and LABELS
		mindcf = compute_min_dcf(numpy.array(labels), numpy.array(llrs), priors[1], 1, 1)
		
		confusion_matrix = compute_confusion_matrix_binary(numpy.array(labels), numpy.array(llrs), priors[1], 1, 1)
		#compute norm dcf
		normDcf = compute_normalized_dcf_binary(confusion_matrix, priors[1], 1, 1)


		#append the mean of accuracies to the global_accuracies dictionary, with m as key
		global_accuracies[iteration] = (numpy.mean(accuracies), mindcf, normDcf)
	
	return global_accuracies
	
#endregion






