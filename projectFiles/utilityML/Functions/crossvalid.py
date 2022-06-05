import numpy
from utilityML.Functions.dimred import *
from utilityML.Functions.genpurpose import split_db_2to1
from utilityML.Classifiers.SVM import SVM_linear, SVM_poly
from utilityML.Classifiers.LogReg import LogReg


def split_leave_one_out(D, L, index):
    D_train = numpy.delete(D, index, 1)
    L_train = numpy.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)

#region gaussian pca
def gaussian_pca_crossvalidation(classifier, DTR, LTR, priors, k=None, percentage=2./3.):
	if k is None:
		return gaussian_pca_1_fold_crossvalidation(classifier, DTR, LTR, priors, percentage)
	else:
		return gaussian_pca_k_fold_crossvalidation(classifier, DTR, LTR, priors, k)

def gaussian_pca_k_fold_crossvalidation(classifier, DTR, LTR, priors, k):

	#for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	for m in range(1,11):

		accuracies = []
		for i in range(k):

			#delete i-th component from indices
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]
			
			reduced_cv_dtr, P = pca(cv_dtr, cv_ltr, m)

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
		
		#append the mean of accuracies to the global_accuracies dictionary, with m as key
		global_accuracies[m] = numpy.mean(accuracies)
	
	return global_accuracies
	

def gaussian_pca_1_fold_crossvalidation(classifier, DTR, LTR, priors, percentage=2./3.):

	#for each group, compute the accuracy
	global_accuracies = {}

	(cv_dtr, cv_ltr), (cv_dte, cv_lte) = split_db_2to1(DTR, LTR, percentage)

	for m in range(1,11):

		reduced_cv_dtr, P = pca(cv_dtr, cv_ltr, m)

		# get projected samples of test data
		reduced_cv_dte = numpy.dot(P.T, cv_dte)

		#train the model
		model = classifier(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, priors)
		model.train()
		model.test()

		#append the accuracy to the global_accuracies dictionary, with m as key
		global_accuracies[m] = model.accuracy

	return global_accuracies
	
	
#endregion


#region logreg pca
def logreg_pca_crossvalidation(DTR, LTR, k=None, percentage=2./3.):
	if k is None:
		return logreg_pca_1_fold_crossvalidation(DTR, LTR, percentage)
	else:
		return logreg_pca_k_fold_crossvalidation(DTR, LTR, k)


def logreg_pca_1_fold_crossvalidation(DTR, LTR, percentage=2./3.):

	#for each group, compute the accuracy
	global_accuracies = {}

	(cv_dtr, cv_ltr), (cv_dte, cv_lte) = split_db_2to1(DTR, LTR, percentage)

	for m in range(1,11):

		accuracies = []

		for l in [10**-6, 10**-3, 10**-1, 1.0]:
			reduced_cv_dtr, P = pca(cv_dtr, cv_ltr, m)

			# get projected samples of test data
			reduced_cv_dte = numpy.dot(P.T, cv_dte)

			#train the model
			model = LogReg(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, l)
			model.estimate_model_parameters()
			model.logreg_test()

			accuracies.append(model.accuracy)

		#append the accuracy to the global_accuracies dictionary, with m as key
		global_accuracies[m] = model.accuracy

	return global_accuracies


def logreg_pca_k_fold_crossvalidation(DTR, LTR, k):

	#for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, k)

	for m in range(1,11):

		for l in [10**-6, 10**-3, 10**-1, 1.0]:

			accuracies = []
			for i in range(k):

				#delete i-th component from indices
				cv_dtr = cv_dtr_array[i]
				cv_ltr = cv_ltr_array[i]
				
				reduced_cv_dtr, P = pca(cv_dtr, cv_ltr, m)

				#get the test data
				cv_dte = cv_dte_array[i]
				cv_lte = cv_lte_array[i]

				# get projected samples of test data
				reduced_cv_dte = numpy.dot(P.T, cv_dte)

				#train the model
				model = LogReg(reduced_cv_dtr, cv_ltr, reduced_cv_dte, cv_lte, l)
				model.estimate_model_parameters()
				model.logreg_test()

				accuracies.append(model.accuracy)
		
			#append the mean of accuracies to the global_accuracies dictionary, with m as key
			global_accuracies[(m,l)] = numpy.mean(accuracies)
	
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
def svm_linear_cross_valid_C(DTR, LTR, C_array, K=1, folds=None, percentage=2./3.):
    if folds is None:
        svm_linear_single_cross_valid_C(DTR, LTR, C_array, K)
    else:
        svm_linear_k_cross_valid_C(DTR, LTR, folds, C_array, K)

def svm_linear_single_cross_valid_C(DTR, LTR, C_array, K=1, percentage=2./3.):
    (cv_DTR, cv_LTR), (cv_DTE, cv_LTE) = split_db_2to1(DTR, LTR, percTraining=percentage)
    accuracies = {}
    for C in C_array:
        svm = SVM_linear(cv_DTR, cv_LTR, cv_DTE, cv_LTE, C, K)
        svm.train()
        svm.test()
        accuracies[C] = svm.accuracy
    
    print(accuracies)
    print(max(accuracies.items(), key=lambda x: x[1]))

def svm_linear_k_cross_valid_C(DTR, LTR, folds, C_array, K=1):

	# for each group, compute the accuracy
	global_accuracies = {}

	cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
	for C in C_array:
		accuracies = []
		for i in range(folds):

			#get the training data
			cv_dtr = cv_dtr_array[i]
			cv_ltr = cv_ltr_array[i]

			#get the test data
			cv_dte = cv_dte_array[i]
			cv_lte = cv_lte_array[i]

			#train the model
			svm = SVM_linear(cv_dtr, cv_ltr, cv_dte, cv_lte, C, K)
			svm.train()
			svm.test()

			accuracies.append(svm.accuracy)

		#append the mean of accuracies to the global_accuracies dictionary, with C as key
		global_accuracies[C] = numpy.mean(accuracies)

	print(global_accuracies)
	
		

def svm_poly_cross_valid(DTR, LTR, C_array, costant_array, K=1, folds=None, percentage=2./3., degree=2):
    if folds is None:
        svm_poly_single_cross_valid(DTR, LTR, C_array, costant_array, K, degree=degree)
    else:
        svm_poly_k_cross_valid(DTR, LTR, folds, C_array, costant_array, K, degree=degree)

def svm_poly_single_cross_valid(DTR, LTR, C_array, costant_array, K=1, degree=2, percentage=2./3.):
    (cv_DTR, cv_LTR), (cv_DTE, cv_LTE) = split_db_2to1(DTR, LTR, percTraining=percentage)
    accuracies = {}
    for C in C_array:
        C_accuracies = {}
        for c in costant_array:
            svm = SVM_poly(cv_DTR, cv_LTR, cv_DTE, cv_LTE, C, K, degree=degree, costant=c)
            svm.train()
            svm.test()
            C_accuracies[c] = svm.accuracy
        accuracies[C] = C_accuracies
    
    print(accuracies)

def svm_poly_k_cross_valid(DTR, LTR, folds, C_array, costant_array, K=1, degree=2, costant = 0):

    # for each group, compute the accuracy
    global_accuracies = {}

    cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
    for C in C_array:
        C_accuracies = {}
        for c in costant_array:

            accuracies = []
            for i in range(folds):

                #get the training data
                cv_dtr = cv_dtr_array[i]
                cv_ltr = cv_ltr_array[i]

                #get the test data
                cv_dte = cv_dte_array[i]
                cv_lte = cv_lte_array[i]

                #train the model
                svm = SVM_poly(cv_dtr, cv_ltr, cv_dte, cv_lte, C, K, degree=degree, costant=costant)
                svm.train()
                svm.test()

                accuracies.append(svm.accuracy)
            C_accuracies[c] = numpy.mean(accuracies)

        #append the mean of accuracies to the global_accuracies dictionary, with C as key
        global_accuracies[C] = C_accuracies

    print(global_accuracies)

#endregion