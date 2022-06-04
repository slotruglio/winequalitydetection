import numpy
from Printer import Printer
from utilityML.Functions.dimred import *


def split_leave_one_out(D, L, index):
    D_train = numpy.delete(D, index, 1)
    L_train = numpy.delete(L, index)
    D_test = D[:, index:index+1]
    L_test = L[index]
    return (D_train, L_train), (D_test, L_test)


def pca_k_fold_crossvalidation(classifier, DTR, LTR, priors, k):

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
	
	print(global_accuracies)
	
	#get the entry of global_accuracies corresponding to the max value
	print(max(global_accuracies.items(), key=lambda x: x[1]))


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



	
