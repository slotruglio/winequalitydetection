import numpy

from utilityML.Functions.normalization import normalize
from utilityML.Functions.crossvalid import fold_data
from utilityML.Functions.crossvalid import gmm_pca_k_cross_valid
from utilityML.Functions.bayes import compute_min_dcf
from utilityML.Classifiers.GMM import GMM

from utilityML.Functions.genpurpose import load

def gmm_calculate_best_combo_ds_and_pca(dataset, labels, priors, folds):
        
    normalized, mu, sigma = normalize(dataset)

    results = {}

    for DTR, type in zip([dataset, normalized], ["raw", "norm"]):

        cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, labels, folds)

        no_pca_labels = []
        no_pca_score = []

        for i in range(folds):
                
            #get the training data
            cv_dtr = cv_dtr_array[i]
            cv_ltr = cv_ltr_array[i]
    
            #get the test data
            cv_dte = cv_dte_array[i]
            cv_lte = cv_lte_array[i]

            gmm = GMM(cv_dtr, cv_ltr, cv_dte, cv_lte, priors)
            gmm.train()
            gmm.test()

            no_pca_labels.extend(gmm.LTE)
            no_pca_score.extend(gmm.llrs)

        mindcf = compute_min_dcf(numpy.array(no_pca_labels), numpy.array(no_pca_score), priors[1], 1, 1)
        results[(type, "no pca")] = mindcf

        print("no pca calculated")

        pca_result = gmm_pca_k_cross_valid(DTR, labels, priors, folds)
        for x in pca_result.items():
            results[(type, x[0])] = x[1][1]

    return sorted(results.items(), key=lambda x: x[1][0])

if __name__ == "__main__":

	# testing data

	#LOAD THE DATA
	DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
	DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


	#COMPUTE CLASS PRIORS: label_i / total_labels
	prior_0 = 0.5
	prior_1 = 0.5


	gmm = gmm_calculate_best_combo_ds_and_pca(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/gmm_data_pca.txt", "w") as f:
		for x in gmm:
			f.write(str(x) + "\n")