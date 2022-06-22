from utilityML.Functions.crossvalid import svm_linear_k_cross_valid_C, svm_poly_k_cross_valid, svm_RBF_k_cross_valid, gmm_k_fold_cross_valid_components
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load

# RESULTS OBTAINED FROM gmmPreprocessing.py
# consider only the top 3 combo by mindcf with at least one with
# a different dataset (if 3 norm, 3 norm + 1 raw)

# values have been copied from results/*_data_pca.txt
best_combo_gmm = [("raw", 10), ("raw", 9), ("raw", "no pca"), ("norm", "no pca")]


def calculate_gmm_parameters(dataset, labels, priors, folds):
    results = {}
    for dsType, pca in best_combo_gmm:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            for type in ["full", "diag", "tied_full", "tied_diag"]:
                result = gmm_k_fold_cross_valid_components(DTR, labels, folds, priors, alpha=0.1, psi=0.01, type=type, pcaVal=-1)
                for x in result.items():
                    results[(dsType, pca, type, x[0])] = (x[1][1],x[1][2])
        else:
            for type in ["full", "diag", "tied_full", "tied_diag"]:
                result = gmm_k_fold_cross_valid_components(DTR, labels, folds, priors, alpha=0.1, psi=0.01, type=type, pcaVal=pca)
                for x in result.items():
                    results[(dsType, pca, type, x[0])] = (x[1][1],x[1][2])
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0][0])


if __name__ == "__main__":

    #LOAD THE DATA
	DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
	DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


	#COMPUTE CLASS PRIORS: label_i / total_labels
	prior_0 = 0.5
	prior_1 = 0.5

	gmm = calculate_gmm_parameters(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/gmm_optimization.txt", "w") as f:
		for x in gmm:
			f.write(str(x)[1:-1] + "\n")


	print("done gmm")
