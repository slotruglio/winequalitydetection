from utilityML.Functions.crossvalid import svm_linear_k_cross_valid_C, svm_poly_k_cross_valid, svm_RBF_k_cross_valid, gmm_k_fold_cross_valid_components
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load

# RESULTS OBTAINED FROM svmPreprocessing & gmmPreprocessing.py
# consider only the top 3 combo by mindcf with at least one with
# a different dataset (if 3 norm, 3 norm + 1 raw)

# values have been copied from results/*_data_pca.txt
best_combo_svm_linear = [("norm", 9), ("norm", "no pca"), ("norm", 10), ("raw", 9)]
best_combo_svm_poly = [("norm", "no pca"), ("norm", 10), ("norm", 7), ("raw", 5)]
best_combo_svm_rbf = [("norm", "no pca"), ("norm", 10), ("norm", 8), ("raw", 10)]

def calculate_svm_linear_paramaters(dataset, labels, priors, folds):
    
    results = {}
    for dsType, pca in best_combo_svm_linear:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            result = svm_linear_k_cross_valid_C(DTR, labels, folds, [0.1, 1, 10], priors, pcaVal=-1)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        else:
            result = svm_linear_k_cross_valid_C(DTR, labels, folds, [0.1, 1, 10], priors, pcaVal=pca)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0][0])

def calculate_svm_poly_paramaters(dataset, labels, priors, folds):
    
    results = {}
    for dsType, pca in best_combo_svm_poly:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            
            result = svm_poly_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [0,1], priors, [0,1], pcaVal=-1)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        else:
            result = svm_poly_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [0,1], priors, [0,1], pcaVal=pca)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0][0])

def calculate_svm_rbf_paramaters(dataset, labels, priors, folds):
    
    results = {}
    for dsType, pca in best_combo_svm_rbf:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            
            result = svm_RBF_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [1,10], priors, [0,1], pcaVal=-1)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        else:
            result = svm_RBF_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [1,10], priors, [0,1], pcaVal=pca)
            for x in result.items():
                results[(dsType, pca, x[0])] = (x[1][1],x[1][2])
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0][0])


if __name__ == "__main__":
	# testing data

	#LOAD THE DATA
	DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
	DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


	#COMPUTE CLASS PRIORS: label_i / total_labels
	prior_0 = 0.5
	prior_1 = 0.5

	svm_linear = calculate_svm_linear_paramaters(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/svm_linear_optimization.txt", "w") as f:
		for x in svm_linear:
			f.write(str(x)[1:-1] + "\n")

	print("done linear")

	svm_poly = calculate_svm_poly_paramaters(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/svm_poly_optimization.txt", "w") as f:
		for x in svm_poly:
			f.write(str(x)[1:-1] + "\n")

	print("done poly")

	svm_rbf = calculate_svm_rbf_paramaters(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/svm_rbf_optimization.txt", "w") as f:
		for x in svm_rbf:
			f.write(str(x)[1:-1] + "\n")

	print("done rbf")



