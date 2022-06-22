from utilityML.Functions.crossvalid import logreg_pca_k_fold_crossvalidation
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load


def quad_logreg_calculate_best_combo(dataset, labels, priors, folds):
    normalized, mu, sigma = normalize(dataset)

    results = {}
    for DTR, type in zip([dataset, normalized], ["raw", "norm"]):
        result = logreg_pca_k_fold_crossvalidation(DTR, labels, priors, folds, quadratic=True)

        for x in result.items():
            results[(type, x[0])] = (x[1][1], x[1][2], x[1][3])
    
    return sorted(results.items(), key=lambda x: x[1][0][0])


if __name__ == "__main__":
	# testing data

	#LOAD THE DATA
	DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
	DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


	#COMPUTE CLASS PRIORS: label_i / total_labels
	prior_0 = 0.5
	prior_1 = 0.5

	logreg = quad_logreg_calculate_best_combo(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/quad_logreg_results.txt", "w") as f:
		for x in logreg:
			f.write(str(x)[1:-1] + "\n")