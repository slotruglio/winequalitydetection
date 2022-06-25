from utilityML.Functions.calibration import calibrate_scores_logreg
from utilityML.Functions.crossvalid import logreg_pca_k_fold_crossvalidation
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load


def logreg_calculate_best_combo(dataset, labels, priors, folds):
    normalized, mu, sigma = normalize(dataset)

    results = {}
    for DTR, type in zip([dataset, normalized], ["raw", "norm"]):
        result = logreg_pca_k_fold_crossvalidation(DTR, labels, priors, folds)

        for x in result.items():
            results[(type, x[0])] = (x[1][1], x[1][2])
    
    return sorted(results.items(), key=lambda x: x[1][0][0])


def logreg_compute_calibrated_scores(dataset, labels, best_pca, best_lambda, data_type, priors, folds):
	normalized, mu, sigma = normalize(dataset)
	
	calibrated_dcfs = {}

	if(data_type == "Raw"):
		calibrated_dcfs = calibrate_scores_logreg(DTR, labels, best_pca, best_lambda, folds, data_type)
	else:
		calibrated_dcfs = calibrate_scores_logreg(normalized, labels, best_pca, best_lambda, priors, folds, data_type)

	return calibrated_dcfs


if __name__ == "__main__":
	# testing data

	#LOAD THE DATA
	DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
	DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


	#COMPUTE CLASS PRIORS: label_i / total_labels
	prior_0 = 0.5
	prior_1 = 0.5

	""" logreg = logreg_calculate_best_combo(DTR, LTR, [prior_0, prior_1], 10)

	with open("results/logreg_results.txt", "w") as f:
		for x in logreg:
			f.write(str(x)[1:-1] + "\n")
	 """
	
	
	best_pcas = 11
	best_lambdas = 0.01
	data_type = "Norm"
	


	calibrated_dcfs = logreg_compute_calibrated_scores(DTR, LTR, best_pcas, best_lambdas, data_type, [prior_0, prior_1], 10)


	with open("results/logreg_calibration_results.txt", "w") as f:
		for x in calibrated_dcfs.items():
			f.write(str(x)[1:-1] + "\n")






