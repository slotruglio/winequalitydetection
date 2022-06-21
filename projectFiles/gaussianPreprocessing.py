from utilityML.Functions.crossvalid import gaussian_pca_k_fold_crossvalidation
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load

# Classifiers import
from utilityML.Classifiers.MVG import MVG
from utilityML.Classifiers.NaiveBayes import NaiveBayes
from utilityML.Classifiers.TiedCovariance import TiedCovariance
from utilityML.Classifiers.TiedNaive import TiedNaive


def gaussian_calculate_best_combo(classifier, dataset, labels, priors, folds):

    normalized, mu, sigma = normalize(dataset)

    results = {}
    for DTR, type in zip([dataset, normalized], ["raw", "norm"]):
        result = gaussian_pca_k_fold_crossvalidation(classifier, DTR, labels, priors, folds)
        for x in result.items():
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


	with open("results/gaussian_results.txt", "w") as f:
		f.write("MVG\n")
		mvg = gaussian_calculate_best_combo(MVG, DTR, LTR, [prior_0, prior_1], 10)

		for x in mvg:
			f.write(str(x) + "\n")

		f.write("\n")

		f.write("Naive Bayes\n")
		naive = gaussian_calculate_best_combo(NaiveBayes, DTR, LTR, [prior_0, prior_1], 10)

		for x in naive:
			f.write(str(x) + "\n")

		f.write("\n")

		f.write("Tied Covariance\n")
		tied = gaussian_calculate_best_combo(TiedCovariance, DTR, LTR, [prior_0, prior_1], 10)

		for x in tied:
			f.write(str(x) + "\n")

		f.write("\n")

		f.write("Tied Naive\n")
		tied_naive = gaussian_calculate_best_combo(TiedNaive, DTR, LTR, [prior_0, prior_1], 10)

		for x in tied_naive:
			f.write(str(x) + "\n")
