import numpy

from utilityML.Functions.normalization import normalize
from utilityML.Functions.crossvalid import fold_data
from utilityML.Functions.crossvalid import svm_linear_pca_k_cross_valid, svm_poly_pca_k_cross_valid, svm_rbf_pca_k_cross_valid
from utilityML.Functions.bayes import compute_min_dcf

from utilityML.Classifiers.SVM import SVM_linear, SVM_poly, SVM_RBF

# imoport for testing pourpose
from utilityML.Functions.genpurpose import load

def svm_calculate_best_combo_ds_and_pca(svm_type, svm_pca_function, dataset, labels, priors, folds):
    
    normalized, mu, sigma = normalize(dataset)

    results = {}

    for DTR, type in zip([dataset, normalized], ["raw", "norm"]):

        cv_dtr_array, cv_ltr_array, cv_dte_array, cv_lte_array = fold_data(DTR, LTR, folds)
        
        no_pca_labels = []
        no_pca_score = []
        for i in range(folds):

            #get the training data
            cv_dtr = cv_dtr_array[i]
            cv_ltr = cv_ltr_array[i]

            #get the test data
            cv_dte = cv_dte_array[i]
            cv_lte = cv_lte_array[i]

            if svm_type == "linear":
                svm = SVM_linear(cv_dtr, cv_ltr, cv_dte, cv_lte, priors)
                svm.train()
                svm.test()
                no_pca_labels.extend(svm.LTE)
                no_pca_score.extend(svm.score[0])
            elif svm_type == "poly":
                svm = SVM_poly(cv_dtr, cv_ltr, cv_dte, cv_lte, priors)
                svm.train()
                svm.test()
                no_pca_labels.extend(svm.LTE)
                no_pca_score.extend(svm.score)
            elif svm_type == "rbf":
                svm = SVM_RBF(cv_dtr, cv_ltr, cv_dte, cv_lte, priors)
                svm.train()
                svm.test()
                no_pca_labels.extend(svm.LTE)
                no_pca_score.extend(svm.score)
        
        mindcf = compute_min_dcf(numpy.array(no_pca_labels), numpy.array(no_pca_score), priors[1], 1, 1)
        results[(type, "no pca")] = mindcf

        print("no pca calculated")
        

        pca_result = svm_pca_function(DTR, labels, priors, folds)
        for x in pca_result.items():
            results[(type, x[0])] = x[1][1]

    
    return sorted(results.items(), key=lambda x: x[1][0])


# testing data

#LOAD THE DATA
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


#COMPUTE CLASS PRIORS: label_i / total_labels
prior_0 = (LTR == 0).sum() / LTR.shape[0]
prior_1 = (LTR == 1).sum() / LTR.shape[0]


svm_linear = svm_calculate_best_combo_ds_and_pca("linear", svm_linear_pca_k_cross_valid, DTR, LTR, [prior_0, prior_1], 10)
svm_poly = svm_calculate_best_combo_ds_and_pca("poly", svm_poly_pca_k_cross_valid, DTR, LTR, [prior_0, prior_1], 10)
svm_rbf = svm_calculate_best_combo_ds_and_pca("rbf", svm_rbf_pca_k_cross_valid, DTR, LTR, [prior_0, prior_1], 10)


with open("results/svm_linear_data_pca.txt", "w") as f:
    for x in svm_linear:
        f.write(str(x) + "\n")

with open("results/svm_poly_data_pca.txt", "w") as f:
    for x in svm_poly:
        f.write(str(x) + "\n")

with open("results/svm_rbf_data_pca.txt", "w") as f:
    for x in svm_rbf:
        f.write(str(x) + "\n")