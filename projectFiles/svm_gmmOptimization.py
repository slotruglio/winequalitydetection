from utilityML.Functions.crossvalid import svm_linear_k_cross_valid_C, svm_poly_k_cross_valid, svm_RBF_k_cross_valid, gmm_k_fold_cross_valid_components
from utilityML.Functions.normalization import normalize
from utilityML.Functions.genpurpose import load

# RESULTS OBTAINED FROM svmPreprocessing & gmmPreprocessing.py
# consider only the top 3 combo by mindcf

# TODO - put real data, these below are just for testing
best_combo_svm_linear = [("raw", 9), ("norm", "no pca")]
best_combo_svm_poly = [("raw", 9), ("norm", "no pca")]
best_combo_svm_rbf = [("raw", 9), ("norm", "no pca")]
best_combo_gmm = [("raw", 9), ("norm", "no pca")]

def calculate_svm_linear_paramaters(dataset, labels, priors, folds):
    
    results = {}
    for dsType, pca in best_combo_svm_linear:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            result = svm_linear_k_cross_valid_C(DTR, labels, folds, [0.1, 1, 10], priors, pcaVal=-1)
            for x in result.items():
                results[(dsType, pca, x[0])] = x[1][1]
        else:
            result = svm_linear_k_cross_valid_C(DTR, labels, folds, [0.1, 1, 10], priors, pcaVal=pca)
            for x in result.items():
                results[(dsType, pca, x[0])] = x[1][1]
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0])

def calculate_svm_poly_paramaters(dataset, labels, priors, folds):
    
    results = {}
    for dsType, pca in best_combo_svm_poly:
        DTR = dataset
        if dsType == "norm":
            DTR, mu, sigma = normalize(dataset)
        
        if pca == "no pca":
            
            result = svm_poly_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [0,1], [prior_0, prior_1], [0,1], pcaVal=-1)
            for x in result.items():
                results[(dsType, pca, x[0])] = x[1][1]
        else:
            result = svm_poly_k_cross_valid(DTR, labels, folds, [0.1, 1, 10], [0,1], [prior_0, prior_1], [0,1], pcaVal=pca)
            for x in result.items():
                results[(dsType, pca, x[0])] = x[1][1]
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0])

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
                    results[(dsType, pca, type, x[0])] = x[1][1]
        else:
            for type in ["full", "diag", "tied_full", "tied_diag"]:
                result = gmm_k_fold_cross_valid_components(DTR, labels, folds, priors, alpha=0.1, psi=0.01, type=type, pcaVal=pca)
                for x in result.items():
                    results[(dsType, pca, type, x[0])] = x[1][1]
        print("done {}, {}".format(dsType, pca))

    # sort by mindcf
    return sorted(results.items(), key=lambda x: x[1][0])

if __name__ == "__main__":
    # testing data

    #LOAD THE DATA
    DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
    DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)


    #COMPUTE CLASS PRIORS: label_i / total_labels
    prior_0 = (LTR == 0).sum() / LTR.shape[0]
    prior_1 = (LTR == 1).sum() / LTR.shape[0]

    #svm_linear = calculate_svm_linear_paramaters(DTR, LTR, [prior_0, prior_1], 3)
    svm_poly = calculate_svm_poly_paramaters(DTR, LTR, [prior_0, prior_1], 3)

    print(svm_poly)

