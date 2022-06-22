# Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.plot import *
from utilityML.Functions.normalization import *
from utilityML.Functions.dimred import pca

# LOAD THE DATA
DTR, LTR = load("data/Train.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
DTE, LTE = load("data/Test.txt", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)

# features as array
features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
# labels as array
labels = [
    "bad quality",
    "good quality"
]

# SET SAVE = TRUE TO SAVE AND DON'T VIEW FIGURES
# SET SAVE = FALSE TO VIEW FIGURES AND DON'T SAVE
SAVE = False

# histogram
plotHist(DTR, LTR, features, labels, "Train dataset", save=SAVE)

DTR_gauss, mu, sigma = normalize(DTR)

plotHist(DTR_gauss, LTR, features, labels, "Normalized Train dataset", save=SAVE)

# labels (sum of data for each class)
plotLabels(LTR, labels, save=SAVE)


# scatter with pca m=2 (2 dimensions)
DTR_pca, P = pca(DTR, 2)
DTR_gauss_pca, P_gauss = pca(DTR_gauss, 2)

plot_scatter(DTR_pca, LTR, "PCA m=2", save=SAVE)
plot_scatter(DTR_gauss_pca, LTR, "Norm + PCA  m=2", save=SAVE)