#Functions import
from utilityML.Functions.genpurpose import load
from utilityML.Functions.plot import *
from utilityML.Functions.gaussianization import *

#LOAD THE DATA
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

# histogram
plotHist(DTR, LTR, features, labels, "Dataset's histogram", save=False)

DTR_gauss = gaussianize(DTR)

plotHist(DTR_gauss, LTR, features, labels, "Dataset's histogram (gaussianized)", save=False)

