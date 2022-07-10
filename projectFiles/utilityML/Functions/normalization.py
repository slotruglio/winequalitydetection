import numpy
from utilityML.Functions.genpurpose import compute_mean


def normalize(dataset, mu=None, sigma=None):
    """
    Z Normalize the dataset
    if normalizing training set, then mu and sigma are computed from the training set
    if normalizing test set, provide mu and sigma
    """
    if mu is None:
        mu = compute_mean(dataset)
    if sigma is None:
        sigma = dataset.std(axis=1)
        sigma = sigma.reshape((len(sigma), 1))

    return (dataset-mu) / sigma, mu, sigma
