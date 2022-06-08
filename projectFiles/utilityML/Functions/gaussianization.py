import numpy
from utilityML.Functions.genpurpose import compute_mean


def gaussianize(dataset, mu=None, sigma=None):
    """
    Gaussianize dataset
    """
    if mu is None:
        mu = compute_mean(dataset)
    if sigma is None:
        sigma = dataset.std(axis=1)
        sigma = sigma.reshape((len(sigma), 1))

    return (dataset-mu) / sigma, mu, sigma
