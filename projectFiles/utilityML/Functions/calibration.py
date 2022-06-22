import numpy as np
from ..Classifiers import LogReg

def calibration(train_scores, LTR, scores, prior):

	logreg = LogReg(train_scores, LTR, None, None, 10**-3)

	logreg.estimate_model_parameters()

	w, b = logreg.estimated_w, logreg.estimated_b

	return w*scores + b - np.log(prior/(1-prior))
