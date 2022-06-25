import numpy
import scipy
from ..Functions.genpurpose import *


def compute_confusion_matrix_binary(labels, llrs, pi, Cfn, Cfp, t = None):

	if(t == None):
		t = -numpy.log((pi * Cfn) / ((1-pi) * Cfp))

	#creo la confusion matrix
	confusion_matrix = numpy.zeros((2,2))

	indexes_label_0 = (labels == 0)
	indexes_label_1 = (labels == 1)

	confusion_matrix[0][0] = (llrs[indexes_label_0] <= t).sum()
	confusion_matrix[0][1] = (llrs[indexes_label_1] <= t).sum()	

	confusion_matrix[1][1] = (llrs[indexes_label_1] > t).sum()
	confusion_matrix[1][0] = (llrs[indexes_label_0] > t).sum()

	return confusion_matrix

#LO CHIAMAVANO BAYES RISK
def compute_dcf_binary(confusion_matrix, pi, Cfn, Cfp):
	Bt = pi * Cfn * (confusion_matrix[0][1] / (confusion_matrix[0][1]+confusion_matrix[1][1]))

	Bf = (1-pi) * Cfp * (confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0]))

	return Bt + Bf

#ANCHE CHIAMATO NORMALIZED BAYES RISK
def compute_normalized_dcf_binary(confusion_matrix, pi, Cfn, Cfp):

	best_dummy = min(pi * Cfn, (1-pi) * Cfp)

	return compute_dcf_binary(confusion_matrix, pi, Cfn, Cfp) / best_dummy


def compute_min_dcf(labels, scores, pi, Cfn, Cfp):

	t = numpy.array(scores)
	t.sort()

	threshold_list = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])

	dcf_array = []

	for threshold in threshold_list:
		confusion_matrix = compute_confusion_matrix_binary(labels, scores, pi, Cfn, Cfp, threshold)

		dcf_array.append((compute_normalized_dcf_binary(confusion_matrix, pi, Cfn, Cfp), threshold))


	#return the entry with the minimum first element
	return min(dcf_array, key=lambda x: x[0])


def generate_roc_curve(labels, scores, pi, Cfn, Cfp):

	t = numpy.array(scores)
	t.sort()

	threshold_list = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])

	FPR_array = []
	TPR_array = []

	for threshold in threshold_list:
		confusion_matrix = compute_confusion_matrix_binary(labels, scores, pi, Cfn, Cfp, threshold)
		FPR_array.append(confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0]))
		TPR_array.append(1 - confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]))

	return FPR_array, TPR_array

def bayes_error_plots(plot_name, labels, scores, validation_threshold = None, calibrated_scores = None):

	#IPOTESI DI BASE
	#Una qualsiasi applicazione (pi, Cfn, Cfp), è equivalente
	#All'applicazione (pi_tilde, 1, 1). Dove pi_tilde ha un valore specifico (?)
	#Possiamo quindi considerare tante applicazioni diverse variando pi_tilde

	#DEFINIZIONE FORMALE
	#Il bayes error plot normalizzato verifica la performance del
	#recognizer al variare dell'applicazione
	#Quindi, come una funzione delle prior log-odds p (formula sul lab)

	#APPROCCIO E OSSERVAZIONI
	#1) I valori di p vanno da -3 a 3
	effPriorLogOdds = numpy.linspace(-3, 3, 21)

	DCF_array = []
	DCF_min_array = []

	if(validation_threshold != None):
		DCF_validation = []
	
	if(calibrated_scores is not None):
		DCF_calibrated = []

	#2) Per ogni valore di p, calcolo pi_tilde (usando la formula inversa di p)
	#3) Calcolare DCF attuale e minima (entrambe normalizzate), considerando pi_tilde
	for p in effPriorLogOdds:
		pi_tilde = 1 / (1 + numpy.exp(-p))

		#EMPIRIC DCF
		DCF_array.append(compute_normalized_dcf_binary(compute_confusion_matrix_binary(labels, scores, pi_tilde, 1, 1), pi_tilde, 1, 1))
		#MIN DCF
		DCF_min_array.append(compute_min_dcf(labels, scores, pi_tilde, 1, 1)[0])
		#DCF USING THE VALIDATION THRESHOLD
		if(validation_threshold != None):
			DCF_validation.append(compute_normalized_dcf_binary(compute_confusion_matrix_binary(labels, scores, pi_tilde, 1, 1, t=validation_threshold), pi_tilde, 1, 1))
		if(calibrated_scores is not None):
			DCF_calibrated.append(compute_normalized_dcf_binary(compute_confusion_matrix_binary(labels, calibrated_scores, pi_tilde, 1, 1), pi_tilde, 1, 1))

	#^^^^^ NOTA: nel metodo compute_min_dcf, consideriamo già le DCF normalizzate.

	#4) Plottare i dati come una funzione di p
	#5) PLOT 1: la x è p, la y è la corrispondente DCF normalizzata
	#6) PLOT 2: la x è sempre la p, la y è la corrispondente DCF normalizzata minima

	
	plt.figure()

	
	plt.xlabel("t")
	plt.plot(effPriorLogOdds, DCF_array, label="Empiric DCF", color='r') 
	plt.plot(effPriorLogOdds, DCF_min_array, label="min DCF", color='b')
	if(validation_threshold != None):
		plt.plot(effPriorLogOdds, DCF_validation, label="XVal Threshold DCF", color='y')
	
	if(calibrated_scores is not None):
		plt.plot(effPriorLogOdds, DCF_calibrated, label="Calibrated DCF", color='g')

	

	plt.ylim([0.2, 1.0])
	plt.xlim([-2, 2])
	plt.legend()
	plt.title(plot_name)

	plt.savefig('./img/bayes_error_plots/'+ plot_name)

#region *** MULTICLASS CASE ***
def compute_confusion_matrix(labels, cond_ll, pi_array, c_matrix):

    #Dobbiamo calcolare le post probabilities, a partire dalle conditional log likelihoods
    #Le conditional log likelihoods sono espresse tramite una matrice: ogni riga è una classe, ogni colonna un sample
	logSJoint = cond_ll + numpy.log(pi_array)
	logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
	logSPost = logSJoint - logSMarginal
	SPost = numpy.exp(logSPost)

	optimal_classes = numpy.argmin(c_matrix @ SPost, axis=0)

	confusion_matrix = numpy.zeros((len(cond_ll), len(cond_ll)))
	for i in range(len(confusion_matrix)):
		for j in range(len(confusion_matrix)):
			confusion_matrix[i][j] = (optimal_classes[(labels == j)] == i).sum()


	return confusion_matrix

def compute_dcf_multiclass(pi_array, c_matrix, confusion_matrix):

	dummy_sistem_dcf = numpy.min(numpy.dot(c_matrix, pi_array))

	#divide each element of c_matrix for the sum of its column
	misclass_ratio = confusion_matrix / numpy.sum(confusion_matrix, axis=0)

	DCF_u = numpy.sum(numpy.multiply(misclass_ratio, c_matrix), axis=0) @ pi_array
	DCF = DCF_u / dummy_sistem_dcf

	return DCF_u[0], DCF[0]

#endregion