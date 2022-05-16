import numpy
from scipy import special

from utilityML.utility_functions import vrow, mcol

class Multinomial:
    def create_dictionary_of_words(tercets):
        labeled_words = {}
        progressive_i = 0

        for tercet in tercets:
            for word in tercet.split():
                if word not in labeled_words:
                    labeled_words[word] = progressive_i
                    progressive_i += 1
                else:
                    continue
        
        return labeled_words

    #For every word in every tercet, we increment the counters inside words_count_array.
    #The index used to access the counters is the index of the word in the dictionary.
    def count_word_occurrences(words_count_array, tercets, labeled_words):
        for tercet in tercets:
            for word in tercet.split():
                words_count_array[labeled_words[word]] += 1
        
        return words_count_array

    #For every array of words count, one for each class, we compute the model parameters of that class
    def estimate_model_parameters(array_of_class_words_count):

        array_pi_c_j = []

        #NB: we consider the log of pi_c_j, to optimize the computation
        for word_count in array_of_class_words_count:
            array_pi_c_j.append(numpy.log(word_count) - numpy.log(sum(word_count)))

        return array_pi_c_j

    #For each model parameter, so for each class, we compute the class conditional log likelihoods.
    #This is done by considering all the evaluation tercets and computing the log likelihood of each tercet for each class
    def compute_classconditional_loglikelihoods(class_model_parameters_array, tercetsEvaluation, labeled_words):
        
        array_of_cc_ll = []
        for class_model_parameters in class_model_parameters_array:

            cc_ll = []

            for tercet in tercetsEvaluation:
                cc_ll_tercet = 0
                for word in tercet.split():
                    if word in labeled_words:
                        cc_ll_tercet += class_model_parameters[labeled_words[word]]
                cc_ll.append(cc_ll_tercet)

            array_of_cc_ll.append(cc_ll)

        return array_of_cc_ll


    def mcol(v):
        return v.reshape((v.size, 1))

    #Calculate the class posterior probabilities, which is the endgame of our model
    #We use the logsumexp trick to avoid underflow
    #And the mcol method to reshape the class prior probabilities (to do the matrix multiplication)
    def compute_classposterior_probabilities(class_conditional_loglikelihoods_array, class_prior_probabilities_array):
        #Expressions taken from the Lab5
        logSJoint = numpy.vstack(class_conditional_loglikelihoods_array) * mcol(numpy.array(class_prior_probabilities_array))
        logSMarginal = vrow(special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = numpy.exp(logSPost)

        return SPost


    #With this method, we compute the accuracy of our model.
    #array_of_labels_array contains the labels of the evaluation tercets, splitted by class
    #The array this function returns contains the accuracy of each class followed by the accuracy of the whole model
    def compute_accuracy(array_of_labels_array, predicted_labels):
        accuracy_array = []

        #class-specific accuracy
        concatenated_labels = []
        lower_bound = 0
        for labels_array in array_of_labels_array:
            upper_bound = lower_bound + len(labels_array)
            accuracy_array.append(sum(labels_array == predicted_labels[lower_bound:upper_bound]) / len(labels_array))
            lower_bound = upper_bound

            concatenated_labels.extend(labels_array)
        
        #Total accuracy
        accuracy_array.append(sum(concatenated_labels == predicted_labels) / len(concatenated_labels))

        return accuracy_array