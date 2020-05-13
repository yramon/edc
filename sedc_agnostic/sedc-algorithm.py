"""
Model-agnostic function SEDC for finding Evidence Counterfactuals.
Last update: 13 May 2020.
"""

""" Import libraries """
import time
import numpy as np 
from scipy.sparse import lil_matrix
from ordered_set import OrderedSet
from itertools import compress

from fn_sedc import perturb_fn, expand_and_prune

class SEDC_Explainer(object):
    """Class for generating evidence counterfactuals for classifiers on behavioral/text data"""
    
    def __init__(self, feature_names, classifier_fn, threshold_classifier,
                 max_iter = 50, max_explained = 1, BB = True, max_features = 30, 
                 time_maximum = 120):
        
        """ Init function
        
        Args:
            classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba 
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:
                
                def classifier_fn(X):
                    c=classification_model.predict_proba(X)
                    y_predicted_proba=c[:,1]
                    return y_predicted_proba
            
            threshold_classifier: [float] the threshold that is used for classifying 
            instances as positive or not. When score or probability exceeds the 
            threshold value, then the instance is predicted as positive. 
            We have no default value, because it is important the user decides 
            a good value for the threshold. 
            
            feature_names: [numpy.array] contains the interpretable feature names, 
            such as the words themselves in case of document classification or the names 
            of visited URLs.
            
            max_iter: [int] maximum number of iterations in the search procedure.
            Default is set to 50.
            
            max_explained: [int] maximum number of EDC explanations generated.
            Default is set to 1.
            
            BB: [“True” or “False”]  when the algorithm is augmented with 
            branch-and-bound (BB=True), one is only interested in the (set of) 
            shortest explanation(s). Default is "True".
            
            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.
            
            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """
        
        self.feature_names=feature_names
        self.classifier_fn=classifier_fn
        self.threshold_classifier=threshold_classifier
        self.max_iter=max_iter
        self.max_explained=max_explained
        self.BB=BB
        self.max_features=max_features
        self.time_maximum=time_maximum
        
    def explanation(self, instance):
        """ Generates evidence counterfactual explanation for the instance.
        
        Args:
            instance: [numpy.array or sparse matrix] instance to explain
        
        Returns:
            A tuple (explanation_set[0:self.max_explained], number_active_elements, 
            number_explanations, minimum_size_explanation, time_elapsed, 
            explanations_score_change[0:self.max_explained]), where:
                
                explanation_set: explanation(s) ranked from high to low change 
                in predicted score or probability.
                The number of explanations shown depends on the argument max_explained.
                
                number_active_elements: number of active elements of 
                the instance of interest.
                
                number_explanations: number of explanations found by algorithm.
                
                minimum_size_explanation: number of features in the smallest explanation.
                
                time_elapsed: number of seconds passed to generate explanation(s).
                
                explanations_score_change: change in predicted score/probability
                when removing the features in the explanation, ranked from 
                high to low change.
        """
        
        # *** INITIALIZATION ***
        print("Start initialization...")
        tic = time.time()       
        instance = lil_matrix(instance)
        iteration = 0
        nb_explanations = 0
        minimum_size_explanation = np.nan
        explanations = []
        explanations_sets = []
        explanations_score_change = []
        expanded_combis = []
        score_predicted = self.classifier_fn(instance)
        indices_active_elements = np.nonzero(instance)[1]
        number_active_elements = len(indices_active_elements)
        indices_active_elements = indices_active_elements.reshape((number_active_elements,1))
        
        candidates_to_expand = []
        for features in indices_active_elements:
            candidates_to_expand.append(OrderedSet(features)) 

        explanation_candidates = candidates_to_expand.copy()  
        
        feature_set = [frozenset(x) for x in indices_active_elements]
        
        print('Initialization is complete.')
        print('\n Elapsed time %d \n' %(time.time() - tic))
        
        # *** WHILE LOOP ***
        while (iteration < self.max_iter) and (nb_explanations < self.max_explained) and (len(candidates_to_expand) != 0) and (len(explanation_candidates) != 0) and ((time.time() - tic) < self.time_maximum): 
                        
            iteration += 1
            print('\n Iteration %d \n' %iteration)
            
            if (iteration==1):
                perturbed_instances = [perturb_fn(x, inst = instance.copy()) for x in explanation_candidates]
                scores_explanation_candidates = [self.classifier_fn(x) for x in perturbed_instances]
                scores_candidates_to_expand = scores_explanation_candidates.copy()
            
            scores_perturbed_new_combinations = [x[0] for x in scores_explanation_candidates]
            
            # ***CHECK IF THERE ARE EXPLANATIONS***
            explanations += list(compress(explanation_candidates, scores_perturbed_new_combinations < self.threshold_classifier))
            nb_explanations += len(list(compress(explanation_candidates, scores_perturbed_new_combinations < self.threshold_classifier)))
            explanations_sets += list(compress(explanation_candidates, scores_perturbed_new_combinations < self.threshold_classifier))
            explanations_sets = [set(x) for x in explanations_sets]
            explanations_score_change += list(compress(scores_explanation_candidates, scores_perturbed_new_combinations < self.threshold_classifier))
            
            #Adjust max_length
            if (self.BB == True):
                if (len(explanations)!=0):
                    lengths = []
                    for explanation in explanations:
                        lengths.append(len(explanation))
                    lengths = np.array(lengths)
                    max_length = lengths.min()
                else: 
                    max_length = number_active_elements 
            else: 
                max_length = number_active_elements 
            
            #Eliminate combinations from candidates_to_expand ("best-first" candidates) that can not be expanded
            #Pruning based on Branch & Bound=True, max. features allowed and number of active features
            candidates_to_expand_updated = []
            scores_candidates_to_expand_updated = [] 
            for j, combination in enumerate(candidates_to_expand):
                if ((len(combination) < number_active_elements) and (len(combination) < max_length) and (len(combination) < self.max_features)):
                    candidates_to_expand_updated.append(combination)
                    scores_candidates_to_expand_updated.append(scores_candidates_to_expand[j])
                    
            # *** IF LOOP ***
            if (len(candidates_to_expand_updated) == 0) or (nb_explanations >= self.max_explained):
                
                print("Stop iterations...")
                explanation_candidates = [] #stop algorithm
            
            elif (len(candidates_to_expand_updated) != 0):
                
                explanation_candidates = []
                it = 0 
                indices = []
                
                scores_candidates_to_expand2 = []
                for score in scores_candidates_to_expand_updated:
                    if score[0] < self.threshold_classifier:
                        scores_candidates_to_expand2.append(2 * score_predicted)
                    else:
                        scores_candidates_to_expand2.append(score)
 
                # *** WHILE LOOP ***                       
                while ((len(explanation_candidates) == 0) and (it < len(scores_candidates_to_expand2)) and ((time.time() - tic) < self.time_maximum)):

                    print('While loop iteration %d' %it)

                    if (it != 0):
                        for index in indices:
                            scores_candidates_to_expand2[index] = 2 * score_predicted
                    
                    index_combi_max = np.argmax(score_predicted - scores_candidates_to_expand2)
                    indices.append(index_combi_max)
                    expanded_combis.append(candidates_to_expand_updated[index_combi_max])
                    
                    comb_to_expand = candidates_to_expand_updated[index_combi_max]
                    func = expand_and_prune(comb_to_expand, expanded_combis, feature_set, candidates_to_expand_updated, explanations_sets, scores_candidates_to_expand_updated, instance, self.classifier_fn)
                    explanation_candidates = func[0]
                    candidates_to_expand = func[1]
                    expanded_combis = func[2]
                    scores_candidates_to_expand = func[3]
                    scores_explanation_candidates = func[4]
                   
                    it += 1
                    
            print('\n Elapsed time %d \n' %(time.time() - tic))


        # *** FINAL PART OF ALGORITHM ***                 
        print("Iterations are done.") 
           
        explanation_set=[]
        explanation_feature_names=[]
        for i in range(len(explanations)):
            explanation_feature_names=[]
            for features in explanations[i]:
                explanation_feature_names.append(self.feature_names[features])
            explanation_set.append(explanation_feature_names)
                
        if (len(explanations)!=0):
            lengths_explanation=[]
            for explanation in explanations:
                l=len(explanation)
                lengths_explanation.append(l)
            minimum_size_explanation=np.min(lengths_explanation)
        
        number_explanations=len(explanations)
        if (np.size(explanations_score_change) > 1):
            inds=np.argsort(explanations_score_change, axis=0)
            inds = np.fliplr([inds])[0]
            inds_2=[]
            for i in range(np.size(inds)):
                inds_2.append(inds[i][0])
            explanation_set_adjusted=[]
            for i in range(np.size(inds)):
                j=inds_2[i]
                explanation_set_adjusted.append(explanation_set[j])
            explanations_score_change_adjusted=[]
            for i in range(np.size(inds)):
                j=inds_2[i]
                explanations_score_change_adjusted.append(explanations_score_change[j])
            explanation_set = explanation_set_adjusted
            explanations_score_change = explanations_score_change_adjusted
        
        time_elapsed = time.time() - tic
        print('\n Total elapsed time %d \n' %time_elapsed)

        return (explanation_set[0:self.max_explained], number_active_elements, number_explanations, minimum_size_explanation, time_elapsed, explanations_score_change[0:self.max_explained], iteration)
