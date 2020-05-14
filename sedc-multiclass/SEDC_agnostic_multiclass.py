"""

Model-agnostic version of algorithm for finding Evidence Counterfactuals (SEDC)
for multiclass (one-versus-rest) tasks.
(= minimal set so that removing features the predicted class changes).

"""

import time
import numpy as np 
from scipy.sparse import lil_matrix
from ordered_set import OrderedSet #use pip install or conda install
from function_edc import fn_1


class SEDC_Explainer_Multiclass(object):
    
    #def __init__(self, feature_names, classifier_fn_multiclass, threshold_classifier, max_iter=50, max_explained=1, BB=True, max_features=30, time_maximum=120):
    def __init__(self, feature_names, classifier_fn_multiclass, max_iter=50, max_explained=1, BB=True, max_features=30, time_maximum=120):

        """ Init function
        
        Args:
            classifier_fn_multiclass: [function] classifier prediction probability function
            or decision function for one-vs-rest multiclass predictions. 
            For ScikitClassifiers, this is classifier.predict_proba or
            classifier.decision_function or classifier.predict_log_proba 
            per class of the target variable.
            Make sure the function only returns a tuple of (float) values. 
            For instance, if you use a ScikitClassifier, transform the multiple
            classifier.predict_proba predictions as follows:
                
                def classifier_fn_multiclass(X):
                    c1=classification_model1.predict_proba(X)[:,1] #model 1
                    c2=classification_model2.predict_proba(X)[:,1] #model 2
                    ...
                    ck=classification_modelk.predict_proba(X)[:,1] #model k
                    return (c1,c2,...,ck)
                    
            feature_names: [numpy.array] contains the interpretable feature names, 
            such as the words themselves in case of document classification or the names 
            of visited URLs.
            
            max_iter: [int] maximum number of iterations in the search procedure.
            Default is set to 50.
            
            max_explained: [int] maximum number of EDC explanations generated.
            Default is set to 1.
            
            BB: [“True” or “False”]  when the algorithm is augmented with branch-and-bound (BB=True), 
            one is only interested in the (set of) shortest explanation(s). Default is True.
            
            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.
            
            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """
        
        self.feature_names=feature_names
        self.classifier_fn_multiclass=classifier_fn_multiclass
        self.max_iter=max_iter
        self.max_explained=max_explained
        self.BB=BB
        self.max_features=max_features
        self.time_maximum=time_maximum
        
    def explanation(self, instance):
        """ Generates explanation(s) for a positively predicted instance
        
        Function for explaining classifiers on high-dimensional, sparse data.
        Type of task: MULTICLASS.
        """
        """
        Args:
            instance: [numpy.array or sparse matrix] instance on which 
            to explain the model prediction
        
        Returns:
            A tuple (explanation_set[0:self.max_explained], number_active_elements, 
            number_explanations, minimum_size_explanation, time_elapsed, where:
                
                explanation_set: explanation(s) ranked from high to low change in predicted score or probability.
                The number of explanations shown depends on the argument max_explained.
                
                number_active_elements: number of active elements of the instance of interest.
                
                number_explanations: number of explanations found by algorithm.
                
                minimum_size_explanation: number of features in the smallest explanation.
                
                time_elapsed: number of seconds passed to generate explanation(s).
                
                explanations_score_change: change in predicted score/probability when removing
                the features in the explanation, ranked from high to low change.
        """
        
# *** INITIALIZATION ***
        
        time_max=0
        tic=time.time()
        instance=lil_matrix(instance)
        iteration=0
        nb_explanations=0
        minimum_size_explanation=np.nan
        explanations=[]
        explanations_sets=[]
        explanations_score_change=[]
        
        class_index = np.argmax(self.classifier_fn_multiclass(instance))
        score_predicted = self.classifier_fn_multiclass(instance)[class_index] 
        #a tuple of predicted scores of one vs rest
        #get predicted score for the class that is predicted
        
        indices_active_elements=np.nonzero(instance)[1]
        number_active_elements=len(indices_active_elements)
        indices_active_elements=indices_active_elements.reshape((number_active_elements,1))
        threshold=-1
        stop=0
        expanded_combis=[]
        
        #use orderedset() 
        combinations_to_expand=[]
        for features in indices_active_elements:
            combinations_to_expand.append(OrderedSet(features))
        #in the first iteration, the new combinations to explore
        #whether it are explanations are the combinations_to_expand
        new_combinations=combinations_to_expand.copy()  
        
        #indices of active features are the feature set to explore
        feature_set=[]
        for features in indices_active_elements:
            feature_set.append(frozenset(features))
            
        time_max += (time.time()-tic)
        
        print('Initialization complete.')
        print('\n Elapsed time %d \n' %(time.time()-tic))

        while (iteration < self.max_iter) and (nb_explanations < self.max_explained) and (len(combinations_to_expand)!=0) and (len(new_combinations)!=0) and (time_max<(self.time_maximum)): 
            
            time_extra=time.time()
            
            iteration+=1
            print('\n Iteration %d \n' %iteration)
            
            new_combinations_to_expand=[]
            scores_new_combinations_to_expand=[]
            for combination in new_combinations: #verify each set in new_combinations if it is an explanation or not
                perturbed_instance=instance.copy()
                for feature_in_combination in combination: 
                    perturbed_instance[:,feature_in_combination]=0
                score_new = self.classifier_fn_multiclass(perturbed_instance)[class_index]
                
                if (score_new[0] != np.max(self.classifier_fn_multiclass(perturbed_instance))): #if class_index has no longer the top predicted score, an explanation is found.
                    explanations.append(combination)
                    explanations_sets.append(set(combination))
                    explanations_score_change.append(score_predicted - score_new)
                    nb_explanations+=1
                else:
                    new_combinations_to_expand.append(combination)
                    scores_new_combinations_to_expand.append(score_new)
           
            if (len(new_combinations[0]) == number_active_elements):  
                stop=1
            else:
                stop=0 
                
            if (self.BB==True): #branch-and-bound
                if (len(explanations)!=0):
                    lengths=[]
                    for explanation in explanations:
                        lengths.append(len(explanation))
                    lengths=np.array(lengths)
                    max_length=lengths.min() 
                else: 
                    max_length=number_active_elements 
            else: 
                max_length=number_active_elements
            
            if (len(scores_new_combinations_to_expand) != 0):
                index_combi_max = np.argmax(score_predicted - scores_new_combinations_to_expand) #best-first combination or feature is chosen.
                new_score = scores_new_combinations_to_expand[index_combi_max]
                difference = score_predicted - new_score
                if difference[0] >= threshold:
                    expand = 1
                else:
                    expand = 0
            else:
                expand = 0

            if ((len(new_combinations[0]) < max_length) and (expand == 1) and (stop==0) and (nb_explanations < self.max_explained) and (len(new_combinations[0]) < self.max_features)): 
                    
                print('length of new_combinations is %d features.' %len(new_combinations[0]))
                print('new combinations can be expanded')
                
                comb=new_combinations_to_expand[index_combi_max]
                func=fn_1(comb, expanded_combis, feature_set, combinations_to_expand, explanations_sets)
                new_combinations=func[0]
                combinations_to_expand=func[1]
                expanded_combis=func[2]
                
                #Calculate new threshold
                scores_combinations_to_expand=[]
                for combination in combinations_to_expand:
                    perturbed_instance=instance.copy()
                    for feature_in_combination in combination:
                        perturbed_instance[:,feature_in_combination]=0
                    score_new = self.classifier_fn_multiclass(perturbed_instance)[class_index]
                    
                    if (score_new[0] == np.max(self.classifier_fn_multiclass(perturbed_instance))):
                        scores_combinations_to_expand.append(score_new)
                    
                index_combi_max = np.argmax(score_predicted - scores_combinations_to_expand)
                new_score = scores_combinations_to_expand[index_combi_max]
                threshold = score_predicted - new_score
                
                time_extra2=time.time()
                time_max+=(time_extra2-time_extra)
                print('\n Elapsed time %d \n' %time_max)
                size_COMBIS=len(combinations_to_expand)
                print('\n size combis to expand %d \n' %size_COMBIS)
                
            else:
                
                print('length of new_combinations is %d features.' %len(new_combinations[0]))
                print('new combination cannot be expanded')
                
                combinations=[]
                for combination in combinations_to_expand:
                    if ((len(combination) < number_active_elements) and (len(combination) < (max_length)) and (len(combination) < self.max_features)):
                        combinations.append(combination)
                
                if (len(combinations) == 0) or (nb_explanations >= self.max_explained) or (len(combinations_to_expand) == len(new_combinations)):
                    new_combinations=[]
                
                elif (len(combinations) != 0):
                    
                    new_combinations=[]
                    it=0
                    indices=[]
                    new_score=0
                    combinations_to_expand_copy = combinations.copy()
                    
                    scores_combinations_to_expand2=[]
                    for combination in combinations_to_expand_copy:
                        perturbed_instance=instance.copy()
                        for feature_in_combination in combination:
                            perturbed_instance[:,feature_in_combination]=0
                        score_new = self.classifier_fn_multiclass(perturbed_instance)[class_index]
                        
                        if (score_new[0] != np.max(self.classifier_fn_multiclass(perturbed_instance))):
                            scores_combinations_to_expand2.append(2 * score_predicted)
                        else:
                            scores_combinations_to_expand2.append(score_new)
                    
                    while ((len(new_combinations) == 0) and (it<len(scores_combinations_to_expand2)) and ((time_max+(time.time() - time_extra))<self.time_maximum)):
    
                        print('while loop %d' %it)
                        
                        if (it!=0):
                            for index in indices:
                                scores_combinations_to_expand2[index]= 2 * score_predicted
                                #to make sure this index is never chosen again
                        
                        index_combi_max=np.argmax(score_predicted - scores_combinations_to_expand2) #best-first combi
                        indices.append(index_combi_max)
                        
                        comb=combinations_to_expand_copy[index_combi_max]
                        func=fn_1(comb, expanded_combis, feature_set, combinations_to_expand_copy, explanations_sets)
                        new_combinations=func[0]
                        combinations_to_expand=func[1]
                        expanded_combis=func[2]
                        
                        #Calculate new threshold
                        scores_combinations_to_expand=[]
                        for combination in combinations_to_expand:
                            perturbed_instance=instance.copy()
                            for feature_in_combination in combination:
                                perturbed_instance[:,feature_in_combination]=0
                            score_new = self.classifier_fn_multiclass(perturbed_instance)[class_index]
                            
                            if (score_new[0] == np.max(self.classifier_fn_multiclass(perturbed_instance))):
                                scores_combinations_to_expand.append(score_new)                                
                    
                        if (len(scores_combinations_to_expand)!=0): 
                            index_combi_max=np.argmax(score_predicted - scores_combinations_to_expand) #best-first combi
                            new_score=scores_combinations_to_expand[index_combi_max]
                            threshold=score_predicted - new_score
                        it+=1 
                        print('length of new_combinations is %d features.' %len(new_combinations))
                        print('score_predicted minus new_score is %f.' %(score_predicted - new_score))
                        
                time_max += (time.time()-time_extra)
                print('\n Elapsed time %d \n' %time_max)
                print('\n size combis to expand %d \n' %len(combinations_to_expand))

        print("iterations are done")            
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
        #show explanation in explanation set which is minimum in size and highest score change (delta)
        if (np.size(explanations_score_change)>1):
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
            explanation_set=explanation_set_adjusted
            explanations_score_change=explanations_score_change_adjusted
            
        toc=time.time()
        time_elapsed=toc-tic
        print('\n Elapsed time %d \n' %time_elapsed)

        return (explanation_set[0:self.max_explained], number_active_elements, number_explanations, minimum_size_explanation, time_elapsed, explanations_score_change[0:self.max_explained], iteration)
    
    

