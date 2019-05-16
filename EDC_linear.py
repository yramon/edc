# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:53:59 2019

@author: YRamon
"""

"""
Model-specific version of Evidence Counterfactual for LINEAR MODELS
"""

"""
Functions for explaining classifiers on high-dimensional, sparse data
"""

import time
import numpy as np 
from scipy.sparse import lil_matrix
from ordered_set import OrderedSet
from function_edc import fn_1

def explanation_linear(instance, classifier_fn, threshold_classifier, feature_names, max_iter, max_explained, max_features):
    
    tic=time.time()
    instance=lil_matrix(instance)
    iteration=0
    nb_explanations=0
    i=0
    k=1
    stop=0
    explanations=[]
    explanations_score_change=[]
    score_predicted=classifier_fn(instance)
    indices_active_elements=np.nonzero(instance)[1] #returns indices where feature value is non-zero (active elements)
    number_active_elements=len(indices_active_elements)
    indices_active_elements=indices_active_elements.reshape((number_active_elements,1))
    number_active_elements=len(indices_active_elements)
    
    combinations_to_expand=[]
    for features in indices_active_elements:
        features=[features[0]]
        combinations_to_expand.append(features)
    
    feature_set=[]
    for features in indices_active_elements:
        feature_set.append(frozenset(features)) #frozenset means immutable set, to allow for sets in set
    feature_set=set(feature_set)
    
    score_new_combi=[]
    new_combinations=combinations_to_expand.copy()
    time_max=0
    
    for combination in new_combinations:
        perturbed_instance=instance.copy()
        for feature_in_combination in combination: 
            perturbed_instance[:,feature_in_combination]=0
        score_new=classifier_fn(perturbed_instance)
        score_new_combi.append(score_new)
    
    scores=np.array(score_new_combi)
    scores_ranked = np.argsort(scores, axis=0)
    scores_sorted=scores[scores_ranked]
    new_combinations_array=np.array(new_combinations)
    new_combinations_sorted2=new_combinations_array[scores_ranked]
    
    new_combinations_sorted=[]
    for comb in new_combinations_sorted2:
        new_combinations_sorted.append(comb[0][0])
    
    print('initialization done')

    while not any([(iteration>=max_iter), (np.size(scores_sorted)==0), (nb_explanations>=max_explained), (time_max>300), (stop!=0)]):
        
        time_extra=time.time()
        combination_set=[]
        perturbed_instance=instance.copy()
        
        for combination in new_combinations_sorted[0:k]:
            perturbed_instance[:,combination]=0
            combination_set.append(combination)
        
        score_new=classifier_fn(perturbed_instance)
        if (score_new[0]<threshold_classifier):
            explanations.append(combination_set) #an explanation (set or combination) is added to the explanation of type "list"
            explanations_score_change.append(score_predicted-score_new)
            nb_explanations += 1

        i+=1
        if (np.size(scores_sorted)==i):
            stop+=1
        elif ((scores_sorted[i]-score_predicted)>0):
            stop+=1
        
        k+=1
        iteration += 1
        
        print('\n Iteration %d \n' %iteration)
        
        time_extra2=time.time()
        time_max+=(time_extra2-time_extra)
            
    print("iterations are done")            

    explanation_set=[]
    explanation_feature_names=[]
    for i in range(len(explanations)):
        explanation_feature_names=[]
        for features in explanations[i]:
            explanation_feature_names.append(feature_names[features])
        explanation_set.append(explanation_feature_names)
            
    if (len(explanations)!=0):
        lengths_explanation=[]
        for explanation in explanations:
            l=len(explanation)
            lengths_explanation.append(l)
        minimum_size_explanation=np.min(lengths_explanation)
    else:
        minimum_size_explanation=np.nan
    
    number_explanations=len(explanations)
    toc=time.time()
    time_elapsed=toc-tic
    
    explanation_set_adjusted=explanation_set
    explanations_score_change_adjusted=explanations_score_change
    
    return (explanation_set_adjusted[0:max_explained], number_active_elements, number_explanations, minimum_size_explanation, time_elapsed, explanations_score_change_adjusted[0:max_explained])


