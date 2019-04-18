# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:53:59 2019

@author: YRamon
"""
### Model-agnostic implementation of EDC ###

# Import libraries / modules 
import time
import numpy as np 
from scipy.sparse import lil_matrix
from ordered_set import OrderedSet
from function import function_1

def EDC_sets_tryout2(instance, classification_model, threshold_classifier, feature_names, max_iter, max_explained, BB, max_features):

    tic=time.time()
    instance=lil_matrix(instance)
    iteration=0
    nb_explanations=0
    explanations=[]
    explanations_sets=[]
    explanations_score_change=[]
    score_predicted=classification_model.predict_proba(instance)[:,1]
    indices_active_elements=np.nonzero(instance)[1] #returns indices where feature value is non-zero (active elements)
    number_active_elements=len(indices_active_elements)
    indices_active_elements=indices_active_elements.reshape((number_active_elements,1))
    number_active_elements=len(indices_active_elements)
    combinations_to_expand=[]
    for features in indices_active_elements:
        combinations_to_expand.append(OrderedSet(features))
        
    feature_set=[]
    for features in indices_active_elements:
        feature_set.append(frozenset(features))
    #feature_set=set(feature_set)
    
    new_combinations=combinations_to_expand.copy()  
    threshold=0
    stop=0
    time_max=0
    expanded_combis=[]
    
    print('initialization done')
        
    while (iteration<max_iter) and (nb_explanations<max_explained) and (len(combinations_to_expand)!=0) and (len(new_combinations)!=0) and (time_max<300): 
        
        time_extra=time.time()
        iteration += 1
        print('\n Iteration %d \n' %iteration)
        
        new_combinations_to_expand=[]
        scores_new_combinations_to_expand=[]
        for combination in new_combinations:
            perturbed_instance=instance.copy()
            for feature_in_combination in combination: 
                perturbed_instance[:,feature_in_combination]=0
            score_new=classification_model.predict_proba(perturbed_instance)[:,1]
            
            if (score_new[0]<threshold_classifier):
                explanations.append(combination)
                explanations_sets.append(set(combination))
                explanations_score_change.append(score_predicted-score_new)
                nb_explanations += 1
            else:
                new_combinations_to_expand.append(combination)
                scores_new_combinations_to_expand.append(score_new)
    
        if (BB==True):
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
        
        if (len(new_combinations[0])==number_active_elements):  
            stop=1
        
        if ((len(new_combinations[0]) < (max_length-1*stop)) and (len(new_combinations[0]) < max_features)): 
                
                print('new combinations can be expanded')
                
                index_combi_max=np.argmax(score_predicted-scores_new_combinations_to_expand) #look for best-first (index)
                new_score=scores_new_combinations_to_expand[index_combi_max]
                difference=score_predicted-new_score
                
                if (difference[0]>=threshold): #difference is higher than threshold
                    
                    print('new combination is expanded')
                    
                    comb=new_combinations_to_expand[index_combi_max]

                    func=function_1(comb, expanded_combis, feature_set, combinations_to_expand, explanations_sets)
                    new_combinations=func[0]
                    combinations_to_expand=func[1]
                    
                    #Calculate new threshold
                    scores_combinations_to_expand=[]
                    for combination in combinations_to_expand:
                        perturbed_instance=instance.copy()
                        for feature_in_combination in combination:
                            perturbed_instance[:,feature_in_combination]=0
                        score_new=classification_model.predict_proba(perturbed_instance)[:,1]
                        
                        if (score_new[0]>=threshold_classifier):
                            scores_combinations_to_expand.append(score_new)
                        
                    index_combi_max=np.argmax(score_predicted-scores_combinations_to_expand)
                    new_score=scores_combinations_to_expand[index_combi_max]
                    new_threshold=score_predicted-new_score
                    threshold=new_threshold
                    
                    time_extra2=time.time()
                    time_max+=(time_extra2-time_extra)
                
                else: #difference is not higher than the threshold
                    print('new combination is not expanded, look for other combination')
                    new_combinations=[]
                    it=0
                    indices=[]
                    new_score=0
                    combinations_to_expand_copy=combinations_to_expand.copy()
                    
                    while ((len(new_combinations)==0) and ((score_predicted-new_score)>0)):
        
                        scores_combinations_to_expand=[]
                        for combination in combinations_to_expand_copy:
                            perturbed_instance=instance.copy()
                            for feature_in_combination in combination:
                                perturbed_instance[:,feature_in_combination]=0
                            score_new=classification_model.predict_proba(perturbed_instance)[:,1]
                            
                            if (score_new[0]<threshold_classifier):
                                scores_combinations_to_expand.append(score_predicted)
                            else:
                                scores_combinations_to_expand.append(score_new)
                        
                        if (it!=0):
                            scores_combinations_to_expand_copy=scores_combinations_to_expand.copy()
                            for index in indices:
                                scores_combinations_to_expand_copy[index]=score_predicted
                            scores_combinations_to_expand=scores_combinations_to_expand_copy
                        
                        #BEST FIRST OPERATION        
                        index_combi_max=np.argmax(score_predicted-scores_combinations_to_expand)
                        indices.append(index_combi_max)
                        new_score=scores_combinations_to_expand[index_combi_max]
                        
                        comb=new_combinations_to_expand[index_combi_max]

                        func=function_1(comb, expanded_combis, feature_set, combinations_to_expand, explanations_sets)
                        new_combinations=func[0]
                        combinations_to_expand=func[1]
                        
                        #Calculate new threshold
                        scores_combinations_to_expand=[]
                        for combination in combinations_to_expand:
                            perturbed_instance=instance.copy()
                            for feature_in_combination in combination:
                                perturbed_instance[:,feature_in_combination]=0
                            score_new=classification_model.predict_proba(perturbed_instance)[:,1]
                            
                            if (score_new[0]>=threshold_classifier):
                                scores_combinations_to_expand.append(score_new)
                            
                        index_combi_max=np.argmax(score_predicted-scores_combinations_to_expand)
                        new_score=scores_combinations_to_expand[index_combi_max]
                        new_threshold=score_predicted-new_score
                        threshold=new_threshold
                    
                        it+=1 
                        
                    time_extra2=time.time()
                    time_max+=(time_extra2-time_extra)

        else: #if the combinations in new_combinations are not bigger than the shortest explanation 
                    
                    print('new combination cannot be expanded because of length OR first explanation has been found already')
                    
                    if (len(new_combinations[0])==number_active_elements):
                        stop_2=1
                    else:
                        stop_2=0
                    
                    combinations=[] #Eerst die combinaties eruit halen die groter zijn dan korste explanation
                    for combination in combinations_to_expand:
                        if ((len(combination) < (max_length-1*stop_2)) and (len(combination) < max_features)):
                            combinations.append(combination)
                    
                    if (len(combinations)==0):
                        new_combinations=[]
                    
                    elif (len(combinations)!=0):
                        combinations_to_expand=combinations
                        
                        new_combinations=[]
                        it=0
                        indices=[]
                        new_score=0
                        combinations_to_expand_copy=combinations_to_expand.copy()
                        
                        while ((len(new_combinations)==0) and ((score_predicted-new_score)>0)):
            
                            scores_combinations_to_expand=[]
                            for combination in combinations_to_expand_copy:
                                perturbed_instance=instance.copy()
                                for feature_in_combination in combination:
                                    perturbed_instance[:,feature_in_combination]=0
                                score_new=classification_model.predict_proba(perturbed_instance)[:,1]
                                
                                if (score_new[0]<threshold_classifier):
                                    scores_combinations_to_expand.append(score_predicted)
                                else:
                                    scores_combinations_to_expand.append(score_new)
                            
                            if (it!=0):
                                scores_combinations_to_expand_copy=scores_combinations_to_expand.copy()
                                for index in indices:
                                    scores_combinations_to_expand_copy[index]=score_predicted
                                scores_combinations_to_expand=scores_combinations_to_expand_copy
                            
                            index_combi_max=np.argmax(score_predicted-scores_combinations_to_expand) #look for best-first (index)
                            indices.append(index_combi_max)
                            new_score=scores_combinations_to_expand[index_combi_max]
                            
                            comb=combinations_to_expand_copy[index_combi_max]

                            func=function_1(comb, expanded_combis, feature_set, combinations_to_expand, explanations_sets)
                            new_combinations=func[0]
                            combinations_to_expand=func[1]
                            
                            #Calculate new threshold
                            scores_combinations_to_expand=[]
                            for combination in combinations_to_expand:
                                perturbed_instance=instance.copy()
                                for feature_in_combination in combination:
                                    perturbed_instance[:,feature_in_combination]=0
                                score_new=classification_model.predict_proba(perturbed_instance)[:,1]
                                
                                if (score_new>=threshold_classifier):
                                    scores_combinations_to_expand.append(score_new)                                
                        
                            if (len(scores_combinations_to_expand)!=0): 
                                index_combi_max=np.argmax(score_predicted-scores_combinations_to_expand) #look for best-first (index)
                                new_score=scores_combinations_to_expand[index_combi_max]
                                new_threshold=score_predicted-new_score
                                threshold=new_threshold
                            it+=1 
                            
                    time_extra2=time.time()
                    time_max+=(time_extra2-time_extra)


    print("iterations are done")            
    # Comes at the end of the loop: all found explanations   
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
    
    ### show explanation in explanation set which is minimum in size and highest score change (delta)
    if (np.size(explanations_score_change)>1):
        inds=np.argsort(explanations_score_change, axis=0) #low to high
        inds = np.fliplr([inds])[0] #flip
        inds_2=[]
        for i in range(np.size(inds)):
            inds_2.append(inds[i][0])
        #explanation_set=np.array(explanation_set)
        #explanations_score_change=np.array(explanations_score_change)
        #explanation_set_adjusted2=explanation_set[inds]
        #explanations_score_change_adjusted2=explanations_score_change[inds]
        explanation_set_adjusted=[]
        for i in range(np.size(inds)):
            j=inds_2[i]
            explanation_set_adjusted.append(explanation_set[j])
        explanations_score_change_adjusted=[]
        for i in range(np.size(inds)):
            j=inds_2[i]
            explanations_score_change_adjusted.append(explanations_score_change[j])
    else: 
        explanation_set_adjusted=explanation_set
        explanations_score_change_adjusted=explanations_score_change
    
    toc=time.time()
    time_elapsed=toc-tic
    
    return (explanation_set_adjusted[0:max_explained], number_active_elements, number_explanations, minimum_size_explanation, time_elapsed, explanations_score_change_adjusted[0:max_explained])


