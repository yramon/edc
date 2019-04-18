# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:56:54 2019

@author: YRamon
"""

import numpy as np
from ordered_set import OrderedSet

def function_1(comb, expanded_combis, feature_set, combinations_to_expand, explanations_sets):
                    
    comb=OrderedSet(comb)
    expanded_combis.append(comb)

    feature_set_new=[]
    for feature in feature_set:
        if (len(comb & feature)==0):
            feature_set_new.append(feature)
            
    combinations_to_expand_notpruned=[]
    for element in feature_set_new:
        union=(comb|element)
        combinations_to_expand_notpruned.append(union)
                    
    combinations_to_expand_all_notpruned=combinations_to_expand.copy()
    for combi in combinations_to_expand_notpruned:
        combinations_to_expand_all_notpruned.append(combi)
    
    combinations_to_expand_all_list=[]
    for combi in combinations_to_expand_all_notpruned:
        pruning=0
        for explanation in explanations_sets:
            if ((explanation.issubset(combi)) or (explanation==combi)):
                pruning = pruning + 1
        if (pruning == 0):
            combinations_to_expand_all_list.append(combi)
    
    combinations_to_expand_bis_=[]
    for features in combinations_to_expand_all_list:
        combinations_to_expand_bis_.append(frozenset(features))
    combinations_to_expand_all=set(combinations_to_expand_bis_)
    
    expanded_combinations=[]
    for features in expanded_combis:
        expanded_combinations.append(frozenset(features))
    expanded_combinations_bis=set(expanded_combinations)
        
    combinations_to_expand_pruned_bis=(combinations_to_expand_all-expanded_combinations_bis)
    
    combinations_to_expand_all_list_frozen=[frozenset(x) for x in combinations_to_expand_all_list]
    ind_dict = dict((k,i) for i,k in enumerate(combinations_to_expand_all_list_frozen))
    indices = [ind_dict[x] for x in combinations_to_expand_pruned_bis]
    combinations_to_expand_pruned_bis_list=[combinations_to_expand_all_list[i] for i in indices]
    
    combinations_to_expand_=[]
    for features in combinations_to_expand:
        combinations_to_expand_.append(frozenset(features))
    combinations_to_expand=set(combinations_to_expand_)

    new_combinations_=(combinations_to_expand_pruned_bis-combinations_to_expand) #the new combos are the ones that weren't in the old combos
    
    combinations_to_expand_pruned_bis_list_frozen=[frozenset(x) for x in combinations_to_expand_pruned_bis_list]
    ind_dict2 = dict((k,i) for i,k in enumerate(combinations_to_expand_pruned_bis_list_frozen))
    indices2 = [ ind_dict2[x] for x in new_combinations_]
    new_combinations=[combinations_to_expand_pruned_bis_list[i] for i in indices2]
    
    combinations_to_expand=combinations_to_expand_pruned_bis_list
    
    return (new_combinations, combinations_to_expand)