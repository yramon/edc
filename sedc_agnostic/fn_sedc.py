"""
Last update: 13 May 2020.
"""

from ordered_set import OrderedSet

def perturb_fn(x,inst):
    """ Function to perturb instance x """
    """
    Returns perturbed instance inst
    """
    inst[:,x]=0
    return inst

"""
Input:
    - comb: "best-first" (combination of) feature(s) that is expanded
    (e.g., comb_to_expand)
    - expanded_combis: list of combinations of features that are already 
    expanded as "best-first"
    - feature_set: indices of the active features of the instance 
    - candidates_to_expand: combinations of features that are candidates to be 
    expanded in next iterations or candidates for "best-first"
    - explanations_sets: counterfactual explanations already found
    - scores_candidates_to_expand: scores after perturbation for the candidate
    combinations of features to be expanded
    - instance: instance to be explained
    - cf: classifier prediction probability function
    or decision function. For ScikitClassifiers, this is classifier.predict_proba 
    or classifier.decision_function or classifier.predict_log_proba.
    Make sure the function only returns one (float) value. For instance, if you
    use a ScikitClassifier, transform the classifier.predict_proba as follows:
            
        def classifier_fn(X):
            c=classification_model.predict_proba(X)
            y_predicted_proba=c[:,1]
            return y_predicted_proba
    
Returns:
    - explanation_candidates: combinations of features that are explanation
    candidates to be checked in the next iteration
    - candidates_to_expand: combinations of features that are candidates to be 
    expanded in next iterations or candidates for "best-first"
    - expanded_combis: [list] list of combinations of features that are already 
    expanded as "best-first"    
    - scores_candidates_to_expand: scores after perturbation for the candidate
    combinations of features to be expanded
    - scores_explanation_candidates: scores after perturbation of explanation candidates
"""

def expand_and_prune(comb, expanded_combis, feature_set, candidates_to_expand, explanations_sets, scores_candidates_to_expand, instance, cf):
    """ Function to expand "best-first" feature combination and prune explanation_candidates and candidates_to_expand """                
    
    comb = OrderedSet(comb)
    expanded_combis.append(comb)
    
    old_candidates_to_expand = [frozenset(x) for x in candidates_to_expand]
    old_candidates_to_expand = set(old_candidates_to_expand)
    
    feature_set_new = []
    for feature in feature_set:
        if (len(comb & feature) == 0): #set operation: intersection
            feature_set_new.append(feature)
            
    new_explanation_candidates = [] 
    for element in feature_set_new:
        union = (comb|element) #set operation: union
        new_explanation_candidates.append(union)
    
    #Add new explanation candidates to the list of candidates to expand
    candidates_to_expand_notpruned = candidates_to_expand.copy() #voeg de nieuwe combinaties toe aan combinations_to_expand 
    for new_candidate in new_explanation_candidates:
        candidates_to_expand_notpruned.append(new_candidate)
        
    #Calculate scores of new combinations and add to scores_candidates_to_expand
    perturbed_instances = [perturb_fn(x, inst=instance.copy()) for x in new_explanation_candidates]
    scores_perturbed_new = [cf(x) for x in perturbed_instances]
    scores_candidates_to_expand_notpruned = scores_candidates_to_expand + scores_perturbed_new
    dictionary_scores = dict(zip([str(x) for x in candidates_to_expand_notpruned], scores_candidates_to_expand_notpruned))
    
    # *** Pruning step: remove all candidates to expand that have an explanation as subset ***
    candidates_to_expand_pruned_explanations = []
    for combi in candidates_to_expand_notpruned:
        pruning=0
        for explanation in explanations_sets:
            if ((explanation.issubset(combi)) or (explanation==combi)):
                pruning = pruning + 1
        if (pruning == 0):
            candidates_to_expand_pruned_explanations.append(combi)
    
    candidates_to_expand_pruned_explanations_frozen = [frozenset(x) for x in candidates_to_expand_pruned_explanations]
    candidates_to_expand_pruned_explanations_ = set(candidates_to_expand_pruned_explanations_frozen)
    
    expanded_combis_frozen = [frozenset(x) for x in expanded_combis]
    expanded_combis_ = set(expanded_combis_frozen)
        
    # *** Pruning step: remove all candidates to expand that are in expanded_combis ***
    candidates_to_expand_pruned = (candidates_to_expand_pruned_explanations_ - expanded_combis_)  
    ind_dict = dict((k,i) for i,k in enumerate(candidates_to_expand_pruned_explanations_frozen))
    indices = [ind_dict[x] for x in candidates_to_expand_pruned]
    candidates_to_expand = [candidates_to_expand_pruned_explanations[i] for i in indices]
    
    #The new explanation candidates are the ones that are NOT in the old list of candidates to expand
    new_explanation_candidates_pruned = (candidates_to_expand_pruned - old_candidates_to_expand) 
    candidates_to_expand_frozen = [frozenset(x) for x in candidates_to_expand]
    ind_dict2 = dict((k,i) for i,k in enumerate(candidates_to_expand_frozen))
    indices2 = [ind_dict2[x] for x in new_explanation_candidates_pruned]
    explanation_candidates = [candidates_to_expand[i] for i in indices2]
        
    scores_candidates_to_expand = [dictionary_scores[x] for x in [str(c) for c in candidates_to_expand]]
    scores_explanation_candidates = [dictionary_scores[x] for x in [str(c) for c in explanation_candidates]]
    
    return (explanation_candidates, candidates_to_expand, expanded_combis, scores_candidates_to_expand, scores_explanation_candidates)