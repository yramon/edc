# Heuristic best-first search algorithm for finding Evidence Counterfactuals (SEDC)

The SEDC algorithm is a model-agnostic heuristic best-first search algorithm for finding Evidence Counterfactuals, which are instance-level explanations for explaining model predictions of any classifier. It returns a minimal set of features so that removing these features results in a predicted class change. Removing means setting the corresponding feature value to zero. SEDC has been originally proposed [in this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2282998) for explaining document classifications.

At the moment, SEDC supports binary classifiers built on high-dimensional, sparse data where a zero feature value corresponds to the absence of the feature. For instance, for behavioral data such as web browsing data, visiting an URL would set the feature value to 1, else 0. The nonzero value indicates that the behavior is present. Setting the feature value to zero would remove this evidence from the browsing history of a user. Another example is text data, where each token is represented by an individual feature. Setting the feature value (term frequency, tf-idf, etc.) to zero would mean that the corresponding token is removed from the document. 

# Explaining positively predicted instances 
An important sidenote is that the current implementation can only be used to explain positively predicted instances classified by a binary classification model. In other words, the instance you want to explain should have a probability or score that exceeds a certain threshold value (eg, 0.5 for logistic regression or 0 for SVMs).

We are currently working on a more general implementation where also the opposite is possible: explaining negatively predicted instances.

# Installation

To use the SEDC explanation algorithm, save the edc_agnostic.py and function_edc.py in the same directory and run them in an IDE of preference. Note that the default settings apply branch-and-bound in the search and return an explanation once one has been found. The feature names, classification function and threshold have to be entered by the user manually. 

# Demonstration

For an example of using the SEDC explanation algorithm on a classification model built from a behavioral data set, consider the following notebook: [Gender prediction from Movielens data](https://github.com/edc/Tutorial_BehavioralData_SEDC.html). 

# Linear implementation for finding Evidence Counterfactuals (lin-SEDC)

There is also a model-specific implementation of the algorithm for linear models: edc_linear.py. This version is more efficient than the model-agnostic implementation, however, it is less flexible to use.

# Licence

The SEDC explainer is patented in US US9836455B2.
