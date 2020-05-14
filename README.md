# Heuristic best-first search algorithm for finding Evidence Counterfactuals (SEDC)

The SEDC algorithm is a model-agnostic heuristic best-first search algorithm for finding Evidence Counterfactuals, which are instance-level explanations for explaining model predictions of any classifier. It returns a minimal set of features so that removing these features results in a predicted class change. "Removing" means setting the corresponding feature value to zero. SEDC has been originally proposed [in this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2282998) for explaining document classifications.

At the moment, SEDC supports binary classifiers built on high-dimensional, sparse data where a zero feature value corresponds to the absence of the feature (the feature is "missing"). For instance, for web browsing data, each individual URL can be represented by a binary feature, where visiting an URL would set the feature value to 1, else 0. The nonzero value indicates that the behavior is "present". Setting the feature value to zero would remove this evidence from the browsing history of a user. Another example is textual data, where each token is represented by an individual feature. Setting the feature value (term frequency, tf-idf, etc.) to zero would mean that the corresponding token is removed from the document. 

# Explaining positively-predicted instances 
An important sidenote is that the current implementation can only be used to explain positively predicted instances classified by a binary classification model. In other words, the instance you want to explain should have a probability or score that exceeds a certain threshold value (eg, 0.5 for logistic regression or 0 for SVMs).

We are currently working on a more general implementation where also the opposite is possible: explaining negatively predicted instances. For now, if you want to do so, you can use the multi-class implementation and use two binary classifiers (one for each target class). If you do this, then you can immediately explain negatively predicted instances.

# Visualization of the (model-agnostic) SEDC algorithm
The figure below shows how the SEDC algorithm works (sedc_algorithm.py). Moreover, the figure expand_and_prune.png in the folder "img" illustrates how the function expand_and_prune() works.

<br>
<img height="600" src="https://github.com/yramon/edc/blob/master/img/sedc-visualisation.png" />
<br>

# Linear implementation for finding Evidence Counterfactuals (lin-SEDC)

There is also a model-specific implementation of the algorithm for linear models: edc_linear.py (in "LinearEDC"). This version is more efficient than the model-agnostic implementation, however, it is less flexible to use.

# Multi-class classification tasks

We have also written an implementation that can be used for multi-class problems (tasks where the target variable has more than two classes). Here, we explain why an instance is classified as a certain class. The counterfactual explanation shows the set of features such that, when their feature values are set to zero, the predicted class would change to another class. It is important to note that we use a one-vs-rest approach. More specifically, when there are three different classes, then we assume there are three binary trained classifiers (one classifier for each target class).

# Installation

To use the SEDC explanation algorithm, save the sedc_algorithm.py and function_edc.py in the same directory and run them in an IDE of preference. Note that the default settings apply branch-and-bound in the search and return an explanation once one has been found (see [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2282998) for more details). The feature names, classification function and threshold have to be entered by the user manually. 

For the linear implementation, use edc_linear.py. 

For the multi-class implementation, use SEDC_agnostic_multiclass.py and fn_sedc.py (same directory and run them in IDE of preference). 

# Tutorials with behavior and text data

For an example of using the SEDC explanation algorithm on a classification model built from high-dimensional, sparse behavioral data and textual data, consider the following notebooks: Gender prediction from movie viewing data (Movielens1M) using a [Logistic Regression model](https://yramon.github.io/tutorials/Tutorial_BehavioralDataMovielens_LR_SEDC.html) and a [Multilayer Perceptron model](https://yramon.github.io/tutorials/Tutorial_BehavioralDataMovielens_MLP_SEDC.html), and [Topic prediction from news documents](https://yramon.github.io/tutorials/Tutorial_TextData_SEDC.html) (20Newsgroups data) using a linear Support Vector Machine.

# Licence

The SEDC explainer is patented in US US9836455B2.
