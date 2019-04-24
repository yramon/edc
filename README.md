# Evidence Counterfactual (EDC)

The Evidence Counterfactual (EDC) is an instance-level explanation method for explaining model predictions of any classifier. It returns a minimal set of features so that removing these features results in a predicted class change. Removing means setting the corresponding feature value to zero. EDC has been originally proposed in [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2282998) for explaining document classifications.

At the moment, EDC supports binary classifiers built on high-dimensional, sparse data where a zero feature value corresponds to the absence of the feature. For instance, for behavioral data such as web browsing data, visiting an URL would set the feature value to 1, else 0. The nonzero value indicates that the behavior is present. Setting the feature value to zero would remove this evidence from the browsing history of a user. Another example is text data, where each token is represented by an individual feature. Setting the feature value (term frequency, tf-idf, etc.) to zero would mean that the corresponding token is removed from the document. 

# Installation

To use the EDC explainer, save the edc_agnostic.py and function_edc.py in the same directory and run them in an IDE of preference. The initialization function  


# Linear EDC

There is also a model-specific implementation of the algorithm for linear models: edc_linear.py. This version is more efficient than the model-agnostic implementation, however, is less flexible in use.
