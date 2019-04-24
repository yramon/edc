# edc

The Evidence Counterfactual (EDC) is an instance-level explanation method for explaining model predictions of any classifier. It returns an explanation in the form of a minimal set of features so that removing these features results in a predicted class change. Removing means setting the corresponding feature value to zero.

At the moment, EDC supports classifiers built on high-dimensional, sparse data where a zero feature value corresponds to absence of the feature. For instance, for behavioral data such as web browsing data, visiting an URL would set the feature value to 1, else 0. The nonzero value indicates that the behavior is present. Setting the feature value to zero would remove this evidence from the browsing history of a user. Another example is text data, where each token is represented by an individual feature. Setting the feature value (term frequency, tf-idf, etc.) to zero would mean that the corresponding token is removed from the document. 

