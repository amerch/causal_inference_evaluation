# Universal Causal Evaluation Engine

Implementation (pre-alpha) of the Universal Causal Evaluation Engine API from Lin ® Merchant ® Sarkar<sup>1</sup>.

New models and datasets can be submitted via pull request to the Models and Data folders.

To evaluate error metrics for a particular model and dataset, run 

```python outer_loop.py --model [MODEL] --data [DATASET]```

Twins-Specific Instructions:
1. To generate Twins data, run the Convert+Twins.py script to generate replications.
2. The outer loop for model evaluation can be run directly or through the makefile 

<sup>1</sup> The ® symbol indicates the authors are listed in certified random order, as described by [Ray ® Robson 2018](https://www.aeaweb.org/articles?id=10.1257/aer.20161492)
