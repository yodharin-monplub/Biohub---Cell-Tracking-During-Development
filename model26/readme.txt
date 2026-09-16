MODEL 26 — SHORT-TRACK COMPONENT PRUNING

Status
------
Complete and rejected on model22's solved lambda-zero graphs. This is not a
submission.

Hypothesis
----------
The official adjusted edge score penalizes total predicted-node count, while
node recall is diagnostic only. Whole isolated or very short track components
often contain no useful edge and can inflate the node-count term. Remove only
entire weakly connected components of length at most 1, 2, 3, or 5 and measure
the exact metric response.

Safety
------
No edge is rewired and no invalid topology is introduced. Strict validation
and exact TP/FP/FN comparison are required. The branch is rejected if apparent
count gains hide meaningful edge recall loss or fail either embryo family.

Results
-------
The ILP produces no components of length one through three. Removing all
components of length at most four drops 16,312 nodes but also drops useful
edges, scoring 0.8904974603 versus the 0.8922506937 control. Removing through
length five scores 0.8904177715. Both outputs are strict-valid.

Decision
--------
Reject. Short tracks cannot be identified as false purely from component
length; their lost edge recall outweighs the node-count adjustment gain.
