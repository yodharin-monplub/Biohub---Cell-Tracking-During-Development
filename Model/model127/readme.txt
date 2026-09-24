MODEL127 - TWO-SIDED TEMPORAL BRIDGE AUDIT

Status: fixed bridge-only exact 44b6 development-family score complete;
rejected. Difficulty4/5 for pilot,5/5 for0.97 CV target.

Model125 preserved the model118 final graph and added raw-node paths,
but still lost0.001087 exact score on the hard 44b6 family. Model126's
high-probability filter lost on every movie. Test the user's temporal
graduation idea as one change: keep only added paths whose first and last
nodes are both present in the model118 final graph. Such a path has
observed support on both sides of one or more candidate missing cells.
One-sided or free-floating added paths are excluded. Original model118
final nodes and edges remain immutable. This is inference-time topology;
no GT or family label is used to select individual paths.

First audit the complete14-movie development 44b6 family to count actual
two-sided bridges. If present, build and exact-score the bridge-only
candidate on the same movies, then use full39+39 paired gates if promising.
The family has been reused for selection; any result needs independent
confirmation. No Kaggle/cloud spend or alarm.

Run: .venv-gpu/bin/python model127/audit.py

AUDIT RESULT
There are10 two-sided paths containing40 added edges, versus202 one-sided
paths/647 edges and897 free-floating paths/2,077 edges. The small bridge
subset can now be exact-scored without any hyperparameter search.

EXACT SCORE RESULT
The bridge-only replay added30 raw nodes and40 edges while preserving
every model118 final node and edge. On all14 development 44b6 movies,
exact organizer-family score was0.941425202543659 versus model118's
0.941429579648439, delta=-0.000004377105. Division Jaccard and mean
node recall were unchanged. Five movies had tiny losses, nine ties, zero
wins. The fixed positive-gain gate fails; do not promote or deploy.
