# DIGC

This script is utilizing python under torch framework.

Please make sure you already have installed package of torch_geometric.

Consider only manually edit the parameter value on :
1. main.py for line 28 to 38
2. If you want to change number of attention heads, please consider also to change:
   a. main.py for line 66 and 67 (make sure the neuron numbers are appropriate in each layer)
   b. utils.py for line 211
