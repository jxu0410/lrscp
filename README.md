# lrscp
An implementation of Lagrangean relaxation based optimization algorithm for Set Covering Problems (SCP), which are NP-complete problems.

LRSCP.pdf: explains Set Covering Problems and the Lagrangean algorithm in details.

lrscp.py: an implementation of a Lagrangean relaxation based optimization algorithm for SCP.

lrscp_utils.py: provides helper functions supporting lrscp.py

scp_evl.py: an implementation of calling a Gurobi solver to solve SCP to evaluate the Lagrangean algorithm.

lrscp_demo.ipynb: a demo of applying the Lagrangean algorithm to solve three problems including simulated and real optimization problems.

Data folder contains two datasets:

swain.txt: contains the id, x and y coordinates and demand amount of 55 points

matrix_82.txt: contains the network distance matrix between 82 potential facilities and 2070 demand points,
               the first line contains total number of facility-demand pairs, the number of demand points, the number of potential facilities, the network distance of real street network
               the rest lines contain: the id of facility-demand pair, the id of potential facility, the id of demand points, the network distance

sbc.png: a map of SB data
