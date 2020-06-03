import numpy as np
from lrscp_utils import *

def init_lambd(n):
    '''
    A random initialization of Lagrangean multipliers.

    Arguments:
    n: the number of rows, also the dimension of lambda

    Returns:
    lambd: Lagrangean multipliers, a n by 1 array

    '''
    np.random.seed(1)
    return np.random.random((n, 1))


def solve_LRSCP(c, a, lambd):
    '''
    Solve the Lagrangean relaxed set covering problem, which gives a lower bound to the SCP optimal.

    Arguments:
    c: an array of cost coefficient in the objective function
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    lambd: Lagrangean multipliers

    Returns:
    sol: a list of selected columns of the optimal solution of LRSCP
    LB: the optimal objective value of LRSCP, which gives a lower bound to the SCP optimal
    C: the new cost coefficient of X in the objective function of the LRSCP
    '''
    sol = []

    C = c - np.sum(lambd * a, axis = 0) # compute the new cost in the LRSCP
    for i in range(len(C)):
        if C[i] <= 0: # set Xj = 1 if a non-positive C
            sol.append(i)

    # compute the corresponding objective value of LRSCP
    LB = sum([C[i] for i in sol]) + float(np.sum(lambd))

    return sol, LB, np.array(C)


def find_primal_feasible(m, n, c, cost_coef_SCP, a, M, N, funcType = "III"):
    '''
    Find a feasible solution to the primal problem, i.e. the SCP, which gives an upper bound to the SCP optimal.

    The algoriothm is detailed in Balas and Ho (1980) Balas, E., & Ho, A. (1980). Set covering algorithms using
    cutting planes, heuristics, and subgradient optimization: a computational study.
    In Combinatorial Optimization (pp. 37-60). Springer, Berlin, Heidelberg.

    Arguments:
    m: the number of columns
    n: the number of rows
    c: an array of cost coefficient of X in the objective function of SCP or LRSCP
    cost_coeff_SCP: an array of cost coefficient of X in the objective function of SCP
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    N: a list of coverage set, N[i]: a set of columns that can cover row i
    M: a list of coverage set, M[j]: a set of rows that can be covered by column j
    funcType: f(c, k) function options detailed in Balas and Ho (1980)

    Returns:
    sol: a list of selected columns of a primal feasible solution
    UB: a feasible objective value of SCP, which gives an upper bound to the SCP optimal
    '''

    # initialize
    R, sol = set(range(n)), [] # a set of uncovered rows, a list of selected columns

    # greedy add new rows until each row is covered
    while R:
        K = [len(R.intersection(j)) for j in M] # K[j]: number of uncovered rows that can be covered by column j

        temp = {i: len(N[i]) for i in R} # temp dict: key: row i in a, value: number of columns that can cover i
        i_star = min(temp, key = temp.get) # find uncovered i_star that has least amount of columns that can cover

        # choose j based on f(c, K)
        pool = set(range(m)).difference(set(sol)) # unselected columns
        pool = pool.intersection(N[i_star]) # unselected columns that can cover i_star
        temp = float("Inf") # minimum f(c, K)
        j_star = float("Inf") # column index for minimum f(c, K)
        for j in pool:
            if f_kc(c[j], K[j], funcType) < temp: # find the minimum f(c, K)
                temp = f_kc(c[j], K[j], funcType)
                j_star = j

        # add selected column and update others
        sol.append(j_star)
        R = R.difference(M[j_star])


    # check if any unnecessary seleted column
    sol_copy = sol.copy()
    for j in sol_copy:
        sol.remove(j) # try removing column j
        if 0 in np.sum(a[:, sol], axis =1): # if sol cannot cover all rows, backtrack
            sol.append(j)

    # compute the corresponding objective value of SCP
    UB = sum([cost_coef_SCP[j] for j in sol])

    return sol, UB

def update_lambd(a, LB, best_UB, sol, lambd, alpha):
    '''
    Subgradient procedure to update Lagrangean multipliers.

    Arguments:
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    LB: a lower bound of the SCP optimal objective
    best_UB: the best upper of the SCP optimal objective bound so far
    sol: the optimal solution to a LRSCP
    lambd: Lagrangean multipliers
    alpha: a factor to control step length during updating

    Returns:
    new_lambd: the updated Lagrangean multipliers
    '''

    t = alpha * (best_UB - LB)/(np.sum((1 - np.sum(a[:, sol], axis =1))**2)) # step length

    new_lambd = np.maximum(0, lambd + t * (1 - np.sum(a[:, sol], axis =1, keepdims = True))) # update lambda

    return new_lambd

def Lagrangean(m, n, cost_coef, a, M, N, S, alpha, beta, epsilon, maxItr):
    '''
    Compute the solution to a Set Covering Problem (SCP) using Lagrangean relaxation.

    Arguments:
    m: the number of columns
    n: the number of rows
    cost_coef: an array of cost coefficient of X in the objective function of SCP
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    N: a list of coverage set, N[i]: a set of columns that can cover row i
    M: a list of coverage set, M[j]: a set of rows that can be covered by column j
    S: the service coverage standard
    alpha: the initial value of alpha, a factor to control step length during updating lambda
    beta: a hyperparameter to update alpha
    epsilon: the tolerance of the gap between lower and upper bounds
    maxItr: the maximum iteration number allowed

    Returns:
    LB: the lower bound of the SCP optimal (the objective value of LRSCP)
    sol_LRSCP: the solution of LRSCP, a list of selected columns
    best_UB: the best upper bound of the SCP optimal
    sol_best_UB: the solution corresponding to the best UB found, a list of selected columns
    caches: a dictionary containing itration number, LB, UB, best_UB
    '''
    # initialize
    LB, sol_LRSCP = float("Inf"), [] # the lower bound of the SCP optimal
    UB, sol_feas = float("-Inf"), [] # the upper bound of the SCP optimal
    best_UB, sol_best_UB = float("Inf"), [] # the best upper bound found so far and its solution
    itr = 0 # iteration counter
    caches = {"Itr": [], "LB": [], "UB":[], "Best_UB": []}

    # initialize Lagrangean multipliers
    lambd = init_lambd(n)

    while itr < maxItr:  # while the itration counter is less than the maxItr

        # solve the LRSCP
        sol_LRSCP, LB, C = solve_LRSCP(cost_coef, a, lambd)

        # find a feasible solution to SCP
        sol_feas, UB = find_primal_feasible(m, n, C, cost_coef, a,
                                            M, N, funcType = "III")

        # update best_UB if needed
        if UB < best_UB:
            best_UB = UB
            sol_best_UB = sol_feas

        # write caches
        caches["Itr"].append(itr)
        caches["LB"].append(LB)
        caches["UB"].append(UB)
        caches["Best_UB"].append(best_UB)

        # if the gap between the best known upper bound and lower bound is less than epsilon, stop
        if best_UB - LB < epsilon:
            break

        # otherwise, subgradient procedure to update lambda's
        lambd = update_lambd(a, LB, best_UB, sol_LRSCP, lambd, alpha)
        alpha = beta * alpha

        itr += 1 # iteration counter

    return LB, sol_LRSCP, best_UB, sol_best_UB, caches
