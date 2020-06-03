from gurobipy import Model, GRB


def SCP(m, n, c, N):
    '''
    Solve the Set Covering Problem (SCP) using the Gurobi solver.
    A valid Gurobi license is required to call the solver. 

    Arguments:
    m: the number of columns
    n: the number of rows
    c: an array of cost coefficient of X in the objective function of SCP
    N: a list of coverage sets, N[i]: a set of columns that can cover row i

    Returns:
    sol: a list of selected columns
    SCP.objVal: the optimal objective value
    '''
    sol = []

    # start a model
    SCP = Model("SCP")

    # define decision variables
    X = {} # location variable
    # binary constraints
    X = SCP.addVars(range(1, m + 1), vtype=GRB.BINARY) # Xj

    # objective
    SCP.setObjective(sum(c[j-1] * X[j] for j in range(1, m + 1)), GRB.MINIMIZE)

    # constraints
    # each row has to be covered
    for i in range(n):
        SCP.addConstr(sum(X[j+1] for j in N[i]) >= 1, "Cov[%d]" %i)

    # solution
    SCP.Params.OutputFlag = 0
    # StartTime = time.time()
    SCP.optimize()
    # ElapsedTime = time.time()-StartTime

    if SCP.status == GRB.Status.OPTIMAL:
        for j in range(1, m + 1):
            if X[j].getAttr("x") == 1:
                sol.append(j - 1)

    return sol, SCP.objVal
