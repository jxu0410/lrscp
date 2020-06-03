#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np


# %%


def generate_points(circles, num_samples = 10):
    '''
    Generate clustered points within given circles. 
    
    Arguments:
    circles: a dictionary, key: circle centers, values: radius
    num_sample: the number of samples within each circle
    
    Returns
    points_x: an array of x coordinates
    points_y: an array of y coordinates
    points_c: an array of cost coefficients 
    points_cluster: an array of cluster index
    '''
    np.random.seed(1)
    points_x, points_y, points_cluster = np.array([]), np.array([]), np.array([])
    c = 1 # cluster id
    for center in circles:        
        points_x = np.append(points_x, center[0]) # add center
        points_y = np.append(points_y, center[1])
    
        theta = np.random.uniform(0 , 2*np.pi, num_samples) # random angles
        r = circles[center] * np.random.random((num_samples)) # random radius
        x, y = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
        
        points_x = np.append(points_x, x) # add points within the circle
        points_y = np.append(points_y, y)
        points_cluster = np.append(points_cluster, c * np.ones(num_samples + 1))
        
        c += 1 # update cluster id
    points_c = np.ones((num_samples + 1) * len(circles)) # uniform cost 
    
    return points_x, points_y, points_cluster, points_c


# %%


def compute_distance(x1, y1, x2, y2):
    '''
    Compute Euclidean distance between two points. 
    
    Arguments
    x1, y1: x coordinate of point 1, y coordinate of point 2
    x2, y2: x coordinate of point 2, y coordinate of point 2
    
    Returns
    Euclidean distance between points 1 and 2
    '''
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# %%


def compute_a_N(points_x, points_y, S):
    '''
    Compute coverage matrix within given points and coverage standard and coverage set. 
    
    Arguments:
    points_x, points_y: an array of x and y coordinates
    S: the service coverage standard
    
    Returns:
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    N: a list of coverage set, N[i]: a set of columns that can cover row i
    '''
    
    if len(points_x) != len(points_y): # check dimensions
        print("Dimensions not matched.")
        return 
    
    # initialize 
    a = np.zeros((len(points_x), len(points_x))) 
    N = [] 
    for i in range(len(points_x)):
        N.append(set())
     
    for i in range(len(points_x)):
        for j in range(i, len(points_x)): # for j >= i, save computation
            dist = compute_distance(points_x[i], points_y[i], points_x[j], points_y[j])
            if dist <= S: 
                a[i][j] = 1
                N[i].add(j)
                N[j].add(i)
    a = a + a.T - np.diag(np.diag(a)) # copy the upper right triangle into the lower left half
    
    return a, N 


# %%
def compute_cov(matrixFile, S):
    '''
    Compute coverage matrix within given points and coverage standard and coverage set. 
    
    Arguments:
    matrixFile: a file with OD matrix information, 
                this file has a head info in the first line, the rest is organized as 
                column-row pair id|column id|row id|network distance
    S: the service coverage standard
    
    Returns:
    a: an array of coverage information, a[i][j]: 0 or 1, indicating whether jth column can cover ith row
    N: a list of coverage set, N[i]: a set of columns that can cover row i
    M: a list of coverage set, M[j]: a set of rows that can be covered by column j
    '''
    
    infile = open(matrixFile,"r")
    line = infile.readline() # read head info
    n = int(line.strip().split('\t')[1]) # num of total rows/demand
    m = int(line.strip().split('\t')[2]) # num of total columns/potential facilities
    
    # initialize
    a = np.zeros((n, m)) 
    N = [] 
    for i in range(n):
        N.append(set())
    M = []
    for j in range(m):
        M.append(set())
        
    # compute a, N and M
    for line in infile:
        line = line.strip().split("\t") # read data
        if float(line[3]) <= S:
            a[int(line[2])-1, int(line[1])-1] = 1
            N[int(line[2])-1].add(int(line[1])-1)
            M[int(line[1])-1].add(int(line[2])-1)
    return a, N, M

# %%


def f_kc(c, K, funcType):
    '''
    Compute the function f(c, K) given c and K, detailed in Balas and Ho (1980). 
    
    Arguments:
    c: a cost coefficient of X[j] in the objective function of SCP/LRSCP 
    K: the number of uncovered rows that can be covered by column j 
    funcType: f(c, k) function options: "I", "II", "III", "IV", "V"
    
    Returns:
    f(c, K)
    '''
    epsilon = 1e-6 # a small number to avoid zero denominator
    
    if funcType == "I":
        return c
    
    elif funcType == "II":
        return c/(K + epsilon)
    
    elif funcType == "III":
        if K == 0 or 1:
            return c/(1 + epsilon)
        return c/(np.log2(K) + epsilon)
    
    elif funcType == "IV":
        if K == 0 or 1:
            return c/(K * 1 + epsilon)
        return c/(K * np.log2(K) + epsilon)
    
    elif funcType == "V":
        if K == 0 or 1 or 2:
            return  c/(K * 1 + epsilon)
        return c/(K * np.log(K) + epsilon)
    
    else:
        print("Non-supported funcType.")
        return 

# %%
