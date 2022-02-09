##############################################################################
# Functions file for the 3/4 Josephson junction Qubit code.
##############################################################################

import numpy as np
from scipy.signal import argrelextrema
import math as math

# Returns the diagonal z values for a line in the xy plane for the purpose of
# plotting a slice across a mesh.
def take_diagonal_weights(data_in, x1, y1, x2, y2, pts=100):
    data = data_in.copy()
    
    x_fixed = [0]*pts
    y_fixed = [0]*pts
    weights = [0]*pts
    
    m = (y2-y1)/(x2-x1)
    b = (x1*y2 - x2*y1)/(x1-x2)
    
    x_diag = np.linspace(x1, x2, pts)
    y_diag = m*x_diag + b
    
    for i in range(pts):
        x_fixed[i] = int(np.round(x_diag[i]))
        y_fixed[i] = int(np.round(y_diag[i]))
        weights[i] = data[x_fixed[i], y_fixed[i]]      
        
    return weights, x_fixed, y_fixed
    
# Finds the maximum value in a matrix.
def find_maximum(data_in, limx, limy, minx=0, miny=0):
    data = data_in.copy()
    
    m = data[minx:limx, miny:limy]
    maxima = m[0,0]
    maxima_x = 0
    maxima_y = 0
    
    for i in range(len(m[:,0])):
        for j in range(len(m[0,:])):
            if m[i,j] > maxima:
                maxima = m[i,j]
                maxima_x = i
                maxima_y = j
        
    return maxima, maxima_x, maxima_y

# For use in finding minima and maxima. Returns all the points surrounding
# a specified input point for a matrix.
def find_surrounding_points(data, i, j):
    pts = []
    valmaxi = len(data[:,0]) - 1
    valmaxj = len(data[0,:]) - 1
    
    if i < valmaxi and j < valmaxj:
        pts.append(data[i+1,j+1])
    if i < valmaxi:
        pts.append(data[i+1,j])
    if i < valmaxi and j > 0:
        pts.append(data[i+1,j-1])
    if i > 0 and j < valmaxj:
        pts.append(data[i-1,j+1])
    if i > 0:
        pts.append(data[i-1,j])
    if i > 0 and j > 0:
        pts.append(data[i-1,j-1])
    if j > 0:
        pts.append(data[i,j-1])
    if j < valmaxj:
        pts.append(data[i,j+1])
    
    return pts

# Stores information about 3D minima and maxima.
class extrema:
    def __init__(self, value, xi, yi, extr, x=0, y=0, isdupe=False):
        self.value = value
        self.x = x
        self.y = y
        self.xi = xi
        self.yi = yi
        self.extr = extr
        self.isdupe = isdupe

# Stores information about 2D minima and maxima.
class extrema2D:
    def __init__(self, value, ind, extr):
        self.value = value
        self.ind = ind
        self.extr = extr

# Finds all of the minima in a matrix and returns them.
def find_minima(data_in, limx, limy, minx=0, miny=0):
    data = data_in.copy()
    
    m = data[minx:limx, miny:limy]
    minimas = []
    
    for i in range(len(m[:,0])):
        for j in range(len(m[0,:])):                        
            if check_isMinima(data, i, j) == True:
                minimas.append(extrema(m[i,j], i, j, 'minima'))
                
    return minimas

# Finds all of the minima in an array and returns them.
def find_minima1D(array_in):
    arr = np.array(array_in)
    
    ind = argrelextrema(arr, np.less)
    minimas = []
    
    for i in range(len(ind[0])):
        indS = ind[0]
        minimas.append(extrema2D(arr[i], indS[i], 'minima'))
        
    return minimas

# Finds all of the maxima in an array and returns them.
def find_maxima1D(array_in):
    arr = np.array(array_in)
    
    ind = argrelextrema(arr, np.greater)
    maximas = []
    
    for i in range(len(ind[0])):
        indS = ind[0]
        maximas.append(extrema2D(arr[i], indS[i], 'maxima'))
        
    return maximas

# Finds all of the maxima in a matrix and returns them.
def find_maxima(data_in, limx, limy, minx=0, miny=0):
    data = data_in.copy()
    
    m = data[minx:limx, miny:limy]
    maximas = []
    
    for i in range(len(m[:,0])):
        for j in range(len(m[0,:])):                        
            if check_isMaxima(data, i, j) == True:
                maximas.append(extrema(m[i,j], i, j, 'maxima'))
                
    return maximas

# Checks if a provided matrix point is a maxima.
def check_isMinima(data_in, i, j):
    data = data_in.copy()
    
    surr = find_surrounding_points(data, i, j)
    isMinima = True
    for k in range(len(surr)):
        if surr[k] < data[i,j]:
            isMinima = False
    
    return isMinima

# Checks if a provided matrix point is a minima.
def check_isMaxima(data_in, i, j):
    data = data_in.copy()
    
    surr = find_surrounding_points(data, i, j)
    isMaxima = True
    for k in range(len(surr)):
        if surr[k] > data[i,j]:
            isMaxima = False
    
    return isMaxima

# Removes minima that are located at the edge of a mesh.
def remove_false_minima(mins, max_point_value):
    mincounter = len(mins)
    
    # Iterates through all minima and converts invalid minima to NaN.
    for id1 in range(mincounter):
        for id2 in range(mincounter):
            if id1 != id2:
                if mins[id1].value == mins[id2].value:
                    mins[id2].value = np.NaN
    
                   
        if mins[id1].ind == 0 or mins[id1].ind == (max_point_value-1):
            mins[id1].value = np.NaN
           
    # Removes any minima that are NaN.
    NaNindexes = []
    for idmin in range(mincounter):
        if math.isnan(mins[idmin].value):
            NaNindexes.append(idmin)
    
    for idmin in sorted(NaNindexes, reverse=True):
        mins.pop(idmin)
        
    return mins

# Locates a value within an array and returns the index.
def locate_in_array(arr, value):
    indices = []
    for i in range(len(arr)):
        if arr[i] == value:
            indices.append(i)
            
    if len(indices) == 1:
        return indices[0]
    elif len(indices) == 0:
        print("No value found")
        return -1
    else:
        return indices
  
# Locates a value within a matrix and returns the indices.          
def locate_in_matrix(mat, value):
    indices = []
    for i in range(len(mat[:,0])):
        for j in range(len(mat[0,:])):
            if mat[i, j] == value:
                indices.append([i, j])
            
    if len(indices) == 1:
        return indices[0]
    elif len(indices) == 0:
        print("No value found")
        return indices
    else:
        return indices
    
# Checks if v1 and v2 are within some value n of each other.
def iswithin(v1, v2, n):
    if v1 >= v2:
        if v2 + n >= v1:
            return True
        else:
            return False
    else:
        if v1 + n >= v2:
            return True
        else:
            return False
        
# Removes the influence of 'duplicate minima' on the number of minima - minima
# that are so close together the fact they are considered different minima is
# an error caused by plotting resolution.
def remove_dupe_minima_count(m, n):
    invalid_minima = []
    
    for i in range(len(m)):
        dupe_minima = 0
        if m[i].isdupe == False:
            for j in range(len(m)):
                if iswithin(m[i].xi,m[j].xi,3) and iswithin(m[i].yi,m[j].yi,3) and i != j:
                    dupe_minima += 1
                    m[j].isdupe = True
                        
            invalid_minima.append(dupe_minima)
    #print(invalid_minima, "g") #DEBUG
    return len(invalid_minima)

# Combines several 4xn matrices. This is used when combining solutions to
# mesh_solve_allvars() in the main code.
def matrix_4xn_combine(mat_i_, mat_f_):
    mat_i = mat_i_.copy()
    mat_f = mat_f_.copy()
    height = len(mat_i[:,0]) + len(mat_f[:,0])
    comb = np.zeros((height, 4))
    
    for i in range(len(mat_i[:,0])):
        for j in range(4):
            comb[i, j] = mat_i[i, j]
            
    for i in range(len(mat_f[:,0])):
        for j in range(4):
            comb[i+len(mat_i[:,0]), j] = mat_f[i, j]
            
    return comb
