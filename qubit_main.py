##############################################################################
# This piece of code was written to simulate a 3 or 4 Josephson junction Qubit
# system. It is capable of testing various circuit frustration and alpha values
# to find for which parameters the Qubit remains in a stable two-minima system.
##############################################################################

import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib import cm   
import qubit_datafunctions as df     
import time as time
        
# Mesh class
class mesh:
    def __init__(self, grid, x, y, points, alpha, f1, f2, junc):
        self.grid = grid
        self.x = x
        self.y = y
        self.points = points  
        self.alpha = alpha
        self.f1 = f1
        self.f2 = f2 
        self.junc = junc
    
    def initialise_axis(self, ax):
        self.ax = ax
        
    # Plots the mesh.
    def plot(self):
        X, Y = np.meshgrid(self.x, self.y)
        
        # Plot the surface.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.initialise_axis(ax)
        
        surf = ax.plot_surface(X, Y, self.grid, cmap=cm.Purples_r,
                               linewidth=0, antialiased=False)
    
        ax.set_xlabel('g1')
        ax.set_ylabel('g2')
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    # Returns the minima of the mesh.
    def mesh_minima(self):
        minimas = df.find_minima(self.grid, len(self.grid[:,0]), len(self.grid[0,:]))
        return minimas
    
    def mesh_maxima(self):
        minimas = df.find_maxima(self.grid, len(self.grid[:,0]), len(self.grid[0,:]))
        return minimas
    
    # Returns the locations of the minima in the mesh, ignoring edge minima.
    def locate_minima(self):
        minimas = self.mesh_minima()
        indx = np.zeros(len(minimas))
        indy = np.zeros(len(minimas))
        
        if len(minimas) > 0:
            for i in range(len(minimas)):
                indx[i] = minimas[i].xi
                indy[i] = minimas[i].yi
        
        mincounter = len(indx)
        
        # Sets the indices of the edge minima to NaN so they can be popped from
        # the list of valid minima.
        for i in range(mincounter):
            if indx[i] == len(self.grid[:,0])-1 or indx[i] == 0 or indy[i] == len(self.grid[0,:])-1 or indy[i] == 0:
                indx[i] = np.NaN
                indy[i] = np.NaN
        
        NaNindexes = []
        for idmin in range(mincounter):
            if math.isnan(indx[idmin]):
                NaNindexes.append(idmin)
        
        for idmin in sorted(NaNindexes, reverse=True):
            minimas.pop(idmin)
            
        return minimas
       
    # Returns the number of minima in the mesh, with the option to remove
    # minima that are located very close to each other and are hence likely
    # duplicates.         
    def quantity_minima(self, clean=True):
        minimas = self.locate_minima()
        if clean == True:
            return df.remove_dupe_minima_count(minimas, 3)
        else:            
            return len(minimas)
     
    # Plots vertical lines with numbers at either the minima or maxima on the
    # mesh. This was used when determining the indicies of minima and maxima
    # in arrays and when plotting lines between 2 extrema.
    def display_extrema_on_mesh(self, etype):
        if etype == 'minima':
            z_plotlines = [0,4]
            minimas = self.mesh_minima()
            for i in range(len(minimas)):
                x_plotline = [self.x[minimas[i].xi], self.y[minimas[i].xi]]
                y_plotline = [self.x[minimas[i].yi], self.y[minimas[i].yi]]
                
                self.ax.plot(x_plotline, y_plotline, z_plotlines, 'seagreen', zorder=50)
                self.ax.text(self.x[minimas[i].xi], self.y[minimas[i].yi], z_plotlines[1], str(i), color='seagreen')
        else:
            maximas = self.mesh_maxima()
            z_plotlines = [0,8]
            for i in range(len(maximas)):
                x_plotline = [self.x[maximas[i].xi], self.y[maximas[i].xi]]
                y_plotline = [self.x[maximas[i].yi], self.y[maximas[i].yi]]
                
                self.ax.plot(x_plotline, y_plotline, z_plotlines, 'maroon', zorder=50)
                self.ax.text(self.x[maximas[i].xi], self.y[maximas[i].yi], z_plotlines[1], str(i), color='maroon')

    # Plots a line between two minima on the mesh, and plots the slice between
    # those two minima as a side-on slice of the mesh. The input tol allows
    # the plotting of a line further than the minima, such that more of the
    # system can be viewed in the slice.
    def plot_extrlines_min(self, minima1, minima2, labels=["L", "R"], tol=4):
        minimas = self.mesh_minima()
        
        # Sets up the coordinates of the vertical lines placed on the mesh at
        # the two minima the plot will be between.
        x_plotline1 = [self.x[minimas[minima1].xi], self.y[minimas[minima1].xi]]
        x_plotline2 = [self.x[minimas[minima2].xi], self.y[minimas[minima2].xi]]
        y_plotline1 = [self.x[minimas[minima1].yi], self.y[minimas[minima1].yi]]
        y_plotline2 = [self.x[minimas[minima2].yi], self.y[minimas[minima2].yi]]
        z_plotlines = [0, 12]
        
        self.ax.plot(x_plotline1, y_plotline1, z_plotlines, 'r', zorder=100)
        self.ax.plot(x_plotline2, y_plotline2, z_plotlines, 'g', zorder=100)
        self.ax.text(self.x[minimas[minima1].xi], self.y[minimas[minima1].yi], 12, labels[0], zorder=110)
        self.ax.text(self.x[minimas[minima2].xi], self.y[minimas[minima2].yi], 12, labels[1], zorder=110)
        
        # Sets up the parameters for the diagonal slice.
        diag_line_x = np.linspace(self.x[minimas[minima1].xi], self.x[minimas[minima2].xi], self.points)
        diag_line_y = np.linspace(self.y[minimas[minima1].yi], self.y[minimas[minima2].yi], self.points)
        
        # Increases the line by a length decided by the variable tol. This is
        # so that more of the graph can be seen on the plot than just the part
        # between the 2 minima.
        if tol != 1:
            x_1 = diag_line_x[len(diag_line_x)-1]
            x_2 = diag_line_x[0]
            y_1 = diag_line_y[len(diag_line_y)-1]
            y_2 = diag_line_y[0]
            dist = np.sqrt(((x_1 - x_2)**2 + (y_1 - y_2)**2))
            m = (y_2 - y_1)/(x_2 - x_1)
            c = (tol-dist)/2
            del_x = np.sqrt((c**2)/((m**2) + 1))
            del_y = m*del_x
            
            if m > 1:
                y_1 += del_y
                y_2 -= del_y
                diag_line_y = np.linspace(y_2, y_1, self.points)
            else:
                y_1 -= del_y
                y_2 += del_y
                diag_line_y = np.linspace(y_1, y_2, self.points)
            
            if m > 1:
                x_1 += del_x
                x_2 -= del_x
                diag_line_x = np.linspace(x_2, x_1, self.points)
            else:
                x_1 -= del_x
                x_2 += del_x
                diag_line_x = np.linspace(x_1, x_2, self.points)
                
            
        self.ax.plot(diag_line_x, diag_line_y, np.ones(self.points)*12, 'b', zorder=100)
        
        # Calculates the result of the equation for the values along the
        # diagonal slice.
        diag_weights = np.zeros((len(diag_line_x)))
        diag_phi = np.zeros((len(diag_line_x)))
        for i in range(len(diag_line_x)):
            diag_weights[i] = junction(diag_line_x[i]*np.pi, diag_line_y[i]*np.pi, self.alpha, self.f1, self.f2, self.junc)
            diag_phi[i] = np.sqrt((diag_line_x[i]*np.pi)**2 + (diag_line_y[i]*np.pi)**2)
            
        counter = np.linspace(0,1,len(diag_weights))
        diagmins = df.find_minima1D(diag_weights)
        print(len(diagmins))
        labels_pos = [counter[diagmins[0].ind], counter[diagmins[1].ind]]
        
        fig = plt.figure()
        plt.xticks(labels_pos, labels)
        plt.plot(counter, diag_weights) 
        
        return diag_phi, diag_weights
        

# Class to store the initial mesh solver values in.
class initial_mesh_solver_values:
    def __init__(self, alpha, f, alphamax, fcmax):
        self.alpha = alpha
        self.f = f
        self.alphamax = alphamax
        self.fcmax = fcmax

# The equation for each of the mesh points. The function contains the ability to
# specify the number of junctions in the Josephson junction.
def junction(g1, g2, alpha, f1, f2, junc='mooji'):
    if junc == 'mooji' or junc == '4':
        UE = 2 + 2*alpha - np.cos(g1) - np.cos(g2) - 2*alpha*(np.cos(f2*np.pi))*\
            (np.cos(2*f1*np.pi + f2*np.pi + g1 - g2))
    else:
        UE = 2 + alpha - np.cos(g1) - np.cos(g2) - alpha*\
            (np.cos(2*f1*np.pi + g1 - g2))
            
    return UE

# Sets up a mesh object with the parameters passed into it.
def u_mesh(minlim, maxlim, f1=0.33, f2=0.33, alpha=0.75, plot=True, junc='mooji', display=[' '], total_points=1e4):
    #total points sets the approximate total number of points for the code to run over
    points = math.ceil(np.sqrt(total_points))  # sets the number of points for each variable        
    
    # Sets values of gamma to run over
    gamma1 = (np.linspace(minlim, maxlim, points))*np.pi
    gamma2 = gamma1.copy()
    
    # Calulates U/Ej for every point on a new mesh.
    meshdata = np.zeros((points, points))  
    for i in range(points):
        for j in range(points):
            meshdata[i,j] = junction(gamma1[j], gamma2[i], alpha, f1, f2, junc=junc)
    
    # Make data.
    x = gamma1 / np.pi
    y = gamma2 / np.pi
    
    # Create mesh object with the calculated values.
    umesh = mesh(meshdata, x, y, points, alpha, f1, f2, junc)
    
    # Optionally plot the mesh.
    if plot == True:
        umesh.plot()
        # Optionally show the maxima and minima along with their associated
        # indices for analysis.
        if display != [' ']:
            if 'minima' in display:
                umesh.display_extrema_on_mesh('minima')
            if 'maxima' in display:
                umesh.display_extrema_on_mesh('maxima')
    
    return umesh

# Takes a diagonal slices of the mesh at several points of alpha and f for the
# purpose of investigating stability. Created for part (d). Obsolete after the
# writing of the function mesh_fc_alpha().
### OBSOLETE ###
def diagonal_fc_alpha(total_points=1e4):
    fig3 = plt.figure()
    
    points = math.ceil(np.sqrt(total_points))

    steps = 1000
    alp_steps = 200
    ug1 = (np.linspace(-1.2, 1.2, points))*np.pi
    ug2 = (np.linspace(1.2, -1.2, points))*np.pi
    alpha_c = np.linspace(0.5, 2, alp_steps)
    f_c = np.linspace(0, 0.3, steps)
    f = f_c + 0.5
    
    f_db = [0]*alp_steps
    
    for n in range(alp_steps):
        m = 0

        broken = False
        while broken == False:
            
            UE = np.zeros(points)
            for i in range(points):
                #Assuming the maximas don't move we can use the previous line calculation with new parameters
                UE[i] = junction(ug1[i], ug2[i], alpha_c[n], f[m], f[m], junc='orlando')
                
            linemax = df.find_maxima1D(UE)
            UE_appended = UE[linemax[0].ind:linemax[len(linemax)-1].ind]
            UE_minima = df.remove_false_minima(df.find_minima1D(UE_appended), points)
            
            checker = len(UE_minima)
            
            if checker != 2:
                f_db[n]=f[m]
                #when break cut
                broken = True
                print("Break found,", n, '/', alp_steps, '; m value was ', m)
                
            m += 1
            if m == steps:
                broken = True
                
    fplot = [0]*len(f_db)
    for i in range(len(f_db)):
        fplot[i] = f_db[i] - 0.5
    plt.plot(alpha_c, fplot)
    plt.xlabel('alpha')
    plt.ylabel('fc')
    
# Creates a unit cell mesh for several values of alpha and f/f1/f2 (depending
# on the number of junctions) in order to investigate the stability by
# meausring the number of minima in this mesh.
def mesh_fc_alpha(total_points=1e4, alp_steps=200, fc_steps=200, junc='orlando', hold_const=['none'], save_pts=True, tol=0.001):
    # Initialises the values for alpha and f, and sets up the range for which
    # they will run over.
    init = initialise_mesh_values(junc, hold_const)
    alpha_c = np.linspace(init.alpha, init.alphamax, alp_steps)
    f_c = np.linspace(0, init.fcmax, fc_steps)
    f = f_c + init.f
    
    m_prev = 0
    
    # Runs over all alpha steps.
    f_db = [0]*alp_steps
    for n in range(alp_steps):
        broken = False
        m = 0
        
        # Optionally cuts out m values a percentage (tol) below the previous
        # m loop value. This is to prevent the code from generating meshes
        # which will not be used, optimising the algorithm.
        if save_pts == True:
            isvalid = False
            
            if n==0:
                isvalid = True
            elif m_prev - tol*fc_steps > 0:
                m = int(m_prev - tol*fc_steps)
                
                invalidruns = 1
                while isvalid == False:
                    valid_mesh = u_mesh(-1, 1, f1=f[m], f2=f[m], alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)

                    if valid_mesh.quantity_minima() == 2:
                        isvalid = True
                    else:
                        m = int(m - tol*invalidruns*fc_steps)
                        print("Invalid mesh") #DEBUG
                        invalidruns +=1
                    del valid_mesh
            else:
                isvalid = True
        
        while broken == False:
            # Checks to see which values are to be held constant, and whether
            # there are 3 or 4 junctions to consider in the equation.
            if junc == 'orlando' or junc == '3' or hold_const==['none']:
                it_mesh = u_mesh(-1, 1, f1=f[m], f2=f[m], alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)
            else:
                if 'f1' in hold_const:
                    it_mesh = u_mesh(-1, 1, f1=init.f, f2=f[m], alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)
                elif 'f2' in hold_const:
                    it_mesh = u_mesh(-1, 1, f1=f[m], f2=init.f, alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)
                elif '2f1+f2' in hold_const:
                    it_mesh = u_mesh(-1, 1, f1=f[m], f2=(1-2*f[m] + f_c[m]), alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)
                else:
                    it_mesh = u_mesh(-1, 1, f1=f[m], f2=f[m], alpha=alpha_c[n], plot=False, junc=junc, total_points=total_points)
                    
            
            if it_mesh.quantity_minima() != 2:
                f_db[n] = f[m]
                # Breaks the loop when the value of f is found for this alpha.
                broken = True              
                m_prev = m
                if (n % 25) == 0:
                    print("Completed,", n, '/', alp_steps)
                print("Break found,", n, '/', alp_steps, '; m value was ', m) #DEBUG
                
            m += 1
            # Breaks the loop when the nunber of f points has ran out.
            if m == fc_steps:
                broken = True
            
            #del it_mesh
                
    # Plots the graph of fc against alpha.
    fig = plt.figure()         
    fplot = [0]*len(f_db)
    for i in range(len(f_db)):
        fplot[i] = f_db[i] - init.f
    plt.plot(alpha_c, fplot)
    plt.xlabel("alpha")
    plt.ylabel("fc")
    
    return alpha_c, fplot

# Returns whether or not a mesh is stable by calculating the number of minima.
def is_mesh_stable(alpha, f1, f2, junc, res):
    it_mesh = u_mesh(-1, 1, f1=f1, f2=f2, alpha=alpha, plot=False, junc=junc, total_points=res)
    
    if it_mesh.quantity_minima() != 2:
        stable = True
    else:
        stable = False
        
    return stable

# Finds the f1 or f2 point at which a mesh becomes unstable, depending on which
# of those is held constant.
def find_stable_additional_f(const, f_im, fnc, a, total_points, plen):
    broken = False
    p = 0
    f = 0
    while broken == False: # Iterates by p
        if const == 'f1':
            f1 = f_im
            f2 = fnc[p]
        else:
            f1 = fnc[p]
            f2 = f_im
  
        if is_mesh_stable(a, f1, f2, '4', total_points) == False:
            broken = True
            f = fnc[p]
        
        p += 1
        if p == plen and broken == False:
            broken = True
            f = np.NaN
            
    return f, p-1

# Computes the data matricies for investigating the stability for a 4-junction
# system. alp_steps takes an initial, final and total number of points for the
# purpose of splitting the computation of this function across several PCs.
def mesh_solve_allvars(total_points=1e4, alp_steps=[0,2000,2000], fc_steps=2000):
    # Initialises the setup values, then modifies these slightly as they are
    # set up for the 3-junction system and need to be in the context of a 4-
    # junction system.
    init = initialise_mesh_values('4', ['none'])
    init.f = -0.3
    alpha_c = np.linspace(init.alpha, init.alphamax, alp_steps[2])
    fc = np.linspace(0, init.fcmax, fc_steps)
    f_i = np.linspace(init.f, 1.67, fc_steps)
    
    # Concatenates the alpha array such that it respects the points being run
    # over in the piece of simulation.
    alpha_c = alpha_c[alp_steps[0]:alp_steps[1]]
    con_len_a = len(alpha_c)
    con_len_f = fc_steps
    
    data = np.zeros((con_len_a,4))
    s = 0 # s contains the number of valid solutions found for stable setups
    
    for n in range(con_len_a):
        for m in range(con_len_f):
            stime=time.time()
            # For constant f1
            data[s,0] = alpha_c[n]
            data[s,1] = f_i[m] #F1
            stable_f2, p = find_stable_additional_f('f1', f_i[m], f_i, alpha_c[n], total_points, con_len_f) #F2
            data[s,2] = stable_f2
            data[s,3] = 0
            s+=1
            data.resize(s+1, 4)
            # Having found the f1 and f2 values for a stable system, backwards
            # calculates how these could have been formed from the addition of
            # some fc.
            i=0
            cont = True
            if math.isnan(stable_f2) == False:
                while f_i[m] - fc[i] > init.f and cont==True:
                    data[s,0] = alpha_c[n]
                    data[s,1] = f_i[m] - fc[i]
                    data[s,2] = stable_f2 - fc[i]
                    data[s,3] = fc[i]
                    s+=1
                    data.resize(s+1, 4)
                    if i == fc_steps-1:
                        cont=False
                    else:
                        i+=1
            
            # For constant f2
            data[s,0] = alpha_c[n]
            data[s,2] = f_i[m] #F2
            stable_f1, p = find_stable_additional_f('f2', f_i[m], f_i, alpha_c[n], total_points, con_len_f) #F1
            data[s,1] = stable_f1
            data[s,3] = 0
            s+=1
            data.resize(s+1, 4)
            # Repeats the step finding the addition of some fc for constant f2.
            i=0
            cont = True
            if math.isnan(stable_f1) == False:
                while f_i[m] - fc[i] > init.f and cont==True:
                    data[s,0] = alpha_c[n]
                    data[s,2] = f_i[m] - fc[i]
                    data[s,1] = stable_f1 - fc[i]
                    data[s,3] = fc[i]
                    s+=1
                    data.resize(s+1, 4)
                    if i == fc_steps-1:
                        cont=False
                    else:
                        i+=1
                        
            etime=time.time()
            
            print("n=",n,"s=",s,"Time elapsed =",etime-stime)
        print("n=",n)
    
    # Removes the invalid solutions containing NaN values.
    con_data_nonan = data[~np.isnan(data).any(axis=1), :]
    con_data = np.zeros((s-(len(data[:,0]) - len(con_data_nonan[:,0])),4))
    # Removes the invalid empty solutions at the end of the matrix if they
    # exist.
    for i in range(len(con_data[:,0])):
        if data[i,0] != 0:
            con_data[i,:] = con_data_nonan[i,:].copy()
            
    return con_data
                    
# Sets up the inital values of alpha and f/f1/f2 for the mesh solver.
def initialise_mesh_values(junc, hold_const):
    names = ['alpha', 'f', 'alpha_max', 'fc_max']
    
    if junc == 'mooji' or junc == '4':
        values = [0.5, 0.33, 8, 0.5]
        
    else:
        values = [0.5, 0.5, 2, 0.3]
        
    init = initial_mesh_solver_values(values[df.locate_in_array(names, 'alpha')], 
                                      values[df.locate_in_array(names, 'f')], 
                                      values[df.locate_in_array(names, 'alpha_max')], 
                                     values[df.locate_in_array(names, 'fc_max')])

    return init

plt.close('all')

# SECTION FOR PLOTTING MESH FOR REPORT #
# The following lines are used to call the above functions for use in plotting.
# The code placed below here changed throughout the project depending on which
# functions needed to be run for each part with varying function input
# parameters.

pltmesh = u_mesh(-3,3, f1=0.33, f2=0.33, alpha=0.8, junc='4', total_points=1e4)
pltmesh.plot_extrlines_min(7, 10, ["L00", "R00"], tol=4)
pltmesh.plot_extrlines_min(10, 13, ["R00", "L10"], tol=6)