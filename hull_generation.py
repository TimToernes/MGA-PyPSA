# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
from IPython.display import display, clear_output


#%%
import pypsa
import pandas as pd
import numpy as np
import time
#import cufflinks as cf
#import plotly.offline as pltly
#pltly.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull,  Delaunay
import logging

#%% ## Notes

# 1. Search in direction normal to faces
# 2. Create measure for searched area
# 3. Gini coeficient 
# 4. Place points by linear combination of points 
# 5. subdivide spaces by introducing mass center 

#%% Loading network from file
network = pypsa.Network()

network.import_from_hdf5('Scandinavia')

#network.import_from_hdf5('euro-30')
#network.snapshots = network.snapshots[0:2]
#network.snapshots

#%% Initial solution of network
timer = time.time()
logging.disable()
solver_options = {  'method': 2,  
                    'threads': 8, 
                    'logfile': 'solver.log', 
                    'BarConvTol' : 1.e-8, #1.e-12 ###1e-8 # [0..1]
                    'crossover' : 0,   # or -1
                    'FeasibilityTol' : 1.e-6 }#1e-2 ###1e-6 # [1e-9..1e-2]
#blockPrint()
network.lopf(network.snapshots, 
            solver_name='gurobi',solver_options=solver_options),
#enablePrint()
old_objective_value = network.model.objective()
elapsed = time.time()-timer
logging.disable(False)

print(elapsed)

original_solution = [sum(network.generators[network.generators.type=='ocgt'].p_nom_opt),
                     sum(network.generators[network.generators.type=='wind'].p_nom_opt),
                     sum(network.generators[network.generators.type=='solar'].p_nom_opt)]

#%% Define extra functionality 
import pyomo.environ as pyomo_env
# Defining exstra functionality, that updates the objective function of the network
def direction_search(network, snapshots,  MGA_slack = 0.05, point=[0,0,0],dim=3):
# Identify the nonzero decision variables that should enter the MGA objective function.
    objective = 0
    if dim == 3:
        variables = [gen_p for gen_p in network.model.generator_p_nom]
        types = ['ocgt','wind','olar']
        for i in range(3):
            gen_p_type = [gen_p  for gen_p in variables if gen_p[-4:]==types[i]]
            objective += point[i]*sum([network.model.generator_p_nom[gen_p] for gen_p in gen_p_type])
    elif dim == 9:
        generators = [gen_p for gen_p in network.model.generator_p_nom]
        for gen_p,i in zip(generators,range(len(generators))):
            objective += point[i]*network.model.generator_p_nom[gen_p]

    # Add the new MGA objective function to the model.
    network.model.mga_objective = pyomo_env.Objective(expr=objective)
    # Deactivate the old objective function and activate the MGA objective function.
    network.model.objective.deactivate()
    network.model.mga_objective.activate()
    # Add the MGA slack constraint.
    network.model.mga_constraint = pyomo_env.Constraint(expr=network.model.objective.expr <= 
                                          (1 + MGA_slack) * old_objective_value)


#%% Initialize dataframe with original solution
original_solution
df_points = pd.DataFrame(columns=['type','feasable','ocgt','wind','solar'])
df_points.loc[0] = ['original',True] + original_solution

df_detail = pd.DataFrame(columns=network.generators.p_nom_opt.index)
df_detail.loc[0] = list(network.generators.p_nom_opt.values)


#%% Search hull

MGA_slack = 0.1
dim = 3
tol = 0.1
old_volume = 0 
epsilon = 1
epsilon_log = [epsilon]
directions_searched = []
computations = 0 
logging.disable()
while not all(np.array(epsilon_log[-2:])<tol) : # The last two itterations must satisfy convergence tollerence
    # Generate list of directions to search in for this batch
    if len(df_points)<=1 : # if only original solution exists, max/min directions are chosen
        directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
    else : # Otherwise search in directions normal to 
        directions = np.array(hull.equations)[:,0:-1]
    # Itterate over directions in batch 
    for direction,i in zip(directions,range(len(directions))) :
        # Check if the direction to search in has already been searched 
        if not any([abs(np.linalg.norm(dir_searched-direction))<0.01  for dir_searched in directions_searched]):
            
            clear_output()
            display('Itteration {}'.format(len(epsilon_log)-1))
            display('direction {} of {}'.format(i,len(directions)))
            # Solve network
            network.lopf(network.snapshots,                                 
                        solver_name='gurobi',                                 
                        extra_functionality=lambda network,                                 
                        snapshots: direction_search(network,snapshots,MGA_slack,direction,dim),
                        solver_options=solver_options)
            
            point3 = [sum(network.generators[network.generators.type=='ocgt'].p_nom_opt),
                      sum(network.generators[network.generators.type=='wind'].p_nom_opt),
                      sum(network.generators[network.generators.type=='solar'].p_nom_opt)]
            point9 = list(network.generators.p_nom_opt.values)
            
            df_points.loc[df_points.index.max()+1]=['maxmin',True,*point3]                                    

            df_detail.loc[df_detail.index.max()+1]=point9
        
            directions_searched.append(direction)

            computations += 1

    if dim == 3:
        hull = ConvexHull(df_points[['ocgt','wind','solar']])
    elif dim == 9:
        hull = ConvexHull(df_detail.values,qhull_options='A-0.99')
            

    delta_v = hull.volume - old_volume
    old_volume = hull.volume
    epsilon = delta_v/hull.volume
    epsilon_log.append(epsilon)
    
    print('####### EPSILON ###############')
    print(epsilon)

#%% Plot of convergence
fig = plt.figure()
plt.plot(range(len(epsilon_log)),epsilon_log)


#%% Hull volume

hull3 = ConvexHull(df_points[df_points.type=='maxmin'][['ocgt','wind','solar']],incremental=True)
hull3.volume

#hull3V_9dim = 128776590064.37102
#hull3V_3dim = 185691445079.29614


#%% Plot of max and min points 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
  
ax.plot(df_points[1:]['ocgt'],df_points[1:]['wind'],df_points[1:]['solar'], "go")

ax.set_xlabel('gas')
ax.set_ylabel('wind')
ax.set_zlabel('solar')

#%% saving data

#np.save('hull_points',df_detail)

df_detail.to_csv('hull_points_detail.csv',index=False)
df_points.to_csv('hull_points.csv',index=False)

#%%
