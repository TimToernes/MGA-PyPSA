# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
#from IPython import get_ipython
#from IPython.display import display, clear_output


#%% import packages
import pypsa
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import logging
import pyomo.environ as pyomo_env
import yaml
import time
import os

# Defining exstra functionality, that updates the objective function of the network
def direction_search(network, snapshots,  MGA_slack = 0.05, point=[0,0,0],dim=3,old_objective_value=0):
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



#%% Search hull
def search_hull(network,options):

    timer = time.time()
    # Original solution
    logging.disable()
    print('starting initial solution')
    solver_options = options['solver_options']
   
    network.lopf(network.snapshots, 
                solver_name='gurobi',solver_options=solver_options),

    old_objective_value = network.model.objective()

    original_solution = [sum(network.generators[network.generators.type=='ocgt'].p_nom_opt),
                        sum(network.generators[network.generators.type=='wind'].p_nom_opt),
                        sum(network.generators[network.generators.type=='solar'].p_nom_opt)]
    
    #%% Initialize dataframe with original solution
    original_solution
    df_points = pd.DataFrame(columns=['type','feasable','ocgt','wind','solar'])
    df_points.loc[0] = ['original',True] + original_solution

    df_detail = pd.DataFrame(columns=network.generators.p_nom_opt.index)
    df_detail.loc[0] = list(network.generators.p_nom_opt.values)

    print('finished initial solution in {} sec'.format(time.time()-timer))

    # MGA solutions
    MGA_slack = options['MGA_slack']
    dim = options['dim']
    MGA_convergence_tol = options['MGA_convergence_tol']

    old_volume = 0 
    epsilon = 1
    epsilon_log = [epsilon]
    directions_searched = []
    computations = 0 
    hull = None
    while not all(np.array(epsilon_log[-2:])<MGA_convergence_tol) : # The last two itterations must satisfy convergence tollerence
        timer = time.time()
        print('Starting itteration {} '.format(len(epsilon_log)-1))
        # Generate list of directions to search in for this batch
        if len(df_points)<=1 : # if only original solution exists, max/min directions are chosen
            directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
        else : # Otherwise search in directions normal to 
            directions = np.array(hull.equations)[:,0:-1]
        # Itterate over directions in batch 
        for direction,i in zip(directions,range(len(directions))) :
            # Check if the direction to search in has already been searched 
            if not any([abs(np.linalg.norm(dir_searched-direction))<0.01  for dir_searched in directions_searched]):
                
                #clear_output()
                #print('Itteration {} -'.format(len(epsilon_log)-1) + ' direction {} of {}'.format(i,len(directions)),end='\r')
                #print('direction {} of {}'.format(i,len(directions)),end='\r')
                # Solve network
                network.lopf(network.snapshots,                                 
                            solver_name='gurobi',                                 
                            extra_functionality=lambda network,                                 
                            snapshots: direction_search(network,snapshots,MGA_slack,direction,dim,old_objective_value),
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
        print('Itteration time {} sec'.format(time.time()-timer))
    
    return df_detail, df_points



#%%
if __name__=='__main__':

    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    options = yaml.load(open(dir_path+'setup.yml',"r"),Loader=yaml.FullLoader)
    timer = time.time()


    network = pypsa.Network()

    network.import_from_hdf5(dir_path+ 'data/networks/' +options['network_name'])
    print('Loaded network {}'.format(options['network_name']))


    df_detail,df_points = search_hull(network,options)

    #df_detail.to_csv('output/hull_points_detail.csv',index=False)
    #df_points.to_csv('output/hull_points.csv',index=False)

    print('Time elapsed {}'.format(time.time()-timer))

