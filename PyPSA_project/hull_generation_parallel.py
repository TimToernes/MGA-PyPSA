#%%
import warnings
warnings.simplefilter("ignore")
import pypsa
import time
import logging
import numpy as np
import os 
import yaml
import pyomo.environ as pyomo_env
from multiprocessing import Pool
from functools import partial
import pandas as pd
from scipy.spatial import ConvexHull

#%%
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

def inital_solution(network,options):

    logging.disable()

    solver_options = options['solver_options']

    network = network.copy()
   
    network.lopf(network.snapshots, 
                solver_name='gurobi',solver_options=solver_options),

    old_objective_value = network.model.objective()
    options['old_objective_value'] = old_objective_value

    original_solution = [sum(network.generators[network.generators.type=='ocgt'].p_nom_opt),
                        sum(network.generators[network.generators.type=='wind'].p_nom_opt),
                        sum(network.generators[network.generators.type=='solar'].p_nom_opt)]
    
    #%% Initialize dataframe with original solution
    #df_points = pd.DataFrame(columns=['type','feasable','ocgt','wind','solar'])
    #df_points.loc[0] = ['original',True] + original_solution

    columns=network.generators.p_nom_opt.index
    data_detail= np.array([network.generators.p_nom_opt.values])


    return data_detail,columns


def parallel_job(direction,network,options):
    
    logging.disable()
    MGA_slack = options['MGA_slack']
    solver_options = options['solver_options']
    dim = options['dim']
    old_objective_value = options['old_objective_value']
    network.lopf(network.snapshots,                                 
                solver_name='gurobi',                                 
                extra_functionality=lambda network,                                 
                snapshots: direction_search(network,snapshots,MGA_slack,direction,dim,old_objective_value),
                solver_options=solver_options)
    
    point3 = [sum(network.generators[network.generators.type=='ocgt'].p_nom_opt),
            sum(network.generators[network.generators.type=='wind'].p_nom_opt),
            sum(network.generators[network.generators.type=='solar'].p_nom_opt)]
    point9 = list(network.generators.p_nom_opt.values)
    
    #df_points.loc[df_points.index.max()+1]=['maxmin',True,*point3]                                    

    #df_detail.loc[df_detail.index.max()+1]=point9

    #directions_searched.append(direction)

    return point9

def run_mga(network,options,data_detail):

    p = Pool()
    MGA_convergence_tol = options['MGA_convergence_tol']
    dim = options['dim']
    old_volume = 0 
    epsilon_log = [1]
    directions_searched = []
    hull = None
    pool_size =os.cpu_count()/2
    print('Starting pool size {}'.format(pool_size))
    with Pool(processes=int(pool_size)) as p:
        while not all(np.array(epsilon_log[-2:])<MGA_convergence_tol) : # The last two itterations must satisfy convergence tollerence
            # Generate list of directions to search in for this batch
            if len(data_detail)<=1 : # if only original solution exists, max/min directions are chosen
                directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
            else : # Otherwise search in directions normal to 
                directions = np.array(hull.equations)[:,0:-1]
            # Itterate over directions in batch 

            # Write something here that filters already searched direction
            for direction in directions:
                if any([abs(np.linalg.norm(dir_searched-direction))<0.01  for dir_searched in directions_searched]):
                    directions.remove(direction)
                    
            result = p.map(partial(parallel_job, network=network,options=options), directions)


            data_detail = np.concatenate([data_detail,result])
            
            if dim == 3:
                points = np.array([sum(data_detail[:,0:3].T),sum(data_detail[:,3:6].T),sum(data_detail[:,6:9].T)])
                hull = ConvexHull(points.T)
            elif dim == 9:
                hull = ConvexHull(data_detail.values,qhull_options='A-0.99')
                    

            delta_v = hull.volume - old_volume
            old_volume = hull.volume
            epsilon = delta_v/hull.volume
            epsilon_log.append(epsilon)
            
            print('####### EPSILON ###############')
            print(epsilon)

    return data_detail
    

#%%
if __name__=='__main__':
    #__file__='multiprocessing_test.py'
    # Import options and start timer
    timer2 = time.time()
    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    options = yaml.load(open(dir_path+'setup.yml',"r"),Loader=yaml.FullLoader)
    # Import network
    network = pypsa.Network()
    network.import_from_hdf5(dir_path+'data/networks/'+options['network_name'])
    # Run initial solution
    data_detail,columns = inital_solution(network,options)
    # Run MGA using parallel
    data_detail = run_mga(network,options,data_detail)
    # Save data
    df_detail = pd.DataFrame(columns=columns,data=data_detail)
    df_detail.to_csv(dir_path+'output/hull_points_detail.csv',index=False)

    print(time.time()-timer2)
#%%
