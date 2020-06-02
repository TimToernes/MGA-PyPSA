#%%
import warnings
warnings.simplefilter("ignore")
import pypsa
import time
import logging
import numpy as np
import os 
import sys
import yaml
import pyomo.environ as pyomo_env
import pandas as pd
from scipy.spatial import ConvexHull
from multiprocessing import Lock, Process, Queue, current_process
import queue # imported for using queue.Empty exception
sys.path.append(os.getcwd())

#%%



#%%
# Defining exstra functionality, that updates the objective function of the network
def direction_search(network, snapshots,options,point): #  MGA_slack = 0.05, point=[0,0,0],dim=3,old_objective_value=0):
# Identify the nonzero decision variables that should enter the MGA objective function.
    old_objective_value = options['old_objective_value']
    dim = options['dim']
    MGA_slack = options['MGA_slack']
    variables = []
    
    for bus in network.buses.index:
        var = []
        for generator in network.model.generator_p_nom:
            if network.generators.loc[generator].type == 'wind' and network.generators.loc[generator].bus == bus :
                var.append(network.model.generator_p_nom[generator])
        variables.append(sum(var))
        #print(sum(var))

    objective = 0
    for i in range(len(variables)):
        #print(variables[i])
        objective += point[i]*variables[i]

    # Add the new MGA objective function to the model.
    #objective += network.model.objective.expr * 1e-9
    network.model.mga_objective = pyomo_env.Objective(expr=objective)
    # Deactivate the old objective function and activate the MGA objective function.
    network.model.objective.deactivate()
    network.model.mga_objective.activate()
    # Add the MGA slack constraint.
    network.model.mga_constraint = pyomo_env.Constraint(expr=network.model.objective.expr <= 
                                          (1 + MGA_slack) * old_objective_value)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def calc_gini(network):
    # Add generator production info to network.generators
    generator_prod = [sum(network.generators_t.p[generator])for generator in network.generators_t.p.columns]
    network.generators['g'] = generator_prod
    # Add bus total porduction info to network.buses
    prod_total = [sum(network.generators.g[network.generators.bus==bus]) for bus in network.buses.index]
    network.buses['total_prod']=prod_total
    # Add network total load info to network.buses
    load_total= [sum(network.loads_t.p_set[load]) for load in network.loads_t.p_set.columns]
    network.buses['total_load']=load_total


    rel_demand = network.buses.total_load/sum(network.buses.total_load)
    rel_generation = network.buses.total_prod/sum(network.buses.total_prod)
    
    # Rearange demand and generation to be of increasing magnitude
    idy = np.argsort(rel_generation/rel_demand)
    rel_demand = rel_demand[idy]
    rel_generation = rel_generation[idy]


    # Calculate cumulative sum and add [0,0 as point
    rel_demand = np.cumsum(rel_demand)
    rel_demand = np.concatenate([[0],rel_demand])
    rel_generation = np.cumsum(rel_generation)
    rel_generation = np.concatenate([[0],rel_generation])

    lorenz_integral= 0

    for i in range(len(rel_demand)-1):
        lorenz_integral += (rel_demand[i+1]-rel_demand[i])*(rel_generation[i+1]-rel_generation[i])/2 + (rel_demand[i+1]-rel_demand[i])*rel_generation[i]
        
    gini = 1- 2*lorenz_integral
        
    return network, gini

def save_network_data(network):
    try :
        co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
    except :
        co2_emission = 0 
    network, gini = calc_gini(network)
    transmission = sum(network.links.p_nom_opt.values)
    #objective_value = network.model.objective()
    generator_sizes = list(network.generators.p_nom_opt.values)
    generator_sizes = generator_sizes+list(network.generators.g.values)
    generator_sizes = generator_sizes+list(network.links.p_nom_opt.values)
    

    scenario_cost = sum(network.generators.capital_cost*network.generators.p_nom_opt)
    scenario_cost +=sum(network.generators.g * network.generators.marginal_cost)
    scenario_cost +=sum(network.links.p_nom_opt*network.links.capital_cost)

    generator_sizes.extend([co2_emission,scenario_cost,transmission,gini])

    return generator_sizes


def inital_solution(network,options):
    print('starting initial solution')
    timer = time.time()
    logging.disable()

    solver_options = options['solver_options']
    # Network is coppied, error will occur if not
    network = network.copy()
    # Solving network
    network.lopf(network.snapshots, 
                solver_name='gurobi',solver_options=solver_options),
    # Storing original objective value in options 
    old_objective_value = network.model.objective()
    options['old_objective_value'] = old_objective_value
    
    columns=list(network.generators.p_nom_opt.index)
    columns.extend([item+' g' for item in list(network.generators.index)])
    columns.extend(list(network.links.index))
    columns.extend(['co2_emission','objective_value','transmission','gini'])

    generator_sizes = save_network_data(network)
    data_detail= np.array([generator_sizes])
    print('finished initial solution in {} sec'.format(time.time()-timer))
    return data_detail,columns


def job(tasks_to_accomplish, tasks_that_are_done,finished_processes,options):
    proc_name = current_process().name
    network = import_network(options)
    while True:
        try:
            #try to get task from the queue. get_nowait() function will 
            #raise queue.Empty exception if the queue is empty. 
            #queue(False) function would do the same task also.
            direction = tasks_to_accomplish.get(False)
        except queue.Empty:
            print('no more jobs')
            break
        else:
            network = network.copy()
            logging.disable()
            MGA_slack = options['MGA_slack']
            solver_options = options['solver_options']
            #dim = options['dim']
            old_objective_value = options['old_objective_value']
            try : 
                network.lopf(network.snapshots,                                 
                            solver_name='gurobi',                                 
                            extra_functionality=lambda network,                                 
                            snapshots: direction_search(network,snapshots,options,direction),
                            solver_options=solver_options)

            except Exception as e:
                print('did not solve {} direction, process {}'.format(direction,proc_name))
                print(e)
            else :
                # Add result data to result queue 
                generator_sizes = save_network_data(network)

                tasks_that_are_done.put(generator_sizes)
    
    print('finishing process {}'.format(proc_name))
    finished_processes.put('done')
    return

def start_parallel_pool(directions,network,options):

    number_of_processes = int(os.cpu_count()/2 if len(directions)>os.cpu_count()/2 else len(directions))
    print('starting {} processes for {} jobs'.format(number_of_processes,len(directions)))
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    finished_processes = Queue()
    processes = []

    # Adding tasks to task queue
    for direction in directions:
        tasks_to_accomplish.put(direction)
    time.sleep(1) # Wait for queue to finsih filling 

    # creating processes
    for w in range(number_of_processes):
        if not tasks_to_accomplish.empty():
            p = Process(target=job, args=(tasks_to_accomplish, tasks_that_are_done,finished_processes,options))
            processes.append(p)
            p.start()
            print('{} started'.format(p.name))
        else :
            print('no more jobs - not starting any more processes')
    # wait for all processes to finish
    print('waiting for processes to finish ')
    wait_timer = time.time()
    wait_timeout = 36000
    while not len(processes) == finished_processes.qsize():
        if time.time()-wait_timer > wait_timeout :
            print('wait timed out')
            break
        time.sleep(5)

    
    for p in processes:
        print('waiting to join {}'.format(p.name))
        try :
            p.join(1)
            #p.close()
        except :
            p.terminate()
            p.join(60)
            print('killed {}'.format(p.name))
        else :
            print('joined {}'.format(p.name))



    # print the output

    result = np.empty([0,len(network.generators)*2+len(network.links)+4])
    #tasks_that_are_done.close()
    while tasks_that_are_done.qsize() > 0 :
        part_result = np.array([tasks_that_are_done.get()])
        #result.append(part_result)
        result = np.concatenate([result,part_result],axis=0)

    for p in processes:
        p.kill()
        time.sleep(1)
        p.close()
    
    tasks_that_are_done.close()
    tasks_that_are_done.join_thread()
    tasks_to_accomplish.close()
    tasks_to_accomplish.join_thread()
    finished_processes.close()
    finished_processes.join_thread()
    return result


def run_mga(network,options,data_detail):

    
    MGA_convergence_tol = options['MGA_convergence_tol']
    dim = options['dim']
    if dim > 100:
        dim = len(network.generators)
    old_volume = 0 
    epsilon_log = [1]
    directions_searched = np.empty([0,dim])
    hull = None
    pool_size =os.cpu_count()/2
    computations = 0

    while not all(np.array(epsilon_log[-2:])<MGA_convergence_tol) : # The last two itterations must satisfy convergence tollerence
        # Generate list of directions to search in for this batch
        if len(data_detail)<=1 : # if only original solution exists, max/min directions are chosen
            directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
        else : # Otherwise search in directions normal to faces
            directions = np.array(hull.equations)[:,0:-1]
        # Itterate over directions in batch 

        # Filter already searched directions out 
        obsolete_directions = []
        for direction,i in zip(directions,range(len(directions))):
            if any([abs(angle_between(dir_searched,direction))<1e-2  for dir_searched in directions_searched]):
                obsolete_directions.append(i)
        directions = np.delete(directions,obsolete_directions,axis=0)
        # Start parallelpool of workers
        result = start_parallel_pool(directions,network,options)

        computations += len(directions)
        directions_searched = np.concatenate([directions_searched,directions],axis=0)


        data_detail = np.concatenate([data_detail,result])
        save_csv(data_detail) # Save csv here to avoid loss of data
        data_without_extra_info = data_detail[:,:len(network.generators)]
        

        points = []
    
        for bus in network.buses.index:
            var = []
            for generator in  network.generators.index:
                if network.generators.loc[generator].type == 'wind' and network.generators.loc[generator].bus == bus :
                    var.append(data_detail[:,columns.index(generator)])
            points.append([sum(item) for item in list(map(list, zip(*var)))])
            #print(sum(var))

        points = np.array(points).T


        try :
            hull = ConvexHull(points)#,qhull_options='Qs C-1e-32')#,qhull_options='A-0.99')
        except Exception as e: 
            print('did not manage to create hull first try')
            try :
                hull = ConvexHull(points,qhull_options='Qx C-1e-32 QJ')
            except Exception as e:
                print('did not manage to create hull second try')
                print(e)
                return

        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        epsilon_log.append(epsilon)
        
        print('####### EPSILON ###############')
        print(epsilon)
    print('performed {} computations'.format(computations))
    return data_detail

def import_network(options):

    path = options['network_path']
    network = pypsa.Network()
    network.import_from_hdf5(path)
    dim = options['dim']

    bus_list = ['DK','SE','NO','DE','PL','CZ','NL','AT','CH']

    bus_list = bus_list[:dim]

    for bus in network.buses.index:
        if bus not in bus_list:
            network.remove('Bus',name=bus)
            network.remove('Load',bus)

    for link in network.links.index:
        if not (network.links.loc[link].bus0 in bus_list and network.links.loc[link].bus1 in bus_list ):
            network.remove('Link',link)

    for generator in network.generators.index:
        if not network.generators.loc[generator].bus in bus_list:
            network.remove('Generator',generator)

    #network.snapshots = network.snapshots[0:4]
    return network

def save_csv(data_detail):
    df_detail = pd.DataFrame(columns=columns,data=data_detail)
    outputfile = options['output_file']+'_'+options['network_name']+'_'+str(options['dim'])+'D'+'_eta_'+str(options['MGA_slack'])+'.csv'
    df_detail.to_csv(dir_path+outputfile,index=False)
    print('seaved file {}'.format(outputfile))


#%%
if __name__=='__main__':

    try :
        setup_file = sys.argv[1]
    except :
        setup_file = 'setup'
    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    options = yaml.load(open(dir_path+'setup_files/'+setup_file+'.yml',"r"),Loader=yaml.FullLoader)
    # Import network
    options['network_path'] = dir_path+'data/networks/'+options['network_name']
    #__file__='multiprocessing_test.py'
    # Import options and start timer
    dims = [2,3,4,5,6,7,8,9]
    for dim in dims:
        options['dim'] = dim
        timer2 = time.time()
        network = import_network(options)
        # Run initial solution
        data_detail,columns = inital_solution(network,options)
        # Run MGA using parallel
        data_detail = run_mga(network,options,data_detail)
        # Save data
        save_csv(data_detail)

        print('finished in time {}'.format( time.time()-timer2))
