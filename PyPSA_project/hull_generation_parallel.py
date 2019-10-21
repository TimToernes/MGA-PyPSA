#%%
# Author: Tim Pedersen
# Contact: timtoernes@gmail.com
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
# Defining exstra functionality, that updates the objective function of the network
def direction_search(network, snapshots,options,point): #  MGA_slack = 0.05, point=[0,0,0],dim=3,old_objective_value=0):
# Identify the nonzero decision variables that should enter the MGA objective function.
    old_objective_value = options['old_objective_value']
    dim = options['dim']
    MGA_slack = options['MGA_slack']
    objective = 0
    if dim == 3:
        variables = [gen_p for gen_p in network.model.generator_p_nom]
        types = ['ocgt','wind','olar']
        for i in range(3):
            gen_p_type = [gen_p  for gen_p in variables if gen_p[-4:]==types[i]]
            objective += point[i]*sum([network.model.generator_p_nom[gen_p] for gen_p in gen_p_type])
    else :
        generators = [gen_p for gen_p in network.model.generator_p_nom]
        for gen_p,i in zip(generators,range(len(generators))):
            objective += point[i]*network.model.generator_p_nom[gen_p]

    # Add the new MGA objective function to the model.
    #objective += network.model.objective.expr * 1e-9
    network.model.mga_objective = pyomo_env.Objective(expr=objective)
    # Deactivate the old objective function and activate the MGA objective function.
    network.model.objective.deactivate()
    network.model.mga_objective.activate()
    # Add the MGA slack constraint.
    network.model.mga_constraint = pyomo_env.Constraint(expr=network.model.objective.expr <= 
                                          (1 + MGA_slack) * old_objective_value)

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
    columns.append('co2_emission')
    columns.append('objective_value')

    try :
        co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
    except :
        co2_emission = 0 
    objective_value = network.model.objective()
    generator_sizes = list(network.generators.p_nom_opt.values)
    generator_sizes.append(co2_emission)
    generator_sizes.append(objective_value)

    data_detail= np.array([generator_sizes])
    #data_detail = np.append(data_detail,[[old_objective_value]],axis=1)

    print('finished initial solution in {} sec'.format(time.time()-timer))

    return data_detail,columns


def do_job(tasks_to_accomplish, tasks_that_are_done,finished_processes,options):
    proc_name = current_process().name
    network = import_network(options['network_path'])
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
                try :
                    co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
                except :
                    co2_emission = 0
                objective_value = network.model.objective()
                generator_sizes = list(network.generators.p_nom_opt.values)
                generator_sizes.append(co2_emission)
                generator_sizes.append(objective_value)
                tasks_that_are_done.put(generator_sizes)
    
    print('finishing process {}'.format(proc_name))
    finished_processes.put('done')
    return

def start_job(directions,network,options):

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
            p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done,finished_processes,options))
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
        except :
            p.terminate()
            p.join(60)
            print('killed {}'.format(p.name))
        print('joined {}'.format(p.name))



    # print the output

    result = np.empty([0,len(network.generators)+2])
    while not tasks_that_are_done.empty():
        part_result = np.array([tasks_that_are_done.get()])
        #result.append(part_result)
        result = np.concatenate([result,part_result],axis=0)
    tasks_that_are_done.close()
    tasks_that_are_done.join_thread()
    tasks_to_accomplish.close()
    tasks_to_accomplish.join_thread()
    return result


def run_mga(network,options,data_detail):

    
    MGA_convergence_tol = options['MGA_convergence_tol']
    dim = options['dim']
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
        else : # Otherwise search in directions normal to 
            directions = np.array(hull.equations)[:,0:-1]
        # Itterate over directions in batch 

        # Filter already searched directions out 
        obsolete_directions = []
        for direction,i in zip(directions,range(len(directions))):
            if any([abs(np.linalg.norm(dir_searched-direction))<1e-6  for dir_searched in directions_searched]):
                obsolete_directions.append(i)
        directions = np.delete(directions,obsolete_directions,axis=0)

        result = start_job(directions,network,options)

        computations += len(directions)
        directions_searched = np.concatenate([directions_searched,directions],axis=0)


        data_detail = np.concatenate([data_detail,result])
        save_csv(data_detail)
        data_without_extra_info = data_detail[:,:-2]
        
        if dim == 3:
            types = network.generators.type
            type_def = ['ocgt','wind','solar']
            idx = [[type_==type_def[i] for type_ in types] for i in range(3)]
            points = np.array([sum(data_without_extra_info[:,idx[0]].T),
                               sum(data_without_extra_info[:,idx[1]].T),
                               sum(data_without_extra_info[:,idx[2]].T)])
            hull = ConvexHull(points.T)#,qhull_options='C-0.1')
        else :
            try :
                hull = ConvexHull(data_without_extra_info,qhull_options='Qx C-1e-32 Qj')#,qhull_options='A-0.99')
            except Exception as e: 
                print('did not manage to create hull')
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

def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    #network.snapshots = network.snapshots[0:2]
    return network

def save_csv(data_detail):
    df_detail = pd.DataFrame(columns=columns,data=data_detail)
    outputfile = options['output_file']+'_'+options['network_name']+'_eta_'+str(options['MGA_slack'])+'.csv'
    df_detail.to_csv(dir_path+outputfile,index=False)
    print('seaved file {}'.format(outputfile))


#%%
if __name__=='__main__':

    try :
        setup_file = sys.argv[1]
    except :
        setup_file = 'setup'
    #__file__='multiprocessing_test.py'
    # Import options and start timer
    timer2 = time.time()
    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    options = yaml.load(open(dir_path+setup_file+'.yml',"r"),Loader=yaml.FullLoader)
    # Import network
    options['network_path'] = network_path = dir_path+'data/networks/'+options['network_name']
    network = import_network(options['network_path'])
    # Run initial solution
    data_detail,columns = inital_solution(network,options)
    # Run MGA using parallel
    data_detail = run_mga(network,options,data_detail)
    # Save data
    save_csv(data_detail)

    print('finished in time {}'.format( time.time()-timer2))