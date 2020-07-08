from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective
import numpy as np 
import time
import solutions.py
import logging
#%% Helper functions

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'::
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

#%%

def inital_solution(network,options):
    # This function performs the initial optimization of the techno-economic PyPSA model
    print('starting initial solution')
    timer = time.time()
    logging.disable()
    # Solving network
    network.lopf(network.snapshots, 
                solver_name='gurobi',
                solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 2,
                                            #'NumericFocus' : 3,
                                            'method':2,
                                            'threads':options['cpus'],
                                            #'NumericFocus' : numeric_focus,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                pyomo=False,
                keep_references=True,
                formulation='kirchhoff',
                solver_dir = options['tmp_dir']
                ),
    # initializing solutions class, to keep all network data
    sol = solutions(network)
    print('finished initial solution in {} sec'.format(time.time()-timer))
    return network,sol

#%% MGA function 

def mga_constraint(network,snapshots,options):
    scale = 1e-6
    # This function creates the MGA constraint 
    gen_capital_cost   = linexpr((scale*network.generators.capital_cost,get_var(network, 'Generator', 'p_nom'))).sum()
    gen_marginal_cost  = linexpr((scale*network.generators.marginal_cost,get_var(network, 'Generator', 'p'))).sum().sum()
    store_capital_cost = linexpr((scale*network.storage_units.capital_cost,get_var(network, 'StorageUnit', 'p_nom'))).sum()
    link_capital_cost  = linexpr((scale*network.links.capital_cost,get_var(network, 'Link', 'p_nom'))).sum()
    # total system cost
    cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,store_capital_cost,link_capital_cost]))
    # MGA slack
    if options['mga_slack_type'] == 'percent':
        slack = network.old_objective*options['mga_slack']+network.old_objective
    elif options['mga_slack_type'] == 'fixed':
        slack = options['baseline_cost']*options['mga_slack']+options['baseline_cost']

    define_constraints(network,cost_scaled,'<=',slack*scale,'GlobalConstraint','MGA_constraint')


def mga_objective(network,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for i,variable in enumerate(mga_variables):
        if variable == 'transmission':
            expr_list.append(linexpr((direction[i],get_var(network,'Link','p_nom'))).sum())
        if variable == 'co2_emission':
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p').filter(network.generators.index[network.generators.type == 'ocgt']))).sum().sum())
        elif variable == 'H2' or variable == 'battery':
            expr_list.append(linexpr((direction[i],get_var(network,'StorageUnit','p_nom').filter(network.storage_units.index[network.storage_units.carrier == variable]))).sum())
        else : 
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.type == variable]))).sum())

    mga_obj = join_exprs(np.array(expr_list))
    write_objective(network,mga_obj)

def extra_functionality(network,snapshots,direction,options):
    mga_constraint(network,snapshots,options)
    mga_objective(network,snapshots,direction,options)


def solve(network,options,direction):
    stat = network.lopf(network.snapshots,
                            pyomo=False,
                            solver_name='gurobi',
                            solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 0,
                                            'ObjScale' : 1e6,
                                            'NumericFocus' : 3,
                                            'method':2,
                                            'threads':int(np.ceil(options['cpus']/options['number_of_processes'])),
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                            keep_references=True,
                            skip_objective=True,
                            formulation='kirchhoff',
                            solver_dir = options['tmp_dir'],
                            extra_functionality=lambda network,snapshots: extra_functionality(network,snapshots,direction,options))
    return network,stat