#%%
from IPython import get_ipython
#from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
from scipy.spatial import ConvexHull,  Delaunay
from scipy.interpolate import griddata,interpn
import sys
#import pypsa_tools as pt
from pypsa_tools import *
import pypsa
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective


# %%

network = pypsa.Network()
network.import_from_hdf5('data/networks/euro_00_storage')
network.snapshots = network.snapshots[0:1000]

# %%

network.lopf(network.snapshots, 
            solver_name='gurobi',
            solver_options={'LogToConsole':0,
                                        'crossover':0,
                                        #'presolve': 2,
                                        #'NumericFocus' : 3,
                                        'method':2,
                                        'threads':4,
                                        #'NumericFocus' : numeric_focus,
                                        'BarConvTol' : 1.e-6,
                                        'FeasibilityTol' : 1.e-2},
            pyomo=False,
            keep_references=True,
            formulation='kirchhoff',
            #solver_dir = options['tmp_dir']
            ),
network.old_objective = network.objective
# %%

def mga_constraint(network,snapshots,options):
    scale = 1e-6
    # This function creates the MGA constraint 
    gen_capital_cost   = linexpr((scale*network.generators.capital_cost,get_var(network, 'Generator', 'p_nom'))).sum()
    gen_marginal_cost  = linexpr((scale*network.generators.marginal_cost,get_var(network, 'Generator', 'p'))).sum().sum()
    store_capital_cost = linexpr((scale*network.storage_units.capital_cost,get_var(network, 'StorageUnit', 'p_nom'))).sum()
    link_capital_cost  = linexpr((scale*network.links.capital_cost,get_var(network, 'Link', 'p_nom'))).sum()
    # total system cost
    cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,store_capital_cost,link_capital_cost]))
    #cost_scaled = linexpr((scale,cost))
    #cost_increase = cost_scaled[0]+'-'+str(network.old_objective*scale)
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
    #print(mga_obj)
    write_objective(network,mga_obj)

def extra_functionality(network,snapshots,direction,options):
    mga_constraint(network,snapshots,options)
    mga_objective(network,snapshots,direction,options)

#%%

options = dict(mga_variables=['wind','solar','H2','battery'],mga_slack_type='percent',mga_slack=0.1)
direction = [0,0,1,0]

network.lopf(network.snapshots,
                            pyomo=False,
                            solver_name='gurobi',
                            solver_options={'LogToConsole':1,
                                            'crossover':0,
                                            #'presolve': 0,
                                            'ObjScale' : 1e6,
                                            #'Aggregate' : 0,
                                            'NumericFocus' : 3,
                                            'method':2,
                                            'threads':4,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                            keep_references=True,
                            skip_objective=True,
                            formulation='kirchhoff',
                            extra_functionality=lambda network,snapshots: extra_functionality(network,snapshots,direction,options))




# %%
dim = 4
directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)

for direction in directions:
    stat = network.lopf(network.snapshots,
                            pyomo=False,
                            solver_name='gurobi',
                            solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 2,
                                            'ObjScale' : 1e3,
                                            'NumericFocus' : 3,
                                            'method':2,
                                            'threads':4,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                            keep_references=True,
                            skip_objective=True,
                            formulation='kirchhoff',
                            extra_functionality=lambda network,snapshots: extra_functionality(network,snapshots,direction,options))
    print(direction,stat)




# %%
