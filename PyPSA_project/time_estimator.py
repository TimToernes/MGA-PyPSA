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

#%% Search hull
def search_hull(network,options):

    timer = time.time()

    # Original solution
    logging.disable()
    solver_options = options['solver_options']
   
    network.lopf(network.snapshots, 
                solver_name='gurobi',solver_options=solver_options),

    elapsed = timer-time.time()
    return elapsed


#%%
if __name__=='__main__':

    options = yaml.load(open('setup.yml',"r"),Loader=yaml.FullLoader)


    network = pypsa.Network()

    network.import_from_hdf5('data/networks/'+options['network_name'])
    print('Loaded network {}'.format(options['network_name']))

    elapsed = search_hull(network,options)

    print('time elapsed {}'.format(elapsed))

