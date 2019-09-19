# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
from IPython.display import display, clear_output

#%%
#import pypsa
import pandas as pd
import numpy as np
import time
#import cufflinks as cf
#import plotly.offline as pltly
#pltly.init_notebook_mode(connected=True)
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial import ConvexHull,  Delaunay
#import logging

#%% Load data

df_detail = pd.read_csv('hull_points_detail.csv')
df_points = pd.read_csv('hull_points.csv')

#%%
def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)for eq in hull.equations)


#%% Generate hulls
hull9 = ConvexHull(df_detail.values,qhull_options='A-0.99')
hull = ConvexHull(df_points[['ocgt','wind','solar']])

#%% Plot points in hull

fig = px.scatter_3d(df_points,x='ocgt',y='wind',z='solar')
fig.show()

#%% Create data for hull
dim = len(hull.min_bound)
dens = 20

test = ()

for i in range(dim):
    points = (np.linspace(hull.min_bound[i],hull.max_bound[i],dens),)
    test = test + points

mesh = np.meshgrid(*test)

new_mesh = np.reshape(np.ravel(mesh),[dim,(dens)**dim])
mesh = new_mesh.T

#%% discard points outside hull 
points_in_hull = np.array([point for point in mesh if point_in_hull(point, hull, tolerance=1e3)])

#%% Histogram

labels = ['ocgt','wind','solar']
hist_data = [points_in_hull[:,0],points_in_hull[:,1],points_in_hull[:,2]]
fig = ff.create_distplot(hist_data,labels,bin_size=1000)
fig.show()


#%%
df = pd.DataFrame(data=points_in_hull)

fig = go.Figure(go.Heatmap(z=df.corr().values,x=labels,y=labels))
fig.show()

#%%
import matplotlib.pyplot as plt

plt.matshow(df.corr())
plt.show()
cb = plt.colorbar()


#%%
import matplotlib.pyplot as plt

plt.matshow(df_detail.corr())
plt.show()
cb = plt.colorbar()

