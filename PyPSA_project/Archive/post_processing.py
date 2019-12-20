# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
from IPython.display import display, clear_output

#%%
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

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
hull = ConvexHull(df_points[['ocgt','wind','solar']],qhull_options='A-0.999')




#%% Randomness function 
def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list

#%% Create interrior points 
m = 2000

tri = Delaunay(hull.points)
tri.volumes = []
for i in range(len(tri.simplices)):
    tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
tri.volumes = np.array(tri.volumes)
tri.volumes_norm = tri.volumes/sum(tri.volumes)

tri.n_points_in_tri =  (tri.volumes_norm*m).astype(int)

interrior_points = []
for i in range(len(tri.simplices)):
    tri_face = tri.points[tri.simplices[i,:]]
    for j in range(tri.n_points_in_tri[i]):
        dim = len(tri.points[0,:])
        rand_list = rand_split(dim+1)
        
        new_point = sum([face*rand_n for face,rand_n in zip(tri_face,rand_list)])
        interrior_points.append(new_point)
interrior_points = np.array(interrior_points)


#%% Plot interrior points 

trace1 = (go.Scatter3d(x=df_points['ocgt'].values,
                            y=df_points['wind'].values,
                            z=df_points['solar'].values,
                            mode='markers'))

trace2 = (go.Scatter3d(x=interrior_points[:,0],
                            y=interrior_points[:,1],
                            z=interrior_points[:,2],
                            mode='markers',
                            marker=dict(size=2)))

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False},
                data=[trace1,trace2])



points = hull.points
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    fig.add_trace(go.Mesh3d(x=points[s, 0], 
                                y=points[s, 1], 
                                z=points[s, 2],
                                opacity=0.3
                                ))


fig.update_layout(scene = dict(
                    xaxis_title='ocgt',
                    yaxis_title='wind',
                    zaxis_title='solar'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
fig.update_layout(scene_aspectmode='cube')



fig.show()

#%% Histogram

labels = ['ocgt','wind','solar']
hist_data = [interrior_points[:,0],interrior_points[:,1],interrior_points[:,2]]
fig = ff.create_distplot(hist_data,labels,bin_size=500)
fig.update_xaxes(title_text='MW installed capacity')
fig.show()



#%%
df = pd.DataFrame(data=interrior_points)

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

