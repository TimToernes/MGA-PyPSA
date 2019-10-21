# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
#from IPython.display import display, clear_output

#%%
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
from scipy.interpolate import griddata
#import logging


#df_detail = pd.read_csv('./output/df_local_Scandinavia_co2_eta_0.1.csv')
#%% Randomness function 
def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list

#%% Dataset class

class dataset:
    def __init__(self,path):
        self.path = path
        self.df_detail = pd.read_csv(path)
        self.co2_emission = self.df_detail['co2_emission']
        self.objective_value = self.df_detail['objective_value']

        self.create_3d_dataset()

        self.hull = ConvexHull(self.df_points[['ocgt','wind','solar']],qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')

        self.create_interior_points()
        self.calc_interrior_points_cost()

    def create_3d_dataset(self):
        type_def = ['ocgt','wind','olar']
        types = [column[-4:] for column in self.df_detail.columns]
        #sort_idx = np.argsort(types)
        idx = [[type_ == type_def[i] for type_ in types] for i in range(len(type_def))]

        points_3D = []
        for row in self.df_detail.iterrows():
            row = np.array(row[1])
            point = [sum(row[idx[0]]),sum(row[idx[1]]),sum(row[idx[2]])]
            points_3D.append(point)

        self.df_points = pd.DataFrame(columns=['ocgt','wind','solar'],data=points_3D)

        return(self)

    def create_interior_points(self):
        m = 10000

        # Generate Delunay triangulation of hull
        try :
            tri = Delaunay(self.hull.points[self.hull.vertices],qhull_options='Qs')#,qhull_options='A-0.999')
        except : 
            points = np.append(self.hull.points[self.hull.vertices],[np.mean(self.hull.points,axis=0)],axis=0)            
            tri = Delaunay(points,qhull_options='Qs')
        # Distribute number of points based on simplex size 
        tri.volumes = []
        for i in range(len(tri.simplices)):
            tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
        tri.volumes = np.array(tri.volumes)
        tri.volumes_norm = tri.volumes/sum(tri.volumes)

        tri.n_points_in_tri =  (tri.volumes_norm*m).astype(int)
        # Generate interrior points of each simplex
        interrior_points = []
        for i in range(len(tri.simplices)):
            tri_face = tri.points[tri.simplices[i,:]]
            for j in range(tri.n_points_in_tri[i]):
                dim = len(tri.points[0,:])
                rand_list = rand_split(dim+1)
                
                new_point = sum([face*rand_n for face,rand_n in zip(tri_face,rand_list)])
                interrior_points.append(new_point)
        self.interrior_points = np.array(interrior_points)
        return self

    def calc_interrior_points_cost(self):
        self.interrior_points_cost = griddata(self.df_points.values, 
                                                self.objective_value, 
                                                self.interrior_points,
                                                method='linear')

        self.interrior_points_co2 = griddata(self.df_points.values, 
                                                self.co2_emission, 
                                                self.interrior_points, 
                                                method='linear')
        return self

    def plot_hull(self):
        hull = self.hull
        df_points = self.df_points
        co2_emission = self.co2_emission
        objective_value = self.objective_value
        interrior_points = self.interrior_points

        """
        trace1 = (go.Scatter3d(x=hull.points[hull.vertices][:,0],
                            y=hull.points[hull.vertices][:,1],
                            z=hull.points[hull.vertices][:,2],
                            mode='markers',
                            marker={'color':objective_value,'colorbar':dict(thickness=20)}))
        """
        trace1 = (go.Scatter3d(x=df_points['ocgt'],
                                    y=df_points['wind'],
                                    z=df_points['solar'],
                                    mode='markers',
                                    marker={'color':objective_value}))
        # Points generated randomly
        trace2 = (go.Scatter3d(x=interrior_points[:,0],
                                    y=interrior_points[:,1],
                                    z=interrior_points[:,2],
                                    mode='markers',
                                    marker={'size':2,
                                            'color':self.interrior_points_cost,
                                            'colorbar':{'thickness':20,'title':'Scenario cost'}}))

        fig = go.Figure(layout={'width':900,
                                'height':800,
                                'showlegend':False},
                        data=[trace1,trace2])
        """
        # Plot of facets
        points = hull.points
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            fig.add_trace(go.Mesh3d(x=points[s, 0], 
                                        y=points[s, 1], 
                                        z=points[s, 2],
                                        opacity=1,
                                        color='white'
                                        ))
                                        """
        
        fig.update_layout(scene = dict(
                            xaxis_title='ocgt',
                            yaxis_title='wind',
                            zaxis_title='solar'))
                            
        fig.update_layout(scene_aspectmode='cube')
        return fig

    def plot_histogram(self):
        interrior_points = self.interrior_points
        labels = ['ocgt','wind','solar']
        hist_data = [interrior_points[:,0],interrior_points[:,1],interrior_points[:,2]]
        fig = ff.create_distplot(hist_data,labels,bin_size=2000)
        fig.update_xaxes(title_text='MW installed capacity')
        return fig

    def plot_cost(self):

        df_points = self.df_points
        #co2_emission = self.co2_emission
        objective_value = self.objective_value
        interrior_points = self.interrior_points
        interrior_points_cost = self.interrior_points_cost
        #interrior_points_co2 = self.interrior_points_co2


        fig = make_subplots(rows = 1, cols=3,shared_yaxes=True)

        fig.add_trace(go.Scatter(y=df_points['ocgt'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=1)

        fig.add_trace(go.Scatter(y=interrior_points[:,0],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=1)
        fig.add_trace(go.Scatter(y=df_points['wind'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=2)

        fig.add_trace(go.Scatter(y=interrior_points[:,1],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=2)

        fig.add_trace(go.Scatter(y=df_points['solar'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=3)

        fig.add_trace(go.Scatter(y=interrior_points[:,2],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=3)



        
        fig.update_yaxes(title_text='Installed ocgt [MW]',col=1)
        fig.update_xaxes(title_text='Scenario cost [€]',col=1)
        fig.update_yaxes(title_text=' Installed wind [MW]',col=2)
        fig.update_xaxes(title_text='Scenario cost [€]',col=2)
        fig.update_yaxes(title_text='Installed solar [MW]',col=3)
        fig.update_xaxes(title_text='Scenario cost [€]',col=3)
        #fig.show()
        return fig
        
###################################################################################
#%% ########################## Data processing ####################################
###################################################################################


#local_co2 = dataset('./output/df_local_test_Scandinavia_co2_eta_0.1.csv')
ds_co2_10 = dataset('./output/df_prime_3D_euro_30_co2_eta_0.1.csv')

ds_bau_10 = dataset('./output/df_prime_3D_euro_30_eta_0.1.csv')
ds_bau_05 = dataset('./output/df_prime_3D_euro_30_eta_0.05.csv')
ds_bau_02 = dataset('./output/df_prime_3D_euro_30_eta_0.02.csv')

#%% plot of hulls
"""
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Buisness as usual','95% CO2 reduction'),
    specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]])

fig1 = local.plot_hull()
fig2 = local_co2.plot_hull()


fig.add_traces(fig1.data[:],rows=[1]*len(fig1.data),cols=[1]*len(fig1.data))
fig.add_traces(fig2.data[:],rows=[1]*len(fig2.data),cols=[2]*len(fig2.data))

fig.update_layout(scene = dict(
                    xaxis_title='ocgt',
                    yaxis_title='wind',
                    zaxis_title='solar'))

fig.show()
"""
#%% plot of hulls

fig = ds_bau_10.plot_hull()
fig.update_layout(
    title=go.layout.Title(
        text="Business as usual"))
fig.show()

fig = ds_bau_02.plot_hull()
fig.update_layout(
    title=go.layout.Title(
        text="95% CO2 reduction"))
fig.show()

#%% plot histogram

fig = make_subplots(rows=3,cols=1,shared_xaxes=True,shared_yaxes=True,
            subplot_titles=('Buisness as usual 2% Slack',
                            'Buisness as usual 5% Slack',
                            'Buisness as usual 10% Slack'))

fig1 = ds_bau_02.plot_histogram()
fig2 = ds_bau_05.plot_histogram()
fig3 = ds_bau_10.plot_histogram()

fig.add_traces(fig1.data[:],rows=[1]*len(fig1.data),cols=[1]*len(fig1.data))
fig.add_traces(fig2.data[:],rows=[2]*len(fig2.data),cols=[1]*len(fig2.data))
fig.add_traces(fig3.data[:],rows=[3]*len(fig3.data),cols=[1]*len(fig3.data))
fig.update_xaxes(title_text="MW installed capacity", row=3, col=1)

fig.show()

#%% Plot capacity vs cost

fig = ds_bau_10.plot_cost()
fig.update_layout(
        title=go.layout.Title(
            text="Buisness as usual"))
fig.show()


fig = ds_co2_10.plot_cost()
fig.update_layout(
        title=go.layout.Title(
            text="95% CO2 reduction"))
fig.show()


#%% Test Section 

self = ds_co2_10

hull = self.hull
df_points = self.df_points
co2_emission = self.co2_emission
objective_value = self.objective_value
interrior_points = self.interrior_points
interrior_points_cost = self.interrior_points_cost
interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')

#%% CO2 vs price 
fig = make_subplots(rows = 1, cols=1)

fig.add_trace(go.Scatter(y=interrior_points_co2,
                            x=interrior_points_cost,
                            mode='markers'),row=1,col=1)

fig.add_trace(go.Scatter(y=[co2_emission[0]],
                            x=[objective_value[0]],
                            mode='markers',marker={'size':10}),row=1,col=1)

fig.update_yaxes(title_text='co2 emission [ton/year]')
fig.update_xaxes(title_text='cost [€]')

#%% Step by step

#interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')
df_points = pd.concat([ds_bau_02.df_points,ds_bau_05.df_points,ds_bau_10.df_points])
cost = list(ds_bau_02.objective_value)+list(ds_bau_05.objective_value)+list(ds_bau_10.objective_value)
trace1 = (go.Scatter3d(x=df_points['ocgt'][0:1],
                            y=df_points['wind'][0:1],
                            z=df_points['solar'][0:1],
                            mode='markers',
                            marker={'color':'blue'}))

trace2 = (go.Scatter3d(x=df_points['ocgt'][1:],
                            y=df_points['wind'][1:],
                            z=df_points['solar'][1:],
                            mode='markers',
                            marker={'color':cost,'colorbar':{'thickness':20,'title':'Scenario cost'}}))

trace3 = (go.Scatter3d(x=interrior_points[:,0],
                                    y=interrior_points[:,1],
                                    z=interrior_points[:,2],
                                    mode='markers',
                                    marker={'size':2,'color':'pink'}))#,
                                            #'color':self.interrior_points_cost,
                                            #'colorbar':{'thickness':20,'title':'Scenario cost'}}))

# Points generated randomly

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False},
                data=[trace2,trace1,trace3])

ds_co2_10.hull = ConvexHull(self.df_points[['ocgt','wind','solar']][0:],qhull_options='C-1e3')#,qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
ds_co2_10=ds_co2_10.create_interior_points()
ds_co2_10=ds_co2_10.calc_interrior_points_cost()

# Plot of hull facets
points = hull.points
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    fig.add_trace(go.Mesh3d(x=points[s, 0], 
                                y=points[s, 1], 
                                z=points[s, 2],
                                opacity=0.2,
                                color='aquamarine'
                                ))
# Plot of vectors
"""
for i in range(len(hull.equations)):
    fig.add_trace(go.Cone(  x=[np.mean(hull.points[hull.simplices[i]],axis=0)[0]], 
                            y=[np.mean(hull.points[hull.simplices[i]],axis=0)[1]], 
                            z=[np.mean(hull.points[hull.simplices[i]],axis=0)[2]], 
                            u=[hull.equations[i,0]*100000], 
                            v=[hull.equations[i,1]*100000], 
                            w=[hull.equations[i,2]*100000],
                            showscale=False))
"""
fig.update_layout(scene = dict(
                    xaxis_title='ocgt',
                    yaxis_title='wind',
                    zaxis_title='solar',
                    camera=dict(eye=dict(x=-1.25,y=1.25,z=1.25))))

fig.show()

#%%

import copy

df_points = pd.concat([ds_bau_02.df_points,ds_bau_05.df_points,ds_bau_10.df_points])

ds_test = copy.copy(ds_bau_10)

ds_test.df_points = df_points
ds_test.objective_value = list(ds_bau_02.objective_value)+list(ds_bau_05.objective_value)+list(ds_bau_10.objective_value)

ds_test.co2_emission = list(ds_bau_02.co2_emission)+list(ds_bau_05.co2_emission)+list(ds_bau_10.co2_emission)

ds_test.hull = ConvexHull(ds_test.df_points[['ocgt','wind','solar']][0:],qhull_options='C-1e3')#,qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
ds_test=ds_test.create_interior_points()
ds_test=ds_test.calc_interrior_points_cost()

fig = ds_test.plot_hull()
fig.show()

fig = ds_test.plot_cost()
fig.show()

#%% correlation heatmap

df_detail = ds_bau_10.df_detail

fig = go.Figure(go.Heatmap(z=df_detail.corr().values,x=df_detail.columns,y=df_detail.columns))
fig.show()

#%% 2D plots 

#%%
