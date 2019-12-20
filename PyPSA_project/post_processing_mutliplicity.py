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
        
        if type(path)==list :
            data_frames = []
            for p in path:
                data_frames.append(pd.read_csv(p))
            self.df_detail = pd.concat(data_frames,ignore_index=True)
        else:
            self.df_detail = pd.read_csv(path)

        self.co2_emission = self.df_detail['co2_emission']
        self.objective_value = self.df_detail['objective_value']
        self.transmission = self.df_detail['transmission']
        self.gini = self.df_detail['gini']

        self.create_3d_dataset()
        #self.create_4d_dataset()

        #self.hull = ConvexHull(self.df_points[['ocgt','wind','solar']],qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
        self.hull = ConvexHull(self.df_points.values)#,qhull_options='C-1')#,qhull_options='A-0.999')

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
        #print(self.df_points.head())

        return(self)

    def create_4d_dataset(self):
        transmission = pd.DataFrame({'transmission':self.df_detail['transmission']})

        self.df_points = pd.concat([self.df_points,transmission],axis=1)
        #print(self.df_points.head())
        return self

    def create_interior_points(self):
        m = 20000

        # Generate Delunay triangulation of hull
        try :
            tri = Delaunay(self.hull.points[self.hull.vertices],qhull_options='Qx, Q12')#,qhull_options='Qs')#,qhull_options='A-0.999')
        except : 
            points = np.append(self.hull.points[self.hull.vertices],[np.mean(self.hull.points,axis=0)],axis=0)            
            tri = Delaunay(points,qhull_options='Qs')
        # Distribute number of points based on simplex size 
        tri.volumes = []
        for i in range(len(tri.simplices)):
            try :
                tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
            except : 
                tri.volumes.append(0)
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
        self.interrior_points_transmission = griddata(self.df_points.values, 
                                                self.transmission, 
                                                self.interrior_points, 
                                                method='linear')
        self.interrior_points_gini = griddata(self.df_points.values, 
                                                self.gini, 
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
        trace0 = (go.Scatter3d(x=[df_points['ocgt'][0]],
                                    y=[df_points['wind'][0]],
                                    z=[df_points['solar'][0]],
                                    mode='markers',
                                    marker={'color':'blue','size':3}))

        trace1 = (go.Scatter3d(x=df_points['ocgt'][1:],
                                    y=df_points['wind'][1:],
                                    z=df_points['solar'][1:],
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
                        data=[trace1,trace2,trace0])
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
        labels = ['ocgt','wind','solar']#,'transmission']
        hist_data = [interrior_points[:,0],interrior_points[:,1],interrior_points[:,2]]#,interrior_points[:,3]]
        fig = ff.create_distplot(hist_data,labels,
                                    bin_size=2000,
                                    colors=['#8c564b','#1f77b4','#ff7f0e','#2ca02c'])
        fig.update_xaxes(title_text='MW installed capacity')
        return fig
    """ Colors
    '#1f77b4',  // muted blue
    '#ff7f0e',  // safety orange
    '#2ca02c',  // cooked asparagus green
    '#d62728',  // brick red
    '#9467bd',  // muted purple
    '#8c564b',  // chestnut brown
    '#e377c2',  // raspberry yogurt pink
    '#7f7f7f',  // middle gray
    '#bcbd22',  // curry yellow-green
    '#17becf'   // blue-teal
"""
    def plot_cost(self):

        df_points = self.df_points
        #co2_emission = self.co2_emission
        objective_value = self.objective_value
        interrior_points = self.interrior_points
        interrior_points_cost = self.interrior_points_cost
        #interrior_points_co2 = self.interrior_points_co2


        fig = make_subplots(rows = 1, cols=4,shared_yaxes=True)

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
        
        fig.add_trace(go.Scatter(y=self.transmission,
                                    x=objective_value,
                                    mode='markers'),row=1,col=4)

        fig.add_trace(go.Scatter(y=self.interrior_points_transmission,
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=4)


        
        fig.update_yaxes(title_text='Installed ocgt [MW]',col=1)
        fig.update_xaxes(title_text='Scenario cost [€]',col=1)
        fig.update_yaxes(title_text=' Installed wind [MW]',col=2)
        fig.update_xaxes(title_text='Scenario cost [€]',col=2)
        fig.update_yaxes(title_text='Installed solar [MW]',col=3)
        fig.update_xaxes(title_text='Scenario cost [€]',col=3)
        fig.update_yaxes(title_text='Installed transmission [MW]',col=4)
        fig.update_xaxes(title_text='Scenario cost [€]',col=4)
        #fig.show()
        return fig
        
###################################################################################
#%% ########################## Data processing ####################################
###################################################################################


#ds_local_co2 = dataset('./output/local_Scandinavia_co2_4D_eta_0.1.csv')

ds_2D = dataset('./output/prime_multi_euro_80_2D_eta_0.02.csv')
ds_3D = dataset('./output/prime_multi_euro_80_3D_eta_0.02.csv')
ds_4D = dataset('./output/prime_multi_euro_80_4D_eta_0.02.csv')


#%%
bus_list = ['DK','SE','NO','DE','PL','CZ','NL','AT','CH']
fig = make_subplots(rows=3,cols=1,shared_xaxes=True,shared_yaxes=True,
            subplot_titles=('Buisness as usual',
                            '50% CO2 reduction',
                            '80% CO2 reduction',
                            '95% CO2 reduction'))

for dim,ds in zip([2,3,4],[ds_2D,ds_3D,ds_4D]):
    buses = bus_list[:dim]
    df_wind = []
    for country in buses:
        data = []
        for label in ds.df_detail.columns:
            if label[0:2]==country and label[-4:]=='wind':
                data.append(ds.df_detail[label].values)
        df_wind.append(sum(data))
    df_wind = np.array(df_wind).T

    hull = ConvexHull(df_wind,qhull_options='Qx')
    ds.hull = hull
    ds.create_interior_points()

    labels = buses

    fig_ff = ff.create_distplot(ds.interrior_points.T,labels,
                                bin_size=500,
                                colors=['#8c564b','#1f77b4','#ff7f0e','#2ca02c'])

    fig.add_traces(fig_ff.data[:],rows=[dim-1]*len(fig_ff.data),cols=[1]*len(fig_ff.data))
    
    fig.update_xaxes(title_text='MW installed capacity')

fig.show()
#%% Plot of capacity vs cost

fig = make_subplots(rows=4,cols=4,shared_xaxes=False,shared_yaxes=True,
            subplot_titles=('ocgt',
                            'wind',
                            'solar',
                            'transmission'))


datasets = [ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]

for k in range(4):
    for i in range(4):

        ds = datasets[k]
        x = ds.interrior_points_cost
        y = ds.interrior_points[:,i]


        # Calculate quantiles
        q_0 = []
        q_08 = []
        q_341 = []
        q_5 = []
        q_659 = []
        q_92 = []
        q_1 = []
        q_x = []

        for j in [0,1,2,5,10]:
            filter = x <= min(x)*(1+j/100)
            q_x.append(j/100)

            q_0.append(np.quantile(y[filter],0.0))
            q_08.append(np.quantile(y[filter],0.08))
            q_341.append(np.quantile(y[filter],0.341))
            q_5.append(np.quantile(y[filter],0.5))
            q_659.append(np.quantile(y[filter],0.659))
            q_92.append(np.quantile(y[filter],0.92))
            q_1.append(np.quantile(y[filter],1))
        # Plot of points
        #fig.add_trace(go.Scatter(x=x/min(x)-1,y=y,mode='markers'))
        # Plot of 100% and 0 % quantiles
        fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                y=np.concatenate([np.flip(q_1),q_0]),
                                fill='toself',
                                fillcolor='#1f77b4',
                                mode='lines+markers',
                                marker=dict(color='#1f77b4'),
                                line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
        # Plot of 2 sigma quantiles
        fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                y=np.concatenate([np.flip(q_92),q_08]),fill='toself',
                                mode='lines+markers',
                                marker=dict(color='#ff7f0e'),
                                fillcolor='#ff7f0e',
                                line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
        # Plot of 1 sigma quantiles
        fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                y=np.concatenate([np.flip(q_659),q_341]),fill='toself',
                                mode='lines+markers',
                                marker=dict(color='#2ca02c'),
                                fillcolor='#2ca02c',
                                line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
        # Plot of 50% quantile
        fig.add_trace(go.Scatter(x=q_x,y=q_5,marker=dict(color='#7f7f7f')),row=k+1,col=i+1)


        fig.update_xaxes(title_text="MGA slack", row=4, col=i+1)

        fig.update_yaxes(title_text="MW installed capacity", row=k+1, col=1,range=[0,2.3e6])

    fig.update_layout(title_text='Capacity vs slack',showlegend=False)


fig.show()



#%% Test section
self = ds_co2_50

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

#%% Corelation plots

ocgt = ds_co2_80.interrior_points[:,0]
wind = ds_co2_80.interrior_points[:,1]
solar = ds_co2_80.interrior_points[:,2]
transmission = ds_co2_80.interrior_points[:,3]
gini = ds_co2_80.interrior_points_gini

variables= [ocgt,wind,solar,transmission]
labels= ['ocgt','wind','solar','transmission']

fig = make_subplots(rows = 4, cols=4,shared_xaxes=False,shared_yaxes=False)

for i in range(4):
    for j in range(4):

        if i != j:
            fig.add_trace(go.Scatter(y=variables[i],
                                        x=variables[j],
                                        mode='markers',yaxis='y2',xaxis='x2'),row=i+1,col=j+1)
            fig.update_yaxes(title_text=labels[i],row=i+1,col=j+1)
            fig.update_xaxes(title_text=labels[j],row=i+1,col=j+1)
        else :
            fig.add_trace(go.Histogram(x=variables[i]),row=i+1,col=j+1)

fig.show()


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

ds_co2_80.hull = ConvexHull(self.df_points[['ocgt','wind','solar']][0:],qhull_options='C-1e3')#,qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
ds_co2_80=ds_co2_80.create_interior_points()
ds_co2_80=ds_co2_80.calc_interrior_points_cost()

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

#%% 


self = ds_132D

m = 10000

# Generate Delunay triangulation of hull
points = ds_132D.df_detail.values[:,:-4]
try :
    tri = Delaunay(self.hull.points[self.hull.vertices])#,qhull_options='Qs')#,qhull_options='A-0.999')
except : 
    points = np.append(self.hull.points[self.hull.vertices],[np.mean(self.hull.points,axis=0)],axis=0)            
    tri = Delaunay(points,qhull_options='Qs')
# Distribute number of points based on simplex size 
tri.volumes = []
for i in range(len(tri.simplices)):
    try :
        tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
    except : 
        tri.volumes.append(0)
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







#%% Network plots 
import pypsa 

def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    network.snapshots = network.snapshots[0:2]
    return network

network = import_network('data/networks/euro_30')


#%% Plot of topology

fig = go.Figure()

# Nodes
fig.add_trace(go.Scattergeo(
    locationmode = 'country names',
    lon = network.buses.x,
    lat = network.buses.y,
    hoverinfo = 'text',
    text = network.buses.index,
    mode = 'markers',
    marker = dict(
        size = 5,
        color = 'black',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))

fig.add_trace(go.Scattergeo(lon=[min(network.buses.x),max(network.buses.x)],
                            lat=[np.mean(network.buses.y),np.mean(network.buses.y)],
                            mode='lines'
                            ))

# Links
import matplotlib.cm as cm
for link in network.links.iterrows():

    bus0 = network.buses.loc[link[1]['bus0']]
    bus1 = network.buses.loc[link[1]['bus1']]
    cap = link[1]['p_nom_opt']

    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = [bus0.x,bus1.x],
        lat = [bus0.y,bus1.y],
        mode = 'lines',
        line = dict(width = cap/2e-13,color = 'green'),
        ))

# Bar plots 
network.generators.p_nom_opt=ds_co2_95.df_detail.iloc[0,0:111]


for bus in network.buses.iterrows():

    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='wind')]
    wind = sum(network.generators[filter].p_nom_opt)
    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='solar')]
    solar = sum(network.generators[filter].p_nom_opt)
    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='ocgt')]
    ocgt = sum(network.generators[filter].p_nom_opt)

    fig.add_trace(go.Scattergeo(
    locationmode = 'country names',
    lon = [bus[1]['x'],bus[1]['x']+0.5,bus[1]['x']-0.5 ],
    lat = [bus[1]['y'], bus[1]['y'],bus[1]['y']],
    hoverinfo = 'text',
    text = bus[0],
    mode = 'markers',
    marker = dict(
        size = np.array([wind,solar,ocgt])/2000,
        color = ['blue','yellow','black'],
        symbol = 'line-ns',
        line = dict(
            width = 10,
            color = ['blue','yellow','black'],
        )
    )))

# Legend 
fig.add_trace(go.Scattergeo(
locationmode = 'country names',
lon = [-11,-11,-11],
lat = [62,63,64],
hoverinfo = 'text',
text = 'legend',
mode = 'markers',
marker = dict(
    size = 10,
    color = ['blue','yellow','black'],
    symbol = 'line-ns',
    line = dict(
        width = 10,
        color = ['blue','yellow','black'],
    )
)))



fig.update_layout(
    title_text = 'Euro-30 model',
    showlegend = False,
    geo = go.layout.Geo(
        scope = 'europe',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
        lataxis = dict(
            range = [35, 64],
            showgrid = False
        ),
        lonaxis = dict(
            range = [-11, 26],
            showgrid = False
        )
    ),
)


fig.show()


#%% Plot of wind potential
from iso3166 import countries

country_area = pd.read_csv('data/country_sizes.csv')

#codes = ['AUT', 'BEL', 'BGR', 'BIH', 'HRV', 'CHE', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB','SVK', 'SVN', 'ESP', 'SWE', 'GBR']
codes = [countries.get(alpha2).alpha3 for alpha2 in network.buses.index]
names = [countries.get(alpha2).name for alpha2 in network.buses.index]
values = network.generators.p_nom_max[network.generators.carrier == 'onwind']
area = [(country_area[country_area['Country Code']==country]['2018']).values[0] for country in codes]

fig = go.Figure(data=go.Choropleth(
                    locations = codes,
                    z = values/area,
                    text = names,
                    colorscale = 'Blues',
                    autocolorscale=True,
                    reversescale=False,
                    marker_line_color='darkgray',
                    marker_line_width=0.5,
                    colorbar_tickprefix = '',
                    colorbar_title = 'Potential [MWh/km^2]')) 

fig.update_layout(
    title_text = 'Euro-30 model',
    showlegend = False,
    geo = go.layout.Geo(
        scope = 'europe',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
        lataxis = dict(
            range = [35, 64],
            showgrid = False
        ),
        lonaxis = dict(
            range = [-11, 26],
            showgrid = False
        )
    ),
)


fig.show()

#%% Plot of optimal solutions sumary

data = []

datasets = [ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]

for ds in datasets:
    data.append([ds.df_points['wind'][0],
                ds.df_points['solar'][0],
                ds.df_points['ocgt'][0],
                ds.df_detail['transmission'][0],
                ds.df_detail['co2_emission'][0],
                ds.df_detail['objective_value'][0],
                ds.df_detail['gini'][0]   ])

data = np.array(data)

fig = go.Figure()

names = ['wind','solar','ocgt','transmission']
colors = ['#1f77b4','#ff7f0e','#8c564b','#2ca02c']

for i in range(4):
    fig.add_trace(go.Scatter(
                    name=names[i],
                    x=[0,50,80,95],
                    y=data[:,i],
                    line=dict(color=colors[i])
                    #yaxis='y'+str(i+1)
    ))

fig.update_layout(xaxis_title='% CO2 reduction',
                yaxis_title='Installed capacity')


names = ['co2','cost','gini']

fig.show()
fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Scatter(
                    name=names[i],
                    x=[0,50,80,95],
                    y=data[:,i+4],
                    yaxis='y'+str(i+1)
    ))

fig.update_layout(xaxis_title='% CO2 reduction',
                yaxis_title='Installed capacity',
                    yaxis2=dict(
        title="yaxis2 title",
        titlefont=dict(
            color="#ff7f0e"
        ),
        tickfont=dict(
            color="#ff7f0e"
        ),
        anchor="free",
        overlaying="y",
        side="left",
        position=0.15
    ),
    yaxis3=dict(
        title="yaxis4 title",
        titlefont=dict(
            color="#d62728"
        ),
        tickfont=dict(
            color="#d62728"
        ),
        anchor="x",
        overlaying="y",
        side="right"
    ),
)

fig.show()


""" Colors
    '#1f77b4',  // muted blue
    '#ff7f0e',  // safety orange
    '#2ca02c',  // cooked asparagus green
    '#d62728',  // brick red
    '#9467bd',  // muted purple
    '#8c564b',  // chestnut brown
    '#e377c2',  // raspberry yogurt pink
    '#7f7f7f',  // middle gray
    '#bcbd22',  // curry yellow-green
    '#17becf'   // blue-teal
"""

# %% Gini plot

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Equality'))

datasets = [ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]
names = ['0% reduction','50% reduction','80% reduction','95% reduction']

for ds,k in zip(datasets,range(4)):
    network.generators['g'] = ds.df_detail.iloc[0,111:-4].values
    network.generators.p_nom_opt=ds_co2_95.df_detail.iloc[0,0:111]

    prod_total = [sum(network.generators.g[network.generators.bus==bus]) for bus in network.buses.index]
    network.buses['total_prod']=prod_total

    load_total= [sum(network.loads_t.p_set[load]) for load in network.loads_t.p_set.columns]
    network.buses['total_load']=load_total


    x = network.buses.total_load/sum(network.buses.total_load)

    y = network.buses.total_prod/sum(network.buses.total_prod)

    idy = np.argsort(y/x)
    y = y[idy]
    x = x[idy]


    x = np.cumsum(x)
    x = np.concatenate([[0],x])
    y = np.cumsum(y)
    y = np.concatenate([[0],y])

    lorenz_integral= 0

    for i in range(len(x)-1):
        lorenz_integral += (x[i+1]-x[i])*(y[i+1]-y[i])/2 + (x[i+1]-x[i])*y[i]
        
    gini = 1- 2*lorenz_integral

    print(gini)

    

    fig.add_trace(go.Scatter(x=x,y=y,name=names[k]))
    
fig.update_layout(xaxis_title='Cumulative share of demand',
                yaxis_title='Cumulative share of generation',
                )
fig.show()
# %%
