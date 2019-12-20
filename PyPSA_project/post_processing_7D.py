# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
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
import pypsa 
#import logging
im_dir="C:/Users\Tim\OneDrive - Aarhus universitet\Speciale\Report\Images/"

#df_detail = pd.read_csv('./output/df_local_Scandinavia_co2_eta_0.1.csv')
#%% Randomness function 
def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list



def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    network.snapshots = network.snapshots[0:2]
    return network


#%% Dataset class

class dataset:
    def __init__(self,path):
        self.path = path

        self.network = import_network('data/networks/euro_30')
        
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

        self.create_7d_dataset()

        self.hull = ConvexHull(self.points,qhull_options='Qx C-1e-32')

        #self.hull = ConvexHull(self.df_points[['ocgt','wind','solar']],qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
        #self.hull = ConvexHull(self.df_points.values)#,qhull_options='C-1')#,qhull_options='A-0.999')

        self.create_interior_points()
        #self.calc_interrior_points_cost()

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

    def create_7d_dataset(self):
        network = self.network
        y_sep = np.mean(network.buses.y)
        bus_filter = network.buses.y > y_sep

        buses = [list(network.buses[bus_filter].index)]
        buses.append(list(network.buses[~bus_filter].index))
        #buses.append(list(network.buses.index))

        gen_types = ['ocgt','wind','solar']
        points = []
        for gen_type in gen_types:
            for bus in buses:
                variable = []
                for generator in network.generators.iterrows():
                    if generator[1].bus in bus and generator[1].type == gen_type:
                        variable.append(self.df_detail[generator[0]])
                points.append([sum(item) for item in list(map(list, zip(*variable)))])
                #print(sum(variable))

        
        points.append(sum([self.df_detail[link] for link in list(network.links.index)]))

        self.points = np.array(points).T
        return self

    def create_interior_points(self):
        m = 10000

        # Generate Delunay triangulation of hull
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
        return self

    def calc_interrior_points_cost(self):
        self.interrior_points_cost = griddata(self.points, 
                                                self.objective_value, 
                                                self.interrior_points,
                                                method='linear')

        self.interrior_points_co2 = griddata(self.points, 
                                                self.co2_emission, 
                                                self.interrior_points, 
                                                method='linear')
        self.interrior_points_transmission = griddata(self.points, 
                                                self.transmission, 
                                                self.interrior_points, 
                                                method='linear')
        self.interrior_points_gini = griddata(self.points, 
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
        interrior_points = self.interrior_points*1e-3
        labels = ['N ocgt','S ocgt','N wind','S wind','N solar','S solar','transmission']
        hist_data = [interrior_points[:,0],
                    interrior_points[:,1],
                    interrior_points[:,2],
                    interrior_points[:,3],
                    interrior_points[:,4],
                    interrior_points[:,5],
                    interrior_points[:,6],
                    ]
        fig = ff.create_distplot(hist_data,labels,
                                    bin_size=2000*1e-3,
                                    colors=['#8c564b','brown','#1f77b4','deepskyblue','#ff7f0e','goldenrod','#2ca02c'])
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

ds_00 = dataset('./output/prime_euro_00_7D_eta_0.1.csv')
ds_50 = dataset('./output/prime_euro_50_7D_eta_0.1.csv')
ds_80 = dataset('./output/prime_euro_80_7D_eta_0.1.csv')
ds_95 = dataset('./output/prime_euro_95_7D_eta_0.1.csv')
#ds_7D = dataset('./output/prime_euro_80_4D_eta_0.1.csv')
datasets = [ds_00,ds_50,ds_80,ds_95]

ds_all = dataset(['./output/prime_euro_00_7D_eta_0.1.csv',
                    './output/prime_euro_50_7D_eta_0.1.csv',
                    './output/prime_euro_80_7D_eta_0.1.csv',
                    './output/prime_euro_95_7D_eta_0.1.csv'])


#%% plot histogram

fig = make_subplots(rows=5,cols=1,shared_xaxes=True,shared_yaxes=True,
            row_heights=[0.25,0.25,0.25,0.25,0.0001],
            subplot_titles=('Buisness as usual',
                            '50% CO2 reduction',
                            '80% CO2 reduction',
                            '95% CO2 reduction',
                            ''),
                    vertical_spacing= 0.05)

fig1 = ds_00.plot_histogram()
fig2 = ds_50.plot_histogram()
fig3 = ds_80.plot_histogram()
fig4 = ds_95.plot_histogram()


fig.add_traces(fig1.data[:],rows=[1]*len(fig1.data),cols=[1]*len(fig1.data))
fig.add_traces(fig2.data[:],rows=[2]*len(fig2.data),cols=[1]*len(fig2.data))
fig.add_traces(fig3.data[:],rows=[3]*len(fig3.data),cols=[1]*len(fig3.data))
fig.add_traces(fig4.data[:],rows=[4]*len(fig4.data),cols=[1]*len(fig4.data))

fig.update_xaxes(title_text="Installed capacity [GW]",range=[0,1500],showticklabels=True, row=4, col=1)

fig.update_xaxes(showline=True, 
                linewidth=1, 
                linecolor='black',
                gridcolor="LightGray",
                zerolinecolor="LightGray",
                zerolinewidth=1,
                #range=[0,100],
                )
fig.update_yaxes(showline=True,
                showticklabels=True,
                title_text='Probability Density',
                linewidth=1, 
                linecolor='black',
                gridcolor="LightGray",
                zerolinecolor="LightGray",
                zerolinewidth=1,
                range=[0,0.01],
                )

# Legend
fig.add_trace(go.Scatter(x=[80,80,80,80,80,80,80],y=[1,2,3,4,5,6,7],
                        mode='markers',
                        xaxis='x2',
                        yaxis='y2',
                        marker=dict(size=10,
                                    color=['#8c564b','brown','#1f77b4','deepskyblue','#ff7f0e','goldenrod','#2ca02c'],)), 
                        row=5, col=1)
fig.add_trace(go.Scatter(x=[2e2,2e2,2e2,2e2,2e2,2e2,2e2],y=[1,2,3,4,5,6,7],
                        mode='text',
                        xaxis='x2',
                        yaxis='y2',
                        text=['N ocgt','S ocgt','N wind','S wind','N solar','S solar','Transmission'],
                        textposition="middle right" ),
                        row=5, col=1)
# Legend
fig.update_xaxes(domain=[0.85,1],
                range=[0,1500],
                showline=False,
                showticklabels=False,
                showgrid=False,
                zerolinecolor='rgba(0,0,0,0)',
                linewidth=0,
                zerolinewidth=0,  
                row=5,col=1)
fig.update_yaxes(domain=[0.65,1],
                range=[0,8],
                showline=False,
                showticklabels=False,
                showgrid=False,
                zerolinecolor='rgba(0,0,0,0)',
                linewidth=0,
                zerolinewidth=0, 
                title_text='',
                row=5,col=1)


fig.update_layout(
    autosize=False,
    width=800,
    height=1000,
    showlegend=False,
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(255,255,255,1)',
)
fig.write_image(im_dir+"7D_study_histogram.pdf")
fig.show()



#%% Corelations 

interrior_points = ds_all.interrior_points

fig = go.Figure()

fig.add_trace(go.Scatter(
                        x= interrior_points[:,4],
                        y=interrior_points[:,5],
                        mode='markers'
))

#%%

df = pd.DataFrame(data=ds_all.interrior_points,
            columns =['N ocgt','S ocgt','N wind','S wind','N solar','S solar','Transmission'] )

df.corr()
fig = go.Figure(go.Heatmap(z=df.corr(),
                            y=df.columns,
                            x=df.columns,
                            autocolorscale=False,
                            zmin=-0.5,
                            zmax=0.5,
                            colorscale=px.colors.sequential.Viridis))
fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    showlegend=False,
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(255,255,255,1)',
)
fig.write_image(im_dir+"7D_study_corr.pdf")
fig.show()

#%% Network plots 

import pypsa 

def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    network.snapshots = network.snapshots[0:2]
    return network

network = import_network('data/networks/euro_30')

#%%

fig = go.Figure()


# Links
import matplotlib.cm as cm
for link in network.links.iterrows():

    bus0 = network.buses.loc[link[1]['bus0']]
    bus1 = network.buses.loc[link[1]['bus1']]
    cap = link[1]['p_nom_opt']
    cap_max = max(network.links.p_nom_opt)

    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        geo = 'geo1',
        lon = [bus0.x,bus1.x],
        lat = [bus0.y,bus1.y],
        mode = 'lines',
        line = dict(width =0.5,color = 'green'),
        ))

seperating_line = np.median(network.buses.y)

lon_line=np.linspace(np.min(network.buses.x)-2,np.max(network.buses.x)+2,1000)
lat_line = np.ones(1000)*seperating_line

# Parting line 
fig.add_trace(go.Scattergeo(
    locationmode = 'country names',
    lon = lon_line,
    lat = lat_line,
    mode = 'lines',
    marker = dict(
        size = 5,
        color = 'red',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )),
     textfont={
        "color": "MidnightBlue",
        "family": "Balto, sans-serif",
        "size": 18
    },
    ))
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
        )),
     textfont={
        "color": "MidnightBlue",
        "family": "Balto, sans-serif",
        "size": 18
    },
    ))



"""

# Bar plots 
#network.generators.p_nom_opt=ds_co2_95.df_detail.iloc[0,0:111]


for bus in network.buses.iterrows():

    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='wind')]
    wind = sum(network.generators[filter].p_nom_opt)
    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='solar')]
    solar = sum(network.generators[filter].p_nom_opt)
    filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='ocgt')]
    ocgt = sum(network.generators[filter].p_nom_opt)

    fig.add_trace(go.Scattergeo(
    locationmode = 'country names',
    lon = [bus[1]['x']],
    lat = [bus[1]['y']],
    geo = 'geo1',
    mode = 'markers',
    marker = dict(
        size = np.sum(network.loads_t.p_set)[bus[0]]*1e-7,
        color = ['blue','yellow','black'],
        symbol = 'line-ns',
        line = dict(
            width = 10,
            color = ['black'],
        )
    )),row=1,col=1)

"""


fig.update_geos(
        scope = 'europe',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
        showocean=False,
        #domain=dict(x=[0,1],y=[0,1]),
        lataxis = dict(
            range = [35, 64],
            showgrid = False
        ),
        lonaxis = dict(
            range = [-11, 26],
            showgrid = False
        ))

fig.update_layout(
    geo1=dict(
        domain=dict(x=[0,1],y=[0.1,1])),#Top Left
)

fig.update_layout(
    autosize=False,
    showlegend=False,
    xaxis=dict(showticklabels=False,showgrid=False,range=[0,0.7],domain=[0,1]),
    yaxis=dict(showticklabels=False,showgrid=False,range=[0,11.5],domain=[0,0.08]),
        paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    width=700,
    height=700,
    margin=dict(l=5, r=5, t=5, b=5,pad=0),
    )

fig.write_image(im_dir+"7D_study_topology.pdf")
fig.show()


# %%
