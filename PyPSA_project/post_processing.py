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
import sys
#import pypsa_tools as pt
from pypsa_tools import *
#import logging
im_dir="C:/Users\Tim\Dropbox\MGA paper\Paper/figures/"

#df_detail = pd.read_csv('./output/df_local_Scandinavia_co2_eta_0.1.csv')


###################################################################################
#%% ########################## Data processing ####################################
###################################################################################


#ds_local_co2 = dataset('./output/local_Scandinavia_co2_4D_eta_0.1.csv')

ds_co2_00 = dataset(['./output/prime_euro_00_4D_eta_0.1.csv',
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_00_4D_eta_0.02.csv',
                    './output/prime_euro_00_4D_eta_0.01.csv'])

ds_co2_50 = dataset(['./output/prime_euro_50_4D_eta_0.1.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.02.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv'])

ds_co2_80 = dataset(['./output/prime_euro_80_4D_eta_0.1.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.02.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv'])

ds_co2_95 = dataset(['./output/prime_euro_95_4D_eta_0.1.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.02.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv'])


ds_all = dataset(['./output/prime_euro_00_4D_eta_0.1.csv',
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_00_4D_eta_0.02.csv',
                    './output/prime_euro_00_4D_eta_0.01.csv',
                    './output/prime_euro_50_4D_eta_0.1.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.02.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv',
                    './output/prime_euro_80_4D_eta_0.1.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.02.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv',
                    './output/prime_euro_95_4D_eta_0.1.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.02.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv']
                    )

ds_all_05 = dataset([
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv']
                    )

ds_all_01 = dataset([
                    './output/prime_euro_00_4D_eta_0.01.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv']
                    )

#%% plot histogram

fig = plot_histogram(ds_co2_00,
                ds_co2_50,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"4D_study_histogram.pdf")
fig.show()

#%% Plot of capacity vs cost

fig = plot_capacity_vs_cost(ds_co2_00,
                ds_co2_50,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"Capacaty_vs_cost.pdf")
fig.show()

#%% Plot of optimal solutions sumary - Capacity

fig = plot_optimal_solutions_power(ds_co2_00,
            ds_co2_50,
            ds_co2_80,
            ds_co2_95)
#fig.write_image(im_dir+"optimal_solutions_summary.pdf")

fig.show()
#%% Plot of optimal solutions sumary - Production

fig = plot_optimal_solutions_energy(ds_co2_00,
                ds_co2_50,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"optimal_solutions_summary_production.pdf")
fig.show()

#%% Network plots 

fig = plot_network('data/networks/euro_30',
                    [ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95])
#fig.write_image(im_dir+"Optimal_solutions.pdf")
fig.show()

#Plot of topology
plot_topology('data/networks/euro_30',ds_co2_95)
fig.show()

#%% Corelation plots

datasets=[ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]
#datasets=[ds_co2_95]
fig = plot_correlation(datasets)
#fig.write_image(im_dir+"Corelation_4D_95.pdf")
fig.show()


#%% New GINI plot

fig = go.Figure()
ds = ds_all_01
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
gini_hull = ConvexHull(np.array([co2_reduction,ds.interrior_points_gini]).T)

x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='1% slack',
                    fill='toself'
                    ))

ds = ds_all_05
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
gini_hull = ConvexHull(np.array([co2_reduction,ds.interrior_points_gini]).T)

x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='5% slack',
                    fill='tonexty'
                    ))

ds = ds_all
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
gini_hull = ConvexHull(np.array([co2_reduction,ds.interrior_points_gini]).T)

x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='10% slack',
                    fill='tonexty'
                    ))

fig.update_yaxes(title_text='CO2 emission reduction [%]',showgrid=False)
fig.update_xaxes(title_text='Gini coefficient',showgrid=False)

fig.update_layout(width=800,
                    height=500,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showline=True,linecolor='black'),
                    yaxis=dict(showline=True,linecolor='black'),
                    )

#fig.write_image(im_dir+"Corelation_gini_co2.pdf")
fig.show()

#%% GINI on cost

network = pypsaTools.import_network('data/networks/euro_30')


def calc_gini_capex(ds,network):
    gini_list = []

    buses = network.buses.index
    techs = ['onwind','offwind','solar','ocgt']
    expenses = dict.fromkeys(buses,0)

    capex = dict(onwind=136428.03135482676, 
                solar=76486.31033239936, 
                ocgt=47234.561404444474,
                offwind=295041.156154988)

    for i in range(ds.df_detail.shape[0]):
        row = ds.df_detail.iloc[i,0:111]

        for bus in buses:
            for tech in techs:
                try:
                    expenses[bus] = expenses[bus] + row[bus+' '+tech]*capex[tech]
                except : 
                    pass
    

        network.buses.total_expenses = list(expenses.values())

        
        # Add network total load info to network.buses
        load_total= [sum(network.loads_t.p_set[load]) for load in network.loads_t.p_set.columns]
        network.buses['total_load']=load_total


        rel_demand = network.buses.total_load/sum(network.buses.total_load)
        rel_generation = network.buses.total_expenses/sum(network.buses.total_expenses)
        
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

        gini_list.append(gini)

    try :
        ds.df_detail = ds.df_detail.join(pd.DataFrame(dict(gini_capex=gini_list)))
    except :
        ds.df_detail['gini_capex'] = gini_list

    ds.interrior_points_gini_capex = griddata(ds.df_points.values, 
                                    ds.df_detail['gini_capex'], 
                                    ds.interrior_points, 
                                    method='linear') 


    return ds


ds_all = calc_gini_capex(ds_all,network)
ds_all_05 = calc_gini_capex(ds_all_05,network)
ds_all_01 = calc_gini_capex(ds_all_01,network)



  

fig = go.Figure()
ds = ds_all_01
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100

co2_data = co2_reduction
gini_data = ds.interrior_points_gini_capex

gini_hull = ConvexHull(np.array([co2_data,gini_data]).T)

x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='1% slack',
                    fill='tonexty'
                    ))

ds = ds_all_05
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100

co2_data = np.concatenate([co2_data,co2_reduction])
gini_data = np.concatenate([gini_data,ds.interrior_points_gini_capex])

gini_hull = ConvexHull(np.array([co2_data,gini_data]).T)


x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='5% slack',
                    fill='tonexty'
                    ))

ds = ds_all
co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100

co2_data = np.concatenate([co2_data,co2_reduction])
gini_data = np.concatenate([gini_data,ds.interrior_points_gini_capex])

gini_hull = ConvexHull(np.array([co2_data,gini_data]).T)
x_data = gini_hull.points[gini_hull.vertices][:,1]
y_data = gini_hull.points[gini_hull.vertices][:,0]


fig.add_trace(go.Scatter(x=np.append(x_data,x_data[0]),
                    y=np.append(y_data,y_data[0]),
                    mode='lines',
                    name='10% slack',
                    fill='tonexty'
                    ))

fig.update_yaxes(title_text='CO2 emission reduction [%]',showgrid=False)
fig.update_xaxes(title_text='Gini coefficient',showgrid=False)

fig.update_layout(width=800,
                    height=500,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showline=True,linecolor='black'),
                    yaxis=dict(showline=True,linecolor='black'),
                    )
fig.write_image(im_dir+"Gini_capex.pdf")

#%% CO2 vs wind/solar mix

fig = go.Figure()
co2_reduction = (1-ds_all.interrior_points_co2/max(ds_all.interrior_points_co2)) *100
#mix = ds_all.interrior_points[:,1]/(ds_all.interrior_points[:,2]+ds_all.interrior_points[:,1])
mix = ds_all.interrior_points_gini_capex


fig.add_trace(go.Scatter(x=mix,
                        y = co2_reduction,
                        mode='markers' ,
                        name='10% slack'))


co2_reduction = (1-ds_all_01.interrior_points_co2/max(ds_all.interrior_points_co2)) *100
#mix = ds_all_01.interrior_points[:,1]/(ds_all_01.interrior_points[:,2]+ds_all_01.interrior_points[:,1])
mix = ds_all_01.interrior_points_gini_capex

fig.add_trace(go.Scatter(x=mix,
                        y = co2_reduction,
                        mode='markers',
                        name='1%slack' ))


fig.update_yaxes(title_text='CO2 emission reduction [%]',showgrid=False)
fig.update_xaxes(title_text='$\\alpha$',showgrid=False)

fig.update_layout(width=800,
                    height=500,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showline=True,linecolor='black'),
                    yaxis=dict(showline=True,linecolor='black'),
                    )
#fig.write_image(im_dir+"Corelation_mix_co2.pdf")
fig.show()

#%% Table data 

ds=[ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]
techs = ['wind','solar','ocgt']


for tech in techs:
    d1 = ds[0].df_points[tech][0]*1e-3
    d2 = ds[1].df_points[tech][0]*1e-3
    d3 = ds[2].df_points[tech][0]*1e-3
    d4 = ds[3].df_points[tech][0]*1e-3

    print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format(tech,d1,d2,d3,d4))

techs2 = ['transmission','gini','co2_emission','objective_value']
factor = [1e-3,1,1e-6,1e-9]

for tech,f in zip(techs2,factor):
    d1 = ds[0].df_detail[tech][0]*f
    d2 = ds[1].df_detail[tech][0]*f
    d3 = ds[2].df_detail[tech][0]*f
    d4 = ds[3].df_detail[tech][0]*f   

    print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format(tech,d1,d2,d3,d4))



#%% Title page figure 

fig = plot_titlefig()
fig.show()





#%% Test section ########################################################
##########################################################################
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


fig.add_trace(go.Scatter(y=interrior_points_cost,
                            x=self.interrior_points_gini,
                            mode='markers'),row=1,col=1)

fig.add_trace(go.Scatter(y=self.df_detail['objective_value'],
                            x=self.df_detail['gini'],
                            mode='markers',marker={'size':10}),row=1,col=1)

fig.update_yaxes(title_text='co2 emission [ton/year]')
fig.update_xaxes(title_text='cost [€]')


#%%
fig = go.Figure()

fig.add_trace(go.Histogram(x=self.interrior_points_gini,
                                        marker_color=colors[i]))


#%%
import networkx as nx
import matplotlib.pyplot as plt

df = pd.DataFrame({'wind':wind,'ocgt':ocgt,'solar':solar,'transmission':transmission})
corr = df.corr()
# Transform it in a links data frame (3 columns only):
links = corr.stack().reset_index()
links.columns = ['var1', 'var2','value']
links
 
# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[  (links['var1'] != links['var2']) ]
links_filtered
 
# Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network:
#nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)

pos = nx.spring_layout(G,weight='weight')
nx.draw(G,pos=pos, width=2, with_labels=True)
#%% Step by step

#interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')
df_points = ds_co2_95.df_points
cost = ds_co2_95.objective_value
interrior_points = ds_co2_95.interrior_points
trace1 = (go.Scatter(x=df_points['ocgt'][0:1],
                            y=df_points['wind'][0:1],
                            #z=df_points['solar'][0:1],
                            mode='markers',
                            marker={'color':'blue'}))

trace2 = (go.Scatter(x=df_points['ocgt'][1:5],
                            y=df_points['wind'][1:5],
                            #z=df_points['solar'][1:],
                            mode='markers',
                            marker={'color':cost,'colorbar':{'thickness':20,'title':'Scenario cost'}}))
"""
trace3 = (go.Scatter(x=interrior_points[:,0],
                                    y=interrior_points[:,1],
                                    #z=interrior_points[:,2],
                                    mode='markers',
                                    marker={'size':2,'color':'pink'}))#,
                                            #'color':self.interrior_points_cost,
                                            #'colorbar':{'thickness':20,'title':'Scenario cost'}}))
"""
# Points generated randomly

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False},
                data=[trace2,trace1])

#ds_co2_80.hull = ConvexHull(df_points[['ocgt','wind']][0:],qhull_options='C-1e3')#,qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
#ds_co2_80=ds_co2_80.create_interior_points()
#ds_co2_80=ds_co2_80.calc_interrior_points_cost()

# Plot of hull facets
"""
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
                    camera=dict(eye=dict(x=-1.25,y=1.25,z=1.25))))

fig.show()


















# %%
