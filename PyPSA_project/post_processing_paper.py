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
import iso3166
#import logging
im_dir="C:/Users\Tim\Dropbox\MGA paper\Paper/figures/"


#%%

ds = dataset('output/prime_euro_50_storage_4D_eta_0.05.xlsx',variables=['wind','solar','H2','battery'])

#ds = dataset('output/archive/prime_euro_95_4D_eta_0.1.csv',data_type='csv')

# %% Calculate interror points

ds.df_interrior = pd.DataFrame(data=ds.interrior_points,
                                columns=['wind','solar','H2','battery'])

ds.df_interrior['transmission'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']],
        ds.df_sum_var['transmission'],
        ds.interrior_points, 
        method='linear') 

ds.df_interrior['ocgt'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']],
        ds.df_sum_var['ocgt'],
        ds.interrior_points, 
        method='linear') 

#filter = ds.df_secondary_metrics['gini'].notnull()
filter = ds.df_secondary_metrics.system_cost != 0
ds.df_interrior['gini'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']][filter],
        ds.df_secondary_metrics['gini'][filter],
        ds.interrior_points, 
        method='linear') 
ds.df_interrior['gini'] = ds.df_interrior['gini'].fillna(0)

ds.df_interrior['autoarky'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']][filter],
        ds.df_secondary_metrics['autoarky'][filter],
        ds.interrior_points, 
        method='linear')

ds.df_interrior['system_cost'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']][filter],
        ds.df_secondary_metrics['system_cost'][filter],
        ds.interrior_points, 
        method='linear')

ds.df_interrior['co2_emission'] =  griddata(ds.df_sum_var[['wind','solar','H2','battery']][filter],
        ds.df_secondary_metrics['co2_emission'][filter],
        ds.interrior_points, 
        method='linear')

# %% Histogram with all sum var
"""
fig = ff.create_distplot(ds.df_interrior.values.T*1e-3,list(ds.df_interrior.keys()),
                            bin_size=50000*1e-3,
                            #colors=['#8c564b','#1f77b4','#ff7f0e','#2ca02c']
                            )
fig.update_xaxes(title_text='MW installed capacity')
"""
#%% Figure 2 - Step by step

#ds.df_sum_var = ds.df_sum_var[ds.df_sum_var['ocgt'] != 0]

def contour(x,y):

    gini_hull = ConvexHull(np.array([x,y]).T)

    x = gini_hull.points[gini_hull.vertices][:,1]
    y = gini_hull.points[gini_hull.vertices][:,0]

    trace = go.Scatter(x=np.append(x,x[0]),
                        y=np.append(y,y[0]),
                        mode='lines',
                        #name=names[i],
                        fill='tonexty'
                        )
    return trace

fig = go.Figure()

fig.add_trace(contour(ds.df_sum_var['wind'][1:9]+ds.df_sum_var['solar'][1:9],ds.df_sum_var['solar'][1:9]))
fig.add_trace(contour(ds.df_sum_var['wind']+ds.df_sum_var['solar'],ds.df_sum_var['solar']))



# %% Figure 3 - Correlation/histogram


def plot_contour(x,y,plot_range=None):

    if plot_range == None:
        [[min(x),max(x)],[min(y),max(y)]]

    h,x_bins,y_bins = np.histogram2d(x,y,bins=50,
                                range=plot_range,
                                )

    trace = go.Contour(x=x_bins,y=y_bins,z=h.T,
    showscale = False,
    contours=dict(start=1,end=150,size=40),
    colorscale=[[0, 'rgba(0, 0, 255,0)'],[0.5, 'rgba(0, 0, 255,0.4)'],[1.0, 'rgba(0, 0, 255,1)']]#'Electric'
    )

    return trace


def figure3(df):

    data_shape = df.shape[1]

    corr_matrix = df.corr()
    fig = make_subplots(rows = data_shape, cols=data_shape,shared_xaxes=False,shared_yaxes=False)

    plot_range = [0,1e3]
    #domains = [[0,0.24],[0.26,0.49],[0.51,0.74],[0.76,1]]
    spacing = 0.02
    domains = list(zip(np.linspace(-spacing,1,data_shape+1)[:-1]+spacing,np.linspace(0,1,data_shape+1)[1:]-spacing))
    #domains = domains[-data_shape-1:]
    colors = ['#1f77b4','#ff7f0e','#8c564b','#2ca02c','#17becf','#bcbd22','#bcbd22']
    labels= list(df.keys())
    data = df.values.T
    data_ranges = np.array([df.min().values,df.max().values]).T


    for i in range(data_shape):
        for j in range(data_shape):
            
            if i != j and j>=i:
                # Write correlation value on left side of matrix
                fig.add_trace(go.Scatter(y=[0],
                                            x=[0],
                                            mode='text',
                                            text='{:.2f}'.format(corr_matrix.iloc[i][j]),
                                            textfont_size=20,
                                            yaxis='y2',xaxis='x2'),row=i+1,col=j+1)
                fig.update_yaxes(range=[-1,1],
                                domain=domains[data_shape-1-i],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)
                fig.update_xaxes(range=[-1,1],
                                domain=domains[j],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)

            if i != j and i>=j:
                # Plot scatter
                fig.add_trace(go.Scatter(x=[data[j][0]],y=[data[i][0]],marker_color='red'),row=i+1,col=j+1)
                fig.add_trace(go.Scatter(x=data[j][0:1000],y=data[i][0:1000],mode='markers'),row=i+1,col=j+1)
                fig.add_trace(plot_contour(x=data[j],
                                            y=data[i],
                                            plot_range=[data_ranges[j],data_ranges[i]]),
                                            row=i+1,col=j+1)

                
                fig.update_yaxes(range=data_ranges[i],
                                domain=domains[data_shape-1-i],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_xaxes(range=data_ranges[j],
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)

            elif i == j :
                # Plot histogram on diagonal
                fig.add_trace(go.Histogram(x=data[i],
                                            xaxis='x',
                                            histnorm='probability',
                                            nbinsx=50,
                                            marker_color=colors[i]),
                                            row=i+1,col=j+1)
                fig.update_xaxes(range=data_ranges[i],
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_yaxes(
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                domain=domains[data_shape-1-i],
                                row=i+1,col=j+1)
            # Set axis titles for border plots
            if i == data_shape-1 :
                fig.update_xaxes(title_text=labels[j],showticklabels=True,
                                domain=domains[j],
                                row=i+1,col=j+1)

            if j == 0 :
                fig.update_yaxes(title_text=labels[i],showticklabels=True,
                                domain=domains[data_shape-1-i],
                                row=i+1,col=j+1)
                if i == 0:
                    fig.update_yaxes(showticklabels=False,row=i+1,col=j+1)


    fig.update_layout(width=1000,
                        height=1000,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showline=True,linecolor='black'),
                        yaxis=dict(showline=True,linecolor='black'),
                        )
    fig.show()


# Scaling of variables
#df = ds.df_interrior[['system_cost','co2_emission','gini','wind','solar','battery','H2']]

df = pd.DataFrame(ds.df_sum_var[['wind','solar','H2','battery']])
df['gini'] = ds.df_secondary_metrics['gini']
df['co2_emission'] = ds.df_secondary_metrics['co2_emission']
df['system_cost'] = ds.df_secondary_metrics['system_cost']



df[['wind','solar','H2','battery']] = df[['wind','solar','H2','battery']]*1e-3
df['gini'] = df['gini']

df['co2_emission'] = 100 - (df['co2_emission']/571067122.7405636)*100

figure3(df)


# %%

def contour(x,y):

    gini_hull = ConvexHull(np.array([x,y]).T)

    x = gini_hull.points[gini_hull.vertices][:,1]
    y = gini_hull.points[gini_hull.vertices][:,0]

    trace = go.Scatter(x=np.append(x,x[0]),
                        y=np.append(y,y[0]),
                        mode='lines',
                        #name=names[i],
                        fill='tonexty'
                        )
    return trace