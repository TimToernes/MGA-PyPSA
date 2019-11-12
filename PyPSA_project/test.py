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

#%%

def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list



#%%



points_10= np.array([[-1,1,1],[-1,-1,1]])

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False})

fig.add_trace(go.Scatter(x=points_10.T[:,0],
                                    y=points_10.T[:,1],
                                    mode='markers',
                                    marker={'size':7,'color':'blue'}))#,

#%% 

df_10 = pd.read_csv('./output/prime_euro_80_4D_eta_0.1.csv')
df_05 = pd.read_csv('./output/prime_euro_80_4D_eta_0.05.csv')
df_02 = pd.read_csv('./output/prime_euro_80_4D_eta_0.02.csv')
df_01 = pd.read_csv('./output/prime_euro_80_4D_eta_0.01.csv')

points_10=df_10.drop(columns=['objective_value']).values
points_05=df_05.drop(columns=['objective_value']).values
points_02=df_02.drop(columns=['objective_value']).values
points_01=df_01.drop(columns=['objective_value']).values

data = np.concatenate([points_10,points_05,points_02,points_01])
results = np.concatenate([df_10['objective_value'],df_05['objective_value'],df_02['objective_value'],df_01['objective_value']])


# %%

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

points_10=df_10.drop(columns=['objective_value']).values
points_05=df_05.drop(columns=['objective_value']).values
points_02=df_02.drop(columns=['objective_value']).values
points_01=df_01.drop(columns=['objective_value']).values

#scaler = MinMaxScaler()
#scaler.fit(points_10)
#points_s = scaler.transform(points_10)
#points_s_10 = scaler.transform(points_05)

#objective_values = df_10['objective_value']
#objective_values_10 = df_05['objective_value']

#data = np.concatenate([points_s,points_s_10])
#results = np.concatenate([objective_values,objective_values_10])


#scaler = MinMaxScaler()
scaler = StandardScaler()
results_s=scaler.fit_transform(np.array([results]).T)
results_s = results_s.T[0]

scaler_2 = MinMaxScaler()
scaler_2 = scaler_2.fit(data)
data_s = scaler_2.transform(data)
data_s = data_s


index = np.random.rand(len(data))>0.2

reg = LinearRegression(normalize=False)

reg.fit(data_s[index], results_s[index])

reg.score(data_s[index==False],results_s[index==False])

#reg.predict(data[index==False])

#%%

m = 1000
inter_points = []
inter_objectives = []

while len(inter_points)<m:
    #point = np.random.rand(len(data.T))
    point = np.random.lognormal(mean=np.mean(data_s,axis=0),
                                sigma=np.std(data_s,axis=0),
                                size=len(data.T))

    inter_objective = reg.predict([point])
    if inter_objective <= max(results_s)*10:
        inter_points.append(point)
        inter_objectives.append(inter_objective)


inter_points = np.array(inter_points)
inter_objectives = np.array(inter_objectives)
inter_objectives = inter_objectives[:,0]

inter_objectives = scaler.inverse_transform(np.array([inter_objectives]).T)
inter_objectives = inter_objectives.T[0]
inter_points = scaler_2.inverse_transform(inter_points)
inter_points = inter_points

columns = list(df_10.drop(columns=['objective_value']).columns)

df_inter_points = pd.DataFrame(data=inter_points,columns=columns)

df_inter_3d = create_3d_dataset(df_inter_points)

# %%
def create_3d_dataset(df_detail):
    type_def = ['ocgt','wind','olar']
    types = [column[-4:] for column in df_detail.columns]
    #sort_idx = np.argsort(types)
    idx = [[type_ == type_def[i] for type_ in types] for i in range(len(type_def))]

    points_3D = []
    for row in df_detail.iterrows():
        row = np.array(row[1])
        point = [sum(row[idx[0]]),sum(row[idx[1]]),sum(row[idx[2]])]
        points_3D.append(point)

    df_points = pd.DataFrame(columns=['ocgt','wind','solar'],data=points_3D)
    #print(self.df_points.head())

    return(df_points)

def create_interior_points(hull):
    m = 100000

    # Generate Delunay triangulation of hull
    try :
        tri = Delaunay(hull.points[hull.vertices])#,qhull_options='Qs')#,qhull_options='A-0.999')
    except : 
        points = np.append(hull.points[hull.vertices],[np.mean(hull.points,axis=0)],axis=0)            
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
    interrior_points = np.array(interrior_points)
    return interrior_points

def calc_interrior_points_cost(df_points,objective_value,interrior_points):
    interrior_points_cost = griddata(df_points.values, 
                                            objective_value, 
                                            interrior_points,
                                            method='linear')


    return interrior_points_cost

# %%

columns = list(df_10.drop(columns=['objective_value']).columns)

df_all = pd.DataFrame(data=data,columns=columns)
df_all_3d= create_3d_dataset(df_all)

hull = ConvexHull(df_all_3d.values)

interrior_points = create_interior_points(hull)
interrior_points_cost = calc_interrior_points_cost(df_all_3d,results,interrior_points)



#%%

#interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')
df_points = df_all_3d
cost = results
trace1 = (go.Scatter3d(x=df_points['ocgt'][0:1],
                            y=df_points['wind'][0:1],
                            z=df_points['solar'][0:1],
                            mode='markers',
                            marker={'color':'blue','size':2}))

trace2 = (go.Scatter3d(x=df_points['ocgt'][:],
                            y=df_points['wind'][:],
                            z=df_points['solar'][:],
                            mode='markers',
                            marker={'color':cost,'colorbar':{'thickness':20,'title':'Scenario cost'}}))


trace4 = (go.Scatter3d(x=df_inter_3d['ocgt'][:],
                            y=df_inter_3d['wind'][:],
                            z=df_inter_3d['solar'][:],
                            mode='markers',
                            marker={'color':inter_objectives,'size':2,'colorbar':{'thickness':20,'title':'Scenario cost'}}))



trace3 = (go.Scatter3d(x=interrior_points[:,0],
                            y=interrior_points[:,1],
                            z=interrior_points[:,2],
                            mode='markers',
                            marker={'color':interrior_points_cost,'size':2}))


fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False},
                data=[trace2,trace1,trace4])

fig.update_layout(scene = dict(
                    xaxis_title='ocgt',
                    yaxis_title='wind',
                    zaxis_title='solar',
                    camera=dict(eye=dict(x=-1.25,y=1.25,z=1.25))))

fig.show()

# %% Check distribution of points 

points = np.array([[1,1,-1,-1],[1,-1,-1,1]]).T

hull = ConvexHull(points)

interrior_points = create_interior_points(hull)

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False})

fig.add_trace(go.Scatter(x=points[:,0],
                        y=points[:,1],
                        mode='markers'))

fig.add_trace(go.Scatter(x=interrior_points[:,0],
                        y=interrior_points[:,1],
                        mode='markers',
                        marker=dict(opacity=0.2)))

# %%
