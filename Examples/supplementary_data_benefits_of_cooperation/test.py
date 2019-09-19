


import os 
import pandas as pd

nodes = pd.read_csv("data/graph/nodes.csv",header=None,squeeze=True)

countries = pd.Series([iso_countries.get(country).name for country in nodes])


for country,node in zip(countries,nodes):
    try :
        os.rename('data/load/{}_2011.xls'.format(country),'data/load/{}_2011.xls'.format(node))
    except :
        print(country)

node = nodes.values[6]
def country_centroids(node):
    import pyproj
    import math

    data = pd.read_csv('data/country_centroids.csv',sep=',',encoding='latin-1')

    PROJ='+proj=utm +zone=31, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

    def LatLon_To_XY(Lat,Lon):
        p1=pyproj.Proj(PROJ,preserve_units=True)
        (x,y)=p1(Lat,Lon)
        return(x,y)

    data['x'] = [(LatLon_To_XY(data.latitude[0],data.longitude[0]))[0] for latitude,lonitude in zip(data.latitude,data.longitude)]
    data['y'] = [(LatLon_To_XY(data.latitude[0],data.longitude[0]))[1] for latitude,lonitude in zip(data.latitude,data.longitude)]

    x = data[data.country==node].x
    y = data[data.country==node].y
    return x,y