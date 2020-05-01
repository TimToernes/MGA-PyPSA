import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
from scipy.spatial import ConvexHull


def plot_gini(x_data,y_data,names = ['1% slack', '5% slack', '10% slack',],x_title='Gini coefficient'):

    if len(x_data) != len(y_data):
        print('!! Errror !! \n x_data and y_data must be same length')

    fig = go.Figure()
    for i in range(len(x_data)):

        gini_hull = ConvexHull(np.array([x_data[i],y_data[i]]).T)

        x = gini_hull.points[gini_hull.vertices][:,1]
        y = gini_hull.points[gini_hull.vertices][:,0]

        fig.add_trace(go.Scatter(x=np.append(x,x[0]),
                            y=np.append(y,y[0]),
                            mode='lines',
                            name=names[i],
                            fill='tonexty'
                            ))

    fig.update_yaxes(title_text='CO2 emission reduction [%]',showgrid=False)
    fig.update_xaxes(title_text=x_title,showgrid=False)

    fig.update_layout(width=800,
                        height=500,
                        showlegend=True,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showline=True,linecolor='black'),
                        yaxis=dict(showline=True,linecolor='black'),
                        )

    return fig 