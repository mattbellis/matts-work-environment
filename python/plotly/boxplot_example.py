import plotly 

import numpy as np
from numpy.random import lognormal

py = plotly.plotly(username_or_email="MatthewBellis", key="d6h4et78v5")

x=[0]*1000+[1]*1000+[2]*1000
y=lognormal(0,1,1000).tolist()+lognormal(0,2,1000).tolist()+lognormal(0,3,1000).tolist()
s={'type':'box','jitter':0.5}
l={'title': 'Fun with the Lognormal distribution','yaxis':{'type':'log'}}

response = py.plot(x,y,style=s,layout=l,filename='boxplot_example',fileopt='overwrite')
print response
url = response['url']
print url
filename = response['filename']
print filename
