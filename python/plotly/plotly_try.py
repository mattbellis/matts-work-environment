import plotly 

py = plotly.plotly(username_or_email="MatthewBellis", key="d6h4et78v5")

x0 = [1,2,3,4]; y0 = [10,15,13,17]
x1 = [2,3,4,5]; y1 = [16,5,11,9]
response = py.plot(x0, y0, x1, y1)
url = response['url']
filename = response['filename']
