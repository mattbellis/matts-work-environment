import numpy as np
import matplotlib.pylab as plt

from sklearn.neighbors.kde import KernelDensity

d0 = np.random.normal(5,0.5,1000)
#d1 = np.random.normal(7,0.3,500)
d1 = np.random.lognormal(1,0.25,500) + 5

# The KDE will be expecting an array of arrays, as if it were a multidimensional
# dataset. 
data = np.concatenate([d0,d1])[:, np.newaxis]
#print(data)

plt.figure()
plt.hist(data,bins=50,density=True)

kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data)
#kde = KernelDensity(kernel='tophat', bandwidth=0.2).fit(data)

# The last bit is to turn it into 100 individual "datasets" of one point each.
#X_plot = np.linspace(-5, 10, 100)[:, np.newaxis]
#print(X_plot)
xpts = np.linspace(min(data),max(data),100)[:, np.newaxis]
#print(xpts)

# KDE is returning the log density
# https://stackoverflow.com/questions/25299642/why-does-scikit-learn-return-log-density
# https://stackoverflow.com/questions/20335944/why-use-log-probability-estimates-in-gaussiannb-scikit-learn
# 
# Might want to read this as well
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
ypts = kde.score_samples(xpts)
#print(ypts)

plt.plot(xpts,np.exp(ypts))

plt.show()
