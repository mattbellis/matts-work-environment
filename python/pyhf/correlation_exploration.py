import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

dataset = np.random.normal(5.0,1.0,10000)


nbins = 10

binned_data = np.histogram(dataset,bins=nbins,range=(0,10))

print(binned_data)

nsamples = 1000
samples = []
for i in range(nsamples):
    x = np.random.choice(dataset,500)
    binned_x = np.histogram(x,bins=nbins,range=(0,10))
    samples.append(binned_x[0])

print(samples)
samples = np.array(samples)

print(samples)


cc = []
for i in range(nbins):
    cc.append([])
    for j in range(nbins):
        x = np.corrcoef(samples.T[i], samples.T[j])[0][1]
        if x != x:
            x = 0
        cc[i].append(x)
        print(x)

plt.figure()
ax = sns.heatmap(cc,vmin=-1,vmax=1,annot=True)

plt.show()
