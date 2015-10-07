import matplotlib.pylab as plt
import numpy as np

ngals = 100000

xlims = [-150., 150.]
ylims = [-150., 150.]
zlims = [2500.0,3200.0]

tag = "Gravitationally_Rigid_Intergalactic_Dimensions_%dk" % (ngals/1000)

# Randoms
ngrids = 10
xgwidth = (xlims[1]-xlims[0])/ngrids
ygwidth = (ylims[1]-ylims[0])/ngrids

print ngrids,xgwidth,ygwidth

x = np.zeros(ngals)
y = np.zeros(ngals)
z = np.zeros(ngals)

for i in range(0,ngals):
    if np.random.randint(2)==0:
        xg = np.random.randint(ngrids)
        x[i] = xlims[0] + xg*xgwidth + (xgwidth*0.3)*np.random.random() 
        y[i] = ylims[0] + (ylims[1]-ylims[0])*np.random.random()
    else:
        yg = np.random.randint(ngrids)
        y[i] = ylims[0] + yg*ygwidth + (ygwidth*0.3)*np.random.random()
        x[i] = xlims[0] + (xlims[1]-xlims[0])*np.random.random()

z = zlims[0] + (zlims[1]-zlims[0])*np.random.random(ngals)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(x,y,'bo',markersize=0.1)

plt.subplot(1,2,2)
plt.hist(z,bins=100)

filename = "%s_data.dat" % (tag)
np.savetxt(filename,np.transpose([x,y,z]),fmt=['%-10.4f','%-10.4f','%-10.4f'])



# Randoms
x = xlims[0] + (xlims[1]-xlims[0])*np.random.random(ngals)
y = ylims[0] + (ylims[1]-ylims[0])*np.random.random(ngals)
z = zlims[0] + (zlims[1]-zlims[0])*np.random.random(ngals)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(x,y,'bo',markersize=0.1)

plt.subplot(1,2,2)
plt.hist(z,bins=100)

filename = "%s_random.dat" % (tag)
np.savetxt(filename,np.transpose([x,y,z]),fmt=['%-10.4f','%-10.4f','%-10.4f'])

#plt.show()


