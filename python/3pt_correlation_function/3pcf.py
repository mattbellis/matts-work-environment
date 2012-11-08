import matplotlib.pylab as plt
import numpy as np

################################################################################
################################################################################
def calc_dist(x0,y0,z0,x1,y1,z1):
    dx = x0-x1
    dy = z0-x1
    dz = z0-z1
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    return dist

################################################################################
ngals = 1000
x = np.random.random(ngals)
y = np.random.random(ngals)
z = np.random.random(ngals)

hist_min = 0.0
hist_max = 2.0
nbins = 10
bin_width = (hist_max-hist_min)/nbins 

hist = np.zeros((nbins,nbins,nbins))

for i in range(0,ngals):
    print i
    if i%10==0:
        print hist
    for j in range(i,ngals):
        for k in range(j,ngals):
            d0 = calc_dist(x[i],y[i],z[i],x[j],y[j],z[j])
            d1 = calc_dist(x[i],y[i],z[i],x[k],y[k],z[k])
            d2 = calc_dist(x[j],y[j],z[j],x[k],y[k],z[k])

            bin_index0 = int((d0-hist_min)/bin_width) + 1;
            bin_index1 = int((d1-hist_min)/bin_width) + 1;
            bin_index2 = int((d2-hist_min)/bin_width) + 1;
            hist[bin_index0][bin_index1][bin_index2] += 1

print hist

np.save('output',hist)
