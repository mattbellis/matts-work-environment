import numpy as np
import matplotlib.pylab as plt

import sys

infile = sys.argv[1]

data = np.loadtxt(infile,skiprows=1)

data = data.transpose()

az = data[0]
el = data[1]

az = az[el>3]
el = el[el>3]

az[az<50] += 360

print len(az)

plt.plot(az,el,'o',markersize=0.1)
plt.xlabel('azimuth',fontsize=24)
plt.ylabel('elevation',fontsize=24)

plt.ylim(-20)

plt.tight_layout()

plt.show()


