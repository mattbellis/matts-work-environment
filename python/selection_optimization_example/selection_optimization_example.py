import matplotlib.pylab as plt
import numpy as np

import lichen.lichen as lch

nsig = 50000
nbkg = 200000

sig_x = np.random.normal(0.895,0.010,nsig)
bkg_x = 0.2*np.random.random(nbkg) + 0.8
tot_x = sig_x.copy()
tot_x = np.append(tot_x,bkg_x.copy())

sig_y = np.random.exponential(0.6,nsig)
bkg_y = 3.0*np.random.random(nbkg) 
tot_y = sig_y.copy()
tot_y = np.append(tot_y,bkg_y.copy())

plt.figure()
lch.hist_err(sig_x,bins=50,range=(0.8,1.0))

plt.figure()
lch.hist_err(bkg_x,bins=50,range=(0.8,1.0))

plt.figure()
lch.hist_err(tot_x,bins=50,range=(0.8,1.0))

plt.figure()
lch.hist_err(sig_y,bins=50,range=(0,3.0))
plt.figure()
lch.hist_err(bkg_y,bins=50,range=(0,3.0))
plt.figure()
lch.hist_err(tot_y,bins=50,range=(0,3.0))

plt.show()
