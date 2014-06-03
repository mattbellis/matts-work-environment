import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime,timedelta

import scipy.integrate as integrate

import parameters
from cogent_utilities import *
#from fitting_utilities import *
from lichen.plotting_utilities import *
import lichen.pdfs as pdfs

import lichen.iminuit_fitting_utilities as fitutils

import lichen.lichen as lch

pi = np.pi
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

np.random.seed(200)

yearly_mod = 2*pi/365.0


#infile_name = 'data/high_gain.txt'
#infile_name = 'data/low_gain.txt'
infile_name = 'data/HE.txt'
#infile_name = 'data/LE.txt'

tdays,energies,rise_times = get_3yr_cogent_data(infile_name,first_event=first_event,calibration=0)
data = [energies.copy(),tdays.copy(),rise_times]
ranges,subranges,nbins = parameters.fitting_parameters(0)
data = cut_events_outside_range(data,ranges)
data = cut_events_outside_subrange(data,subranges[1],data_index=1)

#tdays,energies = get_cogent_data(infile_name,first_event=first_event,calibration=0)
#data = [energies.copy(),tdays.copy()]
#ranges,subranges,nbins = parameters.fitting_parameters(0)
#data = cut_events_outside_range(data,ranges)
#data = cut_events_outside_subrange(data,subranges[1],data_index=1)

#energy cut
#index0 = data[0]>7.50
#index0 *= data[0]<8.50
#index0 = data[0]>4.00
#index0 = data[0]>2.00
#index0 *= data[0]<4.50
index0 = data[0]>1.50

#indexelo = data[0]>2.0
#indexehi = data[0]<3.0

indexelo = data[0]<1.1
indexehi = data[0]>1.35
indexehi2 = data[0]<2.0

antiindexelo = data[0]>1.1
antiindexehi = data[0]<1.35

# rt cuts
index2 = data[2]>1.0
index3 = data[2]<0.7

plt.figure()
lch.hist_err(data[0],bins=300)

plt.figure()
lch.hist_err(data[0][indexelo+(indexehi*indexehi2)],bins=100)

plt.figure()
lch.hist_err(data[1][indexelo+(indexehi*indexehi2)],bins=40)

plt.figure()
lch.hist_err(data[0][antiindexelo*antiindexehi],bins=100)

plt.figure()
lch.hist_err(data[1][antiindexelo*antiindexehi],bins=40)

#plt.figure()
#lch.hist_err(data[1][index0*index2],bins=100)

#plt.figure()
#lch.hist_err(data[1][index0*index3],bins=100)

#plt.figure()
#lch.hist_err(data[0][index2],bins=70)

#plt.figure()
#lch.hist_err(data[0][index3],bins=70)

#print indexelo
#plt.figure()
#lch.hist_err(data[0][indexelo*indexehi],bins=70)

#plt.figure()
#lch.hist_err(data[2][indexelo*indexehi],bins=50)


plt.show()
