import numpy as np
import matplotlib.pylab as plt

from python_files.LUX2013 import *
index = np.arange(0,len(data_values[0]),2)
x0 = data_values[0][index]*data_values_rescale[0][0]
y0 = data_values[0][index+1]*data_values_rescale[0][1]
label0 = data_label

from python_files.CoGeNT_PRL107_2011_annmod_roi import *
index = np.arange(0,len(data_values[0]),2)
x1 = data_values[0][index]*data_values_rescale[0][0]
y1 = data_values[0][index+1]*data_values_rescale[0][1]
label1 = data_label

from python_files.Xenon100_2011 import *
index = np.arange(0,len(data_values[0]),2)
x2 = data_values[0][index]*data_values_rescale[0][0]
y2 = data_values[0][index+1]*data_values_rescale[0][0] # Only has one rescaling
label2 = data_label

from python_files.SuperCDMS_2014 import *
index = np.arange(0,len(data_values[0]),2)
x3 = data_values[0][index]*data_values_rescale[0][0]
y3 = data_values[0][index+1]*data_values_rescale[0][0] # Only has one rescaling
label3 = data_label

from python_files.our_cogent_2014 import *
index = np.arange(0,len(data_values[0]),2)
x4 = data_values[0][index]*data_values_rescale[0][0]
y4 = data_values[0][index+1]*data_values_rescale[0][0] # Only has one rescaling
label4 = data_label

fig = plt.figure(figsize=(10,8))


plt.plot(x1,y1,color='green',label=label1,linewidth=3)
#plt.plot(x2,y2,color='blue',label=label2,linewidth=3)
#plt.plot(x0,y0,color='red',label=label0,linewidth=3)
plt.plot(x3,y3,color='yellow',label=label3,linewidth=3)
plt.plot(x4,y4,color='green',linestyle='--',label=label4,linewidth=3)

plt.fill(x1,y1,facecolor='green',alpha=0.9)
#plt.fill_between(x2,y2,max(y2),facecolor='blue',alpha=0.1)
#plt.fill_between(x0,y0,max(y2),facecolor='red',alpha=0.1)
plt.fill_between(x3,y3,max(y3),facecolor='yellow',alpha=0.1)
plt.fill_between(x4,y4,max(y4),facecolor='green',alpha=0.1)

plt.yscale('log')
plt.xlim(0,30)
plt.ylim(1e-42,1e-38)
plt.xlabel(r'WIMP mass [GeV/c$^2$]',fontsize=24)
plt.ylabel(r'WIMP-nucleon $\sigma_{\rm SI}$ [cm$^2$]',fontsize=24)
plt.legend(loc='upper right')

plt.savefig('exclusion_plot.png')

plt.show()
