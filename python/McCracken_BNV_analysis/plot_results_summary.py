import matplotlib.pylab as plt
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#x = [0,1,2,3,4,5]
#x = [1,1,1.0,0.8,1.2,1]
#text_offsets = [0.4,0.4,0.4,-0.6,0.4,0.4,0.4]
y = [0.63,0.47,1.75e-3,8.4e-4,8.32e-4,1.57e-4]
yerr = [0.005, 0.005, 0.15e-3, 1.4e-4, 0.14e-4, 0.35e-4]
x = np.arange(0,len(y))
mylabels = [r'$p\pi^-$',\
            r'$n\pi^0$', \
            r'$n\gamma$', \
            r'$p\pi^-\gamma$', \
            r'$pe^-\bar{\nu}_e$', \
            r'$p\mu^-\bar{\nu}_\mu$' \
            ]

boxes = []
patches = []

ybnv = [2e-6,3e-6,2e-6,3e-6,6e-7,6e-7,4e-7,6e-7,9e-7,2e-57]
#xbnv = np.ones(len(ybnv)) + 0.5*np.arange(0,len(ybnv))
xbnv = np.arange(len(x),len(ybnv)+len(x))
yerrbnvlo = 1e-8*np.ones(len(ybnv))
yerrbnvhi =     np.zeros(len(ybnv))
text_offsetsbnv = 0.4*np.ones(len(ybnv))
mylabelsbnv = [r'$K^+e^-$',\
               r'$K^+\mu^-$',\
               r'$K^-e^+$',\
               r'$K^-\mu^+$',\
               r'$\pi^+e^-$',\
               r'$\pi^+\mu^-$',\
               r'$\pi^-e^+$',\
               r'$\pi^-\mu^+$',\
               r'$\bar{p}\pi^+$',\
               r'$K_s^0\nu$']

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
plt.errorbar(x,y,yerr=yerr,fmt='o',markersize=10,label=r'$\Lambda$ measured branching fractions')
#plt.xticks(x,mylabels,fontsize=24)
#plt.yticks(y,mylabels,fontsize=24)

#plt.show()

print yerrbnvlo
print yerrbnvhi
plt.errorbar(xbnv,ybnv,yerr=[yerrbnvlo,yerrbnvhi],fmt='s',color='red',markersize=15,linewidth=5,label=r'$\Lambda$ BNV upper limits (this work)')
ybnv = np.array(ybnv)


'''
for xpt,ypt in zip(xbnv,ybnv):
    box = mpatches.Rectangle((xpt-0.25,1e-8), width=0.5, height=ypt-1e-8, alpha=1.0,color='red')
    boxes.append(box)
    ax.add_patch(box)
'''

'''
ybnv_others = [7.8e-7,7.5e-7,7.7e-7,6.3e-7,4.1e-7,5.9e-7,4.2e-7,4.4e-7,95e-7]
xbnv_others = np.arange(len(x),len(ybnv_others)+len(x))
yerrbnvlo = 1e-6*np.ones(len(ybnv))
yerrbnvhi =     np.zeros(len(ybnv))
text_offsetsbnv = 0.4*np.ones(len(ybnv))
mylabelsbnv = [r'$K^+e^-$',\
               r'$K^+\mu^-$',\
               r'$K^-e^+$',\
               r'$K^-\mu^+$',\
               r'$\pi^+e^-$',\
               r'$\pi^+\mu^-$',\
               r'$\pi^-e^+$',\
               r'$\pi^-\mu^+$',\
               r'$K_s^0\nu$']
'''
#print x
#print mylabels
#print xbnv
#print mylabelsbnv

#print type(x)
#print type (xbnv)
#print x + xbnv
#print mylabels + mylabelsbnv
plt.xticks(np.append(x,xbnv),mylabels+mylabelsbnv,fontsize=24)
#plt.plot(xbnv,ybnv-1e-7,'ko',marker=r'$\downarrow$',markersize=50)

plt.xlim(min(x)-1,max(xbnv)+1)
plt.ylim(1e-8,1)
plt.yscale('log')
plt.ylabel('Branching fraction',fontsize=24)

locs, labels = plt.xticks()
plt.setp(labels, rotation=-60)
plt.subplots_adjust(bottom=0.20)

'''
for l,xpt,ypt,to in zip(mylabels,x,y,text_offsets):
    ax.annotate(l,xy=(xpt+to,ypt),xycoords='data',fontsize=24)

for l,xpt,ypt,to in zip(mylabelsbnv,xbnv,ybnv,text_offsetsbnv):
    ax.annotate(l,xy=(xpt+to,ypt),xycoords='data',fontsize=24)
    print xpt,ypt
    #ax.arrow(xpt,ypt,xpt,5e-8,fc='r',ec='r',head_width=0.05,head_length=1e-6)
'''

plt.legend()

plt.show()


