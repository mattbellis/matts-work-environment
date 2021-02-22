import numpy as np
import matplotlib.pylab as plt


xpts = np.linspace(-3,3,1000)

ytotr = np.zeros(len(xpts))
ytoti = np.zeros(len(xpts))
yptsr = []
yptsi = []
for i in range(1,10):

    y = (i**2)*np.sin(i*xpts)
    yptsr.append(y)
    ytotr += y

    y = (i**2)*np.cos(i*xpts)
    yptsi.append(y)
    ytoti += y





plt.figure(figsize=(6,3),dpi=300)
plt.plot(xpts,(ytoti**2+ytotr**2)/400,lw=3,c='b')
for yi,yr in zip(yptsi,yptsr):
    plt.plot(xpts,yi,'r',alpha=0.4)
    plt.plot(xpts,yr,'r',alpha=0.4)

#plt.plot([0,0],[0,3],lw=4,c='k')
#plt.plot([0,2],[3,3],lw=4,c='k')
#plt.plot([2,2],[3,0],lw=4,c='k')
#plt.plot([2,5],[0,0],lw=4,c='k')
#plt.ylim(-0.5,3.5)

#plt.gca().set_facecolor('whitesmoke')
#plt.gca().set_facecolor('gainsboro')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig('hup.png')


plt.show()
