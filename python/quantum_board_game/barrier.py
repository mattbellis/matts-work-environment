import numpy as np
import matplotlib.pylab as plt


xin = np.linspace(-5,0,1000)
yin = np.sin(6*xin + 1.0) + 1.5
yinr = 0.3*np.sin(2*xin + 0.5) + 1.5

xout = np.linspace(2,5,1000)
yout = 0.5*np.sin(3*xout + 0.2) + 1.5

xbar = np.linspace(0,2,1000)
ybar = np.exp(-3*xbar) + 1.5
print(ybar)




plt.figure(figsize=(6,3),dpi=300)
plt.plot(xin,yin,lw=3,c='b')
plt.plot(xout,yout,lw=3,c='b')
#plt.plot(xin,yinr,lw=3,c='r',alpha=0.2,ls='--')
plt.plot(xbar,ybar,lw=3,c='b')
plt.plot([-5,0],[0,0],lw=4,c='k')
plt.plot([0,0],[0,3],lw=4,c='k')
plt.plot([0,0],[0,3],lw=4,c='k')
plt.plot([0,2],[3,3],lw=4,c='k')
plt.plot([2,2],[3,0],lw=4,c='k')
plt.plot([2,5],[0,0],lw=4,c='k')
plt.ylim(-0.5,3.5)

#plt.gca().set_facecolor('whitesmoke')
#plt.gca().set_facecolor('gainsboro')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig('quantumbarrier.png')


plt.show()
