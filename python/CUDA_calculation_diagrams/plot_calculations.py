import numpy as np
import matplotlib.pylab as plt

################################################################################
def calc_points(starting_x, starting_y, nx, ny,diag_flag=0):

    xpts = np.array([])
    ypts = np.array([])

    for i in range(0,nx):
        for j in range(0,ny):
            if diag_flag==0:
                xpts = np.append(xpts,i+starting_x)
                ypts = np.append(ypts,j+starting_y)
            elif diag_flag==1:
                if i>j:
                    xpts = np.append(xpts,starting_x+i)
                    ypts = np.append(ypts,ny-j-1+starting_y)
    return xpts,ypts
################################################################################

fig = plt.figure(figsize=(8,8),dpi=100,facecolor='w',edgecolor='w')
ax = fig.add_subplot(1,1,1)
plt.subplots_adjust(wspace=0.1,hspace=0.1,bottom=0.1,top=0.95,left=0.1,right=0.95)


npts = 10
diag_flag = 1
ndiv = 4

#npts = 40
#diag_flag = 0
#ndiv = 1

alternate = 0
for i in range(0,ndiv):
    if alternate==0:
        alternate=1
    elif alternate==1:
        alternate=0

    for j in range(0,ndiv):
        xpts,ypts = calc_points(i*npts,j*npts,npts,npts,diag_flag)

        if alternate==0:
            p = ax.plot(xpts,ypts,'o',marker='o',markeredgecolor='r',markerfacecolor='w',markeredgewidth=1)
            alternate = 1
        elif alternate==1:
            p = ax.plot(xpts,ypts,'o',marker='o',markeredgecolor='k',markerfacecolor='k')
            alternate = 0

frame1 = plt.gca()

ax.set_xlim(-1,40)
ax.set_ylim(-1,40)

ax.set_xlabel('i$^{th}$ galaxy',fontsize=30)
ax.set_ylabel('j$^{th}$ galaxy',fontsize=30)
#ax.set_xlabel('Test',fontsize=20)

#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
for i in frame1.axes.get_xticklabels():
    i.set_visible(False)
for i in frame1.axes.get_xticklines():
    i.set_visible(False)
for i in frame1.axes.get_yticklabels():
    i.set_visible(False)
for i in frame1.axes.get_yticklines():
    i.set_visible(False)

#name = "cuda_calcs_%d.png" % (diag_flag)
name = "cuda_calcs_%d_%d.eps" % (diag_flag,npts)
fig.savefig(name)

plt.show()

