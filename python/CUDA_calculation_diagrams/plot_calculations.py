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
                if i<j:
                    xpts = np.append(xpts,i+starting_x)
                    ypts = np.append(ypts,j+starting_y)
    return xpts,ypts
################################################################################

fig = plt.figure(figsize=(8,8),dpi=100,facecolor='w',edgecolor='w')

npts = 10
diag_flag = 1

alternate = 0
for i in range(0,4):
    if alternate==0:
        alternate=1
    elif alternate==1:
        alternate=0

    for j in range(0,4):
        xpts,ypts = calc_points(i*npts,j*npts,npts,npts,diag_flag)

        if alternate==0:
            p = plt.plot(xpts,ypts,'co')
            alternate = 1
        elif alternate==1:
            p = plt.plot(xpts,ypts,'ko')
            alternate = 0

frame1 = plt.gca()

plt.xlim(-1,40)
plt.ylim(-1,40)

frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

#name = "cuda_calcs_%d.png" % (diag_flag)
name = "cuda_calcs_%d.eps" % (diag_flag)
plt.savefig(name)

plt.show()

