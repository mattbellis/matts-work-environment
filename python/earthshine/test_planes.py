import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from scipy.spatial import distance

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Define a plane
'''
x = np.array([[1, 1], [3, 3]])
y = np.array([[1, 3], [1, 3]])
z = np.array([[5, 5], [5, 5]])
ax.plot_surface(x, y, z)

x = np.array([[1, 1], [3, 3]])
y = np.array([[1, 3], [1, 3]])
z = np.array([[-5,-5], [-5, -5]])
ax.plot_surface(x, y, z)
'''

def rotations_of_planes(x,y,z):
    planes = []
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        theta = np.deg2rad(angle)
        #rot = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
        # Rotate around x-axis
        rot = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]])

        #print(x.shape,y.shape,z.shape)
        #print(x)
        #print(y)
        #print(z)

        x1 = np.ones(shape=x.shape)
        y1 = np.ones(shape=y.shape)
        z1 = np.ones(shape=z.shape)

        imax = x.shape[0]
        jmax = x.shape[1]
        for i in range(imax):
            for j in range(jmax):
                v = [x[i][j], y[i][j], z[i][j]]
                v1 = np.matmul(rot,v)
                #print("here: ",v1)
                x1[i][j] = v1[0]
                y1[i][j] = v1[1]
                z1[i][j] = v1[2]

        #ax.plot_surface(x1, y1, z1,color='grey')
        ax.plot_wireframe(x1, y1, z1,color='tan',alpha=0.2)

        planes.append([x1,y1,z1])

    return planes



planes = []

radii = [4.0, 5.0, 6.0, 7.0] # Meters
#radii = [4.0] # Meters
length = 13.0/2 # Meters
for radius in radii:

    #width = np.sin(np.deg2rad(30))*radius/2
    width = np.deg2rad(30)*radius/2
    #print(width)

    (x, y) = np.meshgrid(np.arange(-length, length+0.0001, 0.5), np.arange(-width, width+0.0001, .10))
    z = radius*np.ones(shape=x.shape)
    planes += rotations_of_planes(x,y,z)

#(x, y) = np.meshgrid(np.arange(-20, 20.1, 1), np.arange(-0.4, 0.41, .1))
#z = 2*np.ones(shape=x.shape)
#rotations_of_planes(x,y,z)



#xline = [0,0]
#yline = [0,0]
#zline = [-radii[-1]-2,radii[-1]+2]

xline = [-9.8,8.3]
yline = [-7.2,10.9]
zline = [-radii[-1]-2,radii[-1]+2]


for ntrks in range(0,10):

    xline = [2*length*np.random.random()-length,2*length*np.random.random()-length]
    yline = [2*length*np.random.random()-length,2*length*np.random.random()-length]
    zline = [-radii[-1]-2,radii[-1]+2]

    plt.plot(xline,yline,zline,color='red',linewidth=1)

    npts = 1000
    xline_pts = np.linspace(xline[0],xline[1],npts)
    yline_pts = np.linspace(yline[0],yline[1],npts)
    zline_pts = np.linspace(zline[0],zline[1],npts)

    line = np.array([xline_pts,yline_pts,zline_pts]).T
    #print(line)
    #plt.plot(xline_pts,yline_pts,zline_pts,'.',color='red',linewidth=4)


    #FLATTEN A PLANE
    hits = []
    for plane in planes:
        p = np.array([plane[0].flatten(), plane[1].flatten(), plane[2].flatten()])
        p = p.T
        #print(p)

        d = distance.cdist(p,line)
        indices = np.where(d == d.min())
        #print(indices)
        # The first entry is the indices for the points in the plane
        # The second entry is the indices for the points on the line
        #print(p[indices[0]])
        for i,idx in enumerate(indices[0]):
            if d[indices][i]<0.3:
                print(d[indices][i])
                hits.append([p[idx][0],p[idx][1],p[idx][2]])
                plt.plot([p[idx][0]],[p[idx][1]],[p[idx][2]],'bo',markersize=10)
        #plt.plot(p[idx[0]][0],'ko',markersize=20)
        #plt.plot(p[idx], 'ko',markersize=20)

    print("--------------")
    for hit in hits:
        print(hit)



















ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.tight_layout()

plt.show()
