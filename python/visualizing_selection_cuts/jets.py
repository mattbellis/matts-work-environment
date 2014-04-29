import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def jet(origin=(0,0),angle=90,length=0.5,opening_angle=20,ntracks=5,show_tracks=False):

    lines = []
    patches = []

    # Edges of cone
    width_at_top = length*np.deg2rad(opening_angle)
    for side in [-1,1]:
        theta0 = np.deg2rad(angle+(side*opening_angle/2.0)) 
        x1 = length*np.cos(theta0)
        y1 = length*np.sin(theta0)
        print x1,y1
        line = mlines.Line2D((origin[0],x1), (origin[1],y1), lw=2., alpha=0.8,color='gray',markeredgecolor='red')
        lines.append(line)

    # End of cone
    arad = np.deg2rad(angle)
    center = (origin[0]+np.cos(arad)*length,origin[1]+np.sin(arad)*length)
    print center
    p = mpatches.Ellipse(center, width_at_top+0.05, width_at_top/2.0,facecolor='gray',alpha=0.8,edgecolor='gray',angle=abs(270-angle))
    patches.append(p)

    return patches,lines


    


