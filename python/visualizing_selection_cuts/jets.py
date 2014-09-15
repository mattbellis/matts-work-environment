import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

################################################################################
################################################################################
def draw_jet(origin=(0,0),angle=90,length=0.5,opening_angle=20,ntracks=5,show_tracks=False):

    lines = []
    patches = []

    # Edges of cone
    width_at_top = length*np.deg2rad(opening_angle)
    for side in [-1,1]:
        theta0 = np.deg2rad(angle+(side*opening_angle/2.0)) 
        x1 = length*np.cos(theta0)
        y1 = length*np.sin(theta0)
        print x1,y1
        line = mlines.Line2D((origin[0],x1), (origin[1],y1), lw=2., alpha=0.4,color='red',markeredgecolor='red')
        lines.append(line)

    # End of cone
    arad = np.deg2rad(angle)
    center = (origin[0]+np.cos(arad)*length,origin[1]+np.sin(arad)*length)
    print center
    p = mpatches.Ellipse(center, width_at_top+0.01, width_at_top/2.0,facecolor='red',alpha=0.4,edgecolor='gray',angle=abs(angle+90))
    patches.append(p)

    return patches,lines


    

################################################################################
################################################################################
def draw_jets(origins=[(0,0)],angles=[90],lengths=[0.5],opening_angles=[20],ntrackss=[5],show_trackss=[False]):

    alllines = []
    allpatches = []

    # Edges of cone
    for origin,angle,length,opening_angle,ntracks,show_tracks in zip(origins,angles,lengths,opening_angles,ntrackss,show_trackss):
        patches,lines = draw_jet(origin=origin,angle=angle,length=length,opening_angle=opening_angle,ntracks=ntracks,show_tracks=show_tracks)
        allpatches += patches
        alllines += lines


    return allpatches,alllines


    
################################################################################
def draw_muons(origin=[(0,0)],angle=[90],length=[0.5]):

    lines = []

    # Edges of cone
    for o,a,l in zip(origin,angle,length):
        theta = np.deg2rad(a)
        x1 = l*np.cos(theta)
        y1 = l*np.sin(theta)
        print x1,y1
        line = mlines.Line2D((o[0],x1), (o[1],y1), lw=4., alpha=0.9,color='blue',markeredgecolor='blue')
        lines.append(line)

    return lines


    

################################################################################
################################################################################
def draw_electrons(origin=[(0,0)],angle=[90],length=[0.5]):

    lines = []

    # Edges of cone
    for o,a,l in zip(origin,angle,length):
        theta = np.deg2rad(a)
        x1 = l*np.cos(theta)
        y1 = l*np.sin(theta)
        print x1,y1
        line = mlines.Line2D((o[0],x1), (o[1],y1), lw=2., alpha=0.9,color='green',markeredgecolor='green')
        lines.append(line)

    return lines


    
################################################################################
################################################################################
def draw_photons(origin=[(0,0)],angle=[90],length=[0.5]):

    lines = []

    # Edges of cone
    for o,a,l in zip(origin,angle,length):
        theta = np.deg2rad(a)
        x1 = l*np.cos(theta)
        y1 = l*np.sin(theta)
        print x1,y1
        line = mlines.Line2D((o[0],x1), (o[1],y1), lw=5., ls=':', alpha=0.9,color='yellow',markeredgecolor='yellow')
        lines.append(line)

    return lines


    

################################################################################
################################################################################
