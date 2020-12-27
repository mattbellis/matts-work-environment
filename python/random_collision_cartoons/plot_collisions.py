import numpy as np

import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib import patches

isMC = False
#isMC = True


# Data
#nmuons_range = (0,2)
#njets_range = (2,12)
#nelectrons_range = (0,2)
#tag = "data"

# Top quark
nmuons_range = (1,2)
njets_range = (4,6)
nelectrons_range = (0,3)
tag = "top"

# QCD
#nmuons_range = (0,1)
#njets_range = (10,15)
#nelectrons_range = (0,1)
#tag = "qcd"

# Weird
#nmuons_range = (2,4)
#njets_range = (2,4)
#nelectrons_range = (0,6)
#tag = "odd"


origin = (0,0)

def jet(angle,mag):
    nlines = np.random.randint(2,6)
    angles = angle + 0.5*np.random.random(nlines)-0.25
    mags = mag + (0.5*np.random.random(nlines) - 0.25)

    return angles,mags


ncollisions = 32

gs = gridspec.GridSpec(4, 8, wspace=0, hspace=0)

# For final prints
#dpi=600
dpi=100
plt.figure(figsize=(12,5),dpi=dpi)



for nc in range(ncollisions):
    plt.subplot(gs[nc])

    if isMC:
        #plt.gca().set_facecolor('palegreen')
        plt.gca().set_facecolor('aquamarine')

        autoAxis = plt.gca().axis()
        #rec = plt.Rectangle((-1.1,-1.1),2.2,2.2,fill=False,lw=4,joinstyle='round',capstyle='round')
        rec = patches.FancyBboxPatch((-0.85,-0.85),1.75,1.75,fill=False,lw=4,joinstyle='round',capstyle='round')
        rec = plt.gca().add_patch(rec)
        rec.set_clip_on(False)

    ###### Muons
    nmuons = np.random.randint(nmuons_range[0], nmuons_range[1])
    nlines = nmuons
    color = 'b'

    angles = np.pi*2*np.random.random(nlines)
    mags = 0.9 + 0.5*np.random.random(nlines)
    xpts = mags*np.cos(angles)
    ypts = mags*np.sin(angles)

    for x,y in zip(xpts,ypts):
        plt.plot([origin[0],x], [origin[1],y],'-',color=color)

    ###### Electrons
    nelectrons = np.random.randint(nelectrons_range[0], nelectrons_range[1])
    nlines = nelectrons
    color = 'g'

    angles = np.pi*2*np.random.random(nlines)
    mags = 0.9 + 0.5*np.random.random(nlines)
    xpts = mags*np.cos(angles)
    ypts = mags*np.sin(angles)

    for x,y in zip(xpts,ypts):
        plt.plot([origin[0],x], [origin[1],y],'--',color=color)
    ####### JETS ###########
    njets = np.random.randint(njets_range[0],njets_range[1])

    print(njets)
    for i in range(njets):
        angle = np.pi*2*np.random.random()
        mag = 0.5 + 0.5*np.random.random()

        print(angle,mag)

        angles,mags = jet(angle,mag)

        xpts = mags*np.cos(angles)
        ypts = mags*np.sin(angles)

        for x,y in zip(xpts,ypts):
            plt.plot([origin[0],x], [origin[1],y],'r-')

    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)

plt.tight_layout()
if isMC:
    tag += "_MC"
plt.savefig('cartoon_event_{0}.png'.format(tag))
plt.show()


