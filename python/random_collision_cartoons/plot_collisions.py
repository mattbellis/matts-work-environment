import numpy as np

import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib import patches

np.random.seed(1)

# For final prints
#dpi=600
#dpi=300
dpi=200
#dpi=100

#isMC = False
isMC = True


################################################################################
def define_event_type(t='top'):

    nmuons_range = (0,2)
    njets_range = (2,12)
    nelectrons_range = (0,2)
    nphotons_range = (0,2)
    tag = "data"

    if t=='data':
        nmuons_range = (0,2)
        njets_range = (2,12)
        nelectrons_range = (0,2)
        nphotons_range = (0,2)
        tag = "data"

    elif t=='top':
        nmuons_range = (1,2)
        njets_range = (4,6)
        nelectrons_range = (0,3)
        nphotons_range = (0,2)
        tag = "top"

    elif t=='qcd':
        nmuons_range = (0,1)
        njets_range = (10,15)
        nelectrons_range = (0,1)
        nphotons_range = (0,8)
        tag = "qcd"

    elif t=='new_physics':
        nmuons_range = (4,8)
        njets_range = (2,4)
        nelectrons_range = (2,8)
        nphotons_range = (3,6)
        tag = "odd"

    return nmuons_range, njets_range, nelectrons_range, nphotons_range, tag
################################################################################


################################################################################
def jet(angle,mag):
    nlines = np.random.randint(2,6)
    angles = angle + 0.3*np.random.random(nlines)-0.25
    mags = mag + (0.5*np.random.random(nlines) - 0.25)

    return angles,mags
################################################################################



ncollisions = 32
#ncollisions = 64
#gs = gridspec.GridSpec(4, 8, wspace=0, hspace=0)
#plt.figure(figsize=(12,5),dpi=dpi)
# For singles
gs = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
#plt.figure(figsize=(5,4),dpi=dpi)

#ncollisions = 36
#gs = gridspec.GridSpec(6, 6, wspace=0, hspace=0)
#plt.figure(figsize=(10,9),dpi=dpi)

#ncollisions = 25
#gs = gridspec.GridSpec(5, 5, wspace=0, hspace=0)
#plt.figure(figsize=(10,9),dpi=dpi)

#ncollisions = 49
#gs = gridspec.GridSpec(7, 7, wspace=0, hspace=0)
#plt.figure(figsize=(10,9),dpi=dpi)

#ncollisions = 64
#gs = gridspec.GridSpec(8, 8, wspace=0, hspace=0)
#plt.figure(figsize=(10,9),dpi=dpi)

#ncollisions = 9
#gs = gridspec.GridSpec(3, 3, wspace=0, hspace=0)
#plt.figure(figsize=(10,9),dpi=dpi)

#ncollisions = 1
#gs = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
#plt.figure(figsize=(5,4),dpi=dpi)

origin = (0,0)

#event_type='data'
#event_type='mix'
#event_type='top'
event_type='qcd'
#event_type='new_physics'

for nc in range(ncollisions):

    # For singles
    plt.clf()

    event_types = ['top','qcd', 'data']

    et = event_type
    if event_type=='mix':
        et = event_types[np.random.randint(0,2)]

    nmuons_range, njets_range, nelectrons_range, nphotons_range, temp = define_event_type(et)

    #plt.subplot(gs[nc])
    # For singles
    #plt.subplot(1)

    #plt.gca().set_facecolor('whitesmoke')
    plt.gca().set_facecolor('gainsboro')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

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
        plt.plot([origin[0],x], [origin[1],y],'-',color=color,lw=3)

    ###### Electrons
    nelectrons = np.random.randint(nelectrons_range[0], nelectrons_range[1])
    nlines = nelectrons
    color = 'g'

    angles = np.pi*2*np.random.random(nlines)
    mags = 0.9 + 0.5*np.random.random(nlines)
    xpts = mags*np.cos(angles)
    ypts = mags*np.sin(angles)

    for x,y in zip(xpts,ypts):
        plt.plot([origin[0],x], [origin[1],y],'--',color=color,lw=2)

    ###### Photons
    nphotons = np.random.randint(nphotons_range[0], nphotons_range[1])
    nlines = nphotons
    #color = 'deepskyblue'
    color = 'cyan'

    angles = np.pi*2*np.random.random(nlines)
    mags = 0.9 + 0.5*np.random.random(nlines)
    xpts = mags*np.cos(angles)
    ypts = mags*np.sin(angles)

    for x,y in zip(xpts,ypts):
        plt.plot([origin[0],x], [origin[1],y],'-.',color=color,lw=2)

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

    ################################
    # For singles
    ################################
    plt.tight_layout()
    tag = event_type
    if isMC:
        tag += "_MC"
    plt.savefig('cartoon_event_frame{0}n{1}_{2}.png'.format(nc,ncollisions,tag))

#plt.tight_layout()
#tag = event_type
#if isMC:
    #tag += "_MC"
#plt.savefig('cartoon_event_n{0}_{1}.png'.format(ncollisions,tag))

if dpi<100:
    plt.show()


# To convert singles to gif
#convert -delay 20 -loop 0 cartoon_event_frame*mix.png data_animation.gif

