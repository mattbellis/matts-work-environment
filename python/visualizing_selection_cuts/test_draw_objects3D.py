from jets import *
from mpl_toolkits.mplot3d import Axes3D
from draw_objects3D import *
import numpy as np
import mpl_toolkits.mplot3d.art3d as a3


fig = plt.figure(figsize=(7,5),dpi=100)
ax = fig.add_subplot(1,1,1)
ax = fig.gca(projection='3d')
plt.subplots_adjust(top=0.98,bottom=0.02,right=0.98,left=0.02)

lines = []
pmom = []
origin = []

lines += draw_beams()

njets = 6
pmom = 4.0*np.random.random((njets,3))-2.0
origin = np.zeros((njets,3))
lines += draw_jet3D(origin=origin,pmom=pmom)

#'''
nmuons = 2
pmom = 4.0*np.random.random((nmuons,3))-2.0
origin = np.zeros((nmuons,3))
lines += draw_muon3D(origin=origin,pmom=pmom)

nelectrons = 3
pmom = 4.0*np.random.random((nelectrons,3))-2.0
origin = np.zeros((nelectrons,3))
lines += draw_electron3D(origin=origin,pmom=pmom)

nphotons = 2
pmom = 4.0*np.random.random((nphotons,3))-2.0
origin = np.zeros((nphotons,3))
lines += draw_photon3D(origin=origin,pmom=pmom)
#'''

for l in lines:
    ax.add_line(l)

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)

plt.show()

