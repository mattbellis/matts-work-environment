from jets import *

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)

# Draw one draw_jet
#patches,lines = draw_jet()
patches,lines = draw_jet(length=1,opening_angle=35,angle=310)

for p in patches:
    print "here"
    print p
    ax.add_patch(p)
for l in lines:
    print "there"
    print l
    ax.add_line(l)

# Draw many draw_jets
patches,lines = draw_jets(origins=[(0,0),(0,0),(0,0)],lengths=[1,0.5,0.7],opening_angles=[25,20,15],angles=[45,130,280],ntrackss=[5,5,5],show_trackss=[False,False,False])

for p in patches:
    print "here"
    print p
    ax.add_patch(p)
for l in lines:
    ax.add_line(l)

# Draw many muons
lines = draw_muons(origin=[(0,0),(0,0),(0,0)],length=[1.2,1.8,1.9],angle=[45,220,150])

for l in lines:
    ax.add_line(l)

# Draw many electrons
lines = draw_electrons(origin=[(0,0),(0,0),(0,0)],length=[0.5,1.5,1.3],angle=[145,210,260])
for l in lines:
    ax.add_line(l)

# Draw many photons
lines = draw_photons(origin=[(0,0),(0,0)],length=[1.5,1.9],angle=[10,160])
for l in lines:
    ax.add_line(l)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-2.0,2.0)
ax.set_ylim(-2.0,2.0)
plt.axis('off')

# plt.savefig('cogent_cutaway_0.png')

plt.show()

