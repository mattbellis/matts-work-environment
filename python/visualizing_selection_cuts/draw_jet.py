from jets import *

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)

#patches,lines = jet()
patches,lines = jet(length=1,opening_angle=45,angle=340)

for p in patches:
    ax.add_patch(p)
for l in lines:
    ax.add_line(l)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-2.0,2.0)
ax.set_ylim(-2.0,2.0)
plt.axis('off')

# plt.savefig('cogent_cutaway_0.png')

plt.show()

