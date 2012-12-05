import numpy as np
import matplotlib.pylab as plt
import XKCD_plots as xkcd
import matplotlib.image as mpimg
import Image


x0 = np.array([])
y0 = np.array([])
infile = open('constrainedList.tsv','r')
for line in infile:
    vals = line.split()
    x0 = np.append(x0,float(vals[0]))
    y0 = np.append(y0,float(vals[1]))

x1 = np.array([])
y1 = np.array([])
infile = open('unconstrainedList.tsv','r')
for line in infile:
    vals = line.split()
    x1 = np.append(x1,float(vals[0]))
    y1 = np.append(y1,float(vals[1]))

np.random.seed(0)

fig = plt.figure(figsize=(10,6),dpi=100,facecolor='w',edgecolor='b')
ax = fig.add_subplot(1,1,1,frame_on=False)
fig.subplots_adjust(top=0.80,bottom=0.15,right=0.80,left=0.20)


# The lines
xpt = np.linspace(0, 200, 100)
ypt = 0.05*np.ones(100)
ax.plot(xpt, ypt, 'b--', lw=0.5)
ypt = 0.32*np.ones(100)
ax.plot(xpt, ypt, 'r--', lw=0.5)
ypt = 1.00*np.ones(100)
ax.plot(xpt, ypt, 'k', lw=0.5)

#x = np.linspace(0, 180, 100)
#ax.plot(x, np.exp(-((x-90)**2)/120), 'b', lw=1, label='BaBar (2012)')
ax.plot(x1, y1, 'k--', lw=2, label='BaBar (2012) - unconstrained')
ax.plot(x0, y0, 'r', lw=1, label='BaBar (2012) - isospin constrained')

ax.set_title('Likelihood scan of $\\alpha$')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('1-Confidence of graduating')

ax.legend(loc='lower right')

ax.text(200.05, 0.04, "\"Maybe I'll do\na second analysis!\"")
ax.text(200.05, 0.35, "\"Freeman Dyson never got\nhis PhD, right?\"")

ax.text(15.0, 0.60, "SL3 -> SL4\nswitch")
ax.plot([20.0, 45.0], [0.55, 0.40], '-k', lw=0.5)

ax.text(90.0, 1.20, "Robustness studies")
ax.plot([95.0, 120.0], [1.15, 1.05], '-k', lw=0.5)

ax.text(160.0, 0.80, "Punzi effect")
ax.plot([158.0, 135.0], [0.78, 0.65], '-k', lw=0.5)

ax.text(220.0, 1.30, "You\'ve managed to\ntransform counting\nphysics events\ninto a physics event\nthat counts!\nCongratulations!\nAdam Edwards" ,rotation=-45)

ax.text(-70.0, 1.70, "Congratulations Tomo!!!\nBest of luck\nin your future\nendeavors.\nParker Lund",rotation=10)

#ax.set_xlim(0, 10)
ax.set_ylim(-0.3, 1.3)

#XKCDify the axes -- this operates in-place
xkcd.XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-')#, expand_axes=True)

#img=mpimg.imread('sbscience.jpg')
img=Image.open('small_sbscience.jpg')
height = img.size[1]
img = np.array(img).astype(np.float) / 255
fig.figimage(img, -140, 180.0)
#rsize = img.resize((img.size[0]/100,img.size[1]/100)) 
#imgplot = plt.imshow(img)
#rsizeArr = np.asarray(rsize)
#imgplot = plt.imshow(rsizeArr)


plt.show()
