import numpy as np
import matplotlib.pylab as plt
import XKCD_plots as xkcd
import matplotlib.image as mpimg
import Image
import matplotlib.font_manager as fm


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

################################################################################
fig = plt.figure(figsize=(10,11),dpi=100,facecolor='w',edgecolor='b')
#ax = fig.add_subplot(3,1,3,frame_on=False)
#ax = fig.add_subplot(3,1,2,frame_on=False)
ax = plt.subplot2grid((4,1), (1, 0), rowspan=2, frame_on=True)
fig.subplots_adjust(top=0.99,bottom=0.02,right=0.99,left=0.01,wspace=0.05,hspace=0.05)


# The lines
xpt = np.linspace(0, 190, 100)
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

ax.text(190.05, 0.04, "\"Maybe I'll do\na 2nd analysis!\"")
ax.text(190.05, 0.30, "\"Freeman Dyson\nnever got\nhis PhD, right?\"")

ax.text(15.0, 0.60, "SL3 -> SL4\nswitch")
ax.plot([20.0, 40.0], [0.55, 0.20], '-k', lw=0.5)

ax.text(90.0, 1.20, "Robustness studies")
ax.plot([95.0, 122.0], [1.15, 0.65], '-k', lw=0.5)

ax.text(160.0, 0.80, "Punzi effect")
ax.plot([158.0, 135.0], [0.78, 0.65], '-k', lw=0.5)


ax.set_ylim(-0.3, 1.3)

#XKCDify the axes -- this operates in-place
xkcd.XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-')#, expand_axes=True)

#img=mpimg.imread('sbscience.jpg')
img=Image.open('small_sbscience.jpg')
height = img.size[1]
img = np.array(img).astype(np.float) / 255
#fig.figimage(img,-140, 180.0)
fig.figimage(img,-125,875)

# Don't work?
#rsize = img.resize((img.size[0]/100,img.size[1]/100)) 
#imgplot = plt.imshow(img)
#rsizeArr = np.asarray(rsize)
#imgplot = plt.imshow(rsizeArr)

prop = fm.FontProperties(fname='Humor-Sans.ttf', size=14)

ax0 = fig.add_subplot(4,4,2,xticks=[],yticks=[])
ax0.text(0.05,0.2,"You\'ve managed to\ntransform\ncounting physics\nevents into a\nphysics event\nthat counts!\nCongratulations!\n\nAdam Edwards")
for text in ax0.texts:
    text.set_fontproperties(prop)

ax1 = fig.add_subplot(4,4,3,xticks=[],yticks=[])
ax1.text(0.05,0.3,"Congratulations\nTomo!!!\nBest of luck\nin your future\nendeavors.\n\nParker Lund")
for text in ax1.texts:
    text.set_fontproperties(prop)

ax2 = fig.add_subplot(4,4,4,xticks=[],yticks=[])
ax2.text(0.05,0.2,"Many\ncongratulations\nDr. Miyashita on\npassing your\ndefense and wish\nyou all the best\nfor your future\ncareer,\n\nEugenia Puccio")
for text in ax2.texts:
    text.set_fontproperties(prop)

ax3 = fig.add_subplot(4,4,13,xticks=[],yticks=[])
ax3.text(0.05,0.3,"A fitting gesture\nfrom those who\ntried to make\nyour success\ntake even longer.\n\nBrian Meadows")
for text in ax3.texts:
    text.set_fontproperties(prop)

ax4 = fig.add_subplot(4,4,14,xticks=[],yticks=[])
ax4.text(0.05,0.3,"Congrats\nDr. Tomo!!\nExcellent work\nand best of luck\nin the future!\n\nStephanie Majewski")
for text in ax4.texts:
    text.set_fontproperties(prop)

ax6 = fig.add_subplot(4,4,15,xticks=[],yticks=[])
ax6.text(0.05,0.3,"import pyphd\n\nfor i in range(137):\n  print \"Congrats!\"\n\npyphd.finished()\n\nMatt Bellis")
for text in ax6.texts:
    text.set_fontproperties(prop)

ax5 = fig.add_subplot(4,4,16,xticks=[],yticks=[])
ax5.text(0.05,0.3,"Congratulations\non making it over\nall those bumps\nin the road on the\nway to the PhD!\n\nPat Burchat")
for text in ax5.texts:
    text.set_fontproperties(prop)

fig.savefig("tomo_congrats.png")
fig.savefig("tomo_congrats.pdf")

plt.show()
