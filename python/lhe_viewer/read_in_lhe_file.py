import lhe_tools as lhe
import matplotlib.pylab as plt
import lichen.lichen as lch
import seaborn as sn

import sys

infile = open(sys.argv[1],'r')

events = lhe.get_events(infile)

Elist = []
for event in events:
    #print "---"
    for p in event:
        #print p
        E,px,py,pz,m,pid,s = p
        if m<10000000:
            #print m,pid
            Elist.append(E)
#print events

lch.hist_err(Elist,ecolor='black')

plt.show()
