import sys 
import numpy as np
import matplotlib.pylab as plt

infilenames = ['tabula-Dept profile - Siena College.csv',
               'tabula-Dept profile - School of Science.csv',
               'tabula-Dept profile - Physics and Astronomy.csv']
               
colors = ['green','red','black']
labels = ['Siena','SoS','Phys & Astr']

figs = []
for i,infilename in enumerate(infilenames):
    #infilename = infilenames[0]
    vals = np.loadtxt(infilename,delimiter=',',unpack=False,dtype=str)

    #print(vals)
    years = vals[0][1:]

    print("Size of figs..........{0}".format(len(figs)))

    figcount = 0
    for j,v in enumerate(vals[2:]):
        if v[1].find('Fall')>=0:
            print('continue')
            print(v)
            continue

        if i==0:
            figs.append(plt.figure())
            print('Opened figure {0}'.format(j))
            figcount += 1

        else:
            print('Figure {0}'.format(figcount))
            plt.figure(figs[figcount].number)
            figcount += 1

        title = v[0]
        print(title)
        pts = v[1:]
        for k,p in enumerate(pts):
            if p.find('%')>=0:
                #print(p)
                pts[k] = p.replace('%','')

            idx0 = pts[k].find('(')
            idx1 = pts[k].find(')')
            if idx0>=0 and idx1>=0:
                pts[k] = pts[k][0:idx0]

            idx0 = pts[k].find('/')
            if idx0>=0:
                pts[k] = pts[k][0:idx0]

            idx0 = pts[k].find('<')
            if idx0>=0:
                pts[k] = pts[k].replace('<','')

        pts[pts==''] = -1
        print(pts)
        pts = pts.astype(float)

        plt.errorbar(years,pts,fmt='o',markersize=10,color=colors[i],label=labels[i],alpha=0.7)
        plt.xticks(rotation=270)
        plt.title(title)
        #plt.ylim(0,1.3*max(pts))

        if i==2:
            plt.legend()
            plt.tight_layout()
            name = 'plots/figure{0}.png'.format(figcount-1)
            plt.savefig(name)
#plt.show()


