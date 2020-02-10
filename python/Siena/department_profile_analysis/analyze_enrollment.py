import sys 
import numpy as np
import matplotlib.pylab as plt

infilenames = ['tabula-Dept profile - Siena College.csv',
               'tabula-Dept profile - School of Science.csv',
               'tabula-Dept profile - Physics and Astronomy.csv',
               ]

colors = ['green','red','black']
labels = ['Siena','SoS','Phys & Astr']

for i,infilename in enumerate(infilenames):
    #infilename = infilenames[0]
    vals = np.loadtxt(infilename,delimiter=',',unpack=False,dtype=str)

    print(vals)
    years = vals[0][1:]

    enrolled = vals[-1][1:].astype(int)
    admitted = vals[-2][1:].astype(int)
    pct = 100*enrolled/admitted

    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    plt.errorbar(years,admitted,yerr=np.sqrt(admitted),fmt='o',markersize=5,color=colors[i],label=labels[i])
    plt.xticks(rotation=270)
    plt.ylim(0,1.2*max(admitted))
    plt.title('Admitted')
    plt.legend()

    plt.subplot(1,3,2)
    plt.errorbar(years,enrolled,yerr=np.sqrt(enrolled),fmt='o',markersize=5,color=colors[i],label=labels[i])
    plt.xticks(rotation=270)
    plt.ylim(0,1.2*max(enrolled))
    plt.title('Enrolled')
    plt.legend()

    plt.subplot(1,3,3)
    err = 100*(1/admitted)*np.sqrt(enrolled*(1-(enrolled/admitted)))
    print(err)
    plt.errorbar(years,pct,yerr=err,fmt='o',markersize=5,color=colors[i],label=labels[i])
    plt.xticks(rotation=270)
    #plt.ylim(0,1.2*max(pct))
    plt.ylim(0,20)
    plt.title('Yield')
    plt.legend()
    plt.tight_layout()

    plt.savefig('plots/figure_yield_{0}.png'.format(labels[i].replace(' ','').replace('&','')))

plt.show()


