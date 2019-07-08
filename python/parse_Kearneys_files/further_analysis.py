import numpy as np
import sys

import matplotlib.pylab as plt

infilename = sys.argv[1]

infile = open(infilename,encoding='utf8', errors='ignore')

absorbances = []
for line in infile:

    vals = line.replace('\x00','').split('\t')


    if len(vals)==1 and vals[0]=='\n':
        continue

    if len(vals)>12:
        if not (vals[0].find('Plate')>=0 or vals[1].find('Temperature')>=0):
            print(vals)
            a = []
            for i in range(0,12):
                a.append(float(vals[2+i]))
            absorbances.append(a)

absorbances = np.array(absorbances)
print(absorbances)

blank = np.mean(absorbances.transpose()[-1])
print(blank)

absorbances -= blank

print(absorbances)

absorbances[absorbances<0] = 0
print(absorbances)

fractions = (absorbances.transpose()/absorbances.transpose()[10]).transpose()*100.
print(fractions)


#xlabels = np.arange(0,11) + 100
#xlabels = xlabels.astype(str)
concentrations = 10*np.ones(12)
xlabels = []
xlabels.append('{0:2.3f} mM'.format(10))
for i in range(1,11):
    concentrations[i] = concentrations[i-1]/2.
    xlabels.append('{0:2.3f} mM'.format(concentrations[i]))

print()
print(concentrations)
print(xlabels)
print(len(fractions[0]),len(xlabels))
#xlabels = ['10 mM', '5 mM', '2.5 mM', '1.25 mM', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

plt.figure()
for frac in fractions:
    plt.plot(xlabels,frac[0:-1],'o',markersize=10,alpha=0.4)
plt.xticks(rotation=90)

plt.tight_layout()

plt.show()






