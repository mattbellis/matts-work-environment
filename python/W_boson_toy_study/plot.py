import sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

################################################################################
def invmass(p4s):

    E,px,py,pz = p4s[0][0],p4s[0][1],p4s[0][2],p4s[0][3]

    print(len(p4s))

    for i in range(1,len(p4s)):
        E += p4s[i][0]
        px += p4s[i][1]
        py += p4s[i][2]
        pz += p4s[i][3]

    m2 = E**2 - (px**2 + py**2 + pz**2)

    mask = m2<0
    m = m2
    m[mask] = -np.sqrt(-m2[mask])
    m[~mask] = np.sqrt(m2[~mask])

    return m

################################################################################

infilename = sys.argv[1]

df = pd.read_csv(infilename)

initial =[df['Ei'] ,df['pxi'] ,df['pyi'] ,df['pzi']]
p1 =[df['E1'] ,df['px1'] ,df['py1'] ,df['pz1']]
p2 =[df['E2'] ,df['px2'] ,df['py2'] ,df['pz2']]
p1smeared =[df['E1smeared'] ,df['px1smeared'] ,df['py1smeared'] ,df['pz1smeared']]
p2smeared =[df['E2smeared'] ,df['px2smeared'] ,df['py2smeared'] ,df['pz2smeared']]

m = invmass([initial])
mtruth = invmass([p1,p2])
mrecon = invmass([p1smeared,p2smeared])

plt.figure()
plt.hist(m,bins=100,range=(75,90))
plt.hist(mtruth,bins=100,range=(70,110))
plt.hist(mrecon,bins=100,range=(70,110),alpha=0.5)

plt.show()


