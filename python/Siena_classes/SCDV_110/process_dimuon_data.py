import sys

import matplotlib.pylab as plt
import numpy as np

infilename = sys.argv[1]

e1,px1,py1,pz1,q1,e2,px2,py2,pz2,q2,mdummy  = np.loadtxt(infilename,unpack=True,dtype=float,delimiter=',')

e = e1+e2
px = px1+px2
py = py1+py2
pz = pz1+pz2

q = q1.astype(int)+q2.astype(int)

qstr = 10*np.ones(len(q))
qstr = qstr.astype('str')

qstr[q<0] = "mm"
qstr[q>0] = "pp"
qstr[q==0] = "pm"

print(qstr)
#exit()

m = np.sqrt(e**2 - (px**2 + py**2 + pz**2))

outfile = open("dimuon_summary_data.csv","w")
output = ""
for i in range(len(q)):
    output = f"{e[i]},{px[i]},{py[i]},{pz[i]},{qstr[i]}\n"
    outfile.write(output)
    if i%1000==0:
        print(i)
outfile.close()

plt.figure()
plt.hist(m,bins=200)

plt.show()
