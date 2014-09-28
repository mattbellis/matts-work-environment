import numpy as np
import sys

infile = open(sys.argv[1])

outfile = open(sys.argv[2],'w')

E1,PX1,PY1,PZ1,Q1,E2,PX2,PY2,PZ2,Q2 = np.loadtxt(infile,unpack=True,skiprows=1,usecols=(3,4,5,6,10,11,12,13,14,18),delimiter=',',dtype=float)

#print E1

count = 0
for e1,px1,py1,pz1,q1,e2,px2,py2,pz2,q2 in zip(E1,PX1,PY1,PZ1,Q1,E2,PX2,PY2,PZ2,Q2):

    output = "Event: %d\n" % (count)

    # Jets (5)
    output += "0\n"
    # Muons (5)
    output += "2\n"
    output += "%f %f %f %f %d\n" % (e1,px1,py1,pz1,q1)
    output += "%f %f %f %f %d\n" % (e2,px2,py2,pz2,q2)
    # Electrons (5)
    output += "0\n"
    # Photons (4)
    output += "0\n"
    # MET (metx,mety)
    output += "0.0 0.0\n"

    #print output
    outfile.write(output)

    count +=1 

outfile.close()


