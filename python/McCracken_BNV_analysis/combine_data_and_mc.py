import sys
import numpy as np

nsig = 10
if len(sys.argv)>1:
    nsig = int(sys.argv[1])

ntoys = 10
if len(sys.argv)>2:
    ntoys = int(sys.argv[2])

infile0 = open('ks_nu_data_optimal.txt')
infile1 = open('ks_nu_mc_optimal.txt')

data = np.loadtxt(infile0)
mc = np.loadtxt(infile1)

#print data
#print mc

nmc = len(mc)

for i in range(ntoys):

    sample = data.copy()

    entries = np.arange(0,nmc)
    np.random.shuffle(entries)
    index = entries[0:nsig]

    sample = np.append(sample,mc[index])

    #print sample[-nsig:]
    name = "toys/toy_nsig%d_%04d.dat" % (nsig,i)
    np.savetxt(name,sample)

    if i%100==0:
        print name

