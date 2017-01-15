import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pylab as plt

ngen = 1000000
ngen4 = int(ngen/4)

################################################################################
def process_event(ev):

    jetmax = []
    for jets in ev:
        if len(jets)>0:
            jmax = np.max(jets)
            jetmax.append(jmax)
        else:
            jetmax.append(0)


    return jetmax
    

################################################################################
def main():

    #input_data = np.random.random(ngen)
    input_data = []
    for i in range(0,ngen):
        njets = np.random.randint(0,8)
        jets = np.random.random(njets)
        input_data.append(jets)

    ############################################################################
    start = time.clock()
    nsum = process_event(input_data)
    end = time.clock()
    print ("basic version: %f" % (end-start))
    #print (nsum)
    print (np.sum(nsum))

    pool = Pool()
    start = time.clock()
    #results = pool.map(process_event,[input_data[0:ngen/4],input_data[ngen/4:ngen/2],input_data[ngen/2:3*ngen/4],input_data[3*ngen/4:]])
    results = pool.map(process_event,[input_data[i*ngen4:(i+1)*ngen4] for i in range(4)])
    pool.close()
    pool.join()
    end = time.clock()
    print ("pool version: %f" %(end-start))
    #print (results)
    print (np.sum(results))

    '''
    tot = 0
    for r in results:
        tot += np.sum(r)
    print(tot)
    '''

    return 0

################################################################################
if __name__=='__main__':
    main()
