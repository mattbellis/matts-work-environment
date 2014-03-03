import numpy as np
import time
from multiprocessing import Pool

ngen = 10000000

################################################################################
def add_numbers(n):

    #x = sum(n)
    x = np.cos(n)*np.sin(n)*np.log(n)*np.tan(n)/n

    return x
    

################################################################################
def main():

    input_data = np.random.random(ngen)

    start = time.clock()
    nsum = add_numbers(input_data)
    end = time.clock()
    print "basic version: ",end-start,nsum

    pool = Pool()
    start = time.clock()
    results = pool.map(add_numbers,[input_data[0:ngen/4],input_data[ngen/4:ngen/2],input_data[ngen/2:3*ngen/4],input_data[3*ngen/4:]])
    pool.close()
    pool.join()
    end = time.clock()
    print "pool version: ",end-start
    print results
    '''
    tot = 0
    for r in results:
        tot += r
    print tot
    '''

    return 0

################################################################################
if __name__=='__main__':
    main()
