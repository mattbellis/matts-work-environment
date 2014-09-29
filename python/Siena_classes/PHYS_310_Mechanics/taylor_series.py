import numpy as np
import matplotlib.pylab as plt

#plt.figure()

# exponential
def taylor_exponential(nterms=1,frange=(0,1),exact_func=False):

    plt.figure()

    x = np.arange(frange[0],frange[1],0.1)
    y = np.zeros(len(x))

    for n in xrange(nterms):
        yterm = x**n/np.math.factorial(n)
        y += yterm

        plt.plot(x,yterm,'--')

    plt.plot(x,y,'b',linewidth=5)

    if exact_func==True:
        yterm = np.exp(x)
        plt.plot(x,yterm,'r',linewidth=5)

    plt.xlabel("x",fontsize=24)
    plt.ylabel("y",fontsize=24)


def taylor_sin(nterms=1,frange=(0,1),exact_func=False):

    x = np.arange(frange[0],frange[1],0.01)
    y = np.zeros(len(x))

    for n in xrange(nterms):

        print np.math.factorial(2*n+1)
        yterm = (((-1.)**n)/np.math.factorial((2*n)+1)) * (x**((2*n)+1))
        y += yterm

        term = "%d" % (n)
        plt.plot(x,yterm,'--',label=term)

    plt.plot(x,y,'b',linewidth=5)

    if exact_func==True:
        yterm = np.sin(x)
        plt.plot(x,yterm,'r--',linewidth=5)

    plt.xlabel("x",fontsize=24)
    plt.ylabel("y",fontsize=24)
    plt.ylim(-2,2)
    plt.legend()


#taylor_exponential(4,exact_func=True)
#plt.show()


#taylor_exponential(4,exact_func=True)
#plt.show()



