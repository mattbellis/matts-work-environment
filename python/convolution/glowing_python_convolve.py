# Adapted from the Glowing Python example.
# http://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html
import numpy
import pylab

################################################################################
def smooth(x,beta):
    """ kaiser window smoothing """
    window_len=11
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = numpy.kaiser(window_len,beta)
    print(w)
    print(len(w))
    print(len(s))
    y = numpy.convolve(w/w.sum(),s,mode='valid')
    print(len(y))
    print(len(y[5:len(y)-5]))
    return y[5:len(y)-5]
################################################################################

beta = [2,4,16,32]

pylab.figure(1)
for b in beta:
    w = numpy.kaiser(101,b) 
    pylab.plot(list(range(len(w))),w,label="beta = "+str(b))
pylab.xlabel('n')
pylab.ylabel('W_K')
pylab.legend()
#pylab.show()


# random data generation
y = numpy.random.random(100)*100 
for i in range(100):
    y[i]=y[i]+i**((150-i)/80.0) # modifies the trend

# smoothing the data
pylab.figure(2)
pylab.plot(y,'-k',label="original signal",alpha=.3)
print(y)
for b in beta:
    yy = smooth(y,b) 
    #print yy
    pylab.plot(yy,label="filtered (beta = "+str(b)+")")
pylab.legend()
pylab.show()
