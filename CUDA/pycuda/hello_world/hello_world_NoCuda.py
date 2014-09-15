import numpy


################################################################################
# Multiply two arrays
################################################################################
def multiply_them(a, b):

    #max = len(a)
    max = 400

    #dest = array(400,'d')
    dest = numpy.zeros_like(a)

    for i in range(0,max):
        dest[i] = a[i] * b[i]

    return dest

################################################################################
# Main
################################################################################
a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

#dest = numpy.zeros_like(a)
#print dest 

dest = multiply_them(a,b)

print "Product!"
print dest
print dest-a*b
