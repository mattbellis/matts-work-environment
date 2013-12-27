import numpy as np
from numpy import sqrt
from math import factorial,frexp

from decimal import Decimal
from decimal import *

k = 0 

invpi = 0

a = Decimal(0)
b = Decimal(0)
c = Decimal(0)

getcontext().prec = 400

invpi = Decimal(0)

for k in range(0,21):

    a = Decimal((2*sqrt(2.0))/9801.0)

    b = Decimal(factorial(4*k)*(1103+26390*k))

    c = Decimal(((factorial(k)**4))*(396**(4*k)))

    invpi += a*b/c
    print a*b/c
    #print k,a*b/c


print Decimal(1.0)/invpi
