#!/usr/bin/env python

# Import the needed modules

from ROOT import *

Ecm = 0.890
m1 = 0.139
m2 = 0.494

# Write this as a string so we can pass it into the TF1 constructor
q_string = "sqrt( (((x**2 - %4.3f**2 - %4.3f**2))**2 - (%4.3f**2*%4.3f))/(4*x**2) )" % (m1,m2,m1,m2)
print q_string

# Write this as a string so we can pass it into the TF1 constructor
bw_factor_string = "sqrt(2*%s/(1+%s))" % (q_string, q_string)
print bw_factor_string

f1 = TF1("f1",q_string,m1+m2,2.0*Ecm)

f1.Draw()

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]
