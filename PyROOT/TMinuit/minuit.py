from ROOT import TMinuit
from ROOT import Double
from ROOT import Long
from array import array as arr

nPoints = 5

#ncount = 0

# y = a*x^2 + b*x +c
# a = 1.0
# b = 2.0
# c = 3.0

pointsx = arr( 'f', ( 0.0, 1.0, 2.0,  3.0,  4.0  ) )
pointsy = arr( 'f', ( 3.0, 6.0, 11.0, 18.0, 27.0 ) )

def calcChi2(params):

    chisq = 0.0

    a = params[0]
    b = params[1]
    c = params[2]

    for i in range(0,nPoints):
        x = pointsx[i]
        curFuncVal = a*x*x + b*x + c
        curYVal = pointsy[i]
        chisq += ( (curYVal - curFuncVal) * (curYVal - curFuncVal) ) / (1.0*nPoints)

    return chisq


def fcn(npar, gin, f, par, iflag):
#    global ncount

    chisq = calcChi2(par)

    f[0] = chisq
#    ncount += 1

finalPars = []
finalParsErr = []

gMinuit = TMinuit(3)
gMinuit.SetFCN(fcn)
arglist = arr('d', 2*[0.01])
ierflg = Long(0)

arglist[0] = 1
gMinuit.mnexcm("SET ERR", arglist ,1,ierflg)

# Set initial parameter values for fit
vstart = arr( 'd', (1.0, 1.0, 1.0) )
# Set step size for fit
step =   arr( 'd', (0.001, 0.001, 0.001) )

# Define the parameters for the fit
gMinuit.mnparm(0, "A",  vstart[0], step[0], 0,0,ierflg)
gMinuit.mnparm(1, "B",  vstart[1], step[1], 0,0,ierflg)
gMinuit.mnparm(2, "C",  vstart[2], step[2], 0,0,ierflg)

arglist[0] = 6000 # Number of calls to FCN before giving up. 
arglist[1] = 0.3  # Tolerance
gMinuit.mnexcm("MIGRAD", arglist ,2,ierflg)

amin, edm, errdef = Double(0.18), Double(0.19), Double(0.20)
nvpar, nparx, icstat = Long(1), Long(2), Long(3)
gMinuit.mnstat(amin,edm,errdef,nvpar,nparx,icstat);

gMinuit.mnprin(3,amin);
  
dumx, dumxerr = Double(0), Double(0)
for i in range(0,3):
    gMinuit.GetParameter(i,dumx, dumxerr)
    finalPars.append(float(dumx))
    finalParsErr.append(float(dumxerr))

print "A:\t", finalPars[0], "+/-", finalParsErr[0]
print "B:\t", finalPars[1], "+/-", finalParsErr[1]
print "C:\t", finalPars[2], "+/-", finalParsErr[2]
