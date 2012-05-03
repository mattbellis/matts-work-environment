"""
Run the example from the website as a doctest:
http://code.google.com/p/pyminuit/
"""
import minuit
def f(x, y):
    return ((x-2) / 3)**2 + y**2 + y**4

m = minuit.Minuit(f, x=10, y=10)

#m.printMode = 1

m.migrad()
m.fval, m.ncalls, m.edm
m.values["x"], m.values["y"]

m.hesse()
print m.errors

print m.covariance
print m.matrix(correlation=True)

m.minos()
print m.merrors


print m.contour("x", "y", 1.)[:3]
print m.scan(("x", 5, 0, 10), ("y", 5, 0, 10), corners=True)[0]

