import numpy as np

#beta = 0.05
beta = 0.2

masses = [88.0, 87.0, 86.0, 84.0]

D = [3.17e-10, 0.0, 0.0, 0.0]

for i in range(1,4):

    # (M1/M2)^beta = (D2/D1)
    # D2 = (M1/M2)^beta * D1

    D[i] = ((masses[0]/masses[i])**beta) * D[0]

print "beta",beta
for m,d in zip(masses,D):
    print m,d
