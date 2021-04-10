import vector
import numpy as np
import awkward as ak  # at least version 1.2.0
import numba as nb

import time

'''
x = vector.obj(x=3, y=4)  
print(x)

@nb.njit
def compute_mass(v1, v2):
    return (v1 + v2).mass

m = compute_mass(vector.obj(px=1, py=2, pz=3, E=4), vector.obj(px=-1, py=-2, pz=-3, E=4))
print(m)
'''

start = time.time()
array = vector.awk([[dict({x: np.random.normal(0, 1) for x in ("px", "py", "pz")}, E=np.random.normal(10, 1)) for inner in range(np.random.poisson(1.5))] for outer in range(500000)])
print(f"Run time = {time.time() - start} seconds")

@nb.njit
def compute_masses(array):
    out = np.empty(len(array), np.float64)
    for i, event in enumerate(array):
        total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for vec in event:
            total = total + vec
        out[i] = total.mass
    return out

print(ak.num(array))

start = time.time()
m = compute_masses(array)
print(m)
print(len(m))
print(f"Run time = {time.time() - start} seconds")

