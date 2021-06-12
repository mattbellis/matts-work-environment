import numpy as np

# https://www.math.kit.edu/ianm2/lehre/am22016s/media/distance-harvard.pdf

# P is point in space
# Q is point in space
# Line is defined as r(t) = Q + t*u, where u is a vector
#
# d = |(PQvector) x u| / |u|
#

P = np.array([2, 0 , 0])
Q = np.array([0, 0, 0])

u = np.array([1, 1, 0])

umag = np.linalg.norm(u)

PQ = P-Q

PQcrossu = np.cross(PQ,u)

PQcrossumag = np.linalg.norm(PQcrossu)

d = PQcrossumag/umag

print(f"umag: {umag}")
print(f"d: {d}")
