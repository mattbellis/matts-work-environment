import matplotlib.pylab as plt
import numpy as np

# m1*v1 = m1*v1' + m2*v2'
# m1^2*v1^2 = m1^2*v1'^2 + m2^2*v2'^2 + 2*m1*m2*v1'*v2'*cos(theta)

# m1*v1^2 = m1*v1'^2 + m2*v2'^2
# m1^2*v1^2 = m1^2*v1'^2 + m1*m2*v2'^2

# m1^2*v1'^2 + m1*m2*v2'^2 = m1^2*v1'^2 + m2^2*v2'^2 + 2*m1*m2*v1'*v2'*cos(theta)

# m1*m2*v2'^2 = m2^2*v2'^2 + 2*m1*m2*v1'*v2'*cos(theta)
# m1*m2*v2'^2 - m2^2*v2'^2 = 2*m1*m2*v1'*v2'*cos(theta)
# m1*m2*v2' - m2^2*v2' = 2*m1*m2*v1'*cos(theta)
# v2'*(m1*m2 - m2^2) = 2*m1*m2*v1'*cos(theta)
# v2' = 2*m1*m2*v1'*cos(theta) / (m1*m2 - m2^2)

m1 = 1.0

m2 = m1

theta = 0.0
v2 = 2*m1*m2*v1'*v2'*np.cos(theta) /(m1*m2)


