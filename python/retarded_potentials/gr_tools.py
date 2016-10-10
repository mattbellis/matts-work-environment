import numpy as np

################################################################################
def calc_forces(positions):

    forces = []

    npos = len(positions)

    for i in xrange(npos):
        ftot = 0.0
        p=positions[i]
        r = positions-p
        r = r[r!=0]
        sign = r/np.abs(r)
        ftot += np.sum((1.0/(r*r))*sign)

        forces.append(ftot)

    forces = np.array(forces)

    return forces

################################################################################

################################################################################
def acc_from_force(forces,masses=1.0):

    acc = forces/masses

    return acc

################################################################################


