import numpy as np

################################################################################
def calc_forces(positions,softening=0.1):

    forces = []

    npos = len(positions)

    for i in xrange(npos):
        ftot = 0.0
        p=positions[i]
        r = positions-p
        r = r[r!=0]
        sign = r/np.abs(r)
        ftot += np.sum((1.0/(r*r + softening))*sign)

        forces.append(ftot)

    forces = np.array(forces)

    return forces

################################################################################

################################################################################
def calc_retarded_forces(positions,velocities,c=10.0,softening=0.1):

    forces = []

    npos = len(positions)

    for i in xrange(npos):
        ftot = 0.0
        p=positions[i]
        local_velocities = velocities[positions!=p]
        other_positions = positions[positions!=p]

        r = other_positions-p
        time_sep = r/c

        #print len(local_velocities)
        #print len(other_positions)
        #print len(time_sep)
        other_positions -= local_velocities * time_sep

        r = other_positions - p
        sign = r/np.abs(r)
        ftot += np.sum((1.0/(r*r + softening))*sign)

        forces.append(ftot)

    forces = np.array(forces)

    return forces

################################################################################
def acc_from_force(forces,masses=1.0):

    acc = forces/masses

    return acc

################################################################################


