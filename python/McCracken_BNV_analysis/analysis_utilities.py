#!/usr/bin/env python

import sys
from math import *

import numpy as np

################################################################################
# Calculate the magnitude of a three-vector
################################################################################
def magnitude_of_3vec(v3):

    magnitude = sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

    return magnitude

################################################################################
# Calculate the mass of a particle using Classical Physics
################################################################################
def mass_from_classical_physics(v4):

    pmag = magnitude_of_3vec(v4[1:4])

    mass = pmag*pmag/(2.0*v4[0])

    return mass

################################################################################
# Calculate the mass of a particle using Special Relativity
################################################################################
def mass_from_special_relativity(v4):

    pmag = magnitude_of_3vec(v4[1:4])

    mass_squared = v4[0]*v4[0] - pmag*pmag

    if mass_squared>0:
        return sqrt(mass_squared)
    else:
        return -sqrt(abs(mass_squared))

################################################################################

mass_e = 0.000511
mass_pi = 0.139570
mass_k = 0.493677
mass_p = 0.938272

target = np.array([mass_p,0.0,0.0,0.0])

