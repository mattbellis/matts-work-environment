#!/usr/bin/env python

from scattering import *

# Lambda C
max = 20
for i in range(1,max):
  dx = 1.0/max
  step = 0.00 + dx*i
  scattering("plots/test_"+str(i),  step)
