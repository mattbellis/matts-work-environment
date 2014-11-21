import numpy as np
import sys

Nul = float(sys.argv[1])
eff = float(sys.argv[2])

Nppi = 3.71e7 # From Section 2.5 in analysis note
Bppi = 0.639

B = Bppi*Nul/(eff*Nppi)

print B
