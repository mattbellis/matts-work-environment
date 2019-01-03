import matplotlib.pylab as plt
import numpy as np

import sys

from p32s_tools import parse_file,check_equivalency,display_schedule,latex_schedule,header,footer,compile_latex


################################################################################
infilenames = ['siena_schedule.tsv','rpi_schedule.tsv']

infiles = [open(name,'r') for name in infilenames]

#print(infiles)

siena = parse_file(infiles[0])
other = parse_file(infiles[1])

#print(siena)
#print(other)

###############################################################################
# Display everything
###############################################################################
output,equivalents = display_schedule(siena,other_schedule=other)

print(output)

print()
'''
print("------------------------------------------\nRPI requirements satisfied while at Siena:\n---------------------------------------")
for eq in equivalents:
    print(eq)

print()
print("------------------------------------------\nRPI requirements NOT satisfied while at Siena:\n---------------------------------------")
rpi_classes = other[0].tolist()
for eq in equivalents:
    rpi_classes.remove(eq)

for rpicl in rpi_classes:
    print(rpicl)
'''

output,equivalents_other = display_schedule(other,taken_elsewhere=equivalents)
print(output)

#latex_schedule(siena)
content = header()
content += latex_schedule(siena,other_schedule=other,school='Siena Applied Physics')
content += latex_schedule(other,taken_elsewhere=equivalents,school='RPI Mechanical Engineering')
content += footer()

compile_latex(content)
