#!/usr/bin/env python

from pyx import *

from epem_ccbar_D0_to_Kshh import *
from D_mixing_GIM import *
from D_mixing_final_state import *
from D_mixing_final_state_simple_version import *

col = color.cmyk.Red
#text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\definecolor{COL}{cmyk}{%(c)g,%(m)g,%(y)g,%(k)g}" % col.color)


D_mix_GIM("d_mix_gim",0)
#D_mix_final_state("d_mix_final_state",0)
#D_mix_final_state_simple_version("d_mix_final_state_simple_version",0)

# Lambda C
'''
for i in range(0,5):
    print i
    epem_ccbar_D0_to_Kshh("cartoon_slow_pion_tagged_D0",i)
'''
