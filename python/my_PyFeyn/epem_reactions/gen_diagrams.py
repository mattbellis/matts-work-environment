#!/usr/bin/env python

#from epem_B_lambdac import *
#from epem_B_lambda0 import *
tag = ""

from epem_B_lambdac_blk_bkg import *
from epem_B_lambda0_blk_bkg import *
tag = "_blk_bkg"

# Lambda C
for i in range(0,5):
    print i
    epem_B_lambdac("cartoon_B_lambdac"+tag,i)
    epem_B_lambda0("cartoon_B_lambda0"+tag,i)
