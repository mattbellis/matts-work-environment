#!/usr/bin/env python

#from epem_B_lambdac import *
#from epem_B_lambda0 import *
tag = ""

from epem_different_stuff import *
tag = "_blk_bkg"

# Lambda C
for i in range(18,21):
    print i
    epem_B_lambdac("epem_different_stuff"+tag,i)
