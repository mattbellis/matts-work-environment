#!/usr/bin/env python

#import b_l_decay 
#from b_l_decay import *
#tag = ""

import b_semilep_blk_bkg
from b_semilep_blk_bkg import *
tag = "_blk_bkg"

# Lambda C
for i in range(0,2):
    b_semilep("B_semi_lep"+tag, i)
