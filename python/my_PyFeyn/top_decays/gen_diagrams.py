#!/usr/bin/env python

import top_decays_blk_bkg
from top_decays_blk_bkg import *
tag = "_blk_bkg"

# Lambda C
for i in range(0,4):
    top_decays("top_decays"+tag, i)
