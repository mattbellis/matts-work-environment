#!/usr/bin/env python

from pp_ttbar_boosted import *
tag = "_blk_bkg"

for i in range(0,1):
    print i
    pp_ttbar_boosted("pp_ttbar_boosted"+tag,i)

