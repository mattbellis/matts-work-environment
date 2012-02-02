#!/usr/bin/env python

from epem_B0B0bar import *
tag = "_blk_bkg"

for i in range(0,6):
    print i
    epem_B0B0bar("epem_B0B0bar"+tag,i)
