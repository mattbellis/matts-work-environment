#!/usr/bin/env python

tag = ""

from diagrams import *
tag = "_blk_bkg"

# Lambda C
for i in range(0,5):
    print i
    epem_lfv_charm("epem_lfv_charm"+tag,i)
