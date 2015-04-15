#!/usr/bin/env python

from diagrams import *
tag = "_blk_bkg"

for i in range(0,3):
    name = "fd_lfv_B_%s_%d" % (tag,i)
    fd_lfv_c_quark(name,i)
