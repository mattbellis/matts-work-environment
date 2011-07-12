#!/usr/bin/env python

from diagrams import *
tag = "_blk_bkg"

for i in range(0,4):
    name = "fd_lfv_c_quark_%s_%d" % (tag,i)
    fd_lfv_c_quark(name,i)
