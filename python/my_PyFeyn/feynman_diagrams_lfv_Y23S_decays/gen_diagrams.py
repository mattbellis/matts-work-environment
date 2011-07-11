#!/usr/bin/env python

from diagrams import *
tag = "_blk_bkg"

#'''
for i in range(0,7):
    name = "bbar_ss_loop_%s_%d" % (tag,i)
    bbbar_ss_loop(name,i)
#'''

for i in range(0,2):
    name = "bbar_leptoq_4pt_%s_%d" % (tag,i)
    bbbar_leptoq_4pt(name,i)
