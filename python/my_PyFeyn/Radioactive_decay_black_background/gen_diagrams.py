#!/usr/bin/env python

from diagrams import *
tag = "blk_bkg"

'''
for i in range(0,1):
    name = "neutron_decay_%s_%d" % (tag,i)
    neutron_decay(name,i)
'''

'''
for i in range(0,3):
    name = "neutron_decay_quark_lines_%s" % (tag)
    neutron_decay_quark_lines(name,i)
'''
    
for i in range(0,3):
    name = "antineutron_decay_quark_lines_%s" % (tag)
    antineutron_decay_quark_lines(name,i)
