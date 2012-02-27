#!/usr/bin/env python

tag = ""

from event_topology_blk_bkg import *
tag = "_blk_bkg"

# Lambda C
for i in range(0,3):
    print i
    event_topology("event_topology"+tag,i)
