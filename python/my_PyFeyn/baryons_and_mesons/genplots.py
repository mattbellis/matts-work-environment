#!/usr/bin/env python

from pyfeyn.user import *
from pyx import *


col = color.cmyk.White
#text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\definecolor{COL}{cmyk}{%(c)g,%(m)g,%(y)g,%(k)g}" % col.color)

from b_and_m import *



#'''
for i in range(0,5):
    baryon("baryon", i)
#'''
for i in range(0,5):
    meson("meson", i)
