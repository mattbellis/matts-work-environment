#!/usr/bin/env python

from pyfeyn.user import *
from pyx import *


col = color.cmyk.White
#text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\definecolor{COL}{cmyk}{%(c)g,%(m)g,%(y)g,%(k)g}" % col.color)

from b_and_m_blk_bkg import *



#'''
for i in range(4,5):
    #baryon("baryon", i)
    baryon("baryon_blk_bkg", i)
#'''
for i in range(4,5):
    #meson("meson", i)
    meson("meson_blk_bkg", i)
