#!/usr/bin/env python

#import b_l_decay 
#from b_l_decay import *
#tag = ""

import b_l_decay_blk_bkg 
from b_l_decay_blk_bkg import *
tag = "_blk_bkg"

# Lambda C
meson_to_baryon_lepton("B_lambdac"+tag,      "Bz",  "qd", "aqb", "qc", "qu", "lm",  "cgLp", 1)
meson_to_baryon_lepton("B_lambdac_flip"+tag, "Bz",  "qd", "aqb", "qu", "qc", "lm",  "cgLp", 1)
meson_to_baryon_lepton("B_lambda"+tag,      "Bp",  "qu", "aqb", "qs", "qd", "lp",  "gL", 1)
meson_to_baryon_lepton("B_lambda_flip"+tag, "Bp",  "qu", "aqb", "qd", "qs", "lp",  "gL", 1)
meson_to_baryon_lepton("B_proton"+tag,      "Bz",  "qd", "aqb", "qu", "qu", "lm",  "p",   1)

meson_to_baryon_lepton("D_lambda"+tag, "Dm",  "qd", "aqc", "qs", "qu","lm",  "gL", 1)
meson_to_baryon_lepton("D_proton"+tag, "aDz",  "qu", "aqc", "qu", "qd","lm",  "p", 1)

baryon_to_meson_lepton("proton_decay"+tag, "p",  "qd", "qu", "aqd", "qs", "gpz", 1)
