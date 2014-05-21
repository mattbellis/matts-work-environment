#!/usr/bin/env python

from b_l_decay import *

# Lambda C
#meson_to_baryon_lepton("lambdac_noellipse", "Bz",  "qd", "aqb", "qc", "qu", "cgLp", 0)
meson_to_baryon_lepton("B_lambdac",      "Bz",  "qd", "aqb", "qc", "qu", "lm",  "cgLp", 1)
meson_to_baryon_lepton("B_lambdac_flip", "Bz",  "qd", "aqb", "qu", "qc", "lm",  "cgLp", 1)
meson_to_baryon_lepton("B_lambda",      "Bp",  "qu", "aqb", "qs", "qd", "lp",  "gL", 1)
meson_to_baryon_lepton("B_lambda_flip", "Bp",  "qu", "aqb", "qd", "qs", "lp",  "gL", 1)
meson_to_baryon_lepton("B_proton",      "Bz",  "qd", "aqb", "qu", "qu", "lm",  "p",   1)

meson_to_baryon_lepton("D_lambda", "Dm",  "qd", "aqc", "qs", "qu","lm",  "gL", 1)
meson_to_baryon_lepton("D_proton", "aDz",  "qu", "aqc", "qu", "qd","lm",  "p", 1)

baryon_to_meson_lepton("proton_decay", "p",  "qd", "qu", "aqd", "qs", "gpz", 1)
