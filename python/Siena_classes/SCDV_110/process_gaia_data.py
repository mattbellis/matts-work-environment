import sys

import numpy as np
import pandas as pd

df = pd.read_hdf(sys.argv[1])

print(df.columns)

output = ""
for a,b,c in zip(df['ra'], df['dec'], df['phot_g_mean_mag']):
    print(a,b,c)
    output += f"{a},{b},{c}"

outfile = open("gaia_10k_stars.csv",'w')
outfile.write(output)
outfile.close()
