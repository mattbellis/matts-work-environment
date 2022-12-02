import pandas as pd
import pyreadstat

import sys

infilename = sys.argv[1]

#df,meta = pyreadstat.read_sav('Downloads/HealthLink18HudsonValley_Weight.Sav')
df,meta = pyreadstat.read_sav(infilename)

print(meta.column_labels)
print(meta.value_labels)

# This is how we get the question names
print(meta.variable_value_labels)


df['School'].value_counts()

meta.value_labels

meta.variable_to_label
