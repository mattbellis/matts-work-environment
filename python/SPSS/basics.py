import pandas as pd
import pyreadstat

df,meta = pyreadstat.read_sav('Downloads/HealthLink18HudsonValley_Weight.Sav')

print(meta.column_labels)
print(meta.value_labels)

# This is how we get the question names
print(meta.variable_value_labels)
