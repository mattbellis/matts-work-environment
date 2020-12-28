import numpy as np
import uproot
import awkward as ak # This is awkward 1

import h5hep as hp

x = [[1, 2], [3,4,5,6]]

ax = ak.Array(x)

print(ak.flatten(x))
print(ak.num(ax))


####################
# h5hep
####################

data = hp.initialize()

hp.create_group(data, "jet", counter="njet")
hp.create_dataset(data, ["e", "px", "py", "pz"], group="jet", dtype=float)

hp.create_group(data, "muons", counter="nmuon")
hp.create_dataset(data, ["e", "px", "py", "pz"], group="muons", dtype=float)

event = hp.create_single_event(data)

#'''
for i in range(0, 1000):

    hp.clear_event(event)

    njet = 5
    event["jet/njet"] = njet

    for n in range(njet):
        event["jet/e"].append(np.random.random())
        event["jet/px"].append(np.random.random())
        event["jet/py"].append(np.random.random())
        event["jet/pz"].append(np.random.random())

    hp.pack(data, event)

print("Writing the file...")

# hdfile = write_to_file('output.hdf5',data)
hdfile = hp.write_to_file("output.hdf5", data, comp_type="gzip", comp_opts=9)
#'''


data, event = hp.load('output.hdf5', verbose=False)




