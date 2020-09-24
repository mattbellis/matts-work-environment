import pyhf
import numpy as np
import matplotlib.pyplot as plt
import pyhf.contrib.viz.brazil

pyhf.set_backend("numpy")
model = pyhf.simplemodels.hepdata_like(
            signal_data=[10.0, 15, 20], bkg_data=[50.0, 55, 52], bkg_uncerts=[7.0, 7, 8]
            )
data = [55.0, 72, 72] + model.config.auxdata

poi_vals = np.linspace(0, 5, 41)
#results = [
            #pyhf.infer.hypotest(test_poi, data, model, qtilde=True, return_expected_set=True)
                #for test_poi in poi_vals
                #]
results = []
for test_poi in poi_vals:
    x = pyhf.infer.hypotest(test_poi, data, model, qtilde=True, return_expected_set=True)
    results.append(x)

print(results)

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
ax.set_xlabel(r"$\mu$ (POI)")
ax.set_ylabel(r"$\mathrm{CL}_{s}$")
pyhf.contrib.viz.brazil.plot_results(ax, poi_vals, results)

plt.show()
