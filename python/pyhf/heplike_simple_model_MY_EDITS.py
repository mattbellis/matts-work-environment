import pyhf
import json

import numpy as np
import matplotlib.pylab as plt

signal = [5.0, 10.0]
background = [50.0, 60.0]
bkg_uncerts=[5.0, 12.0]

model = pyhf.simplemodels.hepdata_like(signal_data=signal,
                                        bkg_data=background, 
                                        bkg_uncerts=bkg_uncerts)
#model

print(f'  channels: {model.config.channels}')
print(f'     nbins: {model.config.channel_nbins}')
print(f'   samples: {model.config.samples}')
print(f' modifiers: {model.config.modifiers}')
print(f'parameters: {model.config.parameters}')
print(f'  nauxdata: {model.config.nauxdata}')
print(f'   auxdata: {model.config.auxdata}')

print(model.config.auxdata)

print(model.expected_data([1.0, 1.0, 1.0]))

print(model.expected_data([1.0, 1.0, 1.0], include_auxdata=False))

print(model.expected_actualdata([1.0, 1.0, 1.0]))

print(model.expected_auxdata([1.0, 1.0, 1.0]))

print(model.config.suggested_init())

print(model.config.suggested_bounds())

print(model.config.suggested_fixed())

init_pars = model.config.suggested_init()
print(model.expected_actualdata(init_pars))

print(model.config.poi_index)

bkg_pars = [*init_pars] # my clever way to "copy" the list by value
bkg_pars[model.config.poi_index] = 0
model.expected_actualdata(bkg_pars)

observations = [52.5, 65.] + model.config.auxdata

bins = [0,1]

fig, ax = plt.subplots()
ax.bar(bins, background, 1.0, label=r'$t\bar{t}$', edgecolor='red', alpha=0.5)
ax.bar(bins, signal, 1.0, label=r'signal', edgecolor='blue', bottom=background, alpha=0.5)
ax.scatter(bins, observations[0:2], color='black', label='observed')
#ax.set_ylim(0,6)
ax.legend();

################################################################################

x = model.logpdf(pars = bkg_pars, data = observations)
print(x)

x = model.logpdf( pars = init_pars, data = observations)
print(x)

fit_results = pyhf.infer.mle.fit(data = observations, pdf = model)
print(fit_results)

fit_background = []
for w,b in zip(fit_results[1:],background):
    fit_background.append(w*b)

fit_signal = fit_results[0]*np.array(signal)

fig, ax = plt.subplots()
ax.bar(bins, fit_background, 1.0, label=r'$t\bar{t}$', edgecolor='red', alpha=0.5)
ax.bar(bins, fit_signal, 1.0, label=r'signal', edgecolor='blue', bottom=background, alpha=0.5)
ax.scatter(bins, observations[0:2], color='black', label='observed')
#ax.set_ylim(0,6)
ax.legend();


CLs_obs, CLs_exp = pyhf.infer.hypotest( 1.0, # null hypothesis
                                        [52.5, 65.] + model.config.auxdata,
                                        model,
                                        return_expected_set = True
                                        )

print(f'      Observed CLs: {CLs_obs}')
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2,3)):
    print(f'Expected CLs({n_sigma:2d} sigma): {expected_value}')

x = model.config.suggested_bounds()[model.config.poi_index]
print(x)


CLs_obs, CLs_exp = pyhf.infer.hypotest(
    1.0, # null hypothesis
    [52.5, 65.] + model.config.auxdata,
    model,
    return_expected_set = True,
    qtilde=True,
)

print(f'      Observed CLs: {CLs_obs}')
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2,3)):
    print(f'Expected CLs({n_sigma:2d} sigma): {expected_value}')

poi_values = np.linspace(0.1, 5, 50)
results = [
    pyhf.infer.hypotest(
        poi_value,
        observations,
        model,
        return_expected_set=True,
        qtilde=True
    )
    for poi_value in poi_values
]

import pyhf.contrib.viz.brazil # not imported by default!
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_size_inches(5, 3)
ax.set_title(u"Hypothesis Tests")
ax.set_xlabel(u"$\mu$")
ax.set_ylabel(u"$\mathrm{CL}_{s}$")

pyhf.contrib.viz.brazil.plot_results(ax, poi_values, results)


observed = np.asarray([h[0] for h in results]).ravel()
expected = np.asarray([h[1][2] for h in results]).ravel()
print(f'Upper limit (obs): mu = {np.interp(0.05, observed[::-1], poi_values[::-1])}')
print(f'Upper limit (exp): mu = {np.interp(0.05, expected[::-1], poi_values[::-1])}')

plt.show()
