import pyhf
import json

import numpy as np
import matplotlib.pylab as plt

pyhf.set_backend('numpy', 'minuit')

np.random.seed(0)

nbins = 15

# Generate a background and signal MC sample`
#MC_signal_events = np.random.normal(5,1.0,200)
#MC_signal_events = np.random.normal(5,1.0,100000)
#MC_background_events = 10*np.random.random(1000)

MC_signal_events = 10.0 - np.random.lognormal(0,1.0,100000)
MC_background_events = np.random.lognormal(0,1.0,1000)

signal = np.histogram(MC_signal_events,bins=nbins,range=(0,10))[0]
background = np.histogram(MC_background_events,bins=nbins,range=(0,10))[0]

# Generate an observed dataset with a slightly different
# number of events
#signal_events = np.random.normal(5,1.0,180)
signal_events = 10.0-np.random.lognormal(0,1.0,180)
background_events = np.random.lognormal(0,1.0,1100)

observed_events = np.array(signal_events.tolist() + background_events.tolist())
observed_sample = np.histogram(observed_events,bins=nbins,range=(0,10))[0]

print("observed_sample")
print(observed_sample)
print()
print('signal')
print(signal)
print()
print('background')
print(background)
print()


# Plot these samples, if you like
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(observed_events,bins=nbins,range=(0,10),label='Observations')
plt.legend()
plt.xlim(0,10)
plt.subplot(1,3,2)
plt.hist(MC_signal_events,bins=nbins,range=(0,10),label='MC signal')
plt.legend()
plt.xlim(0,10)
plt.subplot(1,3,3)
plt.hist(MC_background_events,bins=nbins,range=(0,10),label='MC background')
plt.legend()
plt.xlim(0,10)

# Use a very naive estimate of the background
# uncertainties
bkg_uncerts = np.sqrt(background)


model = pyhf.simplemodels.hepdata_like(signal_data=signal.tolist(),
                                        bkg_data=background.tolist(), 
                                        bkg_uncerts=bkg_uncerts.tolist())
#model

print(f'  channels: {model.config.channels}')
print(f'     nbins: {model.config.channel_nbins}')
print(f'   samples: {model.config.samples}')
print(f' modifiers: {model.config.modifiers}')
print(f'parameters: {model.config.parameters}')
print(f'  nauxdata: {model.config.nauxdata}')
print(f'   auxdata: {model.config.auxdata}')

print("\nmodel.config.auxdata")
print(model.config.auxdata)

#print(model.expected_data([1.0, 1.0, 1.0]))
#
#print(model.expected_data([1.0, 1.0, 1.0], include_auxdata=False))
#
#print(model.expected_actualdata([1.0, 1.0, 1.0]))
#
#print(model.expected_auxdata([1.0, 1.0, 1.0]))

print("\nmodel.config.suggested_init")
print(model.config.suggested_init())

print("\nmodel.config.suggested_bounds")
print(model.config.suggested_bounds())

print("\nmodel.config.suggested_fixed")
print(model.config.suggested_fixed())

init_pars = model.config.suggested_init()
print("\nmodel.expected_actualdata(init_pars)")
print(model.expected_actualdata(init_pars))

print("\nmodel.config.poi_index")
print(model.config.poi_index)

bkg_pars = [*init_pars] # my clever way to "copy" the list by value
bkg_pars[model.config.poi_index] = 0
model.expected_actualdata(bkg_pars)

#observations = [52.5, 65.] + model.config.auxdata
observations = observed_sample.tolist() + model.config.auxdata

bins = np.arange(0,len(background))

fig, ax = plt.subplots()
ax.bar(bins, background, 1.0, label=r'background', edgecolor='red', alpha=0.5)
ax.bar(bins, signal, 1.0, label=r'signal', edgecolor='blue', bottom=background, alpha=0.5)
ax.scatter(bins, observed_sample, color='black', label='observed')
#ax.set_ylim(0,6)
ax.legend();

################################################################################

x = model.logpdf(pars = bkg_pars, data = observations)
print('\nmodel.logpdf - bkg_pars')
print(x)

x = model.logpdf( pars = init_pars, data = observations)
print('\nmodel.logpdf - init_pars')
print(x)

fit_results = pyhf.infer.mle.fit(data = observations, pdf = model, return_uncertainties=True, return_fitted_val=True)
print('\nfit_results')
print(fit_results)
print()
print("mu: {0}".format(fit_results[0]))
for g in fit_results[0][1:]:
    print("gammas: {}".format(g))

fit_background = []
for w,b in zip(fit_results[0][1:],background):
    fit_background.append(w[0]*b)

fit_signal = fit_results[0][0][0]*np.array(signal)

fig, ax = plt.subplots()
ax.bar(bins, fit_background, 1.0, label=r'background', edgecolor='red', alpha=0.5)
ax.bar(bins, fit_signal, 1.0, label=r'signal', edgecolor='blue', bottom=fit_background, alpha=0.5)
ax.scatter(bins, observed_sample, color='black', label='observed')
#ax.set_ylim(0,6)
ax.legend();

######### FIT TO background only #################
result, twice_nll = pyhf.infer.mle.fixed_poi_fit(
            0.0,
            data=observations,
            pdf=model,
            return_uncertainties=True,
            return_fitted_val=True
        )

bestfit_pars, errors = result.T
print()
print("Background only")
print(bestfit_pars)
print()
print(errors)
print()


CLs_obs, CLs_exp = pyhf.infer.hypotest( 1.0, # null hypothesis
                                        observed_sample.tolist() + model.config.auxdata,
                                        model,
                                        return_expected_set = True
                                        )

print(f'      Observed CLs: {CLs_obs}')
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2,3)):
    print(f'Expected CLs({n_sigma:2d} sigma): {expected_value}')
print()

x = model.config.suggested_bounds()[model.config.poi_index]
print('\nmodel.config.suggested_bounds()[model.config.poi_index]')
print(x)
print()


CLs_obs, CLs_exp = pyhf.infer.hypotest(
    1.0, # null hypothesis
    observed_sample.tolist() + model.config.auxdata,
    model,
    return_expected_set = True,
    qtilde=True,
)

print(f'      Observed CLs: {CLs_obs}')
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2,3)):
    print(f'Expected CLs({n_sigma:2d} sigma): {expected_value}')

#poi_values = np.linspace(0.1, 5, 50)
# For when we have lots of signal
poi_values = np.linspace(0.1*fit_results[0][0][0], 5*fit_results[0][0][0], 50)
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


print()
observed = np.asarray([h[0] for h in results]).ravel()
expected = np.asarray([h[1][2] for h in results]).ravel()
print(f'Upper limit (obs): mu = {np.interp(0.05, observed[::-1], poi_values[::-1])}')
print(f'Upper limit (exp): mu = {np.interp(0.05, expected[::-1], poi_values[::-1])}')

plt.show()
