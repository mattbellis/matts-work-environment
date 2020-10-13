import pyhf
import numpy as np
import matplotlib.pyplot as plt

nbins = 15

# Generate a background and signal MC sample`
MC_signal_events = np.random.normal(5,1.0,200)
MC_background_events = 10*np.random.random(1000)

signal_data = np.histogram(MC_signal_events,bins=nbins)[0]
bkg_data = np.histogram(MC_background_events,bins=nbins)[0]


# Generate an observed dataset with a slightly different
# number of events
signal_events = np.random.normal(5,1.0,180)
background_events = 10*np.random.random(1050)

observed_events = np.array(signal_events.tolist() + background_events.tolist())
observed_sample = np.histogram(observed_events,bins=nbins)[0]


# Plot these samples, if you like
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(observed_events,bins=nbins,label='Observations')
plt.legend()
plt.subplot(1,3,2)
plt.hist(MC_signal_events,bins=nbins,label='MC signal')
plt.legend()
plt.subplot(1,3,3)
plt.hist(MC_background_events,bins=nbins,label='MC background')
plt.legend()

# Use a very naive estimate of the background
# uncertainties
bkg_uncerts = np.sqrt(bkg_data)

print("Defining the PDF.......")
pdf = pyhf.simplemodels.hepdata_like(signal_data=signal_data.tolist(), \
                                     bkg_data=bkg_data.tolist(), \
                                     bkg_uncerts=bkg_uncerts.tolist())

print("Fit.......")
data = pyhf.tensorlib.astensor(observed_sample.tolist() + pdf.config.auxdata)

bestfit_pars, twice_nll = pyhf.infer.mle.fit(data, pdf, return_fitted_val=True)

print(bestfit_pars)
print(twice_nll)

plt.show()
