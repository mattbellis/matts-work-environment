import pyhf
import numpy as np
import matplotlib.pyplot as plt
import pyhf.contrib.viz.brazil

nbins = 25

MC_signal_events = np.random.normal(5,1.0,200)
MC_background_events = 10*np.random.random(1000)

signal_data = np.histogram(MC_signal_events,bins=nbins)[0]
bkg_data = np.histogram(MC_background_events,bins=nbins)[0]

signal_events = np.random.normal(5,1.0,200)
background_events = 10*np.random.random(1000)

data_events = np.array(signal_events.tolist() + background_events.tolist())
data_sample = np.histogram(data_events,bins=nbins)[0]


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(data_events,bins=nbins)
plt.subplot(1,3,2)
plt.hist(MC_signal_events,bins=nbins)
plt.subplot(1,3,3)
plt.hist(MC_background_events,bins=nbins)

print(data_sample)
print(signal_data)
print(bkg_data)
bkg_uncerts = np.sqrt(bkg_data)
print(bkg_uncerts)

print()

print(type(signal_data))

print("Defining the PDF.......")
pdf = pyhf.simplemodels.hepdata_like(signal_data=signal_data.tolist(), \
                                     bkg_data=bkg_data.tolist(), \
                                     bkg_uncerts=bkg_uncerts.tolist())

print("Fit.......")
CLs_obs, CLs_exp = pyhf.infer.hypotest(1.0, \
                                       data_sample.tolist() + pdf.config.auxdata, \
                                       pdf, \
                                       qtilde=True, \
                                       return_expected=True)
print('Observed: {}, Expected: {}'.format(CLs_obs, CLs_exp))



plt.show()
