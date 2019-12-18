import pyhf
pdf = pyhf.simplemodels.hepdata_like(signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0])
CLs_obs, CLs_exp = pyhf.infer.hypotest(1.0, [51, 48] + pdf.config.auxdata, pdf, return_expected=True)
print('Observed: {}, Expected: {}'.format(CLs_obs, CLs_exp))
