python read_in_and_fit.py data/sprvars_OffPeak_Run2.root --initial-values-file config_test_continuum_three_gaussians_fixed_means.py --opposite-sign --tag Continuum_fixed_means --batch
python read_in_and_fit.py data/sprvars_OffPeak_Run2.root --initial-values-file config_test_continuum_three_gaussians_float_means.py --opposite-sign --tag Continuum_float_means --batch
python read_in_and_fit.py data/sprvars_OffPeak_Run2.root --initial-values-file config_test_continuum_three_gaussians_fixed_means.py --same-sign --tag Continuum_fixed_means --batch
python read_in_and_fit.py data/sprvars_OffPeak_Run2.root --initial-values-file config_test_continuum_three_gaussians_float_means.py --same-sign --tag Continuum_float_means --batch
