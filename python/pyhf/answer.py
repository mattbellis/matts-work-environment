import pyhf
import pyhf.contrib.viz.brazil

import numpy as np
import matplotlib.pylab as plt


#    - Get the uncertainties on the best fit signal strength
#    - Calculate an 95% CL upper limit on the signal strength

# tag = "ORIGINAL"
tag = "MODIFIED"


def plot_hist(ax, bins, data, bottom=0, color=None, label=None):
    bin_width = bins[1] - bins[0]
    bin_leftedges = bins[:-1]
    bin_centers = [edge + bin_width / 2.0 for edge in bin_leftedges]
    ax.bar(
        bin_centers, data, bin_width, bottom=bottom, alpha=0.5, color=color, label=label
    )


def plot_data(ax, bins, data, label="Data"):
    bin_width = bins[1] - bins[0]
    bin_leftedges = bins[:-1]
    bin_centers = [edge + bin_width / 2.0 for edge in bin_leftedges]
    ax.scatter(bin_centers, data, color="black", label=label)


def invert_interval(test_mus, hypo_tests, test_size=0.05):
    # This will be taken care of in v0.5.3
    cls_obs = np.array([test[0] for test in hypo_tests]).flatten()
    cls_exp = [
        np.array([test[1][idx] for test in hypo_tests]).flatten() for idx in range(5)
    ]
    crossing_test_stats = {"exp": [], "obs": None}
    for cls_exp_sigma in cls_exp:
        crossing_test_stats["exp"].append(
            np.interp(
                test_size, list(reversed(cls_exp_sigma)), list(reversed(test_mus))
            )
        )
    crossing_test_stats["obs"] = np.interp(
        test_size, list(reversed(cls_obs)), list(reversed(test_mus))
    )
    return crossing_test_stats


def main():
    np.random.seed(0)
    pyhf.set_backend("numpy", "minuit")

    observable_range = [0.0, 10.0]
    bin_width = 0.5
    _bins = np.arange(observable_range[0], observable_range[1] + bin_width, bin_width)
    print("nbins =  {0}".format(len(_bins)))

    #n_bkg = 2000
    n_bkg = 20000
    # n_signal = int(np.sqrt(n_bkg))
    n_signal = 200

    # Generate simulation
    bkg_simulation = 10 * np.random.random(n_bkg)
    signal_simulation = np.random.normal(5, 1.0, n_signal)

    bkg_sample, _ = np.histogram(bkg_simulation, bins=_bins)
    signal_sample, _ = np.histogram(signal_simulation, bins=_bins)

    # Generate observations
    signal_events = np.random.normal(5, 1.0, int(n_signal * 0.8))
    #bkg_events = 10 * np.random.random(n_bkg - 300)
    bkg_events = 10 * np.random.random(2000)

    observed_events = np.array(signal_events.tolist() + bkg_events.tolist())
    observed_sample, _ = np.histogram(observed_events, bins=_bins)

    # Visualize the simulation and observations
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)

    plot_hist(ax, _bins, bkg_sample, label="Background")
    plot_hist(ax, _bins, signal_sample, bottom=bkg_sample, label="Signal")
    plot_data(ax, _bins, observed_sample)
    ax.legend(loc="best")
    ax.set_ylim(top=np.max(observed_sample) * 1.4)
    ax.set_xlabel("Observable")
    ax.set_ylabel("Count")
    fig.savefig("components_{0}.png".format(tag))

    # Build the model
    bkg_uncerts = np.sqrt(bkg_sample)
    """
    model = pyhf.simplemodels.hepdata_like(
        signal_data=signal_sample.tolist(),
        bkg_data=bkg_sample.tolist(),
        bkg_uncerts=bkg_uncerts.tolist(),
    )
    model = pyhf.Model(spec)
    """

    spec = {'channels': [{'name': 'singlechannel',
               'samples': [
                   {'name': 'signal', 'data': signal_sample.tolist(),
                            'modifiers': [{'name': 'mu', 'type': 'normfactor', 'data': None}]},
                   {'name': 'background', 'data': bkg_sample.tolist(),
                            'modifiers': [{'name': 'bkgnorm',
                            'type': 'normfactor',
                            'data': None}]
                            #data': bkg_sample.tolist()}]
                            }
                   ]
              }
       ]
     }

    print(spec)

    print("Adding the model....")
    model = pyhf.Model(spec)
    print("Added the model....")


    data = pyhf.tensorlib.astensor(observed_sample.tolist() + model.config.auxdata)

    # Perform inference
    fit_result = pyhf.infer.mle.fit(data, model, return_uncertainties=True)
    bestfit_pars, par_uncerts = fit_result.T
    print("bestfit_pars")
    print(len(bestfit_pars))
    print(bestfit_pars)
    print()
    print(
        f"best fit parameters:\
        \n * signal strength: {bestfit_pars[0]} +/- {par_uncerts[0]}\
        \n * nuisance parameters: {bestfit_pars[1:]}\
        \n * nuisance parameter uncertainties: {par_uncerts[1:]}"
    )

    # Visualize the results
    fit_bkg_sample = []
    for w, b in zip(bestfit_pars[1:], bkg_sample):
        fit_bkg_sample.append(w * b)

    fit_signal_sample = bestfit_pars[0] * np.array(signal_sample)

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)

    plot_hist(ax, _bins, fit_bkg_sample, label="Background")
    plot_hist(ax, _bins, fit_signal_sample, bottom=fit_bkg_sample, label="Signal")
    plot_data(ax, _bins, observed_sample)
    ax.legend(loc="best")
    ax.set_ylim(top=np.max(observed_sample) * 1.4)
    ax.set_xlabel("Observable")
    ax.set_ylabel("Count")
    fig.savefig("components_after_fit_{0}.png".format(tag))

    # Perform hypothesis test scan
    _start = 0.0
    _stop = 5
    _step = 0.1
    poi_tests = np.arange(_start, _stop + _step, _step)

    print("\nPerforming hypothesis tests\n")
    hypo_tests = [
        pyhf.infer.hypotest(
            mu_test,
            data,
            model,
            return_expected_set=True,
            return_test_statistics=True,
            qtilde=True,
        )
        for mu_test in poi_tests
    ]

    # Upper limits on signal strength
    results = invert_interval(poi_tests, hypo_tests)

    print(f"Observed Limit on µ: {results['obs']:.2f}")
    print("-----")
    for idx, n_sigma in enumerate(np.arange(-2, 3)):
        print(
            "Expected {}Limit on µ: {:.3f}".format(
                "       " if n_sigma == 0 else "({} σ) ".format(n_sigma),
                results["exp"][idx],
            )
        )

    # Visualize the "Brazil band"
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)

    ax.set_title("Hypothesis Tests")
    ax.set_ylabel(r"$\mathrm{CL}_{s}$")
    ax.set_xlabel(r"$\mu$")

    pyhf.contrib.viz.brazil.plot_results(ax, poi_tests, hypo_tests)
    fig.savefig("brazil_band_{0}.png".format(tag))


if __name__ == "__main__":
    main()
