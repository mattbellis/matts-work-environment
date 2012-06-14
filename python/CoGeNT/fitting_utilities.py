import numpy as np
import scipy as sp
import scipy.stats as stats

################################################################################
# Extended maximum likelihood function for minuit
################################################################################
def cosmogenic_peaks(means,sigmas,numbers):

    npeaks = len(means)

    pdfs = []

    for mean,sigma,number in zip(means,sigmas,numbers):
        pdf = sp.stats.norm(loc=mean,scale=sigma)
        pdfs.append(pdf)

    return pdfs

################################################################################
# Extended maximum likelihood function for minuit
################################################################################
def emlf_minuit(p):

    norm_func = (pdf_bmixing(mc_data[i],pars)).sum()/len(mc_data)

    num = num # Number of events in fit.

    ret = (-np.log(pdf_bmixing(data[i],pars) / norm_func).sum()) - pois(num,len(data[i]))

    return ret

