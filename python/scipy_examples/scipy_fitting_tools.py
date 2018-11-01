import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import fmin_bfgs,fmin_l_bfgs_b

import numpy as np

np.random.seed(0)

################################################################################
class Parameter:

    value = None
    limits = (None,None)

    def __init__(self):
        self.value = None
        self.limits = (None,None)

    def __init__(self,value,lo,hi):
        if lo>value:
            print("value {0} is lower than the lo limit {1}!".format(value,lo))
            exit()
        if hi<value:
            print("value {0} is greater than the hi limit {1}!".format(value,hi))
            exit()
        self.value = value
        self.limits = (lo,hi)

    def __init__(self,value,limits):
        if limits[0]>value:
            print("value {0} is lower than the lo limit {1}!".format(value,limits[0]))
            exit()
        if limits[1]<value:
            print("value {0} is greater than the hi limit {1}!".format(value,limits[1]))
            exit()
        self.value = value
        self.limits = limits



################################################################################
def pretty_print_parameters(params_dictionary):
    for key in params_dictionary:
        if key=="mapping":
            continue
        for k in params_dictionary[key].keys():
            print("{0:20} {1:20} {2}".format(key,k,params_dictionary[key][k].value))
                

################################################################################
def get_numbers(params_dictionary):
    numbers = []
    for key in params_dictionary:
        if key=="mapping":
            continue
        #print(params_dictionary[key])
        for k in params_dictionary[key].keys():
            if k=="number":
                numbers.append(params_dictionary[key][k].value)
    return numbers
################################################################################
################################################################################
def reset_parameters(params_dict,params):
    mapping = params_dict["mapping"]
    for val,m in zip(params,mapping):
        #print(m,val)
        params_dict[m[0]][m[1]].value = val

################################################################################

################################################################################
def pois(mu, k):
    #mu = p[0]
    ret = -mu + k*np.log(mu)
    return ret
################################################################################


####################################################
# Extended maximum likelihood method 
# This is the NLL which will be minimized
####################################################
def errfunc(pars, x, y, fix_or_float=[],params_dictionary=None,pdf=None):
  ret = None

  ##############################################################################
  # Update the dictionary with the new parameter values
  reset_parameters(params_dictionary,pars)
  ##############################################################################

  #print("------- in errfunc --------")
  #print(pars)
  #print("here")

  '''
  newpars = []
  if len(fix_or_float)==0:
    newpars = pars

  elif len(fix_or_float)==len(pars) + 1:
    pcount = 0
    for val in fix_or_float:
      if val is None:
        newpars.append(pars[pcount])
        pcount += 1
      else:
        newpars.append(val)
  '''

  nums = get_numbers(params_dictionary)
  ntot = sum(nums)
  #nsig = pars[2]
  #nbkg = pars[3]
  #ntot = nsig + nbkg
        
  #print("newpars: ")
  #print(newpars)
  ret = (-np.log(pdf(params_dictionary, x, frange=(0,10))) .sum()) - pois(ntot, len(x))
  print("NLL: ",ret)
  
  return ret
################################################################################

