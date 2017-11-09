import numpy as np
from dateutil.parser import parse
import unicodedata

################################################################################
def read_file(filename):

    data = np.loadtxt(filename,skiprows=1,delimiter=',',dtype=bytes)

    return data


################################################################################
def is_date(x):
    try:
        a = parse(x)
        return a
    except ValueError:
        return False


################################################################################
# From here
# http://pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/
################################################################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
