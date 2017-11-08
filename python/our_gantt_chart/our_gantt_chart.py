import numpy as np

def read_file(filename):

    data = np.loadtxt(filename,skiprows=1,delimiter=',',dtype=bytes)

    return data
