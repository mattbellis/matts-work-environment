from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os

data_comment = '90% C.L. WIMP exclusion limit'
data_label = 'SuperCDMS Soudan LT, 90% C.L.'
data_reference = 'Search for Low Mass WIMPs with SuperCDMS.  <a href = "http://arxiv.org/pdf/1402.7137.pdf">arXiv:1402.7137</a>'
data_values = np.array([2.8910,1.3459e-36, 3.0290,8.7596e-38, 3.1750,1.6686e-38, 3.3270,4.2169e-39, 3.4860,1.6483e-39, 3.6540,7.0771e-40, 3.8290,3.4673e-40, 4.0120,1.9516e-40, 4.2050,1.1925e-40, 4.4060,7.5876e-41, 4.6170,5.2092e-41, 4.8390,3.7392e-41, 5.0710,2.6184e-41, 5.3140,1.8517e-41, 5.5690,1.3108e-41, 5.8360,9.4458e-42, 6.1160,6.9352e-42, 6.4090,5.0358e-42, 6.7160,3.7555e-42, 7.0380,2.7870e-42, 7.3760,1.9905e-42, 7.7290,1.4700e-42, 8.1000,1.1386e-42, 8.4880,9.1732e-43, 8.8950,7.6042e-43, 9.3220,6.4656e-43, 9.7690,5.6277e-43, 10.237,5.0375e-43, 10.728,4.6250e-43, 11.242,4.1328e-43, 11.781,3.7453e-43, 12.346,3.4431e-43, 12.938,3.2033e-43, 13.558,2.9241e-43, 14.208,2.6941e-43, 14.890,2.5103e-43, 15.603,2.3629e-43, 16.352,2.2455e-43, 17.136,2.1517e-43, 17.957,2.0780e-43, 18.818,2.0216e-43, 19.720,1.9796e-43, 20.666,1.9498e-43, 21.657,1.9306e-43, 22.695,1.9193e-43, 23.783,1.9105e-43, 24.924,1.9090e-43, 26.119,1.9152e-43, 27.371,1.9285e-43, 28.683,1.9485e-43, 30.058,1.9746e-43, 31.500,2.0069e-43, 33.010,2.0450e-43, 34.593,2.0885e-43, 36.251,2.1374e-43, 37.989,2.1918e-43, 39.811,2.2516e-43]).reshape(1, -1)
date_of_announcement="2014-03-12"
default_color="Blk"
default_style="Line"
experiment="SuperCDMS"
measurement="Dir"
public="true"
result="Exp"
spin_dependency="SI"
data_values_rescale = np.array([1, 1]).reshape(1, -1)
data_units = ['GeV', 'cm^2']
#x-rescale>1</x-rescale>
#x-units>GeV</x-units>
#y-rescale>1</y-rescale>
#y-units>cm^2</y-units>
year ="2014"
