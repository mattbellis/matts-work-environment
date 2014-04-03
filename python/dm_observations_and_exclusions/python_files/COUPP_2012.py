# Autogenerated with SMOP version 0.23
# /usr/local/bin/smop COUPP_2012.m
from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os

file_name = 'COUPP_2012.limit'
file_status = 'public'
data_label = 'COUPP, SI upper limit (2012)'
data_comment = 'Spin-independent cross section for WIMP-proton elastic scattering'
data_reference = '<a href="http://link.aps.org/doi/10.1103/PhysRevD.86.052001">PhysRevD.86.052001</a>'
private_comment = ''
data_typeoflimit = ['Exclusion Plot', 'SI-Exp']
formdefault_checked = 'checked'
formdefault_color = 'LtB'
formdefault_style = 'Line'
data_units = ['GeV', 'cm^2']
data_values = np.array([10.0911, 9.1708e-41, 10.3065, 8.0541e-41, 10.5265, 6.123e-41, 10.7511, 5.3004e-41, 10.9806, 4.5226e-41, 11.215, 3.8589e-41, 11.4544, 3.2926e-41, 11.6989, 2.7295e-41, 11.9485, 2.3628e-41, 12.2036, 2.0161e-41, 12.4641, 1.6713e-41, 12.7301, 1.4468e-41, 13.0018, 1.3078e-41, 13.2793, 9.9425e-42, 13.5628, 9.118e-42, 13.8521, 7.78e-42, 14.1479, 6.6383e-42, 14.4499, 5.8299e-42, 14.7583, 5.1944e-42, 15.0733, 4.5619e-42, 15.395, 4.0064e-42, 15.7236, 3.5186e-42, 16.0592, 3.0901e-42, 16.402, 2.7138e-42, 16.7521, 2.3492e-42, 17.1096, 2.0632e-42, 17.4748, 1.865e-42, 17.8477, 1.6617e-42, 18.2287, 1.5021e-42, 18.6178, 1.3578e-42, 19.0152, 1.2098e-42, 19.421, 1.1095e-42, 19.8355, 1.0175e-42, 20.2589, 9.1973e-43, 20.6913, 8.4346e-43, 21.133, 7.7352e-43, 21.584, 7.1969e-43, 22.0447, 6.4124e-43, 22.5152, 5.9661e-43, 22.9958, 5.6315e-43, 23.4866, 5.2396e-43, 23.9879, 4.8749e-43, 24.4999, 4.4707e-43, 25.0228, 4.2813e-43, 25.5569, 3.9834e-43, 26.1024, 3.8146e-43, 26.6595, 3.6007e-43, 27.2285, 3.3988e-43, 27.8096, 3.2082e-43, 28.4033, 3.0723e-43, 29.0095, 2.9422e-43, 29.6287, 2.7772e-43, 30.261, 2.6215e-43, 30.907, 2.5104e-43, 31.5667, 2.4041e-43, 32.2404, 2.3023e-43, 32.9285, 2.2047e-43, 33.6314, 2.142e-43, 34.3492, 2.0811e-43, 35.0823, 2.0219e-43, 35.8311, 1.9644e-43, 36.5958, 1.9363e-43, 37.377, 1.8812e-43, 38.1748, 1.8277e-43, 38.9896, 1.7757e-43, 39.8218, 1.7252e-43, 40.6717, 1.7005e-43, 41.5398, 1.6761e-43, 42.4264, 1.6521e-43, 43.332, 1.6285e-43, 44.2569, 1.6051e-43, 45.2015, 1.5822e-43, 46.1663, 1.5595e-43, 47.1517, 1.5372e-43, 48.1581, 1.5151e-43, 49.1859, 1.4934e-43, 50.2358, 1.472e-43, 51.308, 1.451e-43, 52.4031, 1.451e-43, 53.5215, 1.451e-43, 54.664, 1.451e-43, 55.8307, 1.4302e-43, 57.0223, 1.4302e-43, 58.2394, 1.4302e-43, 59.4825, 1.4097e-43, 60.7521, 1.4097e-43, 62.0488, 1.4097e-43, 63.3731, 1.4097e-43, 64.7258, 1.4097e-43, 66.1073, 1.4097e-43, 67.5183, 1.4097e-43, 68.9594, 1.4302e-43, 70.4312, 1.4302e-43, 71.9344, 1.4302e-43, 73.4699, 1.4302e-43, 75.038, 1.451e-43, 76.6396, 1.451e-43, 78.2753, 1.451e-43, 79.9461, 1.472e-43, 81.6524, 1.472e-43, 83.3953, 1.4934e-43, 85.1752, 1.4934e-43, 86.9931, 1.5151e-43, 88.85, 1.5372e-43, 90.7464, 1.5372e-43, 92.6833, 1.5595e-43, 94.6615, 1.5822e-43, 96.682, 1.5822e-43, 98.7456, 1.6051e-43, 100.8532, 1.6285e-43, 103.0058, 1.6285e-43, 105.2043, 1.6521e-43, 107.4498, 1.6761e-43, 109.7431, 1.7005e-43, 112.0856, 1.7252e-43, 114.4779, 1.7503e-43, 116.9213, 1.7757e-43, 119.4169, 1.8015e-43, 121.9657, 1.8015e-43, 124.5689, 1.8277e-43, 127.2277, 1.8543e-43, 129.9433, 1.8812e-43, 132.7168, 1.9085e-43, 135.5495, 1.9363e-43, 138.4426, 1.9644e-43, 141.3975, 1.993e-43, 144.4155, 2.0219e-43, 147.4979, 2.0811e-43, 150.6461, 2.1114e-43, 153.8615, 2.142e-43, 157.1455, 2.1732e-43, 160.4996, 2.2047e-43, 163.9253, 2.2368e-43, 167.4241, 2.2693e-43, 170.9976, 2.3023e-43, 174.6474, 2.3357e-43, 178.375, 2.4041e-43, 182.1822, 2.439e-43, 186.0707, 2.4745e-43, 190.0422, 2.5104e-43, 194.0985, 2.5469e-43, 198.2413, 2.6215e-43, 202.4725, 2.6596e-43, 206.7941, 2.6982e-43, 211.2079, 2.7374e-43, 215.7159, 2.7772e-43, 220.3201, 2.8585e-43, 225.0226, 2.9001e-43, 229.8255, 2.9422e-43, 234.7309, 2.985e-43, 239.741, 3.0723e-43, 244.858, 3.117e-43, 250.0842, 3.1623e-43, 255.422, 3.2082e-43, 260.8737, 3.3022e-43, 266.4418, 3.3501e-43, 272.1287, 3.3988e-43, 277.937, 3.4983e-43, 283.8693, 3.5492e-43, 289.9282, 3.6007e-43, 296.1164, 3.7061e-43, 302.4367, 3.76e-43, 308.8919, 3.8146e-43, 315.4849, 3.8701e-43, 322.2186, 3.9834e-43, 329.096, 4.0413e-43, 336.1202, 4.1596e-43, 343.2943, 4.22e-43, 350.6216, 4.2813e-43, 358.1052, 4.4067e-43, 365.7486, 4.4707e-43, 373.5551, 4.5357e-43, 381.5282, 4.6685e-43, 389.6716, 4.7363e-43, 397.9887, 4.8749e-43, 406.4833, 4.9458e-43, 415.1593, 5.0177e-43, 424.0204, 5.1646e-43, 433.0707, 5.2396e-43, 442.3141, 5.393e-43, 451.7549, 5.4714e-43, 461.3971, 5.5509e-43, 471.2451, 5.7134e-43, 481.3033, 5.7964e-43, 491.5763, 5.9661e-43, 502.0684, 6.0528e-43, 512.7846, 6.1407e-43, 523.7294, 6.3205e-43, 534.9078, 6.4124e-43, 546.3249, 6.5055e-43, 557.9856, 6.696e-43, 569.8952, 6.7933e-43, 582.059, 6.9922e-43, 594.4825, 7.0938e-43, 607.1711, 7.1969e-43, 620.1305, 7.4075e-43, 633.3665, 7.5152e-43, 646.8851, 7.7352e-43, 660.6922, 7.8476e-43, 674.7939, 7.9616e-43, 689.1967, 8.1947e-43, 703.9069, 8.3138e-43, 718.931, 8.5572e-43, 734.2758, 8.6815e-43, 749.9482, 8.8077e-43, 765.955, 9.0655e-43, 782.3035, 9.1973e-43, 799.001, 9.4665e-43, 816.0548, 9.6041e-43, 833.4726, 9.7437e-43, 851.2622, 1.0029e-42, 869.4315, 1.0175e-42, 887.9886, 1.0322e-42, 906.9418, 1.0625e-42, 926.2995, 1.0779e-42, 946.0704, 1.1095e-42, 966.2633, 1.1419e-42, 981.6903, 1.1585e-42]).reshape(1, -1)
data_values_rescale = np.array([1, 1]).reshape(1, -1)
param_halodensity = 0.3
param_halovelocity = 230
param_escvelocity = 544
param_earthvelocity = 244