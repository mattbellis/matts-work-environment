# Autogenerated with SMOP version 0.23
# /usr/local/bin/smop XMASS13.m
from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os

file_name = 'XMASS13.limit'
data_label = 'XMASS(2013) 90\n % U.L.'
data_comment = 'Sigma normalized to WIMP-Nucleon'
data_reference = '33rd INTERNATIONAL COSMIC RAY CONFERENCE (2013), <a href = "http://www.cbpf.br/~icrc2013/papers/icrc2013-0131.pdf">YAMASHITA,ICRC 2013</a>'
data_typeoflimit = ['Sensitivity Goal', 'SI-Exp']
formdefault_checked = ''
formdefault_color = 'Mag'
formdefault_style = 'Dot'
data_units = ['GeV', 'cm^2']
data_values_rescale = np.array([1, 1]).reshape(1, -1)
data_values = np.array([6.5975, 1.0199e-39, 6.5975, 9.9509e-40, 6.6148, 9.7095e-40, 6.632, 9.4739e-40, 6.6494, 9.2441e-40, 6.6667, 9.0199e-40, 6.6842, 8.8011e-40, 6.7016, 8.5876e-40, 6.7017, 8.3787e-40, 6.7191, 8.1754e-40, 6.7367, 7.9771e-40, 6.7543, 7.7836e-40, 6.772, 7.5948e-40, 6.772, 7.41e-40, 6.7896, 7.2303e-40, 6.8074, 7.0549e-40, 6.8252, 6.8838e-40, 6.8431, 6.7168e-40, 6.8609, 6.5538e-40, 6.861, 6.3944e-40, 6.8789, 6.2393e-40, 6.8969, 6.0879e-40, 6.9149, 5.9403e-40, 6.9329, 5.7962e-40, 6.933, 5.6552e-40, 6.951, 5.518e-40, 6.9691, 5.3841e-40, 6.9875, 5.2535e-40, 7.0057, 5.1261e-40, 7.024, 5.0017e-40, 7.0423, 5.0021e-40, 7.0607, 5.0024e-40, 7.0607, 4.8807e-40, 7.0791, 4.8811e-40, 7.0976, 4.8814e-40, 7.0976, 4.7627e-40, 7.1161, 4.763e-40, 7.1347, 4.6475e-40, 7.1533, 4.6478e-40, 7.172, 4.535e-40, 7.1907, 4.425e-40, 7.2095, 4.4253e-40, 7.2283, 4.318e-40, 7.2471, 4.3183e-40, 7.2661, 4.2135e-40, 7.2851, 4.1113e-40, 7.3041, 4.1116e-40, 7.3231, 4.0119e-40, 7.3423, 3.9145e-40, 7.3614, 3.9148e-40, 7.3806, 3.8199e-40, 7.3999, 3.7272e-40, 7.4192, 3.7274e-40, 7.4386, 3.637e-40, 7.458, 3.5488e-40, 7.4775, 3.549e-40, 7.4969, 3.463e-40, 7.5166, 3.379e-40, 7.5362, 3.3792e-40, 7.5558, 3.2972e-40, 7.5756, 3.2172e-40, 7.5953, 3.2175e-40, 7.6152, 3.1394e-40, 7.6351, 3.0632e-40, 7.655, 3.0635e-40, 7.675, 2.9891e-40, 7.6949, 2.9894e-40, 7.7151, 2.9168e-40, 7.7351, 2.8461e-40, 7.7553, 2.8463e-40, 7.7757, 2.7772e-40, 7.796, 2.7099e-40, 7.8163, 2.7101e-40, 7.8367, 2.6443e-40, 7.8572, 2.5802e-40, 7.8777, 2.5803e-40, 7.8982, 2.5178e-40, 7.9189, 2.4567e-40, 7.9395, 2.4568e-40, 7.9603, 2.3972e-40, 7.9811, 2.3391e-40, 8.0019, 2.3393e-40, 8.0228, 2.2825e-40, 8.0437, 2.2271e-40, 8.0647, 2.2273e-40, 8.0858, 2.1733e-40, 8.1069, 2.1205e-40, 8.128, 2.1207e-40, 8.1493, 2.0693e-40, 8.1705, 2.0694e-40, 8.1918, 2.0192e-40, 8.2132, 1.9702e-40, 8.2346, 1.9704e-40, 8.2562, 1.9226e-40, 8.2777, 1.8759e-40, 8.2993, 1.876e-40, 8.321, 1.8305e-40, 8.3427, 1.7861e-40, 8.3645, 1.7863e-40, 8.3862, 1.7429e-40, 8.4082, 1.7006e-40, 8.4301, 1.7008e-40, 8.4522, 1.6595e-40, 8.4742, 1.6193e-40, 8.4963, 1.6194e-40, 8.5185, 1.5801e-40, 8.5408, 1.5418e-40, 8.5631, 1.5419e-40, 8.5854, 1.5045e-40, 8.6078, 1.468e-40, 8.6303, 1.4681e-40, 8.6528, 1.4325e-40, 8.6754, 1.4325e-40, 8.6981, 1.3978e-40, 8.7208, 1.3639e-40, 8.7435, 1.364e-40, 8.7663, 1.3309e-40, 8.7892, 1.2986e-40, 8.8122, 1.2987e-40, 8.8352, 1.2672e-40, 8.8582, 1.2365e-40, 8.8814, 1.2365e-40, 8.9045, 1.2065e-40, 8.9278, 1.1773e-40, 8.951, 1.1774e-40, 8.9745, 1.1488e-40, 8.9979, 1.1209e-40, 9.0214, 1.0937e-40, 9.0449, 1.0938e-40, 9.0686, 1.0673e-40, 9.0922, 1.0674e-40, 9.116, 1.0415e-40, 9.1398, 1.0162e-40, 9.1636, 1.0163e-40, 9.1875, 9.9162e-41, 9.2115, 9.6757e-41, 9.2355, 9.6763e-41, 9.2597, 9.4416e-41, 9.2839, 9.2126e-41, 9.3081, 9.2132e-41, 9.3324, 8.9897e-41, 9.3567, 8.9903e-41, 9.3811, 8.7722e-41, 9.4057, 8.5595e-41, 9.4302, 8.3518e-41, 9.4548, 8.3524e-41, 9.4795, 8.1498e-41, 9.5043, 7.9521e-41, 9.529, 7.9526e-41, 9.5539, 7.7597e-41, 9.5789, 7.5715e-41, 9.6038, 7.572e-41, 9.629, 7.3883e-41, 9.6541, 7.3888e-41, 9.6793, 7.2096e-41, 9.7046, 7.0347e-41, 9.7299, 7.0352e-41, 9.7553, 6.8645e-41, 9.7808, 6.698e-41, 9.8063, 6.6985e-41, 9.8319, 6.536e-41, 9.8576, 6.3774e-41, 9.8833, 6.3779e-41, 9.9091, 6.2232e-41, 9.935, 6.0722e-41, 9.9609, 6.0726e-41, 9.9868, 5.9253e-41, 10.0129, 5.9257e-41, 10.0391, 5.782e-41, 10.0652, 5.7824e-41, 10.0915, 5.6421e-41, 10.1178, 5.6425e-41, 10.1442, 5.6429e-41, 10.1707, 5.506e-41, 10.1972, 5.5064e-41, 10.2238, 5.5068e-41, 10.2505, 5.3732e-41, 10.2772, 5.3736e-41, 10.304, 5.3739e-41, 10.3309, 5.2436e-41, 10.3578, 5.2439e-41, 10.3849, 5.2443e-41, 10.412, 5.1171e-41, 10.4391, 5.1174e-41, 10.4664, 5.1178e-41, 10.4937, 4.9937e-41, 10.521, 4.994e-41, 10.5485, 4.9943e-41, 10.576, 4.8732e-41, 10.6036, 4.8735e-41, 10.6312, 4.8739e-41, 10.659, 4.7556e-41, 10.6867, 4.756e-41, 10.7147, 4.7563e-41, 10.7426, 4.6409e-41, 10.7706, 4.6412e-41, 10.7987, 4.6416e-41, 10.8269, 4.529e-41, 10.8552, 4.5293e-41, 10.8835, 4.5296e-41, 10.9119, 4.4197e-41, 10.9403, 4.42e-41, 10.9687, 4.4203e-41, 10.9975, 4.3131e-41, 11.0262, 4.3134e-41, 11.0549, 4.3137e-41, 11.0838, 4.2091e-41, 11.1127, 4.2094e-41, 11.1417, 4.2096e-41, 11.1708, 4.1075e-41, 11.1999, 4.1078e-41, 11.2291, 4.1081e-41, 11.2584, 4.0084e-41, 11.2878, 4.0087e-41, 11.3172, 4.009e-41, 11.3467, 3.9118e-41, 11.3763, 3.912e-41, 11.406, 3.9123e-41, 11.4358, 3.8174e-41, 11.4656, 3.8177e-41, 11.4955, 3.8179e-41, 11.5255, 3.7253e-41, 11.5554, 3.7256e-41, 11.5857, 3.7258e-41, 11.6159, 3.6354e-41, 11.6461, 3.6357e-41, 11.6766, 3.6359e-41, 11.7071, 3.5477e-41, 11.7376, 3.548e-41, 11.7682, 3.5482e-41, 11.7989, 3.4622e-41, 11.8296, 3.4624e-41, 11.8605, 3.4626e-41, 11.8914, 3.4629e-41, 11.9224, 3.3789e-41, 11.9536, 3.3791e-41, 11.9847, 3.3793e-41, 12.0161, 3.2974e-41, 12.0474, 3.2976e-41, 12.0788, 3.2978e-41, 12.1103, 3.2981e-41, 12.1419, 3.2181e-41, 12.1736, 3.2183e-41, 12.2053, 3.2185e-41, 12.2371, 3.2187e-41, 12.269, 3.2189e-41, 12.3011, 3.1409e-41, 12.3331, 3.1411e-41, 12.3653, 3.1413e-41, 12.3975, 3.1415e-41, 12.4298, 3.1417e-41, 12.4622, 3.142e-41, 12.4948, 3.0657e-41, 12.5272, 3.0659e-41, 12.56, 3.0662e-41, 12.5927, 3.0664e-41, 12.6256, 3.0666e-41, 12.6585, 3.0668e-41, 12.6915, 2.9924e-41, 12.7246, 2.9926e-41, 12.7578, 2.9928e-41, 12.7911, 2.993e-41, 12.8244, 2.9932e-41, 12.8578, 2.9206e-41, 12.8914, 2.9208e-41, 12.925, 2.921e-41, 12.9586, 2.9212e-41, 12.9925, 2.9214e-41, 13.0264, 2.9216e-41, 13.0604, 2.8508e-41, 13.0944, 2.851e-41, 13.1286, 2.8512e-41, 13.1628, 2.8514e-41, 13.1971, 2.8516e-41, 13.2315, 2.8518e-41, 13.2661, 2.7826e-41, 13.3007, 2.7828e-41, 13.3353, 2.783e-41, 13.37, 2.7832e-41, 13.4049, 2.7833e-41, 13.4399, 2.7835e-41, 13.475, 2.716e-41, 13.5101, 2.7162e-41, 13.5453, 2.7164e-41, 13.5806, 2.7166e-41, 13.616, 2.7168e-41, 13.6515, 2.717e-41, 13.6872, 2.6511e-41, 13.7229, 2.6512e-41, 13.7587, 2.6514e-41, 13.7945, 2.6516e-41, 13.8305, 2.6518e-41, 13.8666, 2.5875e-41, 13.9028, 2.5876e-41, 13.939, 2.5878e-41, 13.9752, 2.588e-41, 14.0118, 2.5882e-41, 14.0483, 2.5884e-41, 14.085, 2.5256e-41, 14.1217, 2.5257e-41, 14.1585, 2.5259e-41, 14.1954, 2.5261e-41, 14.2324, 2.5263e-41, 14.2695, 2.5264e-41, 14.3068, 2.4652e-41, 14.3441, 2.4653e-41, 14.3815, 2.4655e-41, 14.419, 2.4657e-41, 14.4566, 2.4658e-41, 14.4942, 2.466e-41, 14.5321, 2.4062e-41, 14.5699, 2.4064e-41, 14.608, 2.4065e-41, 14.646, 2.4067e-41, 14.6842, 2.4069e-41, 14.7226, 2.3485e-41, 14.761, 2.3486e-41, 14.7993, 2.3488e-41, 14.838, 2.349e-41, 14.8767, 2.3491e-41, 14.9155, 2.3493e-41, 14.9544, 2.2923e-41, 14.9933, 2.2925e-41, 15.0325, 2.2926e-41, 15.0717, 2.2928e-41, 15.111, 2.2929e-41, 15.1504, 2.2931e-41, 15.19, 2.2375e-41, 15.2296, 2.2376e-41, 15.2692, 2.2378e-41, 15.3091, 2.2379e-41, 15.349, 2.2381e-41, 15.3891, 2.1838e-41, 15.4292, 2.184e-41, 15.4694, 2.1841e-41, 15.5097, 2.1843e-41, 15.5501, 2.1844e-41, 15.5907, 2.1846e-41, 15.6314, 2.1316e-41, 15.6722, 2.1317e-41, 15.713, 2.1319e-41, 15.754, 2.132e-41, 15.795, 2.1322e-41, 15.8362, 2.1323e-41, 15.8776, 2.0806e-41, 15.919, 2.0807e-41, 15.9605, 2.0809e-41, 16.0021, 2.081e-41, 16.0438, 2.0812e-41, 16.0857, 2.0307e-41, 16.1276, 2.0308e-41, 16.1697, 2.031e-41, 16.2118, 2.0311e-41, 16.2541, 2.0312e-41, 16.2964, 2.0314e-41, 16.339, 1.9821e-41, 16.3816, 1.9822e-41, 16.4243, 1.9824e-41, 16.4671, 1.9825e-41, 16.5101, 1.9827e-41, 16.5531, 1.9828e-41, 16.5963, 1.9347e-41, 16.6396, 1.9348e-41, 16.683, 1.935e-41, 16.7265, 1.9351e-41, 16.7701, 1.9352e-41, 16.8138, 1.8883e-41, 16.8577, 1.8884e-41, 16.9016, 1.8885e-41, 16.9457, 1.8887e-41, 16.9899, 1.8888e-41, 17.0342, 1.8889e-41, 17.0787, 1.8431e-41, 17.1232, 1.8432e-41, 17.1678, 1.8434e-41, 17.2126, 1.8435e-41, 17.2574, 1.8436e-41, 17.3024, 1.8438e-41, 17.3476, 1.799e-41, 17.3928, 1.7992e-41, 17.4382, 1.7993e-41, 17.4836, 1.7994e-41, 17.5292, 1.7995e-41, 17.575, 1.7559e-41, 17.6208, 1.756e-41, 17.6667, 1.7561e-41, 17.7128, 1.7562e-41, 17.759, 1.7564e-41, 17.8053, 1.7565e-41, 17.8518, 1.7139e-41, 17.8982, 1.714e-41, 17.9449, 1.7141e-41, 17.9918, 1.7142e-41, 18.0387, 1.7144e-41, 18.0857, 1.7145e-41, 18.1328, 1.7146e-41, 18.1802, 1.673e-41, 18.2276, 1.6731e-41, 18.2751, 1.6732e-41, 18.3227, 1.6733e-41, 18.3705, 1.6735e-41, 18.4184, 1.6736e-41, 18.4664, 1.6737e-41, 18.5145, 1.6738e-41, 18.5628, 1.6739e-41, 18.6112, 1.674e-41, 18.6597, 1.6742e-41, 18.7083, 1.6743e-41, 18.757, 1.6744e-41, 18.806, 1.6745e-41, 18.8551, 1.6339e-41, 18.9043, 1.634e-41, 18.9536, 1.6341e-41, 19.003, 1.6342e-41, 19.0525, 1.6343e-41, 19.1022, 1.6345e-41, 19.152, 1.6346e-41, 19.2019, 1.6347e-41, 19.2519, 1.6348e-41, 19.3021, 1.6349e-41, 19.3524, 1.635e-41, 19.4029, 1.6351e-41, 19.4535, 1.6352e-41, 19.5042, 1.6354e-41, 19.555, 1.6355e-41, 19.6061, 1.5958e-41, 19.6572, 1.5959e-41, 19.7085, 1.596e-41, 19.7598, 1.5961e-41, 19.8113, 1.5962e-41, 19.863, 1.5964e-41, 19.9148, 1.5965e-41, 19.9667, 1.5966e-41, 20.0188, 1.5578e-41]).reshape(1, -1)
param_halodensity = 0.3
#GeV/cm^3
param_halovelocity = 220
#km/s
param_escvelocity = 650
#km/s
