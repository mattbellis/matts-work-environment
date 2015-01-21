# Autogenerated with SMOP version 0.23
# /usr/local/bin/smop CDMS_091217_Ge_comb.m
from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os

file_name = 'CDMS_091217_Ge_comb.limit'
file_status = 'public'
data_label = 'CDMS: Soudan 2004-2009 Ge'
data_comment = 'Cross-section data for scalar interaction normalised to nucleon'
data_reference = 'Announced 17Dec2009, Science 327, p. 1619, <a href = "http://arxiv.org/abs/0912.3592">arXiv:0912.3592</a>'
private_comment = 'c89 reanalysis (5 keV) + c34 (10 keV) + c58 (10 keV)'
data_typeoflimit = ['Exclusion Plot', 'SI-Exp']
formdefault_checked = 'checked'
formdefault_color = 'Bk'
formdefault_style = 'Line'
data_units = ['GeV', 'cm^2']
data_values = np.array([6.61321405, 1.19944028e-38, 6.78348413, 1.22517091e-39, 6.95813816, 9.53789637e-40, 7.13728899, 5.01694447e-40, 7.32105241, 3.02332412e-40, 7.50954718, 1.75802951e-40, 7.70289511, 1.10815471e-40, 7.90122116, 7.46238377e-41, 8.1046535, 5.25422416e-41, 8.31332361, 4.49334452e-41, 8.52736633, 4.46187856e-41, 8.74692, 3.88072134e-41, 8.97212651, 3.21145564e-41, 9.2031314, 2.55535174e-41, 9.44008396, 1.98923886e-41, 9.68313732, 1.514099e-41, 9.93244857, 1.04590499e-41, 10.1881788, 7.49926054e-42, 10.4504933, 5.56552907e-42, 10.7195617, 4.24216815e-42, 10.9955577, 3.38700183e-42, 11.2786598, 2.87541472e-42, 11.5690508, 2.45967036e-42, 11.8669186, 2.12331501e-42, 12.1724556, 1.93072846e-42, 12.4858592, 1.493489e-42, 12.8073319, 1.28479922e-42, 13.1370817, 1.17651656e-42, 13.4753215, 1.03211491e-42, 13.8222699, 9.12916839e-43, 14.1781512, 8.1382499e-43, 14.5431953, 7.3076708e-43, 14.9176382, 6.57205285e-43, 15.3017219, 5.76579426e-43, 15.6956946, 5.09190645e-43, 16.0998108, 4.52441193e-43, 16.5143318, 4.04340552e-43, 16.9395255, 3.63214401e-43, 17.3756666, 3.27976656e-43, 17.823037, 2.97613979e-43, 18.2819259, 2.71323361e-43, 18.7526297, 2.48436435e-43, 19.2354527, 2.28442723e-43, 19.7307069, 2.10877376e-43, 20.2387124, 1.95426131e-43, 20.7597975, 1.81733913e-43, 21.294299, 1.69615722e-43, 21.8425622, 1.58778993e-43, 22.4049416, 1.49143516e-43, 22.9818005, 1.4050002e-43, 23.5735118, 1.32706661e-43, 24.1804578, 1.25741345e-43, 24.8030308, 1.19412535e-43, 25.4416332, 1.13705313e-43, 26.0966776, 1.08506232e-43, 26.7685875, 1.03839119e-43, 27.4577969, 9.95517228e-44, 28.1647515, 9.5676136e-44, 28.8899079, 9.20782352e-44, 29.6337349, 8.88471568e-44, 30.3967132, 8.58887642e-44, 31.1793358, 8.31798484e-44, 31.9821086, 8.07072588e-44, 32.8055503, 7.84150606e-44, 33.6501932, 7.63253404e-44, 34.516583, 7.44186821e-44, 35.4052797, 7.26485171e-44, 36.3168577, 7.10739219e-44, 37.251906, 6.95173659e-44, 38.2110289, 6.30020559e-44, 39.1948464, 6.07621401e-44, 40.2039941, 5.87649055e-44, 41.2391243, 5.69306494e-44, 42.3009059, 5.52432156e-44, 43.3900252, 5.37576879e-44, 44.507186, 5.23486955e-44, 45.6531103, 5.10739371e-44, 46.8285386, 4.99422141e-44, 48.0342307, 4.89573882e-44, 49.2709656, 4.80122722e-44, 50.5395426, 4.7156356e-44, 51.8407817, 4.63830877e-44, 53.1755237, 4.56866365e-44, 54.5446312, 4.50619993e-44, 55.9489891, 4.45043821e-44, 57.3895049, 4.40095946e-44, 58.8671096, 4.35738484e-44, 60.3827581, 4.31938467e-44, 61.9374299, 4.28701581e-44, 63.5321298, 4.25999064e-44, 65.1678883, 4.23773924e-44, 66.8457626, 4.21981526e-44, 68.5668371, 4.18340258e-44, 70.332224, 3.6714575e-44, 72.1430642, 3.66428572e-44, 74.0005281, 3.66113118e-44, 75.9058159, 3.66032907e-44, 77.8601591, 3.66326774e-44, 79.8648207, 3.67000367e-44, 81.9210962, 3.67849627e-44, 84.0303146, 3.69162433e-44, 86.1938388, 3.70431831e-44, 88.4130673, 3.72311715e-44, 90.689434, 3.74385661e-44, 93.0244103, 3.76761515e-44, 95.4195051, 3.79375336e-44, 97.8762663, 3.82121882e-44, 100.396282, 3.85197044e-44, 102.98118, 3.88522475e-44, 105.632631, 3.91994707e-44, 108.352349, 3.95918641e-44, 111.142091, 3.99986613e-44, 114.003661, 4.04299967e-44, 116.938908, 4.08858692e-44, 119.949728, 4.13662908e-44, 123.038067, 4.18713157e-44, 126.205922, 4.24010317e-44, 129.45534, 4.29555601e-44, 132.78842, 4.35350208e-44, 136.207317, 4.41396294e-44, 139.71424, 4.47695992e-44, 143.311455, 4.54251648e-44, 147.001288, 4.61065894e-44, 150.786123, 4.68141877e-44, 154.668405, 4.75482408e-44, 158.650645, 4.83091521e-44, 162.735415, 4.90972632e-44, 166.925356, 4.9913182e-44, 171.223174, 5.07613365e-44, 175.631649, 5.16364885e-44, 180.153628, 5.25393109e-44, 184.792034, 5.34728957e-44, 189.549865, 5.44364639e-44, 194.430196, 5.54305242e-44, 199.436181, 5.64556703e-44, 204.571054, 5.75124838e-44, 209.838135, 5.86014645e-44, 215.240827, 5.97233398e-44, 220.782621, 6.08789402e-44, 226.4671, 6.206878e-44, 232.297937, 6.32935778e-44, 238.2789, 6.45542732e-44, 244.413855, 6.58515074e-44, 250.706767, 6.71861392e-44, 257.161701, 6.85589827e-44, 263.78283, 6.99708621e-44, 270.574434, 7.1422681e-44, 277.5409, 7.29168429e-44, 284.686731, 7.44513192e-44, 292.016546, 7.60284841e-44, 299.535082, 7.76494281e-44, 307.247196, 7.9315137e-44, 315.157874, 8.1026655e-44, 323.272228, 8.27850604e-44, 331.595502, 8.45914766e-44, 340.133075, 8.64470333e-44, 348.890464, 8.83529491e-44, 357.873329, 9.03104707e-44, 367.087475, 9.23207472e-44, 376.538858, 9.4384971e-44, 386.233585, 9.6504741e-44, 396.177922, 9.8681312e-44, 406.378295, 1.00915999e-43, 416.841296, 1.03210273e-43, 427.573688, 1.05565671e-43, 438.582406, 1.07983595e-43, 449.874565, 1.10465646e-43, 461.457463, 1.13013417e-43, 473.338585, 1.15628549e-43, 485.52561, 1.18312722e-43, 498.026413, 1.21067659e-43, 510.849074, 1.23895137e-43, 524.001879, 1.26796975e-43, 537.493329, 1.29774988e-43, 551.332143, 1.32831122e-43, 565.527264, 1.35967503e-43, 580.087866, 1.39185969e-43, 595.023359, 1.42488644e-43, 610.343395, 1.45877716e-43, 626.057876, 1.49355612e-43, 642.176956, 1.52924033e-43, 658.711054, 1.56585314e-43, 675.670854, 1.60342078e-43, 693.067317, 1.64196782e-43, 710.911687, 1.68151845e-43, 729.215494, 1.72209772e-43, 747.990569, 1.76373215e-43, 767.249044, 1.80644862e-43, 787.003366, 1.85027332e-43, 807.266302, 1.89523731e-43, 828.050947, 1.94144941e-43, 849.370733, 1.98877512e-43, 871.239439, 2.03733029e-43, 893.671196, 2.08713953e-43, 916.680504, 2.13824277e-43, 940.28223, 2.19066991e-43, 964.491629, 2.24445396e-43, 989.324347, 2.29963226e-43, 1014.79643, 2.3562382e-43, 1040.92434, 2.41430849e-43, 1067.72497, 2.47388182e-43, 1095.21563, 2.53499482e-43, 1123.41409, 2.59768863e-43, 1152.33858, 2.66200317e-43, 1182.00778, 2.72797914e-43, 1212.44088, 2.79566027e-43, 1243.65753, 2.8650906e-43, 1275.67792, 2.93631185e-43, 1308.52273, 3.00937474e-43, 1342.2132, 3.08432432e-43, 1376.7711, 3.16120676e-43, 1412.21875, 3.24007578e-43, 1448.57908, 3.32098197e-43, 1485.87557, 3.40397488e-43, 1524.13233, 3.48911364e-43, 1563.37409, 3.57644493e-43, 1603.62621, 3.66602951e-43, 1644.91469, 3.75794145e-43, 1687.26622, 3.85220836e-43, 1730.70818, 3.94889971e-43, 1775.26864, 4.04809101e-43, 1820.97639, 4.14984247e-43, 1867.86098, 4.25421547e-43, 1915.9527, 4.36127579e-43, 1965.28264, 4.47109947e-43, 2015.88267, 4.58375519e-43, 2067.7855, 4.69931694e-43, 2121.02467, 4.8178501e-43, 2175.63459, 4.93944344e-43, 2231.65054, 5.06416392e-43, 2289.10873, 5.19214687e-43, 2348.0463, 5.32340418e-43, 2408.50133, 5.45802436e-43, 2470.51289, 5.59611113e-43, 2534.12106, 5.73776036e-43, 2599.36695, 5.88305955e-43, 2666.29272, 6.03208961e-43, 2734.94162, 6.1849721e-43, 2805.35803, 6.34180145e-43, 2877.58744, 6.50260895e-43, 2951.67654, 6.66761354e-43, 3027.6732, 6.8368616e-43, 3105.62655, 7.0104764e-43, 3185.58696, 7.18856319e-43, 3267.60611, 7.3712522e-43, 3351.737, 7.55863254e-43, 3438.034, 7.75083936e-43, 3526.55289, 7.94799084e-43, 3617.35087, 8.15022089e-43, 3710.48662, 8.35766494e-43, 3806.02033, 8.57045151e-43, 3904.01375, 8.78865942e-43, 4004.53019, 9.01254693e-43, 4107.63463, 9.24218712e-43, 4213.39369, 9.47774937e-43, 4321.87573, 9.719384e-43, 4433.15084, 9.96724158e-43, 4547.29095, 1.02214825e-42, 4664.36982, 1.04822144e-42, 4784.46312, 1.07498357e-42, 4907.64845, 1.10242305e-42, 5034.00543, 1.13056913e-42, 5163.61571, 1.15944001e-42, 5296.56306, 1.18906421e-42, 5432.9334, 1.21944142e-42, 5572.81487, 1.25060088e-42, 5716.29785, 1.28256274e-42, 5863.47508, 1.31534757e-42, 6014.44168, 1.34896468e-42, 6169.29521, 1.38345944e-42, 6328.13574, 1.41884245e-42, 6491.06593, 1.45513657e-42, 6658.19108, 1.49236526e-42, 6829.61919, 1.53056117e-42, 7005.46105, 1.56973202e-42, 7185.8303, 1.60991807e-42, 7370.84351, 1.65112961e-42, 7560.62025, 1.69340252e-42, 7755.28315, 1.73676643e-42, 7954.95804, 1.78124687e-42, 8159.77394, 1.82694985e-42, 8369.86323, 1.87374942e-42, 8585.36168, 1.92174669e-42, 8806.40855, 1.97099508e-42, 9033.14671, 2.02148631e-42, 9265.72268, 2.07329791e-42, 9504.28677, 2.12640955e-42, 9748.99316, 2.18092116e-42, 10000, 2.23684022e-42]).reshape(1, -1)
param_halodensity = 0.3
param_halovelocity = 220
