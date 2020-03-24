#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-21-2020 01:15:25
# Invocation: btc -server brain.brainome.ai Data/BNG_kr-vs-kp_1000_1.csv -o Models/BNG_kr-vs-kp_1000_1.py -v -v -v -stopat 97.38 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                52.18%
Model accuracy:                     88.17% (881704/1000000 correct)
Improvement over best guess:        35.99% (of possible 47.82%)
Model capacity (MEC):               199 bits
Generalization ratio:               4430.67 bits/bit
Model efficiency:                   0.18%/parameter
System behavior
True Negatives:                     42.02% (420152/1000000)
True Positives:                     46.16% (461552/1000000)
False Negatives:                    6.03% (60323/1000000)
False Positives:                    5.80% (57973/1000000)
True Pos. Rate/Sensitivity/Recall:  0.88
True Neg. Rate/Specificity:         0.88
Precision:                          0.89
F-1 Measure:                        0.89
False Negative Rate/Miss Rate:      0.12
Critical Success Index:             0.80

"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii
import faulthandler

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="BNG_kr-vs-kp_1000_1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 36
n_classes = 2

mappings = [{1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {30677878.0: 0, 2517025534.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {476252946.0: 0, 1908338681.0: 1, 2013832146.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2013832146.0: 1, 2238339752.0: 2}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

transform_true = True

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values()))+1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize,mappings):
            if i>=data_arr.shape[1]:
                break
            col = data_arr[:,i]
            normcol = column_norm(col,mapping)
            data_arr[:,i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([0.11244333333333334, 0.07056, 0.075885, 0.06682333333333333, 0.33199666666666666, 0.4575283333333333, 0.393685, 0.2799333333333333, 0.41577, 0.33926, 0.45428666666666667, 0.14144, 0.7306166666666667, 0.17765666666666666, 1.5177566666666666, 0.217535, 0.14455, 0.631785, 0.012588333333333333, 0.22818, 0.26298333333333335, 0.3155133333333333, 0.29200166666666666, 0.375345, 0.004923333333333333, 0.7898083333333333, 0.13661, 0.0006416666666666667, 0.19927333333333333, 0.131055, 0.23431166666666667, 0.27452166666666666, 0.4320866666666667, 0.5681666666666667, 0.602665, 1.18596])
        components = np.array([array([ 4.76499090e-02,  1.85372691e-02, -2.01197107e-03,  1.47607774e-03,
       -5.88388022e-02,  1.19126415e-03, -4.04618633e-02,  3.26554072e-03,
       -1.19698438e-02, -2.35992621e-02,  2.49698413e-01,  7.98502441e-03,
        1.07853472e-01,  5.71087849e-02, -8.67129941e-01,  2.95385895e-02,
        1.01533135e-02, -5.04873533e-02,  3.45186363e-03,  2.44917979e-01,
        7.68559014e-02,  1.57278284e-02,  3.79530245e-02,  1.26962779e-02,
        4.28426023e-04,  7.38565122e-02,  1.70566541e-02, -4.72392716e-07,
        3.68631603e-02,  2.20122852e-02,  2.12883532e-01,  6.41227717e-02,
       -2.07307738e-02, -3.45879359e-02, -1.76902486e-01, -3.35978554e-02]), array([ 1.10710321e-02, -2.45716743e-02,  3.23740528e-02,  4.02747815e-03,
        4.70982410e-01,  6.76199011e-02,  5.13418506e-01,  3.86668313e-01,
        5.23753800e-01,  1.60990674e-02,  1.04751072e-04, -8.08295328e-02,
       -1.64668170e-02, -5.98814672e-02, -1.06190673e-01, -6.96582896e-02,
        5.61716620e-02,  1.11937613e-01, -1.75833421e-03,  4.84950837e-03,
       -8.59411183e-02,  1.68952185e-02, -1.71713926e-01,  4.92525448e-03,
        5.02519905e-04,  1.41644323e-03, -2.63619059e-02,  1.15353716e-04,
       -2.36777752e-02, -1.33785242e-02, -3.88793938e-02, -2.03898124e-02,
        2.81826315e-04,  1.40746943e-02,  5.79214816e-03,  3.34086246e-02]), array([ 5.45181637e-02, -6.53648827e-03, -3.28719542e-02, -1.08538459e-02,
        2.78694669e-03, -1.13525227e-01,  6.88778585e-02, -1.06226594e-02,
       -4.01082067e-02,  5.22228258e-02,  1.93168935e-01,  4.00196983e-02,
       -1.06820862e-01, -1.17809262e-01,  9.44625710e-02, -3.27841978e-02,
       -2.93525777e-03,  1.71516143e-01, -2.34849457e-03,  7.47157439e-04,
       -9.58207264e-02, -1.42901624e-01, -2.37483978e-02, -4.45716123e-02,
       -1.01773964e-03,  6.77724953e-02, -2.43962633e-02,  1.47955844e-05,
        2.94909854e-02, -1.87805970e-02,  1.59087961e-02,  9.07011267e-02,
       -3.95677886e-02,  1.19168551e-01, -1.85496828e-01, -8.79114089e-01]), array([-4.98794587e-02,  6.06948538e-02,  5.39291567e-02,  4.98378573e-02,
        6.88499919e-02,  1.52120384e-01, -9.45921087e-03,  6.58587537e-02,
        1.29425630e-01, -5.27468767e-02, -1.08088764e-01, -4.97660297e-02,
        2.26204470e-01,  1.58173398e-01,  1.26095368e-01,  1.65941923e-01,
        1.21498068e-02, -5.35861576e-01,  9.25375824e-03,  5.73236829e-02,
        1.99936813e-01,  3.03390315e-01,  1.13006149e-02,  6.12062027e-02,
        1.54808037e-03,  4.55502425e-02,  4.60479337e-02, -5.58831972e-05,
        3.60909921e-02,  6.04054532e-02,  3.69001669e-02,  3.36290399e-02,
        9.12440037e-02, -4.83802901e-01,  1.54455282e-02, -3.49255037e-01]), array([ 4.59914090e-02,  4.63263556e-02,  8.69898606e-03, -2.40851140e-02,
       -6.35295232e-02, -2.28730390e-01,  5.27192789e-02,  1.62961295e-01,
        3.17740424e-02,  2.65518887e-01, -1.69464001e-02,  5.19230204e-02,
       -9.59304220e-02, -3.23790881e-02,  2.03442964e-01,  7.88721935e-02,
       -1.50819289e-02,  2.70692437e-03,  3.08173234e-03,  1.02169185e-01,
        1.52973370e-01,  2.53634105e-02,  2.78544401e-02, -7.67901230e-02,
        1.00437676e-03,  1.03230764e-01,  9.71914929e-02,  7.11849671e-05,
        3.17843438e-01,  5.24398304e-02,  8.31001806e-02,  4.74557834e-01,
        1.91900374e-01, -3.70780516e-02, -5.37245317e-01,  2.29821742e-01]), array([-5.54645418e-02, -1.08293502e-02,  6.23077346e-02,  2.90355168e-02,
       -1.57812831e-01,  3.93112966e-01,  1.02665976e-02,  1.45002543e-01,
       -1.03661615e-01,  4.72072969e-01,  7.00268148e-02, -4.17986804e-02,
       -1.13617149e-01, -9.45968480e-02, -7.16615966e-02, -1.18583122e-01,
       -7.94577944e-02,  1.17344607e-01,  3.71996433e-03, -4.54298478e-02,
        2.73690318e-01, -1.33385409e-01, -1.36401216e-01,  1.09175637e-01,
        1.71187376e-03, -9.15242284e-02,  1.49210040e-01,  3.80129062e-05,
       -4.74634727e-02,  3.57036214e-02, -1.12415942e-02, -1.31079422e-01,
        5.28308299e-01, -4.56340245e-02,  1.65926875e-01, -6.79448193e-02]), array([-8.54929193e-02, -7.64885303e-02,  4.63662760e-03,  1.39391973e-02,
        5.95140167e-02,  2.55213490e-01, -9.82452020e-02, -6.51478580e-02,
        2.60564515e-02, -7.06215519e-02, -6.29415950e-01, -1.85482098e-02,
        7.25158012e-02,  9.15805841e-02, -2.57343283e-01, -4.04514059e-02,
        8.53717808e-02,  1.48545624e-01, -6.07261236e-03, -1.63573768e-01,
        9.79703292e-03, -1.63898232e-01,  1.07319690e-01,  5.31550688e-02,
        8.61342364e-04, -1.05188920e-01,  2.71143893e-03,  3.58058523e-06,
        1.99099888e-01,  1.55523557e-03, -2.62684029e-01,  4.20273881e-01,
       -1.17616788e-02,  6.96454587e-02,  2.11376437e-02, -1.45384256e-01]), array([-8.22611087e-02,  8.08483542e-03, -7.74037619e-02, -2.53720695e-02,
        2.81543718e-02, -2.64456758e-01, -5.84631804e-02,  1.41649728e-01,
        1.36768518e-01,  2.58107365e-01, -9.36323781e-02,  5.98375050e-02,
       -5.04657354e-02,  1.17478331e-01, -8.84395442e-02,  1.43817625e-01,
        1.31364927e-01,  1.09381535e-02, -4.38283327e-03, -3.45340342e-02,
        1.63544855e-01,  6.26439222e-02,  4.34697084e-01, -6.26013480e-01,
        2.23715248e-04, -5.49295074e-02,  6.71305843e-02,  3.21545608e-05,
       -5.29014265e-02,  5.58802684e-02, -1.08404468e-02, -1.16765341e-01,
        4.71226387e-02,  9.98494228e-02,  2.71602887e-01, -8.07799372e-02]), array([-1.05877508e-02, -2.83014017e-02, -1.35407807e-02,  4.19122688e-02,
       -4.57733950e-02,  6.11081201e-01, -2.91090267e-02,  1.16027655e-02,
        1.23337246e-01, -2.49476200e-01,  2.65220257e-01, -5.14462129e-02,
       -7.85351131e-02,  2.16697086e-01,  1.98350870e-01, -5.14163658e-02,
        1.53523868e-01,  1.92291988e-01, -1.87292081e-03,  9.64405156e-02,
       -6.21397489e-02, -7.87359611e-02,  4.29530868e-01, -1.70308694e-01,
       -9.59720094e-04,  1.57081311e-01, -5.00343778e-02, -2.33300339e-05,
        4.18775210e-02,  1.24478201e-02,  1.48291365e-01,  1.80224539e-02,
        2.47149214e-02, -3.63985341e-02, -1.58242504e-01,  5.75049931e-02]), array([-4.04966232e-02,  3.26483694e-02, -4.55413848e-02, -4.70191809e-03,
        2.48354913e-01, -1.81257991e-01, -2.75156429e-01, -2.32215773e-01,
        2.68062318e-01, -3.74954722e-01, -1.48213670e-02, -3.35752886e-02,
        9.56633791e-07,  3.11770729e-02,  2.85417752e-02,  9.39122490e-02,
        4.77187997e-02,  7.71943698e-02,  4.38138403e-03,  3.01459522e-02,
        1.75611686e-01,  8.35714883e-02, -2.77061287e-02,  1.37238650e-01,
        1.56346527e-03, -5.03231732e-02,  7.95014272e-02, -6.19568694e-05,
       -7.56056704e-02,  8.28110182e-03,  9.51031553e-02, -8.23646156e-02,
        5.79515700e-01,  3.21436084e-01, -9.30038300e-02, -4.19380232e-02]), array([-4.69828424e-02,  4.09960382e-02,  1.78595969e-02,  6.52218824e-03,
        6.54797643e-03, -1.34566732e-01, -1.65491816e-01,  1.27932310e-01,
        2.26556816e-01,  1.95825105e-01, -4.63565182e-02, -4.73552894e-03,
        1.66810782e-01,  1.63720819e-01,  1.33456918e-01,  1.28808108e-01,
        5.45787243e-02,  1.58466273e-01,  1.17192921e-02,  1.73046278e-01,
        3.13074224e-01, -3.67994976e-01,  1.60288374e-01,  4.83382508e-01,
        9.69204959e-04,  1.39465309e-01,  6.33379098e-02, -3.22557666e-05,
       -4.69387571e-03,  1.01363260e-02,  2.12356755e-01, -7.53515790e-02,
       -3.25164940e-01,  4.98758773e-02,  1.38177354e-01, -1.82238209e-02]), array([ 2.98665876e-02,  2.21069808e-02, -1.04836192e-02,  3.21469098e-03,
        5.66120261e-02, -2.21691142e-01,  3.72637734e-02, -6.79914171e-02,
        2.00469310e-02, -1.52362165e-01,  9.64952775e-02,  7.38898815e-02,
       -3.08861560e-01, -2.80022584e-02, -9.78140506e-02,  2.77716182e-01,
        8.33697491e-02,  1.45902796e-01, -1.33068873e-03, -1.79997238e-01,
       -1.12146844e-01, -3.75002327e-01,  1.52495686e-01,  1.08584669e-01,
       -1.02158109e-03, -1.31456712e-01,  5.31208420e-03,  5.39620507e-05,
       -1.66651650e-02, -1.05333410e-02, -1.22350830e-01, -2.85517647e-02,
        1.70912236e-01, -6.36150582e-01,  1.83255660e-02,  4.76323533e-02]), array([-6.37801429e-02, -2.41377553e-02,  9.05419456e-02, -5.75809342e-03,
       -1.46494235e-01, -2.81285879e-02,  5.11773354e-02,  1.17464938e-01,
        5.39942978e-02,  9.27788729e-02,  1.36531919e-01,  2.42654238e-02,
       -4.35609638e-01,  1.87109490e-01, -9.23654396e-02,  1.11679473e-01,
        4.32665899e-02, -5.24496537e-02,  3.71627962e-03, -2.23965010e-01,
       -8.02875010e-02,  4.82059977e-01,  2.90460619e-01,  4.54522219e-01,
       -5.09671710e-04, -4.79970685e-02, -1.42150834e-02,  1.83403002e-05,
        2.33618631e-02, -2.69222983e-02, -1.76774025e-01,  5.18950877e-02,
       -5.47876592e-02,  2.09777811e-01,  3.17843666e-02, -4.00849131e-02]), array([ 3.60345589e-02,  1.44130104e-03, -4.35880432e-02,  4.60201632e-02,
        7.28130774e-02,  8.65910618e-02, -2.99416501e-01, -6.31731841e-02,
        2.16758611e-01, -4.29040549e-02,  1.99684860e-01, -1.25408464e-01,
       -1.91892681e-01, -1.98229600e-02, -2.50299342e-02, -1.42917198e-01,
       -8.13804192e-02,  5.45136977e-02,  2.88114169e-03, -3.04963598e-01,
        4.17507788e-01,  5.17939544e-02, -2.15173088e-01, -1.56915046e-01,
        2.17706385e-03, -1.96241200e-01,  2.47157325e-01, -2.57896049e-05,
       -1.18764769e-01,  1.65439684e-01, -1.70105883e-01,  2.78022081e-02,
       -3.87494808e-01, -9.25381794e-02, -2.12861890e-01,  1.45581289e-02]), array([ 1.16464377e-01, -6.72227417e-02,  2.59054591e-02,  2.14802744e-02,
        1.46324435e-01, -5.71231911e-02,  2.93196248e-01, -6.81950095e-02,
       -2.06089556e-01,  9.04209982e-03,  1.95430047e-01,  6.70241884e-02,
        3.37114891e-01, -2.79553368e-02,  3.20848064e-02, -2.21428416e-01,
        2.53655254e-01, -2.93406630e-01,  8.50578199e-03, -1.73730214e-01,
        1.10358376e-01, -2.13503360e-01,  3.50517283e-01,  1.43861094e-01,
        1.40161299e-03, -3.06768649e-01,  1.72592535e-01,  1.84854841e-04,
       -1.02540133e-01,  1.36069810e-01, -1.53000687e-01, -5.57889757e-04,
        4.46987767e-02,  1.59378622e-01, -1.69913571e-01,  3.40306329e-02]), array([-4.50150023e-02,  5.42932437e-02,  3.19838470e-02, -9.46471672e-04,
        4.18222958e-01,  4.90423252e-02, -3.86347657e-02, -4.23844333e-01,
       -5.60589911e-02,  2.28340004e-01, -5.31628305e-02,  5.94034482e-02,
       -3.03619607e-01, -2.99939818e-01,  4.54418929e-03, -2.83641114e-01,
        2.60404023e-01, -4.48289286e-02,  8.77552067e-03,  1.56017738e-01,
        4.67913594e-02,  1.55929210e-01,  1.19110225e-01,  6.47167259e-02,
       -1.86723718e-03, -2.04500035e-03, -1.11421912e-02,  2.51995090e-05,
        1.60979109e-01, -3.50722457e-02,  2.97256882e-01,  6.40775148e-02,
       -1.08418778e-01, -1.26899542e-01,  1.56074248e-01,  1.45926280e-02]), array([-9.50317678e-02,  3.44558171e-02, -1.22568577e-02,  1.80541235e-02,
       -1.25191962e-01,  3.72490298e-02, -9.17063601e-03,  6.82737546e-02,
        1.21969853e-01, -1.13983112e-01,  2.10234641e-01,  6.33126009e-02,
        2.28200040e-02,  3.94163438e-02,  6.95511363e-02,  1.03909960e-01,
       -6.82783449e-02, -5.64190583e-02,  1.06586722e-02,  5.83252239e-02,
       -1.76264343e-01, -6.03518926e-02, -1.31330927e-01, -2.90849104e-02,
        1.08390063e-03, -6.17480844e-01,  1.33411773e-01,  1.28262375e-04,
        3.83482949e-01,  1.31955750e-01,  3.10209405e-01,  2.12584296e-01,
       -4.01673818e-02,  9.28380639e-02,  3.06425765e-01, -2.10718610e-02]), array([-2.98468204e-02,  7.12313207e-02, -2.63645615e-02,  2.22878647e-02,
        2.35204991e-01,  2.26734090e-01,  1.48683685e-01, -2.12287231e-01,
       -1.38653519e-01,  1.02575315e-01,  5.96851574e-02, -6.63135078e-02,
       -2.49097520e-01,  9.96858801e-02, -1.93690680e-02,  5.78746852e-01,
       -9.48631492e-03, -3.34612940e-01,  4.40908043e-03, -3.17030302e-02,
        7.34146667e-02, -2.88257700e-01, -1.78847599e-01, -8.83820527e-02,
       -1.73212281e-03,  1.36279086e-01, -4.12025486e-02,  1.28002588e-04,
        3.36753918e-02,  5.93647652e-02, -5.42787489e-02,  1.05436269e-02,
       -8.21673469e-02,  3.09971416e-01, -6.16776618e-03,  4.12838788e-02]), array([-5.56785651e-03,  1.20481589e-01,  6.21330629e-02,  1.75708069e-02,
       -2.10554973e-01, -3.69451653e-03, -1.61911488e-01,  3.11890540e-01,
        1.67271719e-01, -2.04186922e-01, -1.93022651e-01,  5.32714109e-02,
       -3.68739905e-01, -2.97042411e-01, -3.99261341e-02, -2.52344624e-01,
        3.53412956e-02, -4.85760562e-01,  1.32890725e-02,  6.21998046e-02,
       -8.75221477e-02, -3.08414764e-01,  7.52609462e-02,  2.88067890e-03,
        4.23210459e-04,  8.93705608e-02,  6.60246299e-03,  2.07273340e-04,
       -1.35714141e-01,  1.19506569e-01,  8.70238881e-02, -1.97044146e-02,
        1.30463146e-02,  1.22834952e-01, -2.18560520e-02, -8.00451411e-03]), array([ 0.15170191,  0.04298349, -0.00807284, -0.01284961, -0.10988279,
       -0.05621251,  0.4181216 ,  0.0023719 , -0.22733332, -0.40466037,
       -0.08890005,  0.1587966 , -0.1824039 ,  0.03391354,  0.01599757,
       -0.05385583, -0.06257105,  0.16630254,  0.01085592,  0.15535056,
        0.39783601,  0.08808032, -0.02521156, -0.005716  ,  0.00400847,
        0.20625965,  0.2647994 ,  0.00042859,  0.06626121,  0.27622515,
       -0.00482709,  0.07339179, -0.02816216,  0.0049737 ,  0.27577872,
       -0.0232815 ])])
        whiten = False
        explained_variance = np.array([0.6815515961486533, 0.4526344616431186, 0.40336891291237364, 0.3491435889087174, 0.321426146796654, 0.293905963412304, 0.2644113313155481, 0.26203844414694677, 0.24810869307887662, 0.24063863803965793, 0.2308294174961039, 0.2190838103077105, 0.2071795743973155, 0.18693212494740777, 0.1799855984616779, 0.1707113330975, 0.16506946926785637, 0.15895465332244008, 0.14817622110597445, 0.14124970753366187])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files
def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist=[]
    clean.testfile=testfile
    clean.mapping={}
    

    def convert(cell):
        value=str(cell)
        try:
            result=int(value)
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                return result

    def convertclassid(cell):
        if (clean.testfile):
            return convert(cell)
        value=str(cell)
        if (value==''):
            raise ValueError("All cells in the target column must contain a class label.")

        if (not clean.mapping=={}):
            result=-1
            try:
                result=clean.mapping[cell]
            except:
                raise ValueError("Class label '"+value+"' encountered in input not defined in user-provided mapping.")
            if (not result==int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(int(result*100)/100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not result==int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result<0):
                raise ValueError("Integer class labels must be positive and contiguous.")

        return result

    rowcount=0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f=open(outfile,"w+")
        if (headerless==False):
            next(reader,None)
        outbuf=[]
        for row in reader:
            if (row==[]):  # Skip empty rows
                continue
            rowcount=rowcount+1
            rowlen=num_attr
            if (not testfile):
                rowlen=rowlen+1    
            if (not len(row)==rowlen):
                raise ValueError("Column count must match trained predictor. Row "+str(rowcount)+" differs.")
            i=0
            for elem in row:
                if(i+1<len(row)):
                    outbuf.append(str(convert(elem)))
                    outbuf.append(',')
                else:
                    classid=str(convertclassid(elem))
                    outbuf.append(classid)
                i=i+1
            if (len(outbuf)<IOBUF):
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)>=2):
            raise ValueError("Number of classes must be at least 2.")



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    o=[0]*num_output_logits
    h_0 = max((((12.541765 * float(x[0]))+ (-12.1181345 * float(x[1]))+ (-23.549904 * float(x[2]))+ (29.384167 * float(x[3]))+ (-20.135899 * float(x[4]))+ (-18.910402 * float(x[5]))+ (1.8473014 * float(x[6]))+ (3.7518804 * float(x[7]))+ (18.964252 * float(x[8]))+ (-1.5995514 * float(x[9]))+ (52.817825 * float(x[10]))+ (-13.06988 * float(x[11]))+ (-7.253535 * float(x[12]))+ (80.602 * float(x[13]))+ (10.841871 * float(x[14]))+ (-12.445481 * float(x[15]))+ (16.060305 * float(x[16]))+ (24.938606 * float(x[17]))+ (-3.4354253 * float(x[18]))+ (83.2304 * float(x[19]))) + 15.782398), 0)
    h_1 = max((((-12.633487 * float(x[0]))+ (10.397475 * float(x[1]))+ (24.172523 * float(x[2]))+ (-30.70139 * float(x[3]))+ (15.592546 * float(x[4]))+ (14.032105 * float(x[5]))+ (-4.4043303 * float(x[6]))+ (-5.6609783 * float(x[7]))+ (-20.145473 * float(x[8]))+ (0.3411495 * float(x[9]))+ (-56.600697 * float(x[10]))+ (11.495762 * float(x[11]))+ (8.16937 * float(x[12]))+ (-83.38713 * float(x[13]))+ (-14.233433 * float(x[14]))+ (14.846696 * float(x[15]))+ (-19.486774 * float(x[16]))+ (-24.838625 * float(x[17]))+ (4.949468 * float(x[18]))+ (-91.13388 * float(x[19]))) + -13.6159725), 0)
    h_2 = max((((7.4437733 * float(x[0]))+ (-27.248987 * float(x[1]))+ (-19.66778 * float(x[2]))+ (35.69271 * float(x[3]))+ (-9.955938 * float(x[4]))+ (-5.5495625 * float(x[5]))+ (5.8387246 * float(x[6]))+ (0.46124932 * float(x[7]))+ (8.243174 * float(x[8]))+ (35.769894 * float(x[9]))+ (16.191286 * float(x[10]))+ (-9.574409 * float(x[11]))+ (-6.5581737 * float(x[12]))+ (50.98578 * float(x[13]))+ (4.750474 * float(x[14]))+ (3.9204645 * float(x[15]))+ (-5.304042 * float(x[16]))+ (9.128212 * float(x[17]))+ (7.1747127 * float(x[18]))+ (39.288376 * float(x[19]))) + 30.869547), 0)
    h_3 = max((((-16.888475 * float(x[0]))+ (-3.374408 * float(x[1]))+ (5.5665035 * float(x[2]))+ (-1.4702045 * float(x[3]))+ (-6.613604 * float(x[4]))+ (-10.591952 * float(x[5]))+ (-9.085626 * float(x[6]))+ (-3.751845 * float(x[7]))+ (3.8242912 * float(x[8]))+ (10.294914 * float(x[9]))+ (-0.64439267 * float(x[10]))+ (1.225825 * float(x[11]))+ (-4.0582476 * float(x[12]))+ (7.325242 * float(x[13]))+ (7.814238 * float(x[14]))+ (-2.0868688 * float(x[15]))+ (-3.1733902 * float(x[16]))+ (-11.632942 * float(x[17]))+ (4.9357476 * float(x[18]))+ (3.6225948 * float(x[19]))) + -5.0093107), 0)
    h_4 = max((((7.586665 * float(x[0]))+ (0.061937265 * float(x[1]))+ (-1.9616693 * float(x[2]))+ (3.378803 * float(x[3]))+ (6.6561337 * float(x[4]))+ (12.015317 * float(x[5]))+ (4.9549055 * float(x[6]))+ (4.8643856 * float(x[7]))+ (-0.4135232 * float(x[8]))+ (-7.6379585 * float(x[9]))+ (5.3436556 * float(x[10]))+ (-1.0791475 * float(x[11]))+ (2.5855293 * float(x[12]))+ (-0.29971522 * float(x[13]))+ (-3.7353656 * float(x[14]))+ (0.8359255 * float(x[15]))+ (-0.20539628 * float(x[16]))+ (12.529972 * float(x[17]))+ (-6.4868827 * float(x[18]))+ (-5.0516295 * float(x[19]))) + 6.3524375), 0)
    h_5 = max((((1.2976327 * float(x[0]))+ (-1.6130542 * float(x[1]))+ (-1.0520236 * float(x[2]))+ (2.4033632 * float(x[3]))+ (3.2108278 * float(x[4]))+ (3.2028348 * float(x[5]))+ (-0.013111811 * float(x[6]))+ (2.4525084 * float(x[7]))+ (-1.7978983 * float(x[8]))+ (1.5952977 * float(x[9]))+ (3.7012255 * float(x[10]))+ (-1.3031853 * float(x[11]))+ (-0.925331 * float(x[12]))+ (6.245903 * float(x[13]))+ (3.7770245 * float(x[14]))+ (0.41155443 * float(x[15]))+ (-0.5957682 * float(x[16]))+ (-0.052763242 * float(x[17]))+ (-0.44333574 * float(x[18]))+ (7.1266375 * float(x[19]))) + 3.9535608), 0)
    h_6 = max((((0.082791895 * float(x[0]))+ (-0.22508226 * float(x[1]))+ (-1.5454735 * float(x[2]))+ (2.6990726 * float(x[3]))+ (-0.55831724 * float(x[4]))+ (-0.543754 * float(x[5]))+ (2.3963923 * float(x[6]))+ (-0.50172424 * float(x[7]))+ (-2.3521557 * float(x[8]))+ (0.63892525 * float(x[9]))+ (1.5508566 * float(x[10]))+ (-2.9981618 * float(x[11]))+ (-4.086492 * float(x[12]))+ (-1.5189444 * float(x[13]))+ (1.1174058 * float(x[14]))+ (-2.9428833 * float(x[15]))+ (0.4775934 * float(x[16]))+ (-1.6778424 * float(x[17]))+ (-2.3912618 * float(x[18]))+ (-1.2532587 * float(x[19]))) + -1.2628653), 0)
    h_7 = max((((0.11170595 * float(x[0]))+ (-1.9731985 * float(x[1]))+ (-0.5187265 * float(x[2]))+ (1.2137471 * float(x[3]))+ (-0.5410091 * float(x[4]))+ (-0.8688461 * float(x[5]))+ (-0.42717785 * float(x[6]))+ (-0.5602102 * float(x[7]))+ (-0.9136261 * float(x[8]))+ (0.2842302 * float(x[9]))+ (-1.9240729 * float(x[10]))+ (-0.8610587 * float(x[11]))+ (0.25351983 * float(x[12]))+ (-0.11521412 * float(x[13]))+ (0.98422253 * float(x[14]))+ (1.636028 * float(x[15]))+ (-0.7010718 * float(x[16]))+ (0.75908476 * float(x[17]))+ (-0.93761486 * float(x[18]))+ (0.811087 * float(x[19]))) + -0.16122858), 0)
    h_8 = max((((1.1157837 * float(x[0]))+ (-0.19430907 * float(x[1]))+ (0.006320916 * float(x[2]))+ (0.42196128 * float(x[3]))+ (2.282405 * float(x[4]))+ (2.9726942 * float(x[5]))+ (0.39028552 * float(x[6]))+ (1.245633 * float(x[7]))+ (-0.5901519 * float(x[8]))+ (0.6545205 * float(x[9]))+ (0.29559 * float(x[10]))+ (-0.08693455 * float(x[11]))+ (0.35569993 * float(x[12]))+ (-0.07465897 * float(x[13]))+ (0.39526963 * float(x[14]))+ (0.58299536 * float(x[15]))+ (-0.96255183 * float(x[16]))+ (0.2803996 * float(x[17]))+ (-0.49274364 * float(x[18]))+ (-0.80481637 * float(x[19]))) + 2.7392292), 0)
    o[0] = (0.17845191 * h_0)+ (-0.16501366 * h_1)+ (-0.08854093 * h_2)+ (0.2280421 * h_3)+ (-0.39649713 * h_4)+ (-1.499983 * h_5)+ (1.186008 * h_6)+ (1.2139719 * h_7)+ (4.2347035 * h_8) + -4.5998178

    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)

# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    
    if not args.validate: # Then predict
        if not args.cleanfile: # File is not preprocessed
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            test_tensor = np.loadtxt(cleanfile,delimiter=',',dtype='float64')
            os.remove(cleanfile)
        else: # File is already preprocessed
            test_tensor = np.loadtxt(args.File,delimiter = ',',dtype = 'float64')               
        test_tensor = Normalize(test_tensor)
        if transform_true:
            test_tensor = transform(test_tensor)
        with open(args.csvfile,'r') as csvinput:
            writer = csv.writer(sys.stdout, lineterminator='\n')
            reader = csv.reader(csvinput)
            if (not args.headerless):
                writer.writerow((next(reader, None)+['Prediction']))
            i=0
            for row in reader:
                if (classify(test_tensor[i])):
                    pred="1"
                else:
                    pred="0"
                row.append(pred)
                writer.writerow(row)
                i=i+1
    elif args.validate: # Then validate this predictor, always clean first.
        if n_classes==2:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanfile, -1, args.headerless)
            val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
            os.remove(cleanfile)
            val_tensor = Normalize(val_tensor)
            if transform_true:
                trans = transform(val_tensor[:,:-1])
                val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
            count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0 = 0,0,0,0,0,0,0,0
            for i,row in enumerate(val_tensor):
                if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                    correct_count+=1
                    if int(float(row[-1]))==1:
                        num_class_1+=1
                        num_TP+=1
                    else:
                        num_class_0+=1
                        num_TN+=1
                else:
                    if int(float(row[-1]))==1:
                        num_class_1+=1
                        num_FN+=1
                    else:
                        num_class_0+=1
                        num_FP+=1
                count+=1
        else:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanvalfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanvalfile, -1, args.headerless)
            val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
            os.remove(cleanfile)
            val_tensor = Normalize(val_tensor)
            if transform_true:
                trans = transform(val_tensor[:,:-1])
                val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
            numeachclass={}
            count,correct_count = 0,0
            for i,row in enumerate(val_tensor):
                if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                    correct_count+=1
                    if int(float(val_tensor[i,-1])) in numeachclass.keys():
                        numeachclass[int(float(val_tensor[i,-1]))]+=1
                    else:
                        numeachclass[int(float(val_tensor[i,-1]))]=0
                count+=1

        model_cap=199

        if n_classes==2:

            FN=float(num_FN)*100.0/float(count)
            FP=float(num_FP)*100.0/float(count)
            TN=float(num_TN)*100.0/float(count)
            TP=float(num_TP)*100.0/float(count)
            num_correct=correct_count

            if int(num_TP+num_FN)!=0:
                TPR=num_TP/(num_TP+num_FN) # Sensitivity, Recall
            if int(num_TN+num_FP)!=0:
                TNR=num_TN/(num_TN+num_FP) # Specificity, 
            if int(num_TP+num_FP)!=0:
                PPV=num_TP/(num_TP+num_FP) # Recall
            if int(num_FN+num_TP)!=0:
                FNR=num_FN/(num_FN+num_TP) # Miss rate
            if int(2*num_TP+num_FP+num_FN)!=0:
                FONE=2*num_TP/(2*num_TP+num_FP+num_FN) # F1 Score
            if int(num_TP+num_FN+num_FP)!=0:
                TS=num_TP/(num_TP+num_FN+num_FP) # Critical Success Index

            randguess=int(float(10000.0*max(num_class_1,num_class_0))/count)/100.0
            modelacc=int(float(num_correct*10000)/count)/100.0

            print("System Type:                        Binary classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")
            print("Model efficiency:                   {:.2f}%/parameter".format(int(100*(modelacc-randguess)/model_cap)/100.0))
            print("System behavior")
            print("True Negatives:                     {:.2f}%".format(TN)+" ("+str(int(num_TN))+"/"+str(count)+")")
            print("True Positives:                     {:.2f}%".format(TP)+" ("+str(int(num_TP))+"/"+str(count)+")")
            print("False Negatives:                    {:.2f}%".format(FN)+" ("+str(int(num_FN))+"/"+str(count)+")")
            print("False Positives:                    {:.2f}%".format(FP)+" ("+str(int(num_FP))+"/"+str(count)+")")
            if int(num_TP+num_FN)!=0:
                print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
            if int(num_TN+num_FP)!=0:
                print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
            if int(num_TP+num_FP)!=0:
                print("Precision:                          {:.2f}".format(PPV))
            if int(2*num_TP+num_FP+num_FN)!=0:
                print("F-1 Measure:                        {:.2f}".format(FONE))
            if int(num_TP+num_FN)!=0:
                print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
            if int(num_TP+num_FN+num_FP)!=0:    
                print("Critical Success Index:             {:.2f}".format(TS))
        else:
            num_correct=correct_count
            modelacc=int(float(num_correct*10000)/count)/100.0
            randguess=round(max(numeachclass.values())/sum(numeachclass.values())*100,2)
            print("System Type:                        "+str(n_classes)+"-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")






