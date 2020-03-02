#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Mar-01-2020 21:28:28
# Invocation: btc Data/BNG_kr-vs-kp_1000_1.csv -o Models/BNG_kr-vs-kp_1000_1.py -v -v -v -stopat 97.38 -port 8090 -e 3
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                52.18%
Model accuracy:                     86.09% (860919/1000000 correct)
Improvement over best guess:        33.91% (of possible 47.82%)
Model capacity (MEC):               211 bits
Generalization ratio:               4080.18 bits/bit
Model efficiency:                   0.16%/parameter
System behavior
True Negatives:                     40.65% (406476/1000000)
True Positives:                     45.44% (454443/1000000)
False Negatives:                    6.74% (67432/1000000)
False Positives:                    7.16% (71649/1000000)
True Pos. Rate/Sensitivity/Recall:  0.87
True Neg. Rate/Specificity:         0.85
Precision:                          0.86
F-1 Measure:                        0.87
False Negative Rate/Miss Rate:      0.13
Critical Success Index:             0.77
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
        components = np.array([array([ 4.76567327e-02,  1.85381811e-02, -2.01915256e-03,  1.47597316e-03,
       -5.88445905e-02,  1.19200695e-03, -4.04571110e-02,  3.25928641e-03,
       -1.19624362e-02, -2.35971922e-02,  2.49698024e-01,  7.98723076e-03,
        1.07857420e-01,  5.71051997e-02, -8.67130716e-01,  2.95400659e-02,
        1.01478522e-02, -5.04866274e-02,  3.45275104e-03,  2.44903654e-01,
        7.68586916e-02,  1.57290049e-02,  3.79548454e-02,  1.26979611e-02,
        4.28190568e-04,  7.38598627e-02,  1.70444153e-02, -4.46256140e-07,
        3.68616862e-02,  2.20198687e-02,  2.12894202e-01,  6.41267602e-02,
       -2.07288646e-02, -3.45873151e-02, -1.76899494e-01, -3.35973018e-02]), array([ 1.11083288e-02, -2.45366000e-02,  3.23147694e-02,  4.03319274e-03,
        4.70980574e-01,  6.76231933e-02,  5.13419766e-01,  3.86666449e-01,
        5.23755968e-01,  1.61077656e-02,  9.33166529e-05, -8.08444682e-02,
       -1.64830593e-02, -5.98593177e-02, -1.06189626e-01, -6.96698659e-02,
        5.61782440e-02,  1.11927297e-01, -1.76064870e-03,  4.88046873e-03,
       -8.59533604e-02,  1.68920644e-02, -1.71714655e-01,  4.92954689e-03,
        5.02893729e-04,  1.40102683e-03, -2.63335741e-02,  1.15275385e-04,
       -2.36772798e-02, -1.34089846e-02, -3.88982641e-02, -2.03987274e-02,
        2.80290317e-04,  1.40726735e-02,  5.78735538e-03,  3.34067516e-02]), array([ 5.46100411e-02, -6.51854752e-03, -3.30767282e-02, -1.07171895e-02,
        2.72383826e-03, -1.13523809e-01,  6.89192014e-02, -1.06721740e-02,
       -4.00421085e-02,  5.22487145e-02,  1.93143904e-01,  4.00094113e-02,
       -1.06829455e-01, -1.17819466e-01,  9.44607566e-02, -3.27955110e-02,
       -2.94727372e-03,  1.71495072e-01, -2.34667823e-03,  7.25070766e-04,
       -9.58314680e-02, -1.42893336e-01, -2.37356368e-02, -4.45453554e-02,
       -1.01847580e-03,  6.77730875e-02, -2.43978911e-02,  1.47978235e-05,
        2.94786721e-02, -1.88018680e-02,  1.59431842e-02,  9.07140812e-02,
       -3.95589780e-02,  1.19168463e-01, -1.85486628e-01, -8.79110274e-01]), array([-4.97194119e-02,  6.06047623e-02,  5.37699093e-02,  4.98757463e-02,
        6.86440762e-02,  1.52129222e-01, -9.31598174e-03,  6.56607656e-02,
        1.29683211e-01, -5.27020177e-02, -1.08078125e-01, -4.96653388e-02,
        2.26341707e-01,  1.58014686e-01,  1.26078629e-01,  1.66005774e-01,
        1.19858278e-02, -5.35841763e-01,  9.28391673e-03,  5.68521950e-02,
        2.00034596e-01,  3.03438368e-01,  1.13468814e-02,  6.12701731e-02,
        1.54097507e-03,  4.56908928e-02,  4.56793230e-02, -5.50702189e-05,
        3.60510043e-02,  6.06466722e-02,  3.72696931e-02,  3.37722998e-02,
        9.13141839e-02, -4.83776403e-01,  1.55519485e-02, -3.49232784e-01]), array([ 4.60815983e-02,  4.63075686e-02,  8.58483742e-03, -2.38560717e-02,
       -6.35966501e-02, -2.28760725e-01,  5.27241442e-02,  1.62940863e-01,
        3.18307638e-02,  2.65508815e-01, -1.69805699e-02,  5.19043989e-02,
       -9.59987409e-02, -3.23845922e-02,  2.03449154e-01,  7.88172401e-02,
       -1.50212655e-02,  2.65512376e-03,  3.07754018e-03,  1.02273399e-01,
        1.52902548e-01,  2.53622246e-02,  2.78534266e-02, -7.67747439e-02,
        1.00537560e-03,  1.03197302e-01,  9.72966984e-02,  7.09804800e-05,
        3.17828056e-01,  5.23580380e-02,  8.30746263e-02,  4.74556646e-01,
        1.91881914e-01, -3.70769342e-02, -5.37248884e-01,  2.29829975e-01]), array([-5.61544577e-02, -1.08127857e-02,  6.35132813e-02,  2.85244994e-02,
       -1.57282090e-01,  3.93072808e-01,  9.94503037e-03,  1.45443151e-01,
       -1.04265467e-01,  4.71940667e-01,  7.01142550e-02, -4.18354036e-02,
       -1.13657105e-01, -9.44002844e-02, -7.16501742e-02, -1.18549400e-01,
       -7.92548800e-02,  1.17486181e-01,  3.68553848e-03, -4.49183269e-02,
        2.73693071e-01, -1.33466417e-01, -1.36461905e-01,  1.08908035e-01,
        1.72024769e-03, -9.16375680e-02,  1.49470639e-01,  3.74950467e-05,
       -4.73435443e-02,  3.56665211e-02, -1.17880511e-02, -1.31223776e-01,
        5.28195242e-01, -4.56556888e-02,  1.65751807e-01, -6.79794916e-02]), array([-8.60460682e-02, -7.69129441e-02,  5.45689644e-03,  1.29701604e-02,
        5.96539358e-02,  2.55069569e-01, -9.82509949e-02, -6.51866068e-02,
        2.61847559e-02, -7.06856917e-02, -6.29224803e-01, -1.82008067e-02,
        7.30364384e-02,  9.13525702e-02, -2.57382227e-01, -4.00139085e-02,
        8.50446541e-02,  1.48799922e-01, -6.01444752e-03, -1.64577879e-01,
        1.02608669e-02, -1.63779353e-01,  1.07538401e-01,  5.29018328e-02,
        8.49716934e-04, -1.04791782e-01,  1.82581209e-03,  5.53877796e-06,
        1.99109979e-01,  2.29539722e-03, -2.62072780e-01,  4.20457059e-01,
       -1.16758722e-02,  6.98156518e-02,  2.13679014e-02, -1.45397024e-01]), array([-8.32127179e-02,  7.61476063e-03, -7.60203249e-02, -2.63443087e-02,
        2.84349617e-02, -2.64627798e-01, -5.84879184e-02,  1.41707513e-01,
        1.36641500e-01,  2.57855396e-01, -9.30221644e-02,  6.02719216e-02,
       -4.98059154e-02,  1.17085814e-01, -8.83271837e-02,  1.44287945e-01,
        1.30946132e-01,  1.12519757e-02, -4.31497427e-03, -3.55356251e-02,
        1.64011276e-01,  6.28331638e-02,  4.34680611e-01, -6.26095096e-01,
        2.11028881e-04, -5.43659179e-02,  6.60957510e-02,  3.44623265e-05,
       -5.29097911e-02,  5.67731420e-02, -1.01544900e-02, -1.16708133e-01,
        4.71780256e-02,  9.99755319e-02,  2.71665126e-01, -8.06914960e-02]), array([-1.04261080e-02, -2.83504515e-02, -1.30420529e-02,  4.17392316e-02,
       -4.57877583e-02,  6.11026294e-01, -2.91224464e-02,  1.14058334e-02,
        1.23533809e-01, -2.49484504e-01,  2.65249063e-01, -5.14573218e-02,
       -7.84070465e-02,  2.16734436e-01,  1.98364913e-01, -5.12721207e-02,
        1.53385437e-01,  1.92410828e-01, -1.85027461e-03,  9.61776380e-02,
       -6.19626513e-02, -7.87408666e-02,  4.29533240e-01, -1.70278814e-01,
       -9.66808813e-04,  1.57188017e-01, -5.04811340e-02, -2.25162904e-05,
        4.18458753e-02,  1.27321505e-02,  1.48534772e-01,  1.81466010e-02,
        2.48202045e-02, -3.62971796e-02, -1.58122987e-01,  5.74966367e-02]), array([-4.01221579e-02,  3.33675969e-02, -4.55315899e-02, -3.75883837e-03,
        2.49321878e-01, -1.81412763e-01, -2.75751309e-01, -2.31602480e-01,
        2.66556807e-01, -3.75279587e-01, -1.50742957e-02, -3.46961578e-02,
       -1.72865666e-03,  3.21320967e-02,  2.85727766e-02,  9.30828140e-02,
        4.89709361e-02,  7.65736985e-02,  4.14987082e-03,  3.37984591e-02,
        1.74367541e-01,  8.34506710e-02, -2.84376297e-02,  1.36583973e-01,
        1.61152697e-03, -5.18739965e-02,  8.24097713e-02, -6.81727923e-05,
       -7.56481388e-02,  6.23337168e-03,  9.23118682e-02, -8.34216641e-02,
        5.79667117e-01,  3.20944772e-01, -9.40302875e-02, -4.19884521e-02]), array([-4.85109718e-02,  4.04692848e-02,  1.91983493e-02,  7.92135012e-03,
        8.04200646e-03, -1.35303443e-01, -1.66796412e-01,  1.28759288e-01,
        2.25427332e-01,  1.94853954e-01, -4.63181375e-02, -5.31840385e-03,
        1.65837828e-01,  1.64086369e-01,  1.33684543e-01,  1.28551830e-01,
        5.59527050e-02,  1.58440348e-01,  1.15675202e-02,  1.76288431e-01,
        3.12760192e-01, -3.68390048e-01,  1.59872098e-01,  4.83425029e-01,
        1.01591568e-03,  1.38513467e-01,  6.59973321e-02, -3.66468572e-05,
       -4.80716862e-03,  8.89904951e-03,  2.10290209e-01, -7.64352590e-02,
       -3.24705969e-01,  4.98590749e-02,  1.37201855e-01, -1.83042929e-02]), array([ 2.99465547e-02,  2.06297049e-02, -1.05215702e-02,  3.18136677e-03,
        5.52041071e-02, -2.21553246e-01,  3.80920695e-02, -6.84907711e-02,
        2.11304160e-02, -1.52220077e-01,  9.67173930e-02,  7.48099992e-02,
       -3.08567031e-01, -2.91295617e-02, -9.81284168e-02,  2.78020877e-01,
        8.26789656e-02,  1.45789381e-01, -1.18294912e-03, -1.82712916e-01,
       -1.12149554e-01, -3.74261062e-01,  1.52996349e-01,  1.08698836e-01,
       -1.05274090e-03, -1.30695426e-01,  3.59425535e-03,  5.78096563e-05,
       -1.66105302e-02, -9.37354275e-03, -1.20705579e-01, -2.75306756e-02,
        1.71067500e-01, -6.36148743e-01,  1.91766102e-02,  4.76908514e-02]), array([-6.15502405e-02, -2.22713042e-02,  8.72946421e-02, -2.48431434e-03,
       -1.46762043e-01, -2.80467754e-02,  5.08817482e-02,  1.17250824e-01,
        5.44141391e-02,  9.34755171e-02,  1.35457396e-01,  2.27441866e-02,
       -4.36949953e-01,  1.88142019e-01, -9.21266838e-02,  1.10264580e-01,
        4.43040923e-02, -5.32998415e-02,  3.55948249e-03, -2.20620952e-01,
       -8.08497124e-02,  4.82142983e-01,  2.90365736e-01,  4.54832115e-01,
       -4.79551955e-04, -4.92674173e-02, -1.16161551e-02,  1.29307922e-05,
        2.29990041e-02, -2.89359146e-02, -1.78163008e-01,  5.12162988e-02,
       -5.51812421e-02,  2.10430026e-01,  3.13231192e-02, -4.02485300e-02]), array([ 3.62152440e-02,  2.08228602e-03, -4.20698297e-02,  4.45546954e-02,
        7.12276691e-02,  8.68110618e-02, -2.99342008e-01, -6.38334000e-02,
        2.18739802e-01, -4.32189003e-02,  2.00122992e-01, -1.23988545e-01,
       -1.90670615e-01, -2.09207926e-02, -2.53457046e-02, -1.42049800e-01,
       -8.35586136e-02,  5.56607447e-02,  3.09185613e-03, -3.09414375e-01,
        4.17991393e-01,  5.20304835e-02, -2.15895405e-01, -1.57771445e-01,
        2.11381912e-03, -1.94577689e-01,  2.43133736e-01, -1.90456201e-05,
       -1.18185587e-01,  1.67789187e-01, -1.66997744e-01,  2.93974598e-02,
       -3.87077879e-01, -9.30929773e-02, -2.11120571e-01,  1.46573693e-02]), array([ 1.20094243e-01, -6.56971520e-02,  2.40593923e-02,  1.26081465e-02,
        1.44091349e-01, -5.56083328e-02,  2.94553481e-01, -6.96089247e-02,
       -2.03014651e-01,  8.78263967e-03,  1.97265285e-01,  6.84115471e-02,
        3.39715230e-01, -2.72849359e-02,  3.16277301e-02, -2.19634284e-01,
        2.49749343e-01, -2.93467332e-01,  8.79225358e-03, -1.82979570e-01,
        1.12595654e-01, -2.14207641e-01,  3.49718442e-01,  1.43024493e-01,
        1.33449928e-03, -3.06499518e-01,  1.68227054e-01,  1.91343603e-04,
       -1.02098184e-01,  1.38906553e-01, -1.49317448e-01,  1.59737216e-03,
        4.47057052e-02,  1.59750951e-01, -1.68685354e-01,  3.42337654e-02]), array([-3.79741050e-02,  5.49412904e-02,  2.38017956e-02,  4.28942309e-03,
        4.10449127e-01,  4.88329821e-02, -3.55307686e-02, -4.26560639e-01,
       -4.81248742e-02,  2.27789156e-01, -5.16988289e-02,  6.15819297e-02,
       -3.00437273e-01, -3.03137137e-01,  4.79606703e-03, -2.86738499e-01,
        2.58886285e-01, -4.73156105e-02,  9.22077621e-03,  1.49437283e-01,
        4.56652216e-02,  1.54885942e-01,  1.20520306e-01,  6.53768242e-02,
       -1.93085369e-03, -7.36647723e-03, -1.18444126e-02,  3.19756472e-05,
        1.63399280e-01, -3.22444763e-02,  3.04733380e-01,  6.88541018e-02,
       -1.07482928e-01, -1.26374198e-01,  1.61207376e-01,  1.47027608e-02]), array([-8.82900047e-02,  3.96347479e-02, -2.62531926e-02,  2.45722228e-02,
       -1.36031341e-01,  3.54401714e-02, -1.13588594e-02,  7.84123702e-02,
        1.25827692e-01, -1.17542584e-01,  2.03317519e-01,  5.86676943e-02,
        1.98234312e-02,  4.24332310e-02,  6.97062602e-02,  9.88563129e-02,
       -6.82784865e-02, -5.91270181e-02,  1.01139379e-02,  6.82955755e-02,
       -1.82378552e-01, -6.21709553e-02, -1.30887054e-01, -2.77446027e-02,
        1.23863729e-03, -6.21690111e-01,  1.42465687e-01,  1.09611902e-04,
        3.77994357e-01,  1.24827975e-01,  3.05072306e-01,  2.08844986e-01,
       -3.84685666e-02,  9.28779019e-02,  3.02573132e-01, -2.22664806e-02]), array([-5.13168305e-02,  6.01540352e-02, -1.00486503e-02,  2.12304575e-02,
        2.45235258e-01,  2.25858787e-01,  1.45774600e-01, -2.11931153e-01,
       -1.46160251e-01,  1.01386275e-01,  6.89813351e-02, -6.00420020e-02,
       -2.42654909e-01,  9.77585404e-02, -1.82239290e-02,  5.84045081e-01,
       -9.90814600e-03, -3.25352502e-01,  4.63453724e-03, -3.33680786e-02,
        7.49503367e-02, -2.82984147e-01, -1.81707058e-01, -8.96523300e-02,
       -1.73299695e-03,  1.33329810e-01, -4.31541977e-02,  1.40554878e-04,
        4.33866550e-02,  6.53161181e-02, -5.24389085e-02,  1.22962344e-02,
       -8.42751107e-02,  3.07603880e-01, -3.02330516e-03,  4.16072278e-02]), array([ 2.88819525e-02,  1.22590715e-01,  1.66203633e-02,  1.90892489e-02,
       -2.25133821e-01,  3.76898563e-03, -1.45052082e-01,  2.90414163e-01,
        1.85221294e-01, -1.95632127e-01, -1.96606075e-01,  5.40377391e-02,
       -3.66715543e-01, -2.99483524e-01, -4.27537751e-02, -2.43186913e-01,
        3.11301304e-02, -4.96240355e-01,  1.35095463e-02,  3.95397509e-02,
       -8.14501978e-02, -3.12431237e-01,  7.61562917e-02,  4.72360373e-03,
        2.84481433e-04,  1.08857351e-01,  5.57768686e-03,  1.90492988e-04,
       -1.39213542e-01,  1.10048034e-01,  9.24432452e-02, -1.52960510e-02,
        1.57363168e-02,  1.28022040e-01, -1.62254046e-02, -5.83882726e-03])])
        whiten = False
        explained_variance = np.array([0.6815515930544795, 0.4526344545729429, 0.4033689806837877, 0.34914292637616984, 0.3214263280582067, 0.29390718944244165, 0.2644115600905508, 0.2620404193291989, 0.24810850746466095, 0.2406213532740573, 0.2308161449001211, 0.2190779914485933, 0.2071785038345719, 0.18691801869154978, 0.17995786703497138, 0.1707330119043283, 0.16508640108772119, 0.15896431679260994, 0.14856126449254486])
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
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mapped to 0 and 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(result)

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mappable to 0 and 1.")
        finally:
            if (result<0 or result>1):
                raise ValueError("Alpha version restriction: Integer class labels can only be 0 or 1.")
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
                outbuf.append("\n")
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)==2):
            raise ValueError("Number of classes must be 2.")


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    h_0 = max((((23.355984 * float(x[0]))+ (-1.144655 * float(x[1]))+ (-25.222584 * float(x[2]))+ (34.931545 * float(x[3]))+ (-30.09023 * float(x[4]))+ (-25.098967 * float(x[5]))+ (11.073931 * float(x[6]))+ (-2.43786 * float(x[7]))+ (35.74481 * float(x[8]))+ (16.459616 * float(x[9]))+ (42.99218 * float(x[10]))+ (-12.283966 * float(x[11]))+ (0.50164855 * float(x[12]))+ (45.16253 * float(x[13]))+ (4.6006985 * float(x[14]))+ (-32.78983 * float(x[15]))+ (6.233362 * float(x[16]))+ (9.433814 * float(x[17]))+ (-12.171234 * float(x[18]))) + 1.5547813), 0)
    h_1 = max((((-30.278349 * float(x[0]))+ (1.3150494 * float(x[1]))+ (29.826023 * float(x[2]))+ (-41.009514 * float(x[3]))+ (37.98212 * float(x[4]))+ (31.572939 * float(x[5]))+ (-11.913055 * float(x[6]))+ (3.3458757 * float(x[7]))+ (-41.19566 * float(x[8]))+ (-19.2157 * float(x[9]))+ (-51.79445 * float(x[10]))+ (15.051441 * float(x[11]))+ (-1.140062 * float(x[12]))+ (-57.912594 * float(x[13]))+ (-6.716998 * float(x[14]))+ (40.5025 * float(x[15]))+ (-7.144194 * float(x[16]))+ (-9.563942 * float(x[17]))+ (15.708486 * float(x[18]))) + -1.0709727), 0)
    h_2 = max((((24.647444 * float(x[0]))+ (-1.2574291 * float(x[1]))+ (-26.869411 * float(x[2]))+ (37.30588 * float(x[3]))+ (-31.934374 * float(x[4]))+ (-26.50712 * float(x[5]))+ (11.8705635 * float(x[6]))+ (-2.6392581 * float(x[7]))+ (38.023846 * float(x[8]))+ (17.58275 * float(x[9]))+ (45.627342 * float(x[10]))+ (-12.991337 * float(x[11]))+ (0.466313 * float(x[12]))+ (47.933052 * float(x[13]))+ (4.5829396 * float(x[14]))+ (-34.660343 * float(x[15]))+ (6.72716 * float(x[16]))+ (10.301332 * float(x[17]))+ (-12.771076 * float(x[18]))) + 1.7269475), 0)
    h_3 = max((((0.08512307 * float(x[0]))+ (-0.28619462 * float(x[1]))+ (5.521999 * float(x[2]))+ (-8.961931 * float(x[3]))+ (4.002982 * float(x[4]))+ (2.370928 * float(x[5]))+ (-9.125119 * float(x[6]))+ (2.6161678 * float(x[7]))+ (9.45194 * float(x[8]))+ (-2.4709775 * float(x[9]))+ (-4.4174943 * float(x[10]))+ (11.070464 * float(x[11]))+ (14.34681 * float(x[12]))+ (5.3370185 * float(x[13]))+ (-0.08157948 * float(x[14]))+ (10.189047 * float(x[15]))+ (-1.9282902 * float(x[16]))+ (5.382054 * float(x[17]))+ (7.0446925 * float(x[18]))) + 3.7371798), 0)
    h_4 = max((((1.1566626 * float(x[0]))+ (-0.171591 * float(x[1]))+ (1.3176494 * float(x[2]))+ (-1.3136826 * float(x[3]))+ (1.5555086 * float(x[4]))+ (-5.8014154 * float(x[5]))+ (0.9591418 * float(x[6]))+ (1.139169 * float(x[7]))+ (-1.3451976 * float(x[8]))+ (-10.125025 * float(x[9]))+ (6.3830714 * float(x[10]))+ (-3.2892327 * float(x[11]))+ (1.5451391 * float(x[12]))+ (6.4683805 * float(x[13]))+ (0.69163126 * float(x[14]))+ (3.1490595 * float(x[15]))+ (-0.4004612 * float(x[16]))+ (1.1663362 * float(x[17]))+ (-1.2428435 * float(x[18]))) + 8.627201), 0)
    h_5 = max((((2.3628922 * float(x[0]))+ (1.0253208 * float(x[1]))+ (1.5259595 * float(x[2]))+ (-1.8400084 * float(x[3]))+ (2.5691106 * float(x[4]))+ (-0.074656114 * float(x[5]))+ (1.1448107 * float(x[6]))+ (-1.9946674 * float(x[7]))+ (-0.4679394 * float(x[8]))+ (-1.197098 * float(x[9]))+ (-2.3652096 * float(x[10]))+ (-1.550829 * float(x[11]))+ (-1.7760717 * float(x[12]))+ (0.4224447 * float(x[13]))+ (1.3715845 * float(x[14]))+ (2.287754 * float(x[15]))+ (-1.0493511 * float(x[16]))+ (-2.111738 * float(x[17]))+ (1.9985958 * float(x[18]))) + 7.24776), 0)
    h_6 = max((((0.18749982 * float(x[0]))+ (-0.9444684 * float(x[1]))+ (0.10526032 * float(x[2]))+ (-0.29504967 * float(x[3]))+ (-0.21381019 * float(x[4]))+ (-3.0897236 * float(x[5]))+ (1.3275162 * float(x[6]))+ (-0.7341911 * float(x[7]))+ (0.7965452 * float(x[8]))+ (1.6810513 * float(x[9]))+ (-0.44066614 * float(x[10]))+ (0.51758236 * float(x[11]))+ (-0.24273264 * float(x[12]))+ (0.014068166 * float(x[13]))+ (-0.46210656 * float(x[14]))+ (0.15466176 * float(x[15]))+ (0.010487591 * float(x[16]))+ (-0.79295343 * float(x[17]))+ (1.3594533 * float(x[18]))) + -0.36211854), 0)
    h_7 = max((((1.925378 * float(x[0]))+ (0.66436636 * float(x[1]))+ (2.1805992 * float(x[2]))+ (-1.9940814 * float(x[3]))+ (4.275873 * float(x[4]))+ (-0.6734378 * float(x[5]))+ (-1.5013103 * float(x[6]))+ (-0.36065805 * float(x[7]))+ (-2.375857 * float(x[8]))+ (-1.7781576 * float(x[9]))+ (-1.6151761 * float(x[10]))+ (0.65419143 * float(x[11]))+ (0.590377 * float(x[12]))+ (0.6187521 * float(x[13]))+ (1.4410362 * float(x[14]))+ (0.5079879 * float(x[15]))+ (-0.77904046 * float(x[16]))+ (-0.08182907 * float(x[17]))+ (-0.67668724 * float(x[18]))) + 8.890342), 0)
    h_8 = max((((-0.27184018 * float(x[0]))+ (7.511412 * float(x[1]))+ (-0.59307796 * float(x[2]))+ (2.087139 * float(x[3]))+ (-0.09232035 * float(x[4]))+ (-2.3662755 * float(x[5]))+ (-0.40941334 * float(x[6]))+ (1.1813096 * float(x[7]))+ (0.3709433 * float(x[8]))+ (2.2382407 * float(x[9]))+ (-0.04473497 * float(x[10]))+ (-0.9339114 * float(x[11]))+ (1.0842323 * float(x[12]))+ (0.94737476 * float(x[13]))+ (-0.6335611 * float(x[14]))+ (-1.2282779 * float(x[15]))+ (0.75310034 * float(x[16]))+ (-1.9529392 * float(x[17]))+ (0.5819431 * float(x[18]))) + 3.8675792), 0)
    h_9 = max((((3.7538233 * float(x[0]))+ (0.04313261 * float(x[1]))+ (0.80492383 * float(x[2]))+ (-0.47726914 * float(x[3]))+ (2.2717953 * float(x[4]))+ (-0.5268948 * float(x[5]))+ (-0.7509072 * float(x[6]))+ (-1.4217904 * float(x[7]))+ (0.5581308 * float(x[8]))+ (0.66079533 * float(x[9]))+ (-0.89826345 * float(x[10]))+ (0.36178458 * float(x[11]))+ (0.069643 * float(x[12]))+ (2.5078576 * float(x[13]))+ (2.02043 * float(x[14]))+ (-1.0651289 * float(x[15]))+ (-0.7037192 * float(x[16]))+ (-0.0886507 * float(x[17]))+ (-0.3099716 * float(x[18]))) + 4.7640166), 0)
    o_0 = (4.1118264 * h_0)+ (-0.30567697 * h_1)+ (-3.530392 * h_2)+ (0.268642 * h_3)+ (-1.1280348 * h_4)+ (1.9044929 * h_5)+ (-1.0617093 * h_6)+ (4.1665487 * h_7)+ (-0.7418984 * h_8)+ (-5.23823 * h_9) + -13.087616

    if num_output_logits==1:
        return o_0>=0
    else:
        return argmax([eval('o'+str(i)) for i in range(num_output_logits)])

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

        model_cap=211

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


