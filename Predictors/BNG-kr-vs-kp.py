#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Mar-02-2020 03:55:35
# Invocation: btc Data/BNG-kr-vs-kp.csv -o Models/BNG-kr-vs-kp.py -v -v -v -stopat 95.9 -port 8090 -e 3
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                52.18%
Model accuracy:                     93.92% (939242/1000000 correct)
Improvement over best guess:        41.74% (of possible 47.82%)
Model capacity (MEC):               161 bits
Generalization ratio:               5833.80 bits/bit
Model efficiency:                   0.25%/parameter
System behavior
True Negatives:                     45.40% (453977/1000000)
True Positives:                     48.53% (485265/1000000)
False Negatives:                    3.66% (36610/1000000)
False Positives:                    2.41% (24148/1000000)
True Pos. Rate/Sensitivity/Recall:  0.93
True Neg. Rate/Specificity:         0.95
Precision:                          0.95
F-1 Measure:                        0.94
False Negative Rate/Miss Rate:      0.07
Critical Success Index:             0.89
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
TRAINFILE="BNG-kr-vs-kp.csv"


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
        mean = np.array([0.11265, 0.070508, 0.038076, 0.102058, 0.33523, 0.457102, 0.369484, 0.224762, 0.384992, 0.305612, 0.42659, 0.11743, 0.687292, 0.005626, 1.6316, 0.052746, 0.034516, 0.684494, 0.009518, 0.163128, 0.18991, 0.203946, 0.065296, 0.380042, 0.003928, 0.69535, 0.069302, 0.000612, 0.024798, 0.049496, 0.187532, 0.06836, 0.39663, 0.644736, 0.717374, 1.248026])
        components = np.array([array([ 1.00781237e-01,  2.51894342e-02,  6.57101466e-03,  2.06012867e-04,
       -7.47420692e-02, -2.34684347e-02,  1.15678818e-02, -1.32581512e-02,
       -8.50207811e-04, -1.60747457e-02,  4.16127426e-01, -1.41139577e-02,
        1.80671815e-01,  4.18121752e-03, -6.58093334e-01,  3.91764245e-03,
        1.50728411e-02, -9.73659318e-02,  1.12286423e-02,  2.05463432e-01,
        3.26229464e-02,  7.25848148e-03,  1.38363882e-02,  1.65782077e-02,
        2.63269311e-03,  3.07497531e-01, -1.42981267e-02, -5.82382099e-05,
        2.21505308e-02,  9.01547248e-03,  2.36139275e-01,  3.13597112e-02,
       -6.90764632e-02, -6.52888126e-02, -2.51488374e-01, -2.55618260e-01]), array([ 5.74296689e-02, -7.72657540e-02,  2.19688346e-02,  3.15409584e-02,
        3.28061710e-01,  4.10562573e-02,  5.08656512e-01,  4.63980421e-01,
        5.42029132e-01,  8.12926597e-02,  4.73377426e-02, -5.86647732e-02,
       -8.51254453e-02,  5.97727524e-04, -3.73859480e-02, -1.38753243e-02,
       -1.00265987e-02,  1.35309103e-01, -3.29158195e-03, -7.28076003e-03,
       -1.48933346e-01,  1.74994938e-01, -8.57976362e-03, -6.65712323e-02,
        1.10990460e-03,  5.19497999e-02, -2.38954146e-03,  1.12438026e-04,
        4.82942809e-03, -1.86525340e-02, -9.41187777e-02,  1.16067928e-02,
       -8.94113208e-03,  1.08086570e-02, -3.63574380e-02, -4.05000469e-02]), array([-2.68567113e-02,  8.82889089e-02,  3.47280235e-02,  1.11139856e-01,
        1.51806750e-01,  1.71394583e-01,  8.31357213e-02,  8.43960560e-02,
        7.32925614e-02, -1.02210879e-01, -1.31600775e-01, -2.84485594e-02,
        3.36560014e-01,  9.77887862e-04,  2.96851816e-03,  5.05320945e-02,
       -1.32168238e-02, -5.75664470e-01,  1.39193272e-02,  2.73606983e-02,
        1.25458181e-01, -1.56692275e-02, -3.17793760e-02,  1.08787704e-01,
       -1.12856038e-03, -1.34382477e-01,  4.11067128e-02, -1.44407033e-05,
       -4.80677084e-03,  4.63287547e-02,  1.87929776e-02, -2.69268666e-02,
        7.96073747e-02, -5.90467265e-01,  8.13230708e-02,  1.25502421e-01]), array([-5.50875900e-02, -5.07067603e-02, -1.61328984e-02,  1.24245668e-02,
        5.67635774e-02,  2.53503085e-01, -1.30177369e-02, -3.99100609e-02,
       -2.65840809e-02,  2.75632379e-02, -2.40124823e-01, -7.29502825e-03,
       -4.86788683e-02, -6.10332574e-03, -6.59093011e-01, -4.18247603e-02,
       -8.87591835e-03,  1.03349423e-01, -5.55899729e-03, -1.11762309e-01,
        5.01724667e-02,  1.25939505e-02, -2.68084236e-02, -3.30104089e-02,
       -4.61875036e-03, -2.87205924e-01, -6.78160075e-03, -1.38500194e-04,
        2.39132575e-02, -1.60433956e-02, -1.37740602e-01, -5.83925283e-02,
       -2.49076299e-01,  1.22962701e-01,  3.93457779e-01,  2.47976348e-01]), array([-6.16130661e-02,  6.50417060e-02, -5.45632954e-02, -4.04050544e-02,
       -1.23337695e-01, -2.01997680e-01, -3.06906379e-01, -1.09784114e-01,
        4.41968983e-01, -3.61368771e-01, -8.75487775e-02,  7.13147861e-02,
        1.14024806e-01, -1.92874510e-03, -1.25561496e-02,  1.10354868e-02,
        9.43990870e-03, -3.67421075e-02, -6.21928226e-04,  5.39986094e-02,
        1.27235217e-01,  6.03410232e-01,  1.15806116e-02, -1.95716902e-01,
       -1.93740532e-03, -9.92249833e-02,  2.47232735e-02, -2.38865593e-04,
        5.48171644e-03,  1.87161233e-02,  1.33391235e-01,  9.81497050e-03,
       -3.18825388e-03,  1.10169038e-01, -2.19200233e-02,  9.32701131e-02]), array([-7.79078290e-02,  4.12783783e-02, -1.24905398e-02, -1.68231450e-02,
       -1.06415859e-01, -1.35209674e-01, -7.57508429e-02,  1.38931721e-01,
        5.11315375e-02,  5.63884853e-01, -4.69019707e-02,  3.09156229e-02,
        9.12929362e-02, -1.31456661e-03, -1.30447571e-01,  2.42988751e-02,
        1.03264299e-02, -7.61556146e-03, -5.14265418e-03,  3.84547625e-02,
        2.78495905e-01, -6.31552614e-02,  3.16253647e-02, -4.16844083e-01,
        2.18828132e-03, -6.68100443e-02,  6.74344592e-02,  1.11369868e-04,
        9.93889563e-03,  4.46077163e-03,  3.60774439e-02, -1.42041619e-02,
        5.54293856e-01,  3.22790489e-02,  2.73949283e-02,  9.90580679e-02]), array([-4.03778870e-02, -2.98843713e-02, -6.52700634e-03,  1.31279757e-01,
       -4.84542695e-02,  8.09394218e-01, -7.73349488e-02, -1.10513721e-01,
        4.64971582e-02, -1.06898543e-01,  1.76623682e-01, -1.11202100e-01,
       -1.21210005e-01, -6.51590904e-03,  9.20537878e-02,  2.64396653e-02,
       -8.23869116e-03,  1.28627036e-01, -6.09008895e-04,  1.64461998e-02,
        9.09085221e-02,  1.66926297e-01, -5.06889787e-02, -1.18632100e-03,
        1.63291512e-03,  1.14743383e-01,  2.87512425e-02, -9.70496268e-05,
       -3.60245461e-02, -3.32968270e-02,  6.94162045e-02, -1.18655001e-01,
        3.57058363e-01,  5.31669604e-02, -2.43566807e-03, -4.76913742e-02]), array([ 5.60348999e-02, -2.66600063e-02,  4.91353984e-02,  3.12991432e-02,
        1.55030127e-01, -2.48800866e-01,  1.93235842e-02, -5.68474113e-02,
        1.23905731e-02, -1.26562559e-01, -7.07639837e-02, -1.17784839e-02,
       -1.46827891e-01,  2.24694077e-03, -2.09610589e-01,  1.43217826e-02,
       -2.18961496e-02,  8.29860354e-02,  7.37469314e-03, -3.37580587e-02,
        9.74512927e-02,  4.89405218e-02, -5.05517102e-02,  6.75982684e-01,
        1.33031781e-03, -7.74590385e-02,  1.23574753e-01,  1.49029778e-04,
        1.72404659e-02,  5.41143797e-02, -3.91912235e-02,  2.53326832e-02,
        5.19401387e-01,  9.79383555e-02, -1.06974383e-01,  1.36901555e-01]), array([-2.40749342e-02,  5.73402143e-02,  7.65796082e-02, -4.02775363e-02,
       -6.00859331e-01,  1.02734366e-01, -4.96696087e-02,  2.00395634e-01,
        2.61921738e-01,  3.74991133e-01,  2.21392785e-02,  1.29501414e-01,
        8.13761293e-02,  9.05217968e-04,  8.32250640e-02,  3.20812480e-03,
        3.08769234e-02, -4.23911861e-03,  7.39053890e-03,  6.39999949e-04,
        3.75488322e-02,  9.75666579e-02,  5.81640651e-02,  4.97871553e-01,
        1.20938943e-04,  1.45838783e-03, -6.12188360e-02, -4.37156436e-05,
       -6.81213025e-03, -1.08708656e-03,  4.89457774e-05, -3.02111636e-02,
       -1.85942786e-01, -1.39335380e-02,  1.66688372e-01, -6.25583265e-02]), array([ 1.81801193e-01,  2.38068261e-02,  3.46314241e-02,  3.79435980e-02,
       -3.74625990e-01, -2.63701588e-02, -5.08666993e-03,  4.70919179e-02,
        3.17307537e-02, -1.78418325e-01, -5.33162574e-02,  1.74125255e-01,
       -5.47405144e-01,  1.49114089e-02, -1.67944577e-01,  5.42349434e-02,
        4.54266437e-02, -6.87816681e-02, -6.26997154e-03, -2.32363554e-01,
       -1.02476162e-01,  4.44142694e-04,  1.09017474e-01, -1.95482217e-01,
        3.77422866e-04,  4.70018675e-02,  4.60462806e-02,  7.65732089e-05,
        5.65446523e-02,  4.65602313e-04, -2.98495419e-01,  1.00989095e-01,
        1.18141296e-01, -4.04097679e-01, -1.27798950e-01, -3.99833332e-02]), array([-1.81018373e-02,  1.42089988e-02, -2.92697413e-03,  9.02588989e-02,
        3.23849657e-01, -1.44994566e-01, -2.64768611e-01, -5.29793887e-02,
        7.02584688e-02,  2.40544108e-01,  1.77574476e-01, -1.87805411e-01,
       -4.44129822e-01, -4.58116496e-03,  4.75461725e-02, -3.58303926e-02,
       -4.45544055e-02, -1.24330434e-01,  9.75922076e-03, -1.44063331e-01,
        4.27134873e-01,  1.31895373e-01, -1.22448834e-01,  6.98425040e-02,
       -5.98425514e-04,  1.24769062e-01, -4.30576622e-02, -8.20501742e-05,
       -7.07755447e-02,  2.48830685e-02,  1.89383750e-02, -1.34000885e-01,
       -2.12225335e-01, -1.50749092e-01,  2.09662234e-01, -2.25816215e-01]), array([ 9.84018378e-02,  5.04201140e-02,  2.46122638e-02,  1.67361479e-01,
        1.79225524e-01,  2.23732362e-01, -1.50735068e-01, -8.39402181e-03,
        1.68149251e-02,  2.44941665e-01, -3.68589167e-01,  1.49815105e-01,
       -1.06411603e-01,  1.44239008e-02,  7.48754686e-03,  2.81270551e-02,
        9.67869848e-03,  5.68335384e-02, -3.55048252e-03,  2.10932549e-01,
        2.08241164e-01,  3.45790836e-02,  7.32368414e-02,  6.14058796e-02,
       -3.19630231e-03, -3.07513496e-02,  1.66593637e-02, -1.05085149e-04,
        1.03997609e-01, -1.25734139e-02,  4.70870467e-02,  3.83024830e-01,
       -2.49049437e-01,  7.13831626e-03, -5.39143479e-01,  8.84400580e-02]), array([-2.25866121e-02, -3.40549060e-02,  3.54224582e-03, -1.38334096e-01,
        4.33858265e-02,  6.78612265e-02,  1.15308746e-01, -1.28855461e-01,
       -2.72354776e-02, -1.77334041e-02, -1.60910757e-01,  7.65604978e-02,
        2.32917066e-01, -2.28833931e-03, -3.81447547e-02, -2.59111901e-02,
        6.27549619e-03, -7.67990023e-02, -8.68923251e-03, -2.46233675e-01,
        1.80305728e-01,  9.45560255e-02,  2.04019403e-02,  8.57355186e-03,
       -3.70978029e-03, -2.36981709e-01, -1.58069087e-01, -1.00838000e-04,
        3.72113040e-02, -4.49364045e-02, -3.24351677e-01,  1.22004973e-01,
        1.03411209e-01,  1.36045929e-01, -7.54451466e-02, -7.10963914e-01]), array([-9.34986040e-02,  4.78907184e-02, -3.89141518e-03,  5.12381898e-02,
       -4.19134857e-03, -2.59136527e-02, -5.60776020e-02,  9.64056877e-02,
        4.36993412e-03, -4.01263776e-02, -3.36388640e-02, -2.37027962e-02,
       -2.68591260e-01,  1.22485639e-04, -1.80610471e-02,  4.72584844e-02,
        8.80855046e-04,  1.15105783e-01,  3.80563066e-03,  4.12710611e-01,
       -2.18239862e-01, -8.33863524e-02,  1.17453751e-02,  9.90462813e-04,
        2.31694188e-04, -5.59260364e-01, -1.68871219e-02,  1.66248461e-04,
        1.03196376e-02, -1.60958434e-02,  4.01012279e-01,  5.23648974e-03,
        7.38238417e-02, -1.66780718e-01,  1.20313357e-01, -3.67225905e-01]), array([ 1.68287596e-01,  7.38341362e-02, -2.10165981e-03,  3.07697363e-01,
        1.52452839e-01, -2.69900803e-02, -4.68401583e-01,  2.33636063e-02,
        6.96003746e-02,  1.73073839e-01, -1.57895820e-01, -1.44228506e-01,
        2.34234294e-01,  1.49694633e-02, -6.09105221e-02,  9.30896029e-02,
       -4.44845365e-02,  1.42245179e-01, -1.35127760e-02, -1.58746065e-01,
       -5.35348689e-01,  6.52077355e-02, -6.79420413e-02,  2.53187393e-02,
        1.73414416e-03,  1.89942765e-01,  4.68115882e-02,  1.30481398e-04,
        2.76422734e-02, -1.38560031e-02, -9.72070534e-02,  1.15737424e-01,
        1.04467344e-01, -1.12404019e-01,  1.45436839e-01, -1.70752160e-01]), array([-1.59154489e-01,  4.53626924e-02,  5.62748613e-03, -1.05186707e-01,
        1.40938646e-01,  2.84459028e-03,  1.49918358e-01, -1.57037589e-01,
       -5.21907324e-02, -3.80015388e-03, -2.65284737e-01,  3.97191590e-01,
       -1.17711497e-01, -1.47822213e-03, -7.09606104e-03,  3.31809135e-02,
        7.18704546e-02, -5.69729807e-02, -4.21468593e-04,  3.20436271e-01,
       -3.49125063e-02,  1.07266478e-01,  1.63099284e-01,  2.56035437e-02,
       -4.04991856e-03,  4.86679723e-01, -1.09609316e-01,  7.73656490e-05,
        7.18498238e-02, -2.63582445e-02,  5.03012444e-02,  1.63181302e-01,
        1.24142856e-01, -1.75403731e-02,  4.33076902e-01, -1.01844631e-01]), array([ 4.60659314e-01, -9.43737587e-02, -3.14421550e-03,  3.54223938e-01,
        8.77723308e-02, -3.12355478e-03, -2.81261969e-02,  1.17394432e-01,
       -8.79573804e-04, -1.36326261e-01,  1.60832250e-01,  4.74196262e-01,
        1.40337632e-01,  1.43421037e-02,  8.85425616e-02, -4.82950001e-02,
        1.11310902e-01,  1.19512810e-01,  4.72935573e-03, -1.11668493e-01,
        2.39812362e-01, -1.35454992e-01,  2.10533474e-01, -3.09046167e-02,
        4.06312783e-03, -9.52146251e-02,  2.66154864e-01,  1.24284993e-04,
        2.50327028e-02,  7.97019904e-02,  1.34398475e-01, -5.28484233e-02,
       -2.99079021e-02,  8.36989066e-02,  2.23174461e-01, -7.72198894e-02]), array([-1.96443237e-01, -3.86027446e-04, -9.78484876e-02,  2.48299953e-01,
       -9.56592751e-02,  2.76453251e-02, -1.71476503e-01,  3.58941873e-01,
        1.15605615e-01, -2.16771951e-01, -1.43912747e-01, -7.81569272e-02,
       -1.53934404e-01, -8.87326929e-03, -3.78015686e-02, -6.45499930e-02,
       -2.60945328e-02, -4.71633025e-01,  1.83989870e-02,  3.16441225e-02,
       -8.57975872e-03, -3.24780930e-01, -4.63377979e-02, -1.38404034e-02,
       -7.61700446e-04,  1.30522654e-01,  2.91036764e-02,  2.80122395e-04,
        1.16920420e-02,  7.89482295e-02, -1.49026095e-02,  6.57002856e-02,
        3.02477222e-02,  4.81344786e-01,  1.08220953e-02, -1.07747949e-01])])
        whiten = False
        explained_variance = np.array([0.831609408859359, 0.5215126023279564, 0.4016373634269768, 0.315211120140888, 0.2872632409513757, 0.2737663917959896, 0.254755717126051, 0.23709448629083063, 0.20777337268549315, 0.18595225821031947, 0.16558474957612754, 0.14257241874717322, 0.13735268037883183, 0.12760828923154086, 0.11511494657934339, 0.112429263060151, 0.10148308239551054, 0.09274403733465061])
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
    h_0 = max((((-24.18152 * float(x[0]))+ (-4.9948273 * float(x[1]))+ (6.581068 * float(x[2]))+ (12.492879 * float(x[3]))+ (17.40023 * float(x[4]))+ (-62.29923 * float(x[5]))+ (-5.6951704 * float(x[6]))+ (-34.230328 * float(x[7]))+ (4.5377936 * float(x[8]))+ (-4.291315 * float(x[9]))+ (25.614206 * float(x[10]))+ (-26.564245 * float(x[11]))+ (-4.2951446 * float(x[12]))+ (-13.547977 * float(x[13]))+ (-34.554375 * float(x[14]))+ (-0.55292624 * float(x[15]))+ (33.390938 * float(x[16]))+ (2.1779656 * float(x[17]))) + -14.595626), 0)
    h_1 = max((((8.2300005 * float(x[0]))+ (9.791781 * float(x[1]))+ (-8.4725685 * float(x[2]))+ (-9.945468 * float(x[3]))+ (-28.513288 * float(x[4]))+ (57.405533 * float(x[5]))+ (2.2628815 * float(x[6]))+ (22.60005 * float(x[7]))+ (2.3211277 * float(x[8]))+ (-1.7307214 * float(x[9]))+ (-30.178644 * float(x[10]))+ (25.645985 * float(x[11]))+ (-0.40702182 * float(x[12]))+ (20.617573 * float(x[13]))+ (58.57175 * float(x[14]))+ (-4.1011486 * float(x[15]))+ (-38.6297 * float(x[16]))+ (-2.0553277 * float(x[17]))) + 22.67552), 0)
    h_2 = max((((-14.429364 * float(x[0]))+ (-30.272345 * float(x[1]))+ (-5.0133963 * float(x[2]))+ (-10.373171 * float(x[3]))+ (13.065969 * float(x[4]))+ (-15.76945 * float(x[5]))+ (5.945566 * float(x[6]))+ (-4.145516 * float(x[7]))+ (-8.836856 * float(x[8]))+ (-6.9517903 * float(x[9]))+ (11.195919 * float(x[10]))+ (-3.632243 * float(x[11]))+ (9.406217 * float(x[12]))+ (-8.846156 * float(x[13]))+ (-17.65168 * float(x[14]))+ (8.395921 * float(x[15]))+ (-13.463043 * float(x[16]))+ (-19.729725 * float(x[17]))) + -10.034197), 0)
    h_3 = max((((-3.2183514 * float(x[0]))+ (4.823864 * float(x[1]))+ (3.4739325 * float(x[2]))+ (5.419406 * float(x[3]))+ (-3.2250462 * float(x[4]))+ (8.360785 * float(x[5]))+ (-3.1502542 * float(x[6]))+ (-0.08870735 * float(x[7]))+ (5.276186 * float(x[8]))+ (-1.1216428 * float(x[9]))+ (-0.85054904 * float(x[10]))+ (7.7344975 * float(x[11]))+ (1.3860344 * float(x[12]))+ (7.747232 * float(x[13]))+ (1.2248399 * float(x[14]))+ (-8.287908 * float(x[15]))+ (3.9461591 * float(x[16]))+ (2.7113492 * float(x[17]))) + 14.832253), 0)
    h_4 = max((((0.15089831 * float(x[0]))+ (0.51859707 * float(x[1]))+ (-0.7564675 * float(x[2]))+ (2.241678 * float(x[3]))+ (-1.3048655 * float(x[4]))+ (-1.5428132 * float(x[5]))+ (-4.06441 * float(x[6]))+ (-4.5958714 * float(x[7]))+ (3.2055302 * float(x[8]))+ (-1.3819882 * float(x[9]))+ (2.7395785 * float(x[10]))+ (4.256706 * float(x[11]))+ (-0.3761945 * float(x[12]))+ (-0.88988537 * float(x[13]))+ (-0.34176904 * float(x[14]))+ (-1.4193708 * float(x[15]))+ (0.4085256 * float(x[16]))+ (-0.7675094 * float(x[17]))) + 5.0438733), 0)
    h_5 = max((((-2.235084 * float(x[0]))+ (-1.223118 * float(x[1]))+ (2.4544845 * float(x[2]))+ (1.8246588 * float(x[3]))+ (2.3085885 * float(x[4]))+ (0.3019952 * float(x[5]))+ (-0.6134036 * float(x[6]))+ (1.204257 * float(x[7]))+ (-0.6409761 * float(x[8]))+ (-0.42473376 * float(x[9]))+ (-0.90872914 * float(x[10]))+ (0.5761646 * float(x[11]))+ (2.1960306 * float(x[12]))+ (3.396169 * float(x[13]))+ (-2.8140187 * float(x[14]))+ (-3.6816325 * float(x[15]))+ (2.455841 * float(x[16]))+ (-0.3627872 * float(x[17]))) + 3.0853362), 0)
    h_6 = max((((-2.2913263 * float(x[0]))+ (-0.65210533 * float(x[1]))+ (0.36623424 * float(x[2]))+ (2.3133986 * float(x[3]))+ (1.2902507 * float(x[4]))+ (-0.5508148 * float(x[5]))+ (-1.0047174 * float(x[6]))+ (0.4126771 * float(x[7]))+ (-0.25962278 * float(x[8]))+ (0.44759604 * float(x[9]))+ (-1.217553 * float(x[10]))+ (0.60992396 * float(x[11]))+ (1.5057316 * float(x[12]))+ (2.6084368 * float(x[13]))+ (-1.1004019 * float(x[14]))+ (-1.0663056 * float(x[15]))+ (-0.27403948 * float(x[16]))+ (-0.29892197 * float(x[17]))) + 1.562359), 0)
    h_7 = max((((-0.8402554 * float(x[0]))+ (3.2086902 * float(x[1]))+ (1.056441 * float(x[2]))+ (0.8805031 * float(x[3]))+ (-2.2421665 * float(x[4]))+ (3.4508572 * float(x[5]))+ (-0.33285698 * float(x[6]))+ (-0.09955814 * float(x[7]))+ (2.4035928 * float(x[8]))+ (0.13687553 * float(x[9]))+ (-0.11711228 * float(x[10]))+ (1.9073008 * float(x[11]))+ (-0.73416114 * float(x[12]))+ (1.7318829 * float(x[13]))+ (1.4218522 * float(x[14]))+ (-1.7924542 * float(x[15]))+ (1.7714268 * float(x[16]))+ (2.2178702 * float(x[17]))) + 4.190079), 0)
    o_0 = (0.17175566 * h_0)+ (-0.14616054 * h_1)+ (0.13684833 * h_2)+ (2.6238625 * h_3)+ (-1.6168243 * h_4)+ (-2.1814222 * h_5)+ (-1.7204382 * h_6)+ (-4.5736523 * h_7) + 1.5693035

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

        model_cap=161

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


