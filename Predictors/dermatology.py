#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53544/dermatology.arff -o Predictors/dermatology_NN.py -target binaryClass -stopat 100 -f NN -e 20 --yes
# Total compiler execution time: 0:06:33.66. Finished on: Apr-21-2020 13:02:54.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                69.39%
Model accuracy:                     100.00% (366/366 correct)
Improvement over best guess:        30.61% (of possible 30.61%)
Model capacity (MEC):               43 bits
Generalization ratio:               8.51 bits/bit
Model efficiency:                   0.71%/parameter
System behavior
True Negatives:                     69.40% (254/366)
True Positives:                     30.60% (112/366)
False Negatives:                    0.00% (0/366)
False Positives:                    0.00% (0/366)
True Pos. Rate/Sensitivity/Recall:  1.00
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        1.00
False Negative Rate/Miss Rate:      0.00
Critical Success Index:             1.00

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
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "dermatology.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 34
n_classes = 2

mappings = [{0.0: 0, 7.0: 1, 8.0: 2, 9.0: 3, 10.0: 4, 12.0: 5, 13.0: 6, 15.0: 7, 16.0: 8, 17.0: 9, 18.0: 10, 19.0: 11, 20.0: 12, 21.0: 13, 22.0: 14, 23.0: 15, 24.0: 16, 25.0: 17, 26.0: 18, 27.0: 19, 28.0: 20, 29.0: 21, 30.0: 22, 31.0: 23, 32.0: 24, 33.0: 25, 34.0: 26, 35.0: 27, 36.0: 28, 37.0: 29, 38.0: 30, 39.0: 31, 40.0: 32, 41.0: 33, 42.0: 34, 43.0: 35, 44.0: 36, 45.0: 37, 46.0: 38, 47.0: 39, 48.0: 40, 49.0: 41, 50.0: 42, 51.0: 43, 52.0: 44, 53.0: 45, 55.0: 46, 56.0: 47, 57.0: 48, 60.0: 49, 61.0: 50, 62.0: 51, 64.0: 52, 65.0: 53, 67.0: 54, 68.0: 55, 70.0: 56, 1684325040.0: 57, 75.0: 58, 58.0: 59, 63.0: 60}]
list_of_cols_to_normalize = [33]

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
        mean = np.array([2.10546875, 1.83984375, 1.578125, 1.4296875, 0.62109375, 0.46484375, 0.171875, 0.3671875, 0.6171875, 0.578125, 0.13671875, 0.41015625, 0.14453125, 0.59375, 0.32421875, 1.3203125, 1.95703125, 0.52734375, 1.3359375, 0.70703125, 1.02734375, 0.6953125, 0.35546875, 0.39453125, 0.40625, 0.46484375, 0.47265625, 0.9140625, 0.4609375, 0.08984375, 0.1015625, 1.875, 0.578125, 28.4453125])
        components = np.array([array([-1.13253964e-04,  2.26654826e-03,  7.32321826e-03,  2.32189079e-03,
        7.07452337e-03,  7.11194712e-03, -1.36127625e-02,  6.75159592e-03,
       -5.63273978e-03,  2.82397769e-03, -1.32863192e-03,  5.65279166e-03,
        1.05490268e-03,  5.55411438e-03,  2.64422289e-03, -3.11606444e-03,
        2.47600005e-03, -3.21935086e-04,  4.49523116e-03,  1.10601860e-02,
        1.07337481e-02,  1.04288519e-02,  5.25903134e-03,  3.24506780e-03,
        5.16323601e-03,  5.19432089e-03,  7.16926380e-03, -8.41030727e-03,
        6.93982637e-03, -8.99129175e-03, -1.08584834e-02,  2.44397379e-03,
        7.60978266e-03,  9.99292114e-01]), array([-0.00288111, -0.05390238, -0.01224715,  0.1750667 ,  0.08914215,
        0.25684043, -0.01302918,  0.20523727, -0.21496147, -0.21525063,
       -0.03657784,  0.22761949,  0.02321961, -0.12980398, -0.00864049,
        0.26602791, -0.00196437, -0.07271269, -0.11344718, -0.27914736,
       -0.29721586, -0.27348384, -0.12931024, -0.15450532,  0.22390671,
       -0.13841057,  0.26030644,  0.16063638,  0.25199166, -0.00281158,
       -0.00339921,  0.02803314,  0.29864557,  0.00109371]), array([ 0.02142812,  0.064977  ,  0.28881788,  0.08320704,  0.18646866,
        0.24839822, -0.03597438,  0.19475322,  0.20369837,  0.21060648,
        0.02564178,  0.22124741, -0.0319508 ,  0.00485739, -0.12899055,
       -0.15450461,  0.082481  ,  0.01472828,  0.17662199,  0.26174744,
        0.14394837,  0.25360111,  0.10630337,  0.1623004 ,  0.21863603,
        0.17040807,  0.2517868 , -0.26561754,  0.24230842, -0.02386288,
       -0.02574689,  0.0913184 ,  0.2881201 , -0.02796941]), array([-0.21724888, -0.24497908, -0.07915016,  0.17347164, -0.0379394 ,
        0.01935691,  0.02749399,  0.00278494, -0.12822222, -0.11254483,
       -0.03493865,  0.01182091, -0.04066115, -0.24230765,  0.51034426,
       -0.29026673,  0.07271685,  0.05683965, -0.17182042, -0.08682939,
        0.32314473, -0.07860535, -0.10054666, -0.02148799,  0.02134086,
       -0.11363063,  0.0275433 , -0.48136233,  0.0223513 , -0.02307406,
       -0.02694934, -0.04720586,  0.0413527 , -0.00529261]), array([ 0.04481944,  0.08089635, -0.18529688,  0.847922  ,  0.1949872 ,
       -0.02568538, -0.13401297, -0.05848207, -0.08426843,  0.12772618,
       -0.02448408, -0.05315814,  0.07311329,  0.16732103,  0.01249719,
        0.11654798, -0.00103205,  0.04301082,  0.1094236 ,  0.05267628,
        0.06795411,  0.03005899,  0.05814671,  0.14880861, -0.06955393,
       -0.06804206, -0.06870811,  0.09059885, -0.0571611 , -0.10102513,
       -0.105689  , -0.05936854, -0.07685221, -0.00651655]), array([ 0.11330036,  0.1655575 ,  0.18629126,  0.01831831, -0.34845245,
        0.03084328, -0.21280129,  0.06883048, -0.18424057, -0.07456164,
       -0.01152241,  0.02445796,  0.01842397, -0.09028647,  0.24826959,
       -0.04791255,  0.39166563, -0.27112215,  0.28046131,  0.0090364 ,
        0.31680986, -0.01395185, -0.0087904 , -0.1520744 , -0.002196  ,
        0.16710303,  0.03119737,  0.36848992,  0.01306844, -0.09838865,
       -0.12866198,  0.12919749, -0.04867604, -0.00962831]), array([ 0.00698463,  0.09785263, -0.07118549,  0.1604736 , -0.47359174,
        0.05439706,  0.32370327,  0.02613547,  0.2606817 ,  0.04877461,
        0.07025573,  0.03541992,  0.03145699,  0.13605281,  0.04695265,
       -0.08883497,  0.03310569,  0.2543274 ,  0.44871423, -0.06959106,
       -0.06984587, -0.12761011,  0.0064141 , -0.1601911 ,  0.04942708,
       -0.22927339,  0.05613516, -0.0484291 ,  0.05541694,  0.20548123,
        0.23869196, -0.18693955,  0.06333326,  0.01263903]), array([-0.14362131, -0.04059048,  0.23130315,  0.07258036,  0.3176196 ,
       -0.02184017,  0.21851177,  0.01409746,  0.12619301,  0.02367271,
        0.04146218, -0.0405051 , -0.08212812, -0.51541317,  0.0819642 ,
        0.01318762,  0.19949103,  0.35319067,  0.08557041,  0.03557628,
       -0.01246343, -0.01115748, -0.2251311 ,  0.16817348, -0.05718252,
        0.06482127, -0.07292314,  0.38098622, -0.06406714,  0.15085407,
        0.15703217,  0.00226274, -0.09258198,  0.01033069]), array([ 0.34097704,  0.2415912 ,  0.36423225, -0.05235744,  0.10308597,
       -0.04224217, -0.03117743, -0.08449442, -0.18369958, -0.14426901,
        0.01498767, -0.01281082,  0.01358618,  0.01359775,  0.05674801,
        0.3169851 , -0.00639355,  0.43606706,  0.04751689, -0.07587908,
        0.12862201, -0.02799824,  0.15076811, -0.13497723, -0.00802903,
       -0.25473343, -0.02893657, -0.29250884, -0.06270919, -0.01452579,
       -0.03198948,  0.29581306, -0.0994363 , -0.00468671]), array([ 0.05715606,  0.12631724,  0.39194022, -0.00452811,  0.06788502,
        0.00957093, -0.17652128, -0.01287191, -0.2489913 , -0.20579511,
        0.00230278, -0.05075594, -0.03478966, -0.02040384, -0.06116893,
       -0.31587651, -0.31565335, -0.00260372,  0.27038474, -0.1083704 ,
       -0.15363712, -0.0188878 , -0.18410304,  0.09790386, -0.03399033,
        0.02381808, -0.04441295, -0.05901522, -0.02238646, -0.1034829 ,
       -0.10362492, -0.54523145, -0.02605656, -0.00444615]), array([ 0.04287355, -0.06235783, -0.09556292,  0.12958814, -0.33416877,
       -0.02705239, -0.01646452, -0.07723543, -0.10308751,  0.06297532,
        0.06922595, -0.01532984, -0.0331133 , -0.3152809 , -0.08625173,
        0.34024328, -0.31088052,  0.05348704,  0.15320684,  0.07446843,
       -0.06369078,  0.02147865, -0.29355677, -0.01205461, -0.03508482,
        0.54167768,  0.00323109, -0.26542237, -0.00451203, -0.006726  ,
       -0.00152832,  0.14772709,  0.00483637,  0.00188039]), array([ 0.19325036,  0.30016708,  0.29564252,  0.26097424, -0.011241  ,
        0.03049643,  0.11866416, -0.02158202,  0.28741932,  0.01777285,
        0.04954549, -0.05146425, -0.04535998,  0.01543124,  0.08821676,
       -0.14614266,  0.04385713, -0.17351084, -0.45796105, -0.11644161,
       -0.13298828, -0.24636358,  0.11471926, -0.20836397, -0.06487337,
        0.36914127, -0.03359362, -0.09397028, -0.04742695,  0.11489316,
        0.129332  , -0.06111189, -0.0727521 ,  0.00798809]), array([-0.05878959, -0.02161287,  0.054345  ,  0.08679844,  0.22769621,
       -0.05522361,  0.20146253, -0.01064563, -0.00982543, -0.09584111,
       -0.01467994, -0.01440564, -0.10344394, -0.11738176,  0.00095749,
       -0.18234899, -0.39867648, -0.46283445,  0.35667216, -0.0846981 ,
        0.03947139, -0.10432693,  0.16056449, -0.05211226, -0.00887242,
       -0.11711871, -0.04590037, -0.00357628, -0.03442117,  0.12963676,
        0.15639245,  0.45479628, -0.09401506,  0.00533074]), array([ 3.87546026e-01,  2.59859250e-01, -2.50362878e-01,  2.39465697e-02,
       -1.14396031e-01, -1.43128729e-02,  6.98423525e-02,  2.78622429e-02,
       -2.07047547e-01, -1.80495711e-02,  3.80487135e-02, -2.67723068e-04,
        7.88249851e-02, -7.29685563e-02, -1.53286278e-01, -4.37631400e-01,
        1.27957797e-01,  4.40773510e-02, -1.39310423e-01, -1.50296274e-01,
       -1.23670072e-01,  1.49854762e-01, -3.06245904e-01,  3.49248958e-01,
       -2.16870381e-02, -8.92910961e-02,  4.65232332e-02, -2.60333708e-02,
        3.49514984e-02,  8.01619755e-02,  8.07031557e-02,  2.98662593e-01,
        1.99364759e-02,  3.93375743e-03]), array([ 0.09683836,  0.3364009 , -0.07849302, -0.05110575,  0.04896943,
       -0.01679061, -0.03395471, -0.11199246,  0.23732422,  0.06414591,
        0.0660198 ,  0.00135488,  0.01948081, -0.34900034, -0.06673021,
        0.32900459,  0.09618567, -0.44001009, -0.00493481,  0.04260792,
        0.1568476 ,  0.0715089 , -0.21883517,  0.06127992,  0.0550021 ,
       -0.37182716, -0.02190582, -0.20058963, -0.02524945,  0.04449305,
        0.01972945, -0.2854621 , -0.00826307,  0.00372798]), array([-2.47270801e-01,  3.01941575e-01,  1.69378842e-02,  1.13175609e-01,
       -3.11073335e-01,  7.41387574e-02,  9.92681404e-02,  1.42438242e-01,
       -1.45122158e-01, -1.86833769e-01,  1.70133550e-02,  1.21605043e-02,
       -1.62663429e-01, -1.97322139e-01, -1.42044190e-01, -2.56539423e-02,
       -3.59128829e-01,  1.11716735e-01, -3.36591474e-01,  2.26448513e-01,
        2.25017894e-01,  2.37589390e-01,  2.52172089e-01,  3.05603398e-04,
       -1.16829997e-02, -1.50243510e-01,  3.06040428e-02,  2.31342035e-01,
        2.09640849e-02,  1.17057260e-02,  2.44148618e-02, -3.83609367e-02,
        5.51399849e-02, -1.63062386e-03]), array([-0.33103921,  0.04136331,  0.3827259 ,  0.04603335, -0.28236891,
       -0.01566589, -0.12726097, -0.00344829,  0.23828297,  0.10635979,
       -0.08320257, -0.00763955, -0.07874485,  0.24193023,  0.17753512,
        0.12153709, -0.09877713, -0.06160667, -0.10529091, -0.11793076,
       -0.15818106, -0.03421935, -0.2944344 ,  0.3985605 , -0.02629064,
       -0.20539649, -0.0112173 ,  0.01976385, -0.06029872, -0.09109581,
       -0.09786045,  0.28732294,  0.00918234,  0.00050868]), array([-0.36582719,  0.06429647,  0.08507458,  0.1016902 ,  0.07433841,
       -0.01346219,  0.18591913,  0.00615869, -0.2092237 , -0.41583665,
        0.07053308, -0.03232022, -0.08587742,  0.31396385, -0.25571408,
        0.1152829 ,  0.36226453, -0.12136131,  0.02502761,  0.01863873,
        0.01142088,  0.24084362, -0.21743741, -0.08192325,  0.03298555,
        0.12696248, -0.03848275, -0.20205548, -0.0649177 ,  0.19646747,
        0.1518634 ,  0.01395792, -0.11024302,  0.00060561]), array([ 0.39475984, -0.59294756,  0.31548868,  0.12304194, -0.19859787,
        0.12374748,  0.1687467 , -0.09370236, -0.15716777,  0.02321017,
        0.13215908, -0.05303682, -0.06778237,  0.03479056, -0.07967148,
        0.09877139,  0.02510786, -0.19677695, -0.14766043,  0.15924591,
        0.0652755 ,  0.088762  ,  0.03379685,  0.16612587, -0.06069479,
       -0.16716851,  0.00727186,  0.06705728, -0.03136651,  0.14088976,
        0.14855566, -0.10902169,  0.00968688,  0.00358867])])
        whiten = False
        explained_variance = np.array([221.96207647354765, 8.852009778521476, 4.462980900561971, 1.723364185455895, 1.1122525997306658, 0.9233680050746224, 0.7591601610246715, 0.6627783449242894, 0.5761670571321258, 0.5509761482735952, 0.5117350462121951, 0.4810531754781727, 0.41761922470430046, 0.3734963900570006, 0.33533934644204505, 0.29581113380066365, 0.2692852853583126, 0.2488739052151341, 0.23527313097941754])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"


    if (testfile):
        target=''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless==False):
                header=next(reader, None)
                try:
                    if (target!=''): 
                        hc=header.index(target)
                    else:
                        hc=len(header)-1
                        target=header[hc]
                except:
                    raise NameError("Target '"+target+"' not found! Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=header.index(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute '"+ignorecolumns[i]+"' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '"+ignorecolumns[i]+"' not found in header. Header must be same as in file passed to btc.")
                for i in range(0,len(header)):      
                    if (i==hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i]+",", end = '', file=outputfile)
                print(header[hc],file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"'+row[name]+'"'+",",end = '', file=outputfile)
                        else:
                            print (row[name]+",",end = '', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target!=""): 
                        hc=int(target)
                    else:
                        hc=-1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=int(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute "+str(col)+" is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    if (hc==-1):
                        hc=len(row)-1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0,len(row)):
                        if (i in il):
                            continue
                        if (i==hc):
                            continue
                        if (',' in row[i]):
                            print ('"'+row[i]+'"'+",",end = '', file=outputfile)
                        else:
                            print(row[i]+",",end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    clean.mapping={'N': 0, 'P': 1}

    def convert(cell):
        value = str(cell)
        try:
            result = int(value)
            return result
        except:
            try:
                result = float(value)
                if (rounding != -1):
                    result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
                return result
            except:
                result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
                return result

    # function to return key for any value 
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")

    def convertclassid(cell):
        if (clean.testfile):
            return convert(cell)
        value = str(cell)
        if (value == ''):
            raise ValueError("All cells in the target column must contain a class label.")

        if (not clean.mapping == {}):
            result = -1
            try:
                result = clean.mapping[cell]
            except:
                raise ValueError("Class label '" + value + "' encountered in input not defined in user-provided mapping.")
            if (not result == int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist = clean.classlist + [str(result)]
            return result
        try:
            result = float(cell)
            if (rounding != -1):
                result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
            else:
                result = int(int(result * 100) / 100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist = clean.classlist + [str(result)]
        except:
            result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
            if (result in clean.classlist):
                result = clean.classlist.index(result)
            else:
                clean.classlist = clean.classlist + [result]
                result = clean.classlist.index(result)
            if (not result == int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result < 0):
                raise ValueError("Integer class labels must be positive and contiguous.")

        return result

    rowcount = 0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f = open(outfile, "w+")
        if (headerless == False):
            next(reader, None)
        outbuf = []
        for row in reader:
            if (row == []):  # Skip empty rows
                continue
            rowcount = rowcount + 1
            rowlen = num_attr
            if (not testfile):
                rowlen = rowlen + 1    
            if (not len(row) == rowlen):
                raise ValueError("Column count must match trained predictor. Row " + str(rowcount) + " differs.")
            i = 0
            for elem in row:
                if(i + 1 < len(row)):
                    outbuf.append(str(convert(elem)))
                    outbuf.append(',')
                else:
                    classid = str(convertclassid(elem))
                    outbuf.append(classid)
                i = i + 1
            if (len(outbuf) < IOBUF):
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf = []
        print(''.join(outbuf), end="", file=f)
        f.close()

        if (testfile == False and not len(clean.classlist) >= 2):
            raise ValueError("Number of classes must be at least 2.")

        return get_key, clean.mapping

# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def classify(row):
    #inits
    x=row
    o=[0]*num_output_logits


    #Nueron Equations
    h_0 = max((((13.256219 * float(x[0]))+ (-10.762304 * float(x[1]))+ (11.781236 * float(x[2]))+ (-3.0427825 * float(x[3]))+ (0.29761577 * float(x[4]))+ (-2.7001338 * float(x[5]))+ (-1.6952348 * float(x[6]))+ (-2.003382 * float(x[7]))+ (1.3368387 * float(x[8]))+ (-0.4544082 * float(x[9]))+ (1.3504931 * float(x[10]))+ (-2.4470646 * float(x[11]))+ (0.037287608 * float(x[12]))+ (-0.22025594 * float(x[13]))+ (-0.30161417 * float(x[14]))+ (0.16244942 * float(x[15]))+ (-0.9932898 * float(x[16]))+ (2.4228084 * float(x[17]))+ (0.37271696 * float(x[18]))) + -7.456845), 0)
    h_1 = max((((0.1442651 * float(x[0]))+ (-7.157684 * float(x[1]))+ (4.4231896 * float(x[2]))+ (-2.840237 * float(x[3]))+ (2.4695134 * float(x[4]))+ (-1.2122456 * float(x[5]))+ (-4.7005796 * float(x[6]))+ (-1.723252 * float(x[7]))+ (3.139855 * float(x[8]))+ (2.8674297 * float(x[9]))+ (1.9493104 * float(x[10]))+ (-0.5611206 * float(x[11]))+ (-0.8646144 * float(x[12]))+ (2.9158876 * float(x[13]))+ (0.29740226 * float(x[14]))+ (2.2880707 * float(x[15]))+ (0.36125815 * float(x[16]))+ (2.206304 * float(x[17]))+ (-1.1291071 * float(x[18]))) + 0.4352072), 0)
    o[0] = (-0.0041508884 * h_0)+ (0.6234595 * h_1) + -5.4424386

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        writer = csv.writer(sys.stdout, lineterminator=os.linesep)
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            writer.writerow(','.join(next(reader, None) + ["Prediction"]))
        
        
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            writer.writerow(row)


def Validate(arr):
    if n_classes == 2:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        outputs=[]
        for i, row in enumerate(arr):
            outputs.append(int(classify(arr[i, :-1].tolist())))
        outputs=np.array(outputs)
        correct_count = int(np.sum(outputs.reshape(-1) == arr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(arr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(arr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0
    else:
        numeachclass = {}
        count, correct_count = 0, 0
        preds = []
        for i, row in enumerate(arr):
            pred = int(classify(arr[i].tolist()))
            preds.append(pred)
            if pred == int(float(arr[i, -1])):
                correct_count += 1
                if int(float(arr[i, -1])) in numeachclass.keys():
                    numeachclass[int(float(arr[i, -1]))] += 1
                else:
                    numeachclass[int(float(arr[i, -1]))] = 0
            count += 1
        return count, correct_count, numeachclass, preds
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()


    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempdir + os.sep + "clean.csv"
        preprocessedfile = tempdir + os.sep + "prep.csv"
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x,y: x
        classmapping = {}


    #load file
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')


    #Normalize
    cleanarr = Normalize(cleanarr)


    #Transform
    if transform_true:
        if args.validate:
            trans = transform(cleanarr[:, :-1])
            cleanarr = np.concatenate((trans, cleanarr[:, -1].reshape(-1, 1)), axis = 1)
        else:
            cleanarr = transform(cleanarr)


    #Predict
    if not args.validate:
        Predict(cleanarr, args.headerless, preprocessedfile, get_key, classmapping)


    #Validate
    else: 
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
            #Correct Labels
            true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap=43
        if n_classes == 2:
            #Base metrics
            FN = float(num_FN) * 100.0 / float(count)
            FP = float(num_FP) * 100.0 / float(count)
            TN = float(num_TN) * 100.0 / float(count)
            TP = float(num_TP) * 100.0 / float(count)
            num_correct = correct_count

            #Calculated Metrics
            if int(num_TP + num_FN) != 0:
                TPR = num_TP / (num_TP + num_FN) # Sensitivity, Recall
            if int(num_TN + num_FP) != 0:
                TNR = num_TN / (num_TN + num_FP) # Specificity
            if int(num_TP + num_FP) != 0:
                PPV = num_TP / (num_TP + num_FP) # Recall
            if int(num_FN + num_TP) != 0:
                FNR = num_FN / (num_FN + num_TP) # Miss rate
            if int(2 * num_TP + num_FP + num_FN) != 0:
                FONE = 2 * num_TP / (2 * num_TP + num_FP + num_FN) # F1 Score
            if int(num_TP + num_FN + num_FP) != 0:
                TS = num_TP / (num_TP + num_FN + num_FP) # Critical Success Index
            #Best Guess Accuracy
            randguess = int(float(10000.0 * max(num_class_1, num_class_0)) / count) / 100.0
            #Model Accuracy
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            #Report
            print("System Type:                        Binary classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
            print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))
            print("System behavior")
            print("True Negatives:                     {:.2f}%".format(TN) + " (" + str(int(num_TN)) + "/" + str(count) + ")")
            print("True Positives:                     {:.2f}%".format(TP) + " (" + str(int(num_TP)) + "/" + str(count) + ")")
            print("False Negatives:                    {:.2f}%".format(FN) + " (" + str(int(num_FN)) + "/" + str(count) + ")")
            print("False Positives:                    {:.2f}%".format(FP) + " (" + str(int(num_FP)) + "/" + str(count) + ")")
            if int(num_TP + num_FN) != 0:
                print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
            if int(num_TN + num_FP) != 0:
                print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
            if int(num_TP + num_FP) != 0:
                print("Precision:                          {:.2f}".format(PPV))
            if int(2 * num_TP + num_FP + num_FN) != 0:
                print("F-1 Measure:                        {:.2f}".format(FONE))
            if int(num_TP + num_FN) != 0:
                print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
            if int(num_TP + num_FN + num_FP) != 0:
                print("Critical Success Index:             {:.2f}".format(TS))

        #Multiclass
        else:
            num_correct = correct_count
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            randguess = round(max(numeachclass.values()) / sum(numeachclass.values()) * 100, 2)
            print("System Type:                        " + str(n_classes) + "-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")





            def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
                #check for numpy/scipy is imported
                try:
                    from scipy.sparse import coo_matrix #required for multiclass metrics
                    try:
                        np.array
                    except:
                        import numpy as np
                except:
                    raise ValueError("Scipy and Numpy Required for Multiclass Metrics")
                # Compute confusion matrix to evaluate the accuracy of a classification.
                # By definition a confusion matrix :math:C is such that :math:C_{i, j}
                # is equal to the number of observations known to be in group :math:i and
                # predicted to be in group :math:j.
                # Thus in binary classification, the count of true negatives is
                # :math:C_{0,0}, false negatives is :math:C_{1,0}, true positives is
                # :math:C_{1,1} and false positives is :math:C_{0,1}.
                # Read more in the :ref:User Guide <confusion_matrix>.
                # Parameters
                # ----------
                # y_true : array-like of shape (n_samples,)
                # Ground truth (correct) target values.
                # y_pred : array-like of shape (n_samples,)
                # Estimated targets as returned by a classifier.
                # labels : array-like of shape (n_classes), default=None
                # List of labels to index the matrix. This may be used to reorder
                # or select a subset of labels.
                # If None is given, those that appear at least once
                # in y_true or y_pred are used in sorted order.
                # sample_weight : array-like of shape (n_samples,), default=None
                # Sample weights.
                # normalize : {'true', 'pred', 'all'}, default=None
                # Normalizes confusion matrix over the true (rows), predicted (columns)
                # conditions or all the population. If None, confusion matrix will not be
                # normalized.
                # Returns
                # -------
                # C : ndarray of shape (n_classes, n_classes)
                # Confusion matrix.
                # References
                # ----------
                if labels is None:
                    labels = np.array(list(set(list(y_true.astype('int')))))
                else:
                    labels = np.asarray(labels)
                    if np.all([l not in y_true for l in labels]):
                        raise ValueError("At least one label specified must be in y_true")


                if sample_weight is None:
                    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
                else:
                    sample_weight = np.asarray(sample_weight)
                if y_true.shape[0]!=y_pred.shape[0]:
                    raise ValueError("y_true and y_pred must be of the same length")

                if normalize not in ['true', 'pred', 'all', None]:
                    raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


                n_labels = labels.size
                label_to_ind = {y: x for x, y in enumerate(labels)}
                # convert yt, yp into index
                y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
                y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
                # intersect y_pred, y_true with labels, eliminate items not in labels
                ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
                y_pred = y_pred[ind]
                y_true = y_true[ind]
                # also eliminate weights of eliminated items
                sample_weight = sample_weight[ind]
                # Choose the accumulator dtype to always have high precision
                if sample_weight.dtype.kind in {'i', 'u', 'b'}:
                    dtype = np.int64
                else:
                    dtype = np.float64
                cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype,).toarray()


                with np.errstate(all='ignore'):
                    if normalize == 'true':
                        cm = cm / cm.sum(axis=1, keepdims=True)
                    elif normalize == 'pred':
                        cm = cm / cm.sum(axis=0, keepdims=True)
                    elif normalize == 'all':
                        cm = cm / cm.sum()
                    cm = np.nan_to_num(cm)
                return cm


            print("Confusion Matrix:")
            mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)
