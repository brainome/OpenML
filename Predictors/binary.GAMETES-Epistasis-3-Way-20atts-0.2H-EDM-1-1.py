#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class GAMETES-Epistasis-3-Way-20atts-0.2H-EDM-1-1.csv -o GAMETES-Epistasis-3-Way-20atts-0.2H-EDM-1-1_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:18:53.82. Finished on: Sep-03-2020 17:33:48.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 50.00%
Overall Model accuracy:              63.75% (1020/1600 correct)
Overall Improvement over best guess: 13.75% (of possible 50.0%)
Model capacity (MEC):                115 bits
Generalization ratio:                8.86 bits/bit
Model efficiency:                    0.11%/parameter
System behavior
True Negatives:                      31.50% (504/1600)
True Positives:                      32.25% (516/1600)
False Negatives:                     17.75% (284/1600)
False Positives:                     18.50% (296/1600)
True Pos. Rate/Sensitivity/Recall:   0.65
True Neg. Rate/Specificity:          0.63
Precision:                           0.64
F-1 Measure:                         0.64
False Negative Rate/Miss Rate:       0.35
Critical Success Index:              0.47
Confusion Matrix:
 [31.50% 18.50%]
 [17.75% 32.25%]
Overfitting:                         No
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
try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "GAMETES-Epistasis-3-Way-20atts-0.2H-EDM-1-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 20
n_classes = 2

mappings = []
list_of_cols_to_normalize = []

transform_true = True

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values())) + 1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize, mappings):
            if i >= data_arr.shape[1]:
                break
            col = data_arr[:, i]
            normcol = column_norm(col,mapping)
            data_arr[:, i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([0.9635416666666666, 0.6145833333333334, 0.42291666666666666, 0.21145833333333333, 0.4875, 0.21666666666666667, 0.15, 0.69375, 0.2947916666666667, 0.26979166666666665, 0.5010416666666667, 0.3697916666666667, 0.115625, 0.09791666666666667, 0.596875, 0.5916666666666667, 0.8125, 0.39895833333333336, 0.41770833333333335, 0.4041666666666667])
        components = np.array([array([ 0.90343165,  0.13219696,  0.24581539, -0.01188678,  0.03250309,
       -0.00292896, -0.0250531 ,  0.08050638, -0.04249223,  0.05514745,
       -0.00970733, -0.1429342 ,  0.02521161,  0.00655529,  0.22645539,
       -0.01707862, -0.12922893, -0.04728815,  0.02669501,  0.0179518 ]), array([ 0.02993678,  0.16924351,  0.108654  ,  0.0436865 , -0.0261787 ,
        0.01852997, -0.03922188,  0.70351946,  0.04604746,  0.03306089,
       -0.09914333, -0.00164428,  0.01440218,  0.02006894, -0.27884677,
        0.16534562,  0.55638327, -0.05447605,  0.12646334, -0.10939022]), array([-0.20248523,  0.46117448, -0.05204001,  0.01738284,  0.04552854,
        0.01911525, -0.00441615,  0.33835087,  0.04732631,  0.0368821 ,
        0.16352618, -0.05596335, -0.03282144, -0.00969055,  0.10951894,
        0.44822998, -0.60954316, -0.05767918,  0.04686991,  0.04187438]), array([-0.17000623,  0.00904784, -0.07210708,  0.00079195, -0.0480283 ,
        0.04626026, -0.00780169,  0.33675953,  0.01374488, -0.01642805,
        0.22052132,  0.08232117,  0.02778881, -0.00867017,  0.76316064,
       -0.41347361,  0.13326526,  0.11989815,  0.07313219,  0.01818401]), array([ 0.05504471, -0.37876082, -0.02091811, -0.02902815,  0.11973529,
        0.0368908 ,  0.04328459,  0.42674974,  0.00530507, -0.01443931,
        0.21598478,  0.04323004,  0.00719313,  0.01102269, -0.35715076,
       -0.44664961, -0.41459517, -0.13285014, -0.23785496, -0.17449864]), array([ 0.05490365,  0.54209764, -0.24572347, -0.03354076,  0.6103648 ,
       -0.10446059,  0.05938544, -0.17153121,  0.00394961, -0.0483591 ,
        0.25971221,  0.07496819,  0.05370691,  0.03135266, -0.15081156,
       -0.2828367 ,  0.17860545, -0.06634095, -0.03771301, -0.07380808]), array([-0.04043393, -0.34961981,  0.2606317 ,  0.00201675,  0.32576439,
        0.006884  , -0.030475  , -0.05665302,  0.12350061,  0.08881622,
        0.57669838, -0.17486554, -0.03355038, -0.0021829 ,  0.14015827,
        0.41045029,  0.22502512, -0.09959605, -0.24857333,  0.00133031]), array([ 0.14597584,  0.05820911, -0.12631302, -0.00452435, -0.50794001,
       -0.09295799, -0.00249942, -0.12118574,  0.06110346, -0.0098117 ,
        0.55504656,  0.40158789, -0.04046789,  0.02451455, -0.13483062,
        0.01866997,  0.05690252, -0.31108294,  0.28982931, -0.00578036]), array([-0.2154034 ,  0.37748006,  0.55424423,  0.09230841, -0.30741363,
        0.02758136,  0.01379216, -0.1080094 ,  0.00938975,  0.03199603,
        0.04414636, -0.24908739, -0.04906753, -0.03197227, -0.04953142,
       -0.30391926,  0.04159599, -0.29178733, -0.36745394,  0.05508727]), array([ 0.10808939,  0.16748012, -0.00485114,  0.00871954, -0.24844815,
       -0.03766755, -0.00731205,  0.01072824,  0.01027172, -0.01679022,
        0.21589116,  0.18164505,  0.02122673,  0.00481498, -0.12166065,
        0.06686138,  0.02978887,  0.75894898, -0.45303561, -0.12241155]), array([-0.0861986 , -0.03328858,  0.46025403, -0.07352976,  0.17967543,
       -0.14382305,  0.00691916,  0.03751323,  0.02825615,  0.09837916,
        0.12207027,  0.13344866, -0.0214969 ,  0.01414052, -0.19530469,
       -0.15872218, -0.09162308,  0.33716001,  0.4189746 ,  0.56124964]), array([-0.06955443,  0.00585341,  0.42484037, -0.01458479,  0.19717395,
       -0.07894293,  0.00255118, -0.05619977, -0.092511  ,  0.1622962 ,
       -0.20559181,  0.66614086, -0.05239992,  0.00629665,  0.1466682 ,
        0.08400315, -0.0750524 , -0.08846824, -0.02290706, -0.44710591]), array([ 0.06665729, -0.01075224, -0.20456723,  0.12268377,  0.03652272,
        0.09226933, -0.02203149,  0.10585561, -0.29112587,  0.26675616,
       -0.10357677,  0.36826319,  0.05186955,  0.01654301,  0.03383098,
        0.07545646,  0.07363607, -0.21323317, -0.43706966,  0.60251298]), array([ 0.063299  , -0.01323399, -0.02096373, -0.01513403,  0.02013628,
       -0.24031222,  0.00901147,  0.03464631,  0.83854169, -0.24584812,
       -0.20000829,  0.19508661,  0.04589695,  0.01212454,  0.06588847,
        0.008391  , -0.01575277, -0.10055984, -0.20379668,  0.1884098 ]), array([-0.01239504,  0.01026465, -0.1523457 ,  0.07034903, -0.05363786,
       -0.11938397, -0.03763297, -0.04870083,  0.30722906,  0.89349474,
       -0.02593699, -0.14203852,  0.03167606, -0.05473878, -0.02260037,
       -0.10359033, -0.0135476 ,  0.05317462,  0.06034635, -0.12329163]), array([ 0.03437027,  0.02144272,  0.05081593,  0.59020158,  0.0837191 ,
        0.7175934 , -0.09690725, -0.09164557,  0.23731042, -0.02955302,
        0.03916113,  0.11233671,  0.06154648,  0.05582939, -0.05898728,
       -0.04329718, -0.03196675,  0.0748364 ,  0.13129532, -0.0153954 ]), array([-8.62075924e-03, -6.55231547e-02, -2.11995054e-02,  7.34276060e-01,
        2.91300611e-02, -5.84875038e-01, -2.57331782e-01,  2.68706797e-02,
       -1.55671571e-01, -1.13278778e-01,  1.01687296e-02, -6.62302055e-02,
        5.15967031e-04, -1.26405371e-02,  2.00747782e-02, -1.26707067e-02,
       -4.08191184e-02,  1.24142965e-02,  2.53694908e-02, -3.75435114e-02])])
        whiten = False
        explained_variance = np.array([0.5277784872647473, 0.5075948775655422, 0.4955411564123689, 0.4663566156551138, 0.438119524221023, 0.394743983132737, 0.37018569858579137, 0.35506124636726377, 0.34603782682800965, 0.32567316446800443, 0.323120723013934, 0.29303879191358057, 0.2813572623618576, 0.24641901848520983, 0.21382533187129854, 0.1959066726048105, 0.16574943840186576])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="class"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="class"
    if ignorelabels == [] and ignorecolumns == [] and target == "":
        return
    if (testfile):
        target = ''
        hc = -1
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless == False):
                header=next(reader, None)
                try:
                    if not testfile:
                        if (target != ''): 
                            hc = header.index(target)
                        else:
                            hc = len(header) - 1
                            target=header[hc]
                except:
                    raise NameError("Target '" + target + "' not found! Header must be same as in file passed to btc.")
                for i in range(0, len(ignorecolumns)):
                    try:
                        col = header.index(ignorecolumns[i])
                        if not testfile:
                            if (col == hc):
                                raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
                        il = il + [col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
                first = True
                for i in range(0, len(header)):

                    if (i == hc):
                        continue
                    if (i in il):
                        continue
                    if first:
                        first = False
                    else:
                        print(",", end='', file=outputfile)
                    print(header[i], end='', file=outputfile)
                if not testfile:
                    print("," + header[hc], file=outputfile)
                else:
                    print("", file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if target and (row[target] in ignorelabels):
                        continue
                    first = True
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name == target):
                            continue
                        if first:
                            first = False
                        else:
                            print(",", end='', file=outputfile)
                        if (',' in row[name]):
                            print('"' + row[name].replace('"', '') + '"', end='', file=outputfile)
                        else:
                            print(row[name].replace('"', ''), end='', file=outputfile)
                    if not testfile:
                        print("," + row[target], file=outputfile)
                    else:
                        print("", file=outputfile)

            else:
                try:
                    if (target != ""): 
                        hc = int(target)
                    else:
                        hc = -1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0, len(ignorecolumns)):
                    try:
                        col = int(ignorecolumns[i])
                        if (col == hc):
                            raise ValueError("Attribute " + str(col) + " is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il = il + [col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    first = True
                    if (hc == -1) and (not testfile):
                        hc = len(row) - 1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0, len(row)):
                        if (i in il):
                            continue
                        if (i == hc):
                            continue
                        if first:
                            first = False
                        else:
                            print(",", end='', file=outputfile)
                        if (',' in row[i]):
                            print('"' + row[i].replace('"', '') + '"', end='', file=outputfile)
                        else:
                            print(row[i].replace('"', ''), end = '', file=outputfile)
                    if not testfile:
                        print("," + row[hc], file=outputfile)
                    else:
                        print("", file=outputfile)


def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    #This function takes a preprocessed csv and cleans it to real numbers for prediction or validation


    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    

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

    #Function to return key for any value 
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")


    #Function to convert the class label
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


    #Main Cleaning Code
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
def single_classify(row):
    #inits
    x = row
    o = [0] * num_output_logits


    #Nueron Equations
    h_0 = max((((-3.3855824 * float(x[0]))+ (-0.09773637 * float(x[1]))+ (0.89427495 * float(x[2]))+ (6.723951 * float(x[3]))+ (0.48419166 * float(x[4]))+ (1.3054081 * float(x[5]))+ (5.295794 * float(x[6]))+ (-2.1529782 * float(x[7]))+ (5.476336 * float(x[8]))+ (-3.643562 * float(x[9]))+ (-2.8182702 * float(x[10]))+ (-0.57564104 * float(x[11]))+ (-3.4824972 * float(x[12]))+ (2.4104319 * float(x[13]))+ (-3.3316607 * float(x[14]))+ (-5.240546 * float(x[15]))+ (-0.39406517 * float(x[16]))) + 2.0013247), 0)
    h_1 = max((((5.8635516 * float(x[0]))+ (-1.2655734 * float(x[1]))+ (3.1631765 * float(x[2]))+ (-4.5187993 * float(x[3]))+ (-3.5550027 * float(x[4]))+ (-3.089712 * float(x[5]))+ (-5.6352863 * float(x[6]))+ (0.5969295 * float(x[7]))+ (-4.1445236 * float(x[8]))+ (0.7717809 * float(x[9]))+ (2.4255784 * float(x[10]))+ (5.3796678 * float(x[11]))+ (4.7257 * float(x[12]))+ (-3.6520848 * float(x[13]))+ (8.836206 * float(x[14]))+ (0.9180758 * float(x[15]))+ (-1.8355923 * float(x[16]))) + -2.360452), 0)
    h_2 = max((((0.22608756 * float(x[0]))+ (0.6035547 * float(x[1]))+ (1.9330916 * float(x[2]))+ (-0.84649396 * float(x[3]))+ (-0.22776294 * float(x[4]))+ (-1.4843647 * float(x[5]))+ (2.4867625 * float(x[6]))+ (-1.3314658 * float(x[7]))+ (2.2394576 * float(x[8]))+ (5.417208 * float(x[9]))+ (1.0977473 * float(x[10]))+ (-0.51461136 * float(x[11]))+ (2.6249928 * float(x[12]))+ (2.5139196 * float(x[13]))+ (0.8280765 * float(x[14]))+ (-0.57524955 * float(x[15]))+ (1.0770016 * float(x[16]))) + -2.1732755), 0)
    h_3 = max((((-2.3500953 * float(x[0]))+ (-0.19000655 * float(x[1]))+ (-0.21902463 * float(x[2]))+ (1.7356626 * float(x[3]))+ (0.8063977 * float(x[4]))+ (-0.4374208 * float(x[5]))+ (2.8413417 * float(x[6]))+ (-1.1550809 * float(x[7]))+ (2.2245276 * float(x[8]))+ (0.35834944 * float(x[9]))+ (-0.517393 * float(x[10]))+ (0.13071069 * float(x[11]))+ (-1.4367448 * float(x[12]))+ (1.6045272 * float(x[13]))+ (-1.3149571 * float(x[14]))+ (-2.0120275 * float(x[15]))+ (0.9347654 * float(x[16]))) + -0.5362229), 0)
    h_4 = max((((-0.7314846 * float(x[0]))+ (0.04251917 * float(x[1]))+ (0.5875897 * float(x[2]))+ (2.3700006 * float(x[3]))+ (-0.107652724 * float(x[4]))+ (0.71727246 * float(x[5]))+ (1.3900198 * float(x[6]))+ (-0.75132316 * float(x[7]))+ (1.8020146 * float(x[8]))+ (-1.7766099 * float(x[9]))+ (-1.164621 * float(x[10]))+ (-0.12533377 * float(x[11]))+ (-0.8736205 * float(x[12]))+ (0.5309801 * float(x[13]))+ (-0.78788567 * float(x[14]))+ (-1.7973201 * float(x[15]))+ (-0.3345051 * float(x[16]))) + 0.5603319), 0)
    h_5 = max((((0.33009565 * float(x[0]))+ (-0.05582374 * float(x[1]))+ (0.10616732 * float(x[2]))+ (-0.3008542 * float(x[3]))+ (-0.29045343 * float(x[4]))+ (-0.07681323 * float(x[5]))+ (-0.7092757 * float(x[6]))+ (-0.06304967 * float(x[7]))+ (-0.34899515 * float(x[8]))+ (-0.3039939 * float(x[9]))+ (-0.01015383 * float(x[10]))+ (0.3237362 * float(x[11]))+ (0.30839196 * float(x[12]))+ (-0.2116312 * float(x[13]))+ (0.7027851 * float(x[14]))+ (0.03542162 * float(x[15]))+ (-0.23605525 * float(x[16]))) + -0.1457119), 0)
    o[0] = (1.0793293 * h_0)+ (-0.25495288 * h_1)+ (0.28741312 * h_2)+ (-0.96760494 * h_3)+ (-2.4154522 * h_4)+ (3.7884355 * h_5) + -0.6846166

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


def classify(arr, transform=False):
    #apply transformation if necessary
    if transform:
        arr[:,:-1] = transform(arr[:,:-1])
    #init
    w_h = np.array([[-3.385582447052002, -0.09773637354373932, 0.8942749500274658, 6.7239508628845215, 0.4841916561126709, 1.3054081201553345, 5.2957940101623535, -2.1529781818389893, 5.4763360023498535, -3.643562078475952, -2.818270206451416, -0.5756410360336304, -3.482497215270996, 2.4104318618774414, -3.331660747528076, -5.240546226501465, -0.39406517148017883], [5.863551616668701, -1.2655733823776245, 3.1631765365600586, -4.518799304962158, -3.5550026893615723, -3.089711904525757, -5.635286331176758, 0.5969294905662537, -4.144523620605469, 0.7717809081077576, 2.4255783557891846, 5.37966775894165, 4.7256999015808105, -3.6520848274230957, 8.836206436157227, 0.9180757999420166, -1.835592269897461], [0.2260875552892685, 0.6035547256469727, 1.933091640472412, -0.8464939594268799, -0.22776293754577637, -1.4843647480010986, 2.486762523651123, -1.3314658403396606, 2.239457607269287, 5.417208194732666, 1.0977473258972168, -0.5146113634109497, 2.624992847442627, 2.5139195919036865, 0.8280764818191528, -0.5752495527267456, 1.0770015716552734], [-2.350095272064209, -0.1900065541267395, -0.2190246284008026, 1.735662579536438, 0.8063976764678955, -0.4374207854270935, 2.841341733932495, -1.1550809144973755, 2.224527597427368, 0.35834944248199463, -0.5173929929733276, 0.1307106912136078, -1.4367448091506958, 1.604527235031128, -1.3149571418762207, -2.0120275020599365, 0.9347653985023499], [-0.731484591960907, 0.04251917079091072, 0.587589681148529, 2.3700006008148193, -0.10765272378921509, 0.7172724604606628, 1.3900197744369507, -0.7513231635093689, 1.8020145893096924, -1.7766098976135254, -1.1646209955215454, -0.125333771109581, -0.8736205101013184, 0.530980110168457, -0.7878856658935547, -1.7973201274871826, -0.3345051109790802], [0.33009564876556396, -0.05582373961806297, 0.10616731643676758, -0.3008542060852051, -0.2904534339904785, -0.0768132284283638, -0.7092757225036621, -0.06304966658353806, -0.3489951491355896, -0.30399391055107117, -0.01015383005142212, 0.32373619079589844, 0.3083919584751129, -0.2116311937570572, 0.702785074710846, 0.035421621054410934, -0.23605525493621826]])
    b_h = np.array([2.0013246536254883, -2.3604519367218018, -2.1732754707336426, -0.5362228751182556, 0.5603318810462952, -0.14571189880371094])
    w_o = np.array([[1.079329252243042, -0.25495287775993347, 0.2874131202697754, -0.96760493516922, -2.415452241897583, 3.788435459136963]])
    b_o = np.array(-0.6846166253089905)

    #Hidden Layer
    h = np.dot(arr, w_h.T) + b_h
    
    relu = np.maximum(h, np.zeros_like(h))


    #Output
    out = np.dot(relu, w_o.T) + b_o
    if num_output_logits == 1:
        return (out >= 0).astype('int').reshape(-1)
    else:
        return (np.argmax(out, axis=1)).reshape(-1)



def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(reader, None) + ["Prediction"]))
        
        
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(single_classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            print(','.join(row))


def Validate(cleanarr):
    if n_classes == 2:
        #note that classification is a single line of code
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        correct_count = int(np.sum(outputs.reshape(-1) == cleanarr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(cleanarr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(cleanarr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs


    else:
        #validation
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count = 0, 0
        numeachclass = {}
        for k, o in enumerate(outputs):
            if int(o) == int(float(cleanarr[k, -1])):
                correct_count += 1
            if int(float(cleanarr[k, -1])) in numeachclass.keys():
                numeachclass[int(float(cleanarr[k, -1]))] += 1
            else:
                numeachclass[int(float(cleanarr[k, -1]))] = 1
            count += 1
        return count, correct_count, numeachclass, outputs
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    args = parser.parse_args()
    faulthandler.enable()


    #clean if not already clean
    if not args.cleanfile:
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
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
        classifier_type = 'NN'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
        #Correct Labels
        true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap = 115
        if args.json:
            import json
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
            if args.json:
                #                json_dict = {'Instance Count':count, 'classifier_type':classifier_type, 'n_classes':2, 'Number of False Negative Instances': num_FN, 'Number of False Positive Instances': num_FP, 'Number of True Positive Instances': num_TP, 'Number of True Negative Instances': num_TN,   'False Negatives': FN, 'False Positives': FP, 'True Negatives': TN, 'True Positives': TP, 'Number Correct': num_correct, 'Best Guess': randguess, 'Model Accuracy': modelacc, 'Model Capacity': model_cap, 'Generalization Ratio': int(float(num_correct * 100) / model_cap) / 100.0, 'Model Efficiency': int(100 * (modelacc - randguess) / model_cap) / 100.0}
                json_dict = {'instance_count':                        count ,
                            'classifier_type':                        classifier_type ,
                            'n_classes':                            2 ,
                            'number_of_false_negative_instances':    num_FN ,
                            'number_of_false_positive_instances':    num_FP ,
                            'number_of_true_positive_instances':    num_TP ,
                            'number_of_true_negative_instances':    num_TN,
                            'false_negatives':                        FN ,
                            'false_positives':                        FP ,
                            'true_negatives':                        TN ,
                            'true_positives':                        TP ,
                            'number_correct':                        num_correct ,
                            'best_guess':                            randguess ,
                            'model_accuracy':                        modelacc ,
                            'model_capacity':                        model_cap ,
                            'generalization_ratio':                int(float(num_correct * 100) / model_cap) / 100.0,
                            'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0
                             }
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                else:
                    print("Classifier Type:                    Decision Tree")
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
            if args.json:
        #        json_dict = {'Instance Count':count, 'classifier_type':classifier_type, 'Number Correct': num_correct, 'Best Guess': randguess, 'Model Accuracy': modelacc, 'Model Capacity': model_cap, 'Generalization Ratio': int(float(num_correct * 100) / model_cap) / 100.0, 'Model Efficiency': int(100 * (modelacc - randguess) / model_cap) / 100.0, 'n_classes': n_classes}
                json_dict = {'instance_count':                        count,
                            'classifier_type':                        classifier_type,
                            'n_classes':                            n_classes,
                            'number_correct':                        num_correct,
                            'best_guess':                            randguess,
                            'model_accuracy':                        modelacc,
                            'model_capacity':                        model_cap,
                            'generalization_ratio':                int(float(num_correct * 100) / model_cap) / 100.0,
                            'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0
                            }
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                else:
                    print("Classifier Type:                    Decision Tree")
                print("System Type:                        " + str(n_classes) + "-way classifier")
                print("Best-guess accuracy:                {:.2f}%".format(randguess))
                print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
                print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
                print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
                print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
                print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))

        try:
            import numpy as np # For numpy see: http://numpy.org
            from numpy import array
        except:
            print("Note: If you install numpy (https://www.numpy.org) and scipy (https://www.scipy.org) this predictor generates a confusion matrix")

        def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
            #check for numpy/scipy is imported
            try:
                from scipy.sparse import coo_matrix #required for multiclass metrics
            except:
                print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix")
                sys.exit()
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
        mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
            print(json.dumps(json_dict))
        else:
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print("Confusion Matrix:")
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])

    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)


