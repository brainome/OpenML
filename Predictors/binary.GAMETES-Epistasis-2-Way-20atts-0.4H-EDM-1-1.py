#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1.csv -o GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:30:56.27. Finished on: Sep-03-2020 17:38:03.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 50.00%
Overall Model accuracy:              80.62% (1290/1600 correct)
Overall Improvement over best guess: 30.62% (of possible 50.0%)
Model capacity (MEC):                153 bits
Generalization ratio:                8.43 bits/bit
Model efficiency:                    0.20%/parameter
System behavior
True Negatives:                      41.81% (669/1600)
True Positives:                      38.81% (621/1600)
False Negatives:                     11.19% (179/1600)
False Positives:                     8.19% (131/1600)
True Pos. Rate/Sensitivity/Recall:   0.78
True Neg. Rate/Specificity:          0.84
Precision:                           0.83
F-1 Measure:                         0.80
False Negative Rate/Miss Rate:       0.22
Critical Success Index:              0.67
Confusion Matrix:
 [41.81% 8.19%]
 [11.19% 38.81%]
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
TRAINFILE = "GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1.csv"


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
        mean = np.array([0.019791666666666666, 0.27291666666666664, 0.7510416666666667, 0.8895833333333333, 0.35833333333333334, 0.7822916666666667, 0.478125, 0.7427083333333333, 0.35, 0.7552083333333334, 0.023958333333333335, 0.865625, 0.4791666666666667, 0.075, 0.18125, 0.6729166666666667, 0.4083333333333333, 0.7802083333333333, 0.415625, 0.415625])
        components = np.array([array([ 0.00221665, -0.06241342,  0.33485812,  0.25951471, -0.07244139,
       -0.27982633,  0.05214882, -0.24510894,  0.08794787,  0.38467449,
        0.00135817, -0.63260316, -0.02616038,  0.02311465,  0.03447814,
        0.09075392, -0.03683942,  0.29470333,  0.12092326,  0.03003649]), array([ 0.0033697 ,  0.06976366,  0.04410392, -0.28826743, -0.08055553,
        0.02722517,  0.00822878,  0.4954275 , -0.17745421, -0.25742614,
       -0.011473  , -0.24650399, -0.05266841, -0.02358036,  0.01192122,
       -0.2789675 ,  0.07859438,  0.62720609, -0.07106324, -0.11299695]), array([ 0.0014158 ,  0.03414674, -0.31091349,  0.32059663, -0.02761328,
        0.66433616, -0.12218928, -0.32733834,  0.00867175, -0.26061264,
       -0.00378711, -0.20479206, -0.05511546,  0.01528522,  0.03342156,
        0.17898177,  0.06801022,  0.28497846,  0.00593118,  0.05433742]), array([-0.00618248, -0.02210224, -0.0147775 ,  0.48653225, -0.06253508,
        0.24267096,  0.24261608,  0.53324086,  0.04273925,  0.46164208,
        0.0030467 ,  0.28899712,  0.12011066,  0.02153367,  0.00285283,
        0.07641469,  0.03386356,  0.16137335,  0.05837155,  0.08204528]), array([ 0.01162891,  0.03279862,  0.24786795,  0.44766046,  0.01094268,
       -0.33425756,  0.06702464, -0.22703973,  0.08535218, -0.43808327,
       -0.00715686,  0.44901277,  0.04420565, -0.01211545, -0.05173949,
       -0.0932759 , -0.09239624,  0.35267942,  0.03851083, -0.14253618]), array([-1.62574175e-04,  2.54367494e-02, -5.43910253e-01, -2.07304292e-01,
        2.03040209e-03, -3.14550297e-01, -1.44392501e-01, -1.55571232e-01,
       -3.34722315e-02,  3.34174935e-01, -3.06024249e-03,  2.41000467e-01,
       -7.63570023e-02, -7.92845450e-03,  9.69588622e-03,  3.55158362e-01,
       -3.43019315e-03,  4.11833411e-01, -3.09994205e-02, -2.07253025e-01]), array([ 1.67481474e-04,  2.55218582e-02,  5.70862957e-01, -3.87005109e-01,
       -2.28442822e-02,  2.62867198e-01,  1.68225646e-02, -1.37945313e-01,
       -1.08132905e-01,  4.99195535e-02,  1.01313138e-02,  2.52353118e-01,
        2.04311200e-01,  7.25208649e-03, -1.10040988e-02,  4.79895978e-01,
        1.85183627e-01,  1.94429153e-01,  5.47184659e-02,  9.73760526e-02]), array([ 6.29047356e-04, -7.63927575e-02, -6.09012250e-02,  1.54506867e-01,
        1.45541225e-01, -2.68471555e-01, -7.31198557e-02,  3.94469156e-01,
       -8.25236985e-02, -3.89598072e-01, -1.22632804e-03, -1.78252529e-01,
       -1.54908222e-01,  1.54287070e-02,  9.06285034e-03,  6.30469427e-01,
        2.52479827e-02, -1.26129178e-01,  2.16876706e-01,  1.91333503e-01]), array([ 0.00509506,  0.01941551, -0.04805058,  0.1285794 , -0.09633171,
       -0.06396229,  0.09342898,  0.05579196, -0.11875761, -0.10080481,
       -0.00522614, -0.20433675,  0.60259451, -0.02068433, -0.04768989,
        0.23262402, -0.00770545, -0.12689095, -0.5515976 , -0.38618677]), array([ 0.00942171,  0.04388944, -0.04548894,  0.02529369, -0.08436936,
       -0.04638582, -0.66489976,  0.08973787,  0.0638229 ,  0.01930235,
       -0.00548293,  0.00158421,  0.58124064, -0.00643935,  0.03951736,
       -0.13828229, -0.10713791,  0.03477463,  0.36922588,  0.15427797]), array([ 0.00860501, -0.00171144, -0.30232666, -0.12762983, -0.14055287,
       -0.14854592,  0.56974359, -0.14067299,  0.07301448, -0.13819157,
        0.00373307, -0.06034316,  0.38348705, -0.0109437 ,  0.00628329,
       -0.07979278,  0.36949692,  0.03281649,  0.32786477,  0.28150078]), array([ 5.97014116e-04, -5.04978107e-02,  3.75123232e-02,  1.98327180e-01,
        1.49525246e-01, -1.47468122e-01, -3.10388647e-01, -3.39797589e-02,
       -1.85923069e-01,  7.81698003e-02, -1.08376460e-02,  4.49397945e-02,
       -7.68625176e-02,  4.30264838e-03, -9.08546715e-03, -1.36320643e-01,
        8.11254254e-01, -2.46167490e-02, -2.61533105e-01,  1.43395280e-01]), array([ 0.00551247,  0.01406092,  0.04090336, -0.089158  , -0.33771468,
        0.01107935, -0.11632718,  0.13690247,  0.84402859, -0.08074851,
        0.01765096, -0.01175621, -0.10979968, -0.01034883, -0.03372105,
        0.1066268 ,  0.2541538 , -0.02733067, -0.0904499 , -0.16172603]), array([-0.00607704, -0.09233683, -0.04455182, -0.03489783, -0.25746037,
       -0.11231597, -0.0380511 , -0.02649326,  0.06014004, -0.03079496,
       -0.00159614,  0.05946642, -0.00097145, -0.00579298, -0.03418844,
        0.01578119, -0.26194116,  0.13199676, -0.5206831 ,  0.73517358]), array([-0.01123591, -0.29609746, -0.02124717, -0.09818653,  0.80857131,
        0.05236387,  0.05899491,  0.02737496,  0.37869672,  0.00514831,
        0.00956474, -0.04326949,  0.19723063, -0.01586382,  0.01054302,
       -0.04389612, -0.06063846,  0.16484479, -0.13079156,  0.08246442]), array([ 0.0016159 ,  0.93112611,  0.01122265,  0.02533515,  0.25654517,
       -0.05057925,  0.04110976,  0.01584209,  0.11770296,  0.03862816,
        0.01027968, -0.04776194,  0.01103942,  0.01467301,  0.12209113,
        0.03433893, -0.00832955, -0.00923345, -0.08832071,  0.14074201]), array([-9.84999941e-03, -1.13865596e-01,  2.34649927e-02,  2.19674705e-03,
       -5.64053164e-02, -2.30402541e-02,  2.08738798e-02, -2.07878593e-03,
        5.13761648e-03, -3.89272548e-02,  5.17595228e-03,  5.60865142e-02,
        6.39865853e-03,  4.46536772e-02,  9.85285932e-01,  4.34400220e-04,
        5.24071118e-03, -1.45997665e-02, -5.83441822e-02, -3.23634638e-02])])
        whiten = False
        explained_variance = np.array([0.5515554348894117, 0.5366634964523312, 0.5274450138739775, 0.5071651226435566, 0.4745723658625418, 0.45967102908605384, 0.4462999595087371, 0.41849508246560807, 0.38217113131520025, 0.35955013569099703, 0.3382082698516943, 0.32646571074101666, 0.2783342474877305, 0.27066098166679725, 0.2648653774583769, 0.2232343794933315, 0.15730555750111225])
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
    h_0 = max((((0.9391141 * float(x[0]))+ (0.6921544 * float(x[1]))+ (-1.0777534 * float(x[2]))+ (-0.58792186 * float(x[3]))+ (3.2360659 * float(x[4]))+ (3.3340611 * float(x[5]))+ (-1.5261805 * float(x[6]))+ (0.12729365 * float(x[7]))+ (-2.186322 * float(x[8]))+ (3.7484462 * float(x[9]))+ (1.2344232 * float(x[10]))+ (-5.5347037 * float(x[11]))+ (1.6790915 * float(x[12]))+ (-19.249004 * float(x[13]))+ (-3.9035192 * float(x[14]))+ (0.81226665 * float(x[15]))+ (-1.4231478 * float(x[16]))) + -0.5625157), 0)
    h_1 = max((((-3.6899936 * float(x[0]))+ (-0.6271194 * float(x[1]))+ (7.591335 * float(x[2]))+ (-0.8708467 * float(x[3]))+ (0.3696739 * float(x[4]))+ (-2.1742039 * float(x[5]))+ (-2.1341362 * float(x[6]))+ (7.7864175 * float(x[7]))+ (-1.745323 * float(x[8]))+ (2.146835 * float(x[9]))+ (-7.9524074 * float(x[10]))+ (1.9585383 * float(x[11]))+ (-5.2253985 * float(x[12]))+ (1.7011219 * float(x[13]))+ (3.5843747 * float(x[14]))+ (-6.661426 * float(x[15]))+ (-1.2426614 * float(x[16]))) + 0.7154179), 0)
    h_2 = max((((2.3992898 * float(x[0]))+ (-0.5766664 * float(x[1]))+ (1.9451785 * float(x[2]))+ (-1.6057782 * float(x[3]))+ (0.09471745 * float(x[4]))+ (-2.6235995 * float(x[5]))+ (-0.42502186 * float(x[6]))+ (-0.13307567 * float(x[7]))+ (-1.1701821 * float(x[8]))+ (-0.792892 * float(x[9]))+ (0.05605476 * float(x[10]))+ (1.2569263 * float(x[11]))+ (-0.639569 * float(x[12]))+ (0.123665065 * float(x[13]))+ (0.17943336 * float(x[14]))+ (2.214217 * float(x[15]))+ (0.23101394 * float(x[16]))) + -5.829907), 0)
    h_3 = max((((2.0187616 * float(x[0]))+ (0.9702554 * float(x[1]))+ (-1.8947641 * float(x[2]))+ (1.4960042 * float(x[3]))+ (-0.80443794 * float(x[4]))+ (2.1493351 * float(x[5]))+ (0.70161647 * float(x[6]))+ (1.200828 * float(x[7]))+ (0.29116046 * float(x[8]))+ (0.96791464 * float(x[9]))+ (0.3196559 * float(x[10]))+ (0.27983212 * float(x[11]))+ (0.22731557 * float(x[12]))+ (-2.4120183 * float(x[13]))+ (-1.5937477 * float(x[14]))+ (-0.026276018 * float(x[15]))+ (-1.9431375 * float(x[16]))) + 0.82700646), 0)
    h_4 = max((((0.12575081 * float(x[0]))+ (0.633018 * float(x[1]))+ (-2.7158897 * float(x[2]))+ (-0.43654108 * float(x[3]))+ (0.026164278 * float(x[4]))+ (2.486014 * float(x[5]))+ (0.116951615 * float(x[6]))+ (-0.85943717 * float(x[7]))+ (3.178087 * float(x[8]))+ (-0.94860286 * float(x[9]))+ (-1.4751427 * float(x[10]))+ (0.8322086 * float(x[11]))+ (1.5573423 * float(x[12]))+ (-1.5072598 * float(x[13]))+ (-1.2788916 * float(x[14]))+ (-0.5394193 * float(x[15]))+ (-0.09238785 * float(x[16]))) + 1.1940906), 0)
    h_5 = max((((-0.17806047 * float(x[0]))+ (0.052089076 * float(x[1]))+ (-0.5238405 * float(x[2]))+ (-3.168593e-05 * float(x[3]))+ (0.49326605 * float(x[4]))+ (0.41583496 * float(x[5]))+ (-0.18495198 * float(x[6]))+ (0.39345133 * float(x[7]))+ (-0.7931687 * float(x[8]))+ (1.1659431 * float(x[9]))+ (0.41080162 * float(x[10]))+ (-0.6747389 * float(x[11]))+ (0.21771084 * float(x[12]))+ (-2.2729263 * float(x[13]))+ (-0.3828593 * float(x[14]))+ (-0.033664312 * float(x[15]))+ (-0.30130196 * float(x[16]))) + 0.9180923), 0)
    h_6 = max((((-0.09804069 * float(x[0]))+ (0.34132183 * float(x[1]))+ (-0.50529885 * float(x[2]))+ (-0.4592338 * float(x[3]))+ (0.06918209 * float(x[4]))+ (0.7756447 * float(x[5]))+ (-0.06945479 * float(x[6]))+ (-0.86318487 * float(x[7]))+ (2.2589562 * float(x[8]))+ (-1.3424935 * float(x[9]))+ (-1.5476737 * float(x[10]))+ (0.48767048 * float(x[11]))+ (0.750913 * float(x[12]))+ (-0.6020933 * float(x[13]))+ (-0.31853542 * float(x[14]))+ (-0.19485113 * float(x[15]))+ (0.12723096 * float(x[16]))) + -0.41626424), 0)
    h_7 = max((((0.17354232 * float(x[0]))+ (0.20963584 * float(x[1]))+ (-0.28092474 * float(x[2]))+ (-0.27669352 * float(x[3]))+ (0.4909017 * float(x[4]))+ (0.7771782 * float(x[5]))+ (-0.28693077 * float(x[6]))+ (-0.41535556 * float(x[7]))+ (0.4299929 * float(x[8]))+ (0.037553504 * float(x[9]))+ (-0.058224622 * float(x[10]))+ (-0.89905554 * float(x[11]))+ (0.36093584 * float(x[12]))+ (-2.6779723 * float(x[13]))+ (-0.7334633 * float(x[14]))+ (0.060119204 * float(x[15]))+ (0.19312046 * float(x[16]))) + -0.8660888), 0)
    o[0] = (1.3059145 * h_0)+ (0.13043551 * h_1)+ (-4.6994085 * h_2)+ (-0.62311 * h_3)+ (1.8855814 * h_4)+ (-4.1259203 * h_5)+ (-4.490255 * h_6)+ (-5.824314 * h_7) + 1.2468575

    

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
    w_h = np.array([[0.9391140937805176, 0.6921544075012207, -1.0777534246444702, -0.5879218578338623, 3.2360658645629883, 3.3340611457824707, -1.5261805057525635, 0.1272936463356018, -2.186321973800659, 3.748446226119995, 1.2344231605529785, -5.534703731536865, 1.679091453552246, -19.249004364013672, -3.9035191535949707, 0.8122666478157043, -1.4231477975845337], [-3.6899936199188232, -0.6271194219589233, 7.591334819793701, -0.870846688747406, 0.3696739077568054, -2.174203872680664, -2.134136199951172, 7.786417484283447, -1.7453229427337646, 2.1468350887298584, -7.952407360076904, 1.958538293838501, -5.225398540496826, 1.7011219263076782, 3.5843746662139893, -6.661426067352295, -1.2426613569259644], [2.399289846420288, -0.5766664147377014, 1.945178508758545, -1.6057782173156738, 0.09471745043992996, -2.6235995292663574, -0.42502185702323914, -0.13307566940784454, -1.1701821088790894, -0.7928919792175293, 0.05605475977063179, 1.2569262981414795, -0.6395689845085144, 0.12366506457328796, 0.17943336069583893, 2.214216947555542, 0.2310139387845993], [2.01876163482666, 0.9702553749084473, -1.8947640657424927, 1.4960042238235474, -0.8044379353523254, 2.1493351459503174, 0.7016164660453796, 1.200827956199646, 0.2911604642868042, 0.9679146409034729, 0.3196558952331543, 0.279832124710083, 0.22731557488441467, -2.412018299102783, -1.5937477350234985, -0.02627601847052574, -1.943137526512146], [0.1257508099079132, 0.6330180168151855, -2.7158896923065186, -0.4365410804748535, 0.026164278388023376, 2.486013889312744, 0.11695161461830139, -0.8594371676445007, 3.178086996078491, -0.9486028552055359, -1.4751427173614502, 0.8322085738182068, 1.557342290878296, -1.5072598457336426, -1.2788915634155273, -0.5394192934036255, -0.0923878476023674], [-0.17806047201156616, 0.05208907648921013, -0.5238404870033264, -3.168592957081273e-05, 0.4932660460472107, 0.4158349633216858, -0.18495197594165802, 0.3934513330459595, -0.7931687235832214, 1.1659431457519531, 0.41080161929130554, -0.674738883972168, 0.21771083772182465, -2.2729263305664062, -0.3828592896461487, -0.033664312213659286, -0.3013019561767578], [-0.09804069250822067, 0.34132182598114014, -0.5052988529205322, -0.4592337906360626, 0.06918209046125412, 0.7756447196006775, -0.06945478916168213, -0.8631848692893982, 2.25895619392395, -1.3424935340881348, -1.5476737022399902, 0.4876704812049866, 0.7509130239486694, -0.6020932793617249, -0.3185354173183441, -0.1948511302471161, 0.1272309571504593], [0.173542320728302, 0.20963583886623383, -0.2809247374534607, -0.27669352293014526, 0.4909017086029053, 0.7771782279014587, -0.28693076968193054, -0.4153555631637573, 0.4299929141998291, 0.03755350410938263, -0.058224622160196304, -0.899055540561676, 0.36093583703041077, -2.6779723167419434, -0.7334632873535156, 0.060119204223155975, 0.19312046468257904]])
    b_h = np.array([-0.5625156760215759, 0.7154179215431213, -5.829906940460205, 0.827006459236145, 1.1940906047821045, 0.9180923104286194, -0.41626423597335815, -0.8660888075828552])
    w_o = np.array([[1.3059145212173462, 0.130435511469841, -4.699408531188965, -0.6231099963188171, 1.885581374168396, -4.125920295715332, -4.490254878997803, -5.824314117431641]])
    b_o = np.array(1.2468575239181519)

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
        model_cap = 153
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


