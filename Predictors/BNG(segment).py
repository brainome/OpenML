#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1852/BayesianNetworkGenerator_segment.arff -o Predictors/BNG(segment)_NN.py -target class -stopat 87.78 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 6:41:15.93. Finished on: Jun-09-2020 19:33:38.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        7-way classifier
Best-guess accuracy:                16.40%
Model accuracy:                     86.77% (867784/999999 correct)
Improvement over best guess:        70.37% (of possible 83.6%)
Model capacity (MEC):               385 bits
Generalization ratio:               2253.98 bits/bit
Confusion Matrix:
 [13.16% 0.11% 0.78% 0.00% 0.05% 0.00% 0.13%]
 [0.04% 10.25% 0.58% 0.82% 2.50% 0.03% 0.04%]
 [0.12% 1.55% 11.75% 0.39% 0.42% 0.04% 0.03%]
 [0.02% 0.60% 0.22% 13.05% 0.33% 0.04% 0.04%]
 [0.04% 2.78% 0.45% 0.49% 10.47% 0.02% 0.04%]
 [0.00% 0.02% 0.06% 0.01% 0.03% 14.23% 0.01%]
 [0.14% 0.09% 0.03% 0.04% 0.06% 0.02% 13.88%]

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
TRAINFILE = "BayesianNetworkGenerator_segment.csv"


#Number of output logits
num_output_logits = 7

#Number of attributes
num_attr = 19
n_classes = 7

mappings = [{1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {4216701371.0: 0}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

transform_true = False

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
        mean = np.array([])
        components = np.array([])
        whiten = None
        explained_variance = np.array([])
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
    target="class"


    # if (testfile):
    #     target = ''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless == False):
                header=next(reader, None)
                try:
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
                        if (col == hc):
                            raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
                for i in range(0, len(header)):      
                    if (i == hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i] + ",", end='', file=outputfile)
                print(header[hc], file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"' + row[name] + '"' + ",", end='', file=outputfile)
                        else:
                            print (row[name] + ",", end='', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target != ""): 
                        hc = int(target)
                    else:
                        hc =- 1
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
                    if (hc == -1):
                        hc = len(row) - 1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0, len(row)):
                        if (i in il):
                            continue
                        if (i == hc):
                            continue
                        if (',' in row[i]):
                            print ('"' + row[i] + '"'+",", end='', file=outputfile)
                        else:
                            print(row[i]+",", end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    clean.mapping={'path': 0, 'window': 1, 'cement': 2, 'brickface': 3, 'foliage': 4, 'sky': 5, 'grass': 6}

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
def single_classify(row):
    #inits
    x = row
    o = [0] * num_output_logits


    #Nueron Equations
    h_0 = max((((0.00092240877 * float(x[0]))+ (-0.046663646 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.21443765 * float(x[3]))+ (0.03595703 * float(x[4]))+ (0.009508773 * float(x[5]))+ (-0.07123254 * float(x[6]))+ (0.04340069 * float(x[7]))+ (-0.033161126 * float(x[8]))+ (0.3287155 * float(x[9]))+ (0.5908072 * float(x[10]))+ (0.580132 * float(x[11]))+ (0.05616004 * float(x[12]))+ (0.21067609 * float(x[13]))+ (0.5487662 * float(x[14]))+ (-0.56853163 * float(x[15]))+ (0.017252889 * float(x[16]))+ (-0.16548422 * float(x[17]))+ (0.17578658 * float(x[18]))) + -3.0196648), 0)
    h_1 = max((((-0.08927601 * float(x[0]))+ (0.78080034 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.17199585 * float(x[3]))+ (0.12115392 * float(x[4]))+ (-0.020700216 * float(x[5]))+ (-0.016169854 * float(x[6]))+ (-0.013341251 * float(x[7]))+ (-0.038221 * float(x[8]))+ (-0.15909038 * float(x[9]))+ (-0.009557152 * float(x[10]))+ (0.025754552 * float(x[11]))+ (0.057989176 * float(x[12]))+ (-0.119929835 * float(x[13]))+ (0.11151674 * float(x[14]))+ (-0.07073152 * float(x[15]))+ (0.3234035 * float(x[16]))+ (0.09434975 * float(x[17]))+ (0.14941287 * float(x[18]))) + -0.40552673), 0)
    h_2 = max((((-0.016339472 * float(x[0]))+ (0.23530889 * float(x[1]))+ (0.0 * float(x[2]))+ (0.20833738 * float(x[3]))+ (0.017174393 * float(x[4]))+ (0.08866355 * float(x[5]))+ (0.060882255 * float(x[6]))+ (0.5692481 * float(x[7]))+ (0.015931001 * float(x[8]))+ (-0.106718816 * float(x[9]))+ (-0.098523654 * float(x[10]))+ (0.17605668 * float(x[11]))+ (-0.12883683 * float(x[12]))+ (0.041536786 * float(x[13]))+ (-0.100248136 * float(x[14]))+ (0.20009828 * float(x[15]))+ (-1.2589552 * float(x[16]))+ (-0.16291204 * float(x[17]))+ (-0.39078096 * float(x[18]))) + 1.0362895), 0)
    h_3 = max((((0.020375593 * float(x[0]))+ (0.04817359 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.16622423 * float(x[3]))+ (0.046542916 * float(x[4]))+ (0.30624327 * float(x[5]))+ (-0.09808592 * float(x[6]))+ (0.18771684 * float(x[7]))+ (-0.0039248383 * float(x[8]))+ (0.18414308 * float(x[9]))+ (0.34390703 * float(x[10]))+ (0.44385874 * float(x[11]))+ (0.019713037 * float(x[12]))+ (-0.42845175 * float(x[13]))+ (0.6070505 * float(x[14]))+ (0.38303646 * float(x[15]))+ (-0.06676469 * float(x[16]))+ (0.23309286 * float(x[17]))+ (0.14982429 * float(x[18]))) + -3.9274294), 0)
    h_4 = max((((0.022044238 * float(x[0]))+ (0.02451861 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0023855015 * float(x[3]))+ (0.38028574 * float(x[4]))+ (0.2553593 * float(x[5]))+ (-0.14630008 * float(x[6]))+ (0.25123212 * float(x[7]))+ (-0.26147187 * float(x[8]))+ (0.19032384 * float(x[9]))+ (-0.2567864 * float(x[10]))+ (-0.22924706 * float(x[11]))+ (-0.3345359 * float(x[12]))+ (0.01108744 * float(x[13]))+ (-0.25570992 * float(x[14]))+ (0.20163786 * float(x[15]))+ (0.31727663 * float(x[16]))+ (-0.028975556 * float(x[17]))+ (-0.38428843 * float(x[18]))) + 0.6736231), 0)
    h_5 = max((((0.11808402 * float(x[0]))+ (2.1798992 * float(x[1]))+ (0.0 * float(x[2]))+ (0.01014861 * float(x[3]))+ (-0.052589692 * float(x[4]))+ (-0.35344896 * float(x[5]))+ (0.039908964 * float(x[6]))+ (-0.11429875 * float(x[7]))+ (0.008644287 * float(x[8]))+ (-0.473131 * float(x[9]))+ (-0.07008725 * float(x[10]))+ (0.0122707365 * float(x[11]))+ (-0.029958695 * float(x[12]))+ (-0.1296146 * float(x[13]))+ (0.060303867 * float(x[14]))+ (0.055640373 * float(x[15]))+ (0.029203644 * float(x[16]))+ (-0.1763652 * float(x[17]))+ (0.12815398 * float(x[18]))) + 0.2542151), 0)
    h_6 = max((((-0.004472186 * float(x[0]))+ (-0.37834606 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.07316374 * float(x[3]))+ (-0.0013325723 * float(x[4]))+ (-0.0003149639 * float(x[5]))+ (-0.034105275 * float(x[6]))+ (-0.031633876 * float(x[7]))+ (0.028734691 * float(x[8]))+ (-0.25272018 * float(x[9]))+ (0.39579883 * float(x[10]))+ (0.04984842 * float(x[11]))+ (-0.08958421 * float(x[12]))+ (0.026367582 * float(x[13]))+ (0.96444905 * float(x[14]))+ (0.22966024 * float(x[15]))+ (0.020361565 * float(x[16]))+ (0.66512394 * float(x[17]))+ (-0.084773436 * float(x[18]))) + -2.3797953), 0)
    h_7 = max((((-0.02784582 * float(x[0]))+ (0.29465353 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.31340796 * float(x[3]))+ (-0.052139908 * float(x[4]))+ (0.07222582 * float(x[5]))+ (-0.14601898 * float(x[6]))+ (0.07379102 * float(x[7]))+ (-0.089319676 * float(x[8]))+ (1.002955 * float(x[9]))+ (0.64535207 * float(x[10]))+ (0.5668588 * float(x[11]))+ (0.096817516 * float(x[12]))+ (0.27064416 * float(x[13]))+ (0.6873898 * float(x[14]))+ (-0.5158146 * float(x[15]))+ (-0.023551764 * float(x[16]))+ (-0.70135343 * float(x[17]))+ (0.6005093 * float(x[18]))) + -4.698466), 0)
    h_8 = max((((-0.0012980059 * float(x[0]))+ (0.074541934 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.056714922 * float(x[3]))+ (-0.021658748 * float(x[4]))+ (0.034831285 * float(x[5]))+ (0.018744174 * float(x[6]))+ (0.039808378 * float(x[7]))+ (0.024616145 * float(x[8]))+ (0.19050635 * float(x[9]))+ (0.15694457 * float(x[10]))+ (0.059482347 * float(x[11]))+ (-0.07240841 * float(x[12]))+ (-0.14924636 * float(x[13]))+ (0.4823223 * float(x[14]))+ (-0.035985008 * float(x[15]))+ (-0.14009438 * float(x[16]))+ (0.88606775 * float(x[17]))+ (0.4241046 * float(x[18]))) + -2.9354722), 0)
    h_9 = max((((0.012117113 * float(x[0]))+ (0.056051314 * float(x[1]))+ (0.0 * float(x[2]))+ (-0.23320216 * float(x[3]))+ (0.0051334463 * float(x[4]))+ (-0.0005390292 * float(x[5]))+ (0.02623166 * float(x[6]))+ (0.073564604 * float(x[7]))+ (0.014635633 * float(x[8]))+ (-0.032474656 * float(x[9]))+ (0.089391954 * float(x[10]))+ (-0.0054461374 * float(x[11]))+ (-0.12606944 * float(x[12]))+ (-0.05472318 * float(x[13]))+ (0.075996675 * float(x[14]))+ (0.5703705 * float(x[15]))+ (-0.08817529 * float(x[16]))+ (0.33612362 * float(x[17]))+ (0.8833129 * float(x[18]))) + -2.3995264), 0)
    h_10 = max((((-0.047204353 * float(x[0]))+ (0.074770816 * float(x[1]))+ (0.0 * float(x[2]))+ (0.08244133 * float(x[3]))+ (0.09668961 * float(x[4]))+ (0.19296055 * float(x[5]))+ (-0.060669582 * float(x[6]))+ (0.66623944 * float(x[7]))+ (-0.07525823 * float(x[8]))+ (-0.030265534 * float(x[9]))+ (0.067193724 * float(x[10]))+ (0.15655428 * float(x[11]))+ (-0.59690475 * float(x[12]))+ (0.24674036 * float(x[13]))+ (0.11146334 * float(x[14]))+ (-0.069009796 * float(x[15]))+ (-3.070138 * float(x[16]))+ (0.015145963 * float(x[17]))+ (0.027862767 * float(x[18]))) + 1.3556719), 0)
    h_11 = max((((-0.0014864516 * float(x[0]))+ (-0.6899526 * float(x[1]))+ (0.0 * float(x[2]))+ (0.022191871 * float(x[3]))+ (0.006067316 * float(x[4]))+ (0.0946305 * float(x[5]))+ (-0.03749596 * float(x[6]))+ (-0.07648095 * float(x[7]))+ (0.060658567 * float(x[8]))+ (-0.4135409 * float(x[9]))+ (0.36341384 * float(x[10]))+ (0.019821197 * float(x[11]))+ (-0.12410976 * float(x[12]))+ (0.06349554 * float(x[13]))+ (0.6922616 * float(x[14]))+ (0.23074469 * float(x[15]))+ (-0.005674056 * float(x[16]))+ (0.74170184 * float(x[17]))+ (-0.13068737 * float(x[18]))) + -1.9664769), 0)
    h_12 = max((((-0.0085058035 * float(x[0]))+ (2.070457 * float(x[1]))+ (0.0 * float(x[2]))+ (0.018896064 * float(x[3]))+ (0.0037667379 * float(x[4]))+ (-0.20465891 * float(x[5]))+ (0.0007389665 * float(x[6]))+ (-0.005011564 * float(x[7]))+ (-0.015504575 * float(x[8]))+ (-0.5772351 * float(x[9]))+ (-0.10486679 * float(x[10]))+ (0.0027458475 * float(x[11]))+ (-0.018531332 * float(x[12]))+ (-0.112034015 * float(x[13]))+ (0.05872367 * float(x[14]))+ (0.06196478 * float(x[15]))+ (0.08168138 * float(x[16]))+ (-0.1993936 * float(x[17]))+ (0.064820044 * float(x[18]))) + -0.53477496), 0)
    h_13 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))+ (0.0 * float(x[16]))+ (0.0 * float(x[17]))+ (0.0 * float(x[18]))) + 0.0), 0)
    o[0] = (2.7192216 * h_0)+ (0.44194958 * h_1)+ (4.649235 * h_2)+ (-2.654716 * h_3)+ (0.4439459 * h_4)+ (-2.070146 * h_5)+ (-4.316947 * h_6)+ (-0.7651749 * h_7)+ (8.822789 * h_8)+ (1.0657251 * h_9)+ (0.5061184 * h_10)+ (10.495685 * h_11)+ (9.428404 * h_12)+ (0.0 * h_13) + 4.282114
    o[1] = (-3.2434692 * h_0)+ (7.331686 * h_1)+ (3.1719973 * h_2)+ (0.6229179 * h_3)+ (2.6549153 * h_4)+ (0.9416447 * h_5)+ (6.9670362 * h_6)+ (3.0851884 * h_7)+ (2.6597857 * h_8)+ (-2.6635773 * h_9)+ (3.3846214 * h_10)+ (-0.7398671 * h_11)+ (4.7214527 * h_12)+ (0.0 * h_13) + -0.7879164
    o[2] = (0.89748675 * h_0)+ (8.245088 * h_1)+ (6.5099587 * h_2)+ (-2.3584776 * h_3)+ (1.9033504 * h_4)+ (4.847578 * h_5)+ (-0.15200326 * h_6)+ (-0.9111164 * h_7)+ (5.104788 * h_8)+ (4.699402 * h_9)+ (0.7089283 * h_10)+ (4.2041974 * h_11)+ (-0.113868356 * h_12)+ (0.0 * h_13) + -4.053843
    o[3] = (7.677481 * h_0)+ (4.9906993 * h_1)+ (2.5397718 * h_2)+ (9.50568 * h_3)+ (4.035832 * h_4)+ (5.123761 * h_5)+ (-0.051453825 * h_6)+ (-4.839854 * h_7)+ (-5.5897264 * h_8)+ (1.6968364 * h_9)+ (3.675243 * h_10)+ (9.693323 * h_11)+ (1.4374845 * h_12)+ (0.0 * h_13) + -4.0132656
    o[4] = (-0.4543959 * h_0)+ (0.7682607 * h_1)+ (2.3357399 * h_2)+ (-0.5931703 * h_3)+ (2.325881 * h_4)+ (5.796139 * h_5)+ (10.357971 * h_6)+ (3.67345 * h_7)+ (2.6497314 * h_8)+ (0.32441854 * h_9)+ (2.2652707 * h_10)+ (-3.1143112 * h_11)+ (1.9985915 * h_12)+ (0.0 * h_13) + 0.15960868
    o[5] = (4.7424583 * h_0)+ (4.9397564 * h_1)+ (2.435333 * h_2)+ (-4.6479125 * h_3)+ (5.827286 * h_4)+ (3.1993122 * h_5)+ (5.71367 * h_6)+ (-0.629477 * h_7)+ (3.000416 * h_8)+ (4.0937448 * h_9)+ (5.409011 * h_10)+ (0.6110821 * h_11)+ (4.2112393 * h_12)+ (0.0 * h_13) + -7.414596
    o[6] = (13.348638 * h_0)+ (1.4046488 * h_1)+ (2.091486 * h_2)+ (0.9180005 * h_3)+ (3.819156 * h_4)+ (-0.62776065 * h_5)+ (9.465487 * h_6)+ (-4.55651 * h_7)+ (-2.1099885 * h_8)+ (0.9290622 * h_9)+ (3.4457996 * h_10)+ (0.5220627 * h_11)+ (8.233646 * h_12)+ (0.0 * h_13) + 1.111022

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


#for classifying batches
def classify(arr):
    outputs = []
    for row in arr:
        outputs.append(single_classify(row))
    return outputs


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
            pred = str(get_key(int(single_classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            writer.writerow(row)


def Validate(arr):
    if n_classes == 2:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        outputs=[]
        for i, row in enumerate(arr):
            outputs.append(int(single_classify(arr[i, :-1].tolist())))
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
            pred = int(single_classify(arr[i].tolist()))
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
        print("Classifier Type: Neural Network")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
            #Correct Labels
            true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap = 385
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


            print("Confusion Matrix:")
            mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)
