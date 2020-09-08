#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class BNG-segment.csv -o BNG-segment_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 3:54:43.50. Finished on: Sep-04-2020 02:49:44.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         7-way classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 14.36%
Training accuracy:                   86.53% (519239/600000 correct)
Validation accuracy:                 86.64% (346573/400000 correct)
Overall Model accuracy:              86.58% (865812/1000000 correct)
Overall Improvement over best guess: 72.22% (of possible 85.64%)
Model capacity (MEC):                439 bits
Generalization ratio:                1972.23 bits/bit
Model efficiency:                    0.16%/parameter
Confusion Matrix:
 [13.14% 0.10% 0.75% 0.01% 0.06% 0.01% 0.18%]
 [0.04% 10.37% 0.60% 0.83% 2.33% 0.04% 0.05%]
 [0.13% 1.69% 11.55% 0.46% 0.37% 0.04% 0.05%]
 [0.02% 0.59% 0.14% 13.12% 0.34% 0.03% 0.05%]
 [0.05% 2.91% 0.45% 0.53% 10.26% 0.03% 0.05%]
 [0.01% 0.02% 0.05% 0.02% 0.02% 14.22% 0.01%]
 [0.12% 0.07% 0.04% 0.06% 0.05% 0.01% 13.91%]
Overfitting:                         No
Note: Labels have been remapped to 'path'=0, 'window'=1, 'cement'=2, 'brickface'=3, 'foliage'=4, 'sky'=5, 'grass'=6.
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
TRAINFILE = "BNG-segment.csv"


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
    h_0 = max((((-0.5228794 * float(x[0]))+ (-1.2125694 * float(x[1]))+ (-0.39339504 * float(x[2]))+ (0.090265326 * float(x[3]))+ (0.4014969 * float(x[4]))+ (0.4313314 * float(x[5]))+ (-0.01673303 * float(x[6]))+ (0.34331155 * float(x[7]))+ (-0.08257517 * float(x[8]))+ (0.1861814 * float(x[9]))+ (0.09274115 * float(x[10]))+ (-0.24248005 * float(x[11]))+ (0.35390648 * float(x[12]))+ (-0.056372616 * float(x[13]))+ (-0.32465908 * float(x[14]))+ (-0.5540404 * float(x[15]))+ (0.28739554 * float(x[16]))+ (0.9776049 * float(x[17]))+ (0.22334434 * float(x[18]))) + 0.5650725), 0)
    h_1 = max((((0.025577037 * float(x[0]))+ (0.08330399 * float(x[1]))+ (-0.24019481 * float(x[2]))+ (-0.0030763391 * float(x[3]))+ (-0.3485297 * float(x[4]))+ (-0.33016294 * float(x[5]))+ (0.19052199 * float(x[6]))+ (-0.058737878 * float(x[7]))+ (0.17276399 * float(x[8]))+ (0.19619171 * float(x[9]))+ (0.04303775 * float(x[10]))+ (-0.5147814 * float(x[11]))+ (-0.096389525 * float(x[12]))+ (-0.2992976 * float(x[13]))+ (-0.42514205 * float(x[14]))+ (-0.35489857 * float(x[15]))+ (-0.8995887 * float(x[16]))+ (1.2735562 * float(x[17]))+ (1.1834238 * float(x[18]))) + 0.5765786), 0)
    h_2 = max((((0.038087048 * float(x[0]))+ (-0.03283814 * float(x[1]))+ (0.23052062 * float(x[2]))+ (0.26407632 * float(x[3]))+ (0.06688813 * float(x[4]))+ (0.26813963 * float(x[5]))+ (0.34812108 * float(x[6]))+ (0.3468029 * float(x[7]))+ (0.08715636 * float(x[8]))+ (0.04836809 * float(x[9]))+ (0.3760806 * float(x[10]))+ (-1.1884624 * float(x[11]))+ (-0.31280136 * float(x[12]))+ (0.08235818 * float(x[13]))+ (0.03530994 * float(x[14]))+ (-0.05019359 * float(x[15]))+ (-3.0095627 * float(x[16]))+ (0.009992821 * float(x[17]))+ (0.19658166 * float(x[18]))) + 0.87250024), 0)
    h_3 = max((((0.17563461 * float(x[0]))+ (-0.30392534 * float(x[1]))+ (-0.26122907 * float(x[2]))+ (0.11937247 * float(x[3]))+ (0.2954684 * float(x[4]))+ (0.115087576 * float(x[5]))+ (-0.030015033 * float(x[6]))+ (0.2574043 * float(x[7]))+ (-0.1399953 * float(x[8]))+ (0.081065215 * float(x[9]))+ (0.055492934 * float(x[10]))+ (0.12189309 * float(x[11]))+ (0.31517234 * float(x[12]))+ (-0.17962825 * float(x[13]))+ (0.22625338 * float(x[14]))+ (0.30335748 * float(x[15]))+ (0.5043032 * float(x[16]))+ (-0.13041066 * float(x[17]))+ (-1.1040025 * float(x[18]))) + 0.7126812), 0)
    h_4 = max((((-0.08634936 * float(x[0]))+ (0.5441049 * float(x[1]))+ (0.18080638 * float(x[2]))+ (-0.09078462 * float(x[3]))+ (-0.13986672 * float(x[4]))+ (-0.5477329 * float(x[5]))+ (-0.521978 * float(x[6]))+ (0.2323066 * float(x[7]))+ (-0.014261954 * float(x[8]))+ (0.052784324 * float(x[9]))+ (0.37987083 * float(x[10]))+ (0.23991 * float(x[11]))+ (-0.12629484 * float(x[12]))+ (-0.23906009 * float(x[13]))+ (0.8104553 * float(x[14]))+ (-1.3764716 * float(x[15]))+ (0.33282137 * float(x[16]))+ (0.049577262 * float(x[17]))+ (0.36916408 * float(x[18]))) + 0.4226057), 0)
    h_5 = max((((-0.09996668 * float(x[0]))+ (0.46602625 * float(x[1]))+ (-0.21342877 * float(x[2]))+ (0.12225955 * float(x[3]))+ (-0.022253482 * float(x[4]))+ (0.0065722726 * float(x[5]))+ (0.12541011 * float(x[6]))+ (0.042832546 * float(x[7]))+ (-0.22490272 * float(x[8]))+ (-0.22839886 * float(x[9]))+ (-0.06188781 * float(x[10]))+ (0.08748987 * float(x[11]))+ (-0.21099125 * float(x[12]))+ (1.3459098 * float(x[13]))+ (0.8175953 * float(x[14]))+ (-0.8030507 * float(x[15]))+ (0.294887 * float(x[16]))+ (-0.61346835 * float(x[17]))+ (0.0073372857 * float(x[18]))) + 0.6425808), 0)
    h_6 = max((((0.0082752565 * float(x[0]))+ (0.06916284 * float(x[1]))+ (-0.034510944 * float(x[2]))+ (0.33021578 * float(x[3]))+ (0.12879142 * float(x[4]))+ (0.36393997 * float(x[5]))+ (0.07499856 * float(x[6]))+ (0.18027556 * float(x[7]))+ (0.19816762 * float(x[8]))+ (-0.26081815 * float(x[9]))+ (-0.08429605 * float(x[10]))+ (-0.12000691 * float(x[11]))+ (-0.04989809 * float(x[12]))+ (0.24123919 * float(x[13]))+ (0.03865815 * float(x[14]))+ (0.5587819 * float(x[15]))+ (-0.43251455 * float(x[16]))+ (-1.3367898 * float(x[17]))+ (-1.1958958 * float(x[18]))) + 1.2014887), 0)
    h_7 = max((((-0.056871455 * float(x[0]))+ (-0.16277833 * float(x[1]))+ (0.07368532 * float(x[2]))+ (0.43220603 * float(x[3]))+ (-0.2919395 * float(x[4]))+ (-0.025562875 * float(x[5]))+ (0.3131222 * float(x[6]))+ (-0.02507748 * float(x[7]))+ (-0.061243396 * float(x[8]))+ (0.17353249 * float(x[9]))+ (0.1850849 * float(x[10]))+ (-0.42116323 * float(x[11]))+ (0.23221475 * float(x[12]))+ (1.193842 * float(x[13]))+ (-0.5927122 * float(x[14]))+ (-0.36699253 * float(x[15]))+ (0.10550645 * float(x[16]))+ (-1.4881691 * float(x[17]))+ (1.1124836 * float(x[18]))) + 0.8744291), 0)
    h_8 = max((((0.049637336 * float(x[0]))+ (0.18017203 * float(x[1]))+ (-0.09432898 * float(x[2]))+ (-0.056007795 * float(x[3]))+ (0.16681753 * float(x[4]))+ (-0.16375566 * float(x[5]))+ (0.115788266 * float(x[6]))+ (0.02792883 * float(x[7]))+ (0.032336514 * float(x[8]))+ (-0.049274173 * float(x[9]))+ (-0.18261164 * float(x[10]))+ (-0.6171836 * float(x[11]))+ (0.015595824 * float(x[12]))+ (1.0347836 * float(x[13]))+ (-1.2995377 * float(x[14]))+ (-0.2986113 * float(x[15]))+ (0.40801895 * float(x[16]))+ (-0.3038449 * float(x[17]))+ (0.36923003 * float(x[18]))) + 0.8200687), 0)
    h_9 = max((((0.015588964 * float(x[0]))+ (-2.4069664 * float(x[1]))+ (-0.08146744 * float(x[2]))+ (-0.08199896 * float(x[3]))+ (-0.16713656 * float(x[4]))+ (0.23438211 * float(x[5]))+ (0.085838675 * float(x[6]))+ (0.056636713 * float(x[7]))+ (0.12558421 * float(x[8]))+ (0.394012 * float(x[9]))+ (0.065384574 * float(x[10]))+ (-0.07534075 * float(x[11]))+ (0.23323728 * float(x[12]))+ (-0.19102381 * float(x[13]))+ (0.6487648 * float(x[14]))+ (-0.2985801 * float(x[15]))+ (-0.15282233 * float(x[16]))+ (0.5982648 * float(x[17]))+ (-0.20236635 * float(x[18]))) + 0.6058603), 0)
    h_10 = max((((-0.41569284 * float(x[0]))+ (1.4620632 * float(x[1]))+ (0.3516067 * float(x[2]))+ (-0.27378064 * float(x[3]))+ (0.29336286 * float(x[4]))+ (0.070008345 * float(x[5]))+ (0.44410664 * float(x[6]))+ (0.46309015 * float(x[7]))+ (0.0021246613 * float(x[8]))+ (-0.24477695 * float(x[9]))+ (-0.11959226 * float(x[10]))+ (-0.29147366 * float(x[11]))+ (-0.33425972 * float(x[12]))+ (0.06756398 * float(x[13]))+ (-0.116703205 * float(x[14]))+ (-0.06417447 * float(x[15]))+ (-0.08989789 * float(x[16]))+ (0.87919027 * float(x[17]))+ (0.68233216 * float(x[18]))) + 0.64059395), 0)
    h_11 = max((((-0.1117663 * float(x[0]))+ (1.2861887 * float(x[1]))+ (0.32873017 * float(x[2]))+ (-0.3228038 * float(x[3]))+ (-0.15252383 * float(x[4]))+ (0.102086775 * float(x[5]))+ (-0.29029942 * float(x[6]))+ (-0.4203069 * float(x[7]))+ (-0.21708679 * float(x[8]))+ (0.52155954 * float(x[9]))+ (0.0727303 * float(x[10]))+ (0.17915647 * float(x[11]))+ (0.060359243 * float(x[12]))+ (-2.1339903 * float(x[13]))+ (0.42765653 * float(x[14]))+ (0.7153211 * float(x[15]))+ (0.55394375 * float(x[16]))+ (-0.80928326 * float(x[17]))+ (0.32448155 * float(x[18]))) + 0.3102159), 0)
    h_12 = max((((0.008907158 * float(x[0]))+ (-0.4003252 * float(x[1]))+ (-0.049079873 * float(x[2]))+ (-0.25254497 * float(x[3]))+ (0.2488646 * float(x[4]))+ (-0.069819205 * float(x[5]))+ (0.20891383 * float(x[6]))+ (0.2504763 * float(x[7]))+ (-0.17585051 * float(x[8]))+ (-0.034393236 * float(x[9]))+ (0.10521428 * float(x[10]))+ (-0.2751704 * float(x[11]))+ (-0.4430088 * float(x[12]))+ (-0.5189656 * float(x[13]))+ (-0.5010785 * float(x[14]))+ (1.7305955 * float(x[15]))+ (-0.022039147 * float(x[16]))+ (-0.10183682 * float(x[17]))+ (0.28595126 * float(x[18]))) + 0.51906765), 0)
    h_13 = max((((-0.005583419 * float(x[0]))+ (-0.8677605 * float(x[1]))+ (0.2564527 * float(x[2]))+ (0.13325746 * float(x[3]))+ (-0.07674917 * float(x[4]))+ (-0.2698905 * float(x[5]))+ (0.096195914 * float(x[6]))+ (-0.3974184 * float(x[7]))+ (0.2933359 * float(x[8]))+ (-0.19459212 * float(x[9]))+ (-0.044507734 * float(x[10]))+ (0.07948891 * float(x[11]))+ (-0.21749869 * float(x[12]))+ (-0.5213788 * float(x[13]))+ (0.023958215 * float(x[14]))+ (0.24546506 * float(x[15]))+ (-0.26155952 * float(x[16]))+ (1.707196 * float(x[17]))+ (-0.14863943 * float(x[18]))) + 0.7690147), 0)
    h_14 = max((((0.3712745 * float(x[0]))+ (0.032180704 * float(x[1]))+ (0.16406833 * float(x[2]))+ (0.14832787 * float(x[3]))+ (-0.26562315 * float(x[4]))+ (-0.4717252 * float(x[5]))+ (0.4685298 * float(x[6]))+ (-0.19974932 * float(x[7]))+ (0.42807543 * float(x[8]))+ (-0.035853256 * float(x[9]))+ (0.46080706 * float(x[10]))+ (-0.34700537 * float(x[11]))+ (0.6208891 * float(x[12]))+ (0.29719993 * float(x[13]))+ (0.18832631 * float(x[14]))+ (-0.012132887 * float(x[15]))+ (-0.30751938 * float(x[16]))+ (-0.31925416 * float(x[17]))+ (0.48397034 * float(x[18]))) + 0.58736694), 0)
    h_15 = max((((0.008968104 * float(x[0]))+ (0.12946992 * float(x[1]))+ (-0.28341177 * float(x[2]))+ (0.28491744 * float(x[3]))+ (0.27251962 * float(x[4]))+ (0.38565797 * float(x[5]))+ (-0.040779587 * float(x[6]))+ (-0.050446816 * float(x[7]))+ (-0.079847544 * float(x[8]))+ (-0.56899905 * float(x[9]))+ (-0.07546201 * float(x[10]))+ (-0.06805652 * float(x[11]))+ (-0.09281364 * float(x[12]))+ (1.3444625 * float(x[13]))+ (0.19689699 * float(x[14]))+ (-0.744944 * float(x[15]))+ (-0.051511873 * float(x[16]))+ (-1.052565 * float(x[17]))+ (0.0721 * float(x[18]))) + 0.13012537), 0)
    o[0] = (1.0073366 * h_0)+ (1.5974882 * h_1)+ (-1.8076589 * h_2)+ (-0.5730517 * h_3)+ (-1.5109953 * h_4)+ (-0.47839186 * h_5)+ (1.418474 * h_6)+ (0.7326634 * h_7)+ (-1.314617 * h_8)+ (0.84103394 * h_9)+ (-0.61299145 * h_10)+ (-0.8284957 * h_11)+ (-0.062300008 * h_12)+ (-0.01436055 * h_13)+ (0.15245208 * h_14)+ (0.6758828 * h_15) + -0.8039047
    o[1] = (0.17456864 * h_0)+ (-1.3763591 * h_1)+ (0.1717949 * h_2)+ (-0.62632084 * h_3)+ (0.14666328 * h_4)+ (0.037005298 * h_5)+ (0.4098393 * h_6)+ (0.34834597 * h_7)+ (0.18593748 * h_8)+ (-1.0560668 * h_9)+ (0.37506106 * h_10)+ (0.6296309 * h_11)+ (-1.2498068 * h_12)+ (1.0158376 * h_13)+ (-0.03749163 * h_14)+ (-1.0981894 * h_15) + 0.9324111
    o[2] = (-0.111941285 * h_0)+ (0.86750257 * h_1)+ (-1.667255 * h_2)+ (-0.32617092 * h_3)+ (0.55206966 * h_4)+ (-1.0230714 * h_5)+ (0.5259258 * h_6)+ (-0.33569136 * h_7)+ (-0.8250217 * h_8)+ (-1.7636288 * h_9)+ (0.71853715 * h_10)+ (-0.39888227 * h_11)+ (0.28389332 * h_12)+ (-0.90316963 * h_13)+ (0.46891075 * h_14)+ (0.8578248 * h_15) + 0.9526167
    o[3] = (-0.5704595 * h_0)+ (-0.9323731 * h_1)+ (0.48989567 * h_2)+ (0.7833374 * h_3)+ (-0.843971 * h_4)+ (-0.690272 * h_5)+ (-1.9609522 * h_6)+ (-1.632336 * h_7)+ (1.5041057 * h_8)+ (0.51460403 * h_9)+ (0.14142002 * h_10)+ (0.8925127 * h_11)+ (0.37545758 * h_12)+ (-3.3234677 * h_13)+ (0.3448607 * h_14)+ (2.1453085 * h_15) + 1.2028568
    o[4] = (-1.0605861 * h_0)+ (0.13222234 * h_1)+ (-0.7777861 * h_2)+ (-0.19172601 * h_3)+ (0.5686651 * h_4)+ (0.4181204 * h_5)+ (-0.1435716 * h_6)+ (0.6601734 * h_7)+ (0.1818063 * h_8)+ (-0.5008232 * h_9)+ (-0.19134037 * h_10)+ (0.30073297 * h_11)+ (-0.42855936 * h_12)+ (1.2697358 * h_13)+ (0.5329668 * h_14)+ (-0.8213711 * h_15) + 1.0627793
    o[5] = (-1.0669605 * h_0)+ (0.24503385 * h_1)+ (1.027732 * h_2)+ (-0.8157848 * h_3)+ (0.36848757 * h_4)+ (0.43176135 * h_5)+ (0.4373667 * h_6)+ (-0.5147637 * h_7)+ (0.95993716 * h_8)+ (0.07104174 * h_9)+ (0.7183232 * h_10)+ (0.21832897 * h_11)+ (0.10751122 * h_12)+ (-0.16687132 * h_13)+ (-1.4654833 * h_14)+ (-0.10449153 * h_15) + 0.7336815
    o[6] = (0.11271392 * h_0)+ (0.12314407 * h_1)+ (0.33425212 * h_2)+ (0.8598424 * h_3)+ (1.0878747 * h_4)+ (0.55383587 * h_5)+ (-0.57232636 * h_6)+ (0.080893956 * h_7)+ (0.55258477 * h_8)+ (1.3465385 * h_9)+ (-0.9932555 * h_10)+ (-0.8101026 * h_11)+ (1.1097693 * h_12)+ (-1.3373191 * h_13)+ (-0.88666785 * h_14)+ (-0.21025382 * h_15) + 0.6586811

    

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
    w_h = np.array([[-0.522879421710968, -1.2125693559646606, -0.3933950364589691, 0.09026532620191574, 0.40149688720703125, 0.4313313961029053, -0.01673302985727787, 0.3433115482330322, -0.08257517218589783, 0.18618139624595642, 0.09274114668369293, -0.2424800544977188, 0.3539064824581146, -0.056372616440057755, -0.3246590793132782, -0.5540403723716736, 0.28739553689956665, 0.9776049256324768, 0.2233443409204483], [0.025577036663889885, 0.08330398797988892, -0.24019481241703033, -0.003076339140534401, -0.3485296964645386, -0.3301629424095154, 0.1905219852924347, -0.05873787775635719, 0.17276398837566376, 0.1961917132139206, 0.04303774982690811, -0.5147814154624939, -0.0963895246386528, -0.29929760098457336, -0.4251420497894287, -0.3548985719680786, -0.8995887041091919, 1.2735562324523926, 1.1834237575531006], [0.03808704763650894, -0.032838139683008194, 0.23052062094211578, 0.2640763223171234, 0.06688813120126724, 0.26813963055610657, 0.3481210768222809, 0.34680289030075073, 0.08715636283159256, 0.04836808890104294, 0.37608060240745544, -1.1884623765945435, -0.3128013610839844, 0.08235818147659302, 0.035309940576553345, -0.05019358918070793, -3.0095627307891846, 0.009992821142077446, 0.1965816617012024], [0.17563460767269135, -0.3039253354072571, -0.2612290680408478, 0.11937247216701508, 0.29546838998794556, 0.11508757621049881, -0.03001503273844719, 0.25740429759025574, -0.13999530673027039, 0.08106521517038345, 0.055492933839559555, 0.12189309298992157, 0.31517234444618225, -0.17962825298309326, 0.22625337541103363, 0.30335748195648193, 0.5043032169342041, -0.13041065633296967, -1.1040024757385254], [-0.08634936064481735, 0.5441048741340637, 0.18080638349056244, -0.09078461676836014, -0.13986672461032867, -0.5477328896522522, -0.5219780206680298, 0.2323065996170044, -0.014261954464018345, 0.05278432369232178, 0.37987083196640015, 0.23991000652313232, -0.1262948364019394, -0.23906008899211884, 0.810455322265625, -1.3764716386795044, 0.3328213691711426, 0.0495772622525692, 0.3691640794277191], [-0.09996668249368668, 0.466026246547699, -0.21342876553535461, 0.12225954979658127, -0.022253481671214104, 0.006572272628545761, 0.12541010975837708, 0.042832545936107635, -0.22490271925926208, -0.228398859500885, -0.06188780814409256, 0.08748987317085266, -0.2109912484884262, 1.345909833908081, 0.8175953030586243, -0.803050696849823, 0.2948870062828064, -0.61346834897995, 0.0073372856713831425], [0.008275256492197514, 0.06916283816099167, -0.03451094403862953, 0.33021578192710876, 0.1287914216518402, 0.36393997073173523, 0.07499855756759644, 0.180275559425354, 0.198167622089386, -0.26081815361976624, -0.08429604768753052, -0.12000691145658493, -0.049898091703653336, 0.24123919010162354, 0.03865814954042435, 0.5587819218635559, -0.4325145483016968, -1.336789846420288, -1.195895791053772], [-0.056871455162763596, -0.1627783328294754, 0.07368531823158264, 0.43220603466033936, -0.29193949699401855, -0.025562874972820282, 0.3131222128868103, -0.02507748082280159, -0.06124339625239372, 0.17353248596191406, 0.18508489429950714, -0.4211632311344147, 0.23221474885940552, 1.1938420534133911, -0.5927122235298157, -0.3669925332069397, 0.10550644993782043, -1.4881690740585327, 1.1124836206436157], [0.049637336283922195, 0.18017202615737915, -0.09432897716760635, -0.05600779503583908, 0.1668175309896469, -0.1637556552886963, 0.11578826606273651, 0.027928829193115234, 0.0323365144431591, -0.04927417263388634, -0.1826116442680359, -0.6171836256980896, 0.015595824457705021, 1.0347836017608643, -1.2995376586914062, -0.2986113131046295, 0.40801894664764404, -0.3038448989391327, 0.3692300319671631], [0.015588964335620403, -2.4069664478302, -0.08146744221448898, -0.08199895918369293, -0.1671365648508072, 0.23438210785388947, 0.0858386754989624, 0.056636713445186615, 0.125584214925766, 0.3940120041370392, 0.06538457423448563, -0.07534074783325195, 0.23323728144168854, -0.1910238116979599, 0.6487647891044617, -0.2985801100730896, -0.1528223305940628, 0.5982648134231567, -0.20236635208129883], [-0.41569283604621887, 1.462063193321228, 0.35160669684410095, -0.2737806439399719, 0.2933628559112549, 0.07000834494829178, 0.4441066384315491, 0.4630901515483856, 0.002124661346897483, -0.24477694928646088, -0.11959225684404373, -0.2914736568927765, -0.3342597186565399, 0.0675639808177948, -0.11670320481061935, -0.06417447328567505, -0.08989789336919785, 0.8791902661323547, 0.6823321580886841], [-0.11176630109548569, 1.2861887216567993, 0.32873016595840454, -0.322803795337677, -0.15252383053302765, 0.10208677500486374, -0.2902994155883789, -0.42030689120292664, -0.2170867919921875, 0.5215595364570618, 0.07273030281066895, 0.17915646731853485, 0.06035924330353737, -2.1339902877807617, 0.42765653133392334, 0.7153211236000061, 0.5539437532424927, -0.8092832565307617, 0.3244815468788147], [0.008907157927751541, -0.400325208902359, -0.04907987266778946, -0.25254496932029724, 0.24886460602283478, -0.06981920450925827, 0.20891383290290833, 0.25047630071640015, -0.175850510597229, -0.03439323604106903, 0.10521428287029266, -0.27517038583755493, -0.44300881028175354, -0.5189656019210815, -0.5010784864425659, 1.7305954694747925, -0.022039147093892097, -0.10183682292699814, 0.28595125675201416], [-0.005583418998867273, -0.8677604794502258, 0.2564527094364166, 0.13325746357440948, -0.07674916833639145, -0.2698904871940613, 0.09619591385126114, -0.39741840958595276, 0.2933359146118164, -0.19459211826324463, -0.04450773447751999, 0.07948891073465347, -0.21749868988990784, -0.5213788151741028, 0.02395821548998356, 0.24546505510807037, -0.26155951619148254, 1.7071959972381592, -0.14863942563533783], [0.37127450108528137, 0.032180704176425934, 0.1640683263540268, 0.14832787215709686, -0.26562315225601196, -0.471725195646286, 0.4685297906398773, -0.19974932074546814, 0.4280754327774048, -0.03585325554013252, 0.46080705523490906, -0.34700536727905273, 0.6208891272544861, 0.29719993472099304, 0.18832631409168243, -0.012132886797189713, -0.3075193762779236, -0.31925415992736816, 0.4839703440666199], [0.008968103677034378, 0.12946991622447968, -0.28341177105903625, 0.2849174439907074, 0.27251961827278137, 0.3856579661369324, -0.040779586881399155, -0.05044681578874588, -0.0798475444316864, -0.5689990520477295, -0.07546201348304749, -0.06805651634931564, -0.092813640832901, 1.344462513923645, 0.1968969851732254, -0.7449439764022827, -0.05151187255978584, -1.0525649785995483, 0.07209999859333038]])
    b_h = np.array([0.5650724768638611, 0.5765786170959473, 0.8725002408027649, 0.7126811742782593, 0.4226056933403015, 0.6425808072090149, 1.201488733291626, 0.8744291067123413, 0.8200687170028687, 0.6058602929115295, 0.640593945980072, 0.31021589040756226, 0.519067645072937, 0.7690147161483765, 0.5873669385910034, 0.13012537360191345])
    w_o = np.array([[1.0073366165161133, 1.5974881649017334, -1.8076589107513428, -0.5730516910552979, -1.5109952688217163, -0.4783918559551239, 1.4184739589691162, 0.7326633930206299, -1.3146170377731323, 0.841033935546875, -0.612991452217102, -0.8284956812858582, -0.06230000779032707, -0.014360549859702587, 0.15245208144187927, 0.6758828163146973], [0.17456863820552826, -1.3763591051101685, 0.17179490625858307, -0.6263208389282227, 0.14666327834129333, 0.03700529783964157, 0.40983930230140686, 0.34834596514701843, 0.18593747913837433, -1.0560667514801025, 0.3750610649585724, 0.6296309232711792, -1.2498067617416382, 1.015837550163269, -0.037491630762815475, -1.098189353942871], [-0.11194128543138504, 0.8675025701522827, -1.6672550439834595, -0.3261709213256836, 0.5520696640014648, -1.0230714082717896, 0.5259258151054382, -0.3356913626194, -0.8250216841697693, -1.7636288404464722, 0.7185371518135071, -0.39888226985931396, 0.28389331698417664, -0.9031696319580078, 0.46891075372695923, 0.8578248023986816], [-0.5704594850540161, -0.9323731064796448, 0.48989567160606384, 0.783337414264679, -0.8439710140228271, -0.6902719736099243, -1.9609521627426147, -1.6323360204696655, 1.5041056871414185, 0.5146040320396423, 0.14142002165317535, 0.8925126791000366, 0.3754575848579407, -3.32346773147583, 0.344860702753067, 2.145308494567871], [-1.0605860948562622, 0.13222233951091766, -0.7777860760688782, -0.19172601401805878, 0.568665087223053, 0.418120414018631, -0.14357160031795502, 0.6601734161376953, 0.1818062961101532, -0.5008231997489929, -0.191340371966362, 0.30073297023773193, -0.4285593628883362, 1.2697358131408691, 0.5329667925834656, -0.8213710784912109], [-1.0669604539871216, 0.2450338453054428, 1.027732014656067, -0.8157848119735718, 0.36848756670951843, 0.43176135420799255, 0.43736669421195984, -0.5147637128829956, 0.9599371552467346, 0.07104174047708511, 0.7183231711387634, 0.21832896769046783, 0.10751122236251831, -0.1668713241815567, -1.46548330783844, -0.10449153184890747], [0.11271391808986664, 0.12314406782388687, 0.33425211906433105, 0.8598424196243286, 1.0878746509552002, 0.5538358688354492, -0.5723263621330261, 0.08089395612478256, 0.5525847673416138, 1.3465385437011719, -0.9932554960250854, -0.8101025819778442, 1.109769344329834, -1.3373191356658936, -0.8866678476333618, -0.21025381982326508]])
    b_o = np.array([-0.8039047122001648, 0.9324110746383667, 0.9526166915893555, 1.2028567790985107, 1.0627793073654175, 0.7336814999580383, 0.6586810946464539])

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
        model_cap = 439
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

