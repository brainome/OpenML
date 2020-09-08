#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target game kropt.csv -o kropt_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 1:47:11.02. Finished on: Sep-04-2020 13:03:28.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         18-way classifier
Best-guess accuracy:                 16.23%
Overall Model accuracy:              36.47% (10234/28056 correct)
Overall Improvement over best guess: 20.24% (of possible 83.77%)
Model capacity (MEC):                210 bits
Generalization ratio:                48.73 bits/bit
Model efficiency:                    0.09%/parameter
Confusion Matrix:
 [8.09% 0.00% 0.00% 0.02% 0.00% 0.09% 0.02% 0.10% 0.01% 0.21% 0.19% 0.00%
  0.08% 0.01% 0.18% 0.47% 0.36% 0.13%]
 [0.01% 0.00% 0.00% 0.03% 0.00% 0.00% 0.01% 0.00% 0.00% 0.00% 0.00% 0.04%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.07% 0.00% 0.00% 0.06% 0.00% 0.01% 0.01% 0.00% 0.00% 0.03% 0.02% 0.08%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.10% 0.00% 0.00% 0.42% 0.00% 0.01% 0.11% 0.00% 0.00% 0.06% 0.03% 0.14%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.03% 0.00% 0.00% 0.07% 0.00% 0.06% 0.02% 0.04% 0.00% 0.06% 0.01% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.07% 0.00% 0.00% 0.00% 0.00% 0.32% 0.07% 0.14% 0.00% 0.09% 0.00% 0.01%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.25% 0.00% 0.00% 0.04% 0.00% 0.21% 0.25% 0.25% 0.01% 0.47% 0.08% 0.09%
  0.01% 0.01% 0.00% 0.00% 0.00% 0.00%]
 [0.32% 0.00% 0.00% 0.02% 0.00% 0.14% 0.04% 0.59% 0.01% 0.65% 0.12% 0.07%
  0.02% 0.04% 0.07% 0.02% 0.00% 0.00%]
 [0.41% 0.00% 0.00% 0.06% 0.00% 0.05% 0.07% 0.17% 0.03% 0.67% 0.19% 0.21%
  0.08% 0.10% 0.18% 0.20% 0.02% 0.00%]
 [0.76% 0.00% 0.00% 0.19% 0.00% 0.02% 0.09% 0.17% 0.03% 2.09% 0.50% 0.43%
  0.19% 0.06% 0.22% 0.25% 0.12% 0.00%]
 [0.97% 0.00% 0.00% 0.13% 0.00% 0.01% 0.04% 0.05% 0.02% 1.17% 1.02% 0.83%
  0.83% 0.30% 0.36% 0.31% 0.07% 0.00%]
 [1.29% 0.00% 0.00% 0.14% 0.00% 0.00% 0.03% 0.02% 0.01% 0.73% 0.52% 0.82%
  1.14% 0.84% 0.77% 0.58% 0.16% 0.01%]
 [1.85% 0.00% 0.00% 0.05% 0.00% 0.00% 0.01% 0.04% 0.00% 0.26% 0.48% 0.58%
  2.20% 1.73% 1.49% 1.20% 0.25% 0.04%]
 [2.27% 0.00% 0.00% 0.04% 0.00% 0.01% 0.00% 0.02% 0.00% 0.18% 0.23% 0.58%
  1.21% 3.52% 2.47% 2.07% 0.23% 0.00%]
 [2.93% 0.00% 0.00% 0.02% 0.00% 0.00% 0.00% 0.00% 0.01% 0.07% 0.02% 0.23%
  0.31% 1.76% 5.35% 3.81% 0.42% 0.02%]
 [2.55% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.02% 0.02% 0.25%
  0.17% 0.43% 2.60% 8.98% 1.15% 0.06%]
 [1.23% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01% 0.00% 0.06%
  0.09% 0.06% 0.23% 3.48% 2.38% 0.19%]
 [0.19% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01%
  0.00% 0.00% 0.00% 0.22% 0.55% 0.41%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to 'draw'=0, 'zero'=1, 'one'=2, 'two'=3, 'three'=4, 'four'=5, 'five'=6, 'six'=7, 'seven'=8, 'eight'=9, 'nine'=10, 'ten'=11, 'eleven'=12, 'twelve'=13, 'thirteen'=14, 'fourteen'=15, 'fifteen'=16, 'sixteen'=17.
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
TRAINFILE = "kropt.csv"


#Number of output logits
num_output_logits = 18

#Number of attributes
num_attr = 6
n_classes = 18

mappings = [{112844655.0: 0, 1908338681.0: 1, 2564639436.0: 2, 3904355907.0: 3}, {30677878.0: 0, 112844655.0: 1, 1908338681.0: 2, 1993550816.0: 3, 2439710439.0: 4, 2564639436.0: 5, 3904355907.0: 6, 4024072794.0: 7}, {30677878.0: 0, 112844655.0: 1, 1908338681.0: 2, 1993550816.0: 3, 2439710439.0: 4, 2564639436.0: 5, 3904355907.0: 6, 4024072794.0: 7}]
list_of_cols_to_normalize = [0, 2, 4]

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
target="game"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="game"
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
    clean.mapping={'draw': 0, 'zero': 1, 'one': 2, 'two': 3, 'three': 4, 'four': 5, 'five': 6, 'six': 7, 'seven': 8, 'eight': 9, 'nine': 10, 'ten': 11, 'eleven': 12, 'twelve': 13, 'thirteen': 14, 'fourteen': 15, 'fifteen': 16, 'sixteen': 17}

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
    h_0 = max((((0.018664157 * float(x[0]))+ (-1.5544206 * float(x[1]))+ (-0.14637622 * float(x[2]))+ (0.5762373 * float(x[3]))+ (0.3365394 * float(x[4]))+ (0.39191937 * float(x[5]))) + 1.4524614), 0)
    h_1 = max((((-0.0025363916 * float(x[0]))+ (-0.021738682 * float(x[1]))+ (0.0034400625 * float(x[2]))+ (1.0845523 * float(x[3]))+ (-0.0044578426 * float(x[4]))+ (-1.0666338 * float(x[5]))) + 1.0138116), 0)
    h_2 = max((((-1.0058477 * float(x[0]))+ (-0.12834492 * float(x[1]))+ (-0.11251173 * float(x[2]))+ (-0.37922573 * float(x[3]))+ (0.0371404 * float(x[4]))+ (0.80141854 * float(x[5]))) + 2.6621678), 0)
    h_3 = max((((0.034492806 * float(x[0]))+ (0.14370447 * float(x[1]))+ (0.19495136 * float(x[2]))+ (0.29380295 * float(x[3]))+ (0.39868766 * float(x[4]))+ (-0.54366946 * float(x[5]))) + 2.3631446), 0)
    h_4 = max((((-1.8899556 * float(x[0]))+ (1.3380264 * float(x[1]))+ (0.05839899 * float(x[2]))+ (0.011034478 * float(x[3]))+ (0.030276852 * float(x[4]))+ (1.0879122 * float(x[5]))) + 0.35702014), 0)
    h_5 = max((((-0.009844772 * float(x[0]))+ (-0.043737926 * float(x[1]))+ (0.6670608 * float(x[2]))+ (0.014454744 * float(x[3]))+ (-0.6768211 * float(x[4]))+ (-0.02478176 * float(x[5]))) + -2.6416316), 0)
    h_6 = max((((-0.7726168 * float(x[0]))+ (2.3607285 * float(x[1]))+ (0.2278184 * float(x[2]))+ (0.38826203 * float(x[3]))+ (-0.38202742 * float(x[4]))+ (0.52663946 * float(x[5]))) + -0.29855728), 0)
    h_7 = max((((0.053703833 * float(x[0]))+ (0.27639475 * float(x[1]))+ (0.060449056 * float(x[2]))+ (0.4573672 * float(x[3]))+ (-0.018432098 * float(x[4]))+ (-1.4121382 * float(x[5]))) + 3.541628), 0)
    h_8 = max((((-0.11078291 * float(x[0]))+ (-0.9475541 * float(x[1]))+ (-0.12738831 * float(x[2]))+ (0.013451222 * float(x[3]))+ (0.03985341 * float(x[4]))+ (-0.18978783 * float(x[5]))) + 3.1230016), 0)
    h_9 = max((((0.012855873 * float(x[0]))+ (-0.008466049 * float(x[1]))+ (0.0061827367 * float(x[2]))+ (1.2552459 * float(x[3]))+ (-0.0048780204 * float(x[4]))+ (-1.2629355 * float(x[5]))) + -1.2829392), 0)
    h_10 = max((((1.6515346 * float(x[0]))+ (-0.27310503 * float(x[1]))+ (0.3317324 * float(x[2]))+ (-0.09863682 * float(x[3]))+ (0.05983313 * float(x[4]))+ (0.22314823 * float(x[5]))) + 0.35807383), 0)
    h_11 = max((((-0.038636148 * float(x[0]))+ (0.4237588 * float(x[1]))+ (0.048942707 * float(x[2]))+ (0.94255424 * float(x[3]))+ (-0.14032698 * float(x[4]))+ (-0.30722684 * float(x[5]))) + 1.0238461), 0)
    h_12 = max((((-1.7908303 * float(x[0]))+ (0.46140012 * float(x[1]))+ (-0.049587466 * float(x[2]))+ (-0.0036855997 * float(x[3]))+ (0.037184726 * float(x[4]))+ (0.057606675 * float(x[5]))) + 3.0886), 0)
    h_13 = max((((-0.1789984 * float(x[0]))+ (0.07468705 * float(x[1]))+ (-0.036905497 * float(x[2]))+ (0.0067656026 * float(x[3]))+ (-0.75553316 * float(x[4]))+ (0.22413328 * float(x[5]))) + 3.093834), 0)
    h_14 = max((((0.70781875 * float(x[0]))+ (-0.91840416 * float(x[1]))+ (0.0007880142 * float(x[2]))+ (-0.0054507847 * float(x[3]))+ (-0.3730736 * float(x[4]))+ (0.039863456 * float(x[5]))) + 3.1036017), 0)
    o[0] = (0.76237327 * h_0)+ (-2.8390143 * h_1)+ (-1.4146651 * h_2)+ (1.0551822 * h_3)+ (-0.7191578 * h_4)+ (-11.017395 * h_5)+ (0.20180145 * h_6)+ (-0.55042547 * h_7)+ (0.5602921 * h_8)+ (-7.3115754 * h_9)+ (-0.50530523 * h_10)+ (1.6381757 * h_11)+ (0.7895894 * h_12)+ (0.702169 * h_13)+ (0.12852784 * h_14) + -0.75626475
    o[1] = (0.40031812 * h_0)+ (-0.34047276 * h_1)+ (-1.2742788 * h_2)+ (0.016582137 * h_3)+ (-0.42893797 * h_4)+ (0.38054454 * h_5)+ (-0.026658116 * h_6)+ (0.24565972 * h_7)+ (-0.56784356 * h_8)+ (0.3197894 * h_9)+ (0.40500346 * h_10)+ (-1.0132045 * h_11)+ (0.961077 * h_12)+ (1.0026292 * h_13)+ (-9.66418 * h_14) + 0.85237175
    o[2] = (-0.5425578 * h_0)+ (0.8952495 * h_1)+ (0.43042272 * h_2)+ (0.075391255 * h_3)+ (-0.6907067 * h_4)+ (-1.5550319 * h_5)+ (-0.068280384 * h_6)+ (-0.35998714 * h_7)+ (-0.17232703 * h_8)+ (-0.38369095 * h_9)+ (-0.21966544 * h_10)+ (-0.050122853 * h_11)+ (0.42658374 * h_12)+ (-0.6141124 * h_13)+ (-1.8769927 * h_14) + 0.37300986
    o[3] = (-0.68464243 * h_0)+ (0.32243112 * h_1)+ (1.1855652 * h_2)+ (-0.16384266 * h_3)+ (-0.8244731 * h_4)+ (1.3447087 * h_5)+ (-0.3164633 * h_6)+ (0.33149856 * h_7)+ (-0.08989132 * h_8)+ (0.11397109 * h_9)+ (0.16962337 * h_10)+ (0.07493099 * h_11)+ (0.30371928 * h_12)+ (0.6343447 * h_13)+ (-11.848979 * h_14) + 1.2383124
    o[4] = (-0.37917274 * h_0)+ (0.3981394 * h_1)+ (-0.39659384 * h_2)+ (0.40915757 * h_3)+ (-0.7620507 * h_4)+ (-0.84215564 * h_5)+ (-0.092908315 * h_6)+ (1.0515572 * h_7)+ (-1.350088 * h_8)+ (-1.3282734 * h_9)+ (-1.055546 * h_10)+ (0.11296289 * h_11)+ (-0.40664613 * h_12)+ (-0.059423473 * h_13)+ (-0.2231103 * h_14) + 0.9107965
    o[5] = (0.1287514 * h_0)+ (0.23856938 * h_1)+ (1.2233514 * h_2)+ (0.8972706 * h_3)+ (-2.4687185 * h_4)+ (-0.40069243 * h_5)+ (0.5595154 * h_6)+ (0.6889992 * h_7)+ (-0.5682296 * h_8)+ (-2.1773353 * h_9)+ (-0.99496 * h_10)+ (0.8906039 * h_11)+ (-3.016259 * h_12)+ (0.6052195 * h_13)+ (-2.1778543 * h_14) + 1.0592802
    o[6] = (-0.107582785 * h_0)+ (0.70458573 * h_1)+ (2.023043 * h_2)+ (-0.0027835781 * h_3)+ (-0.4939187 * h_4)+ (0.11257811 * h_5)+ (0.17344053 * h_6)+ (1.1089569 * h_7)+ (-0.1187549 * h_8)+ (-1.3517644 * h_9)+ (-0.5297268 * h_10)+ (0.10024355 * h_11)+ (-2.504989 * h_12)+ (-0.14032829 * h_13)+ (-1.2260331 * h_14) + 0.9689684
    o[7] = (-0.23796318 * h_0)+ (0.36673898 * h_1)+ (1.7597969 * h_2)+ (-0.1654006 * h_3)+ (-0.4120879 * h_4)+ (0.32911757 * h_5)+ (0.53812534 * h_6)+ (1.0698323 * h_7)+ (-0.16501369 * h_8)+ (-1.0361012 * h_9)+ (-0.798661 * h_10)+ (0.0029149023 * h_11)+ (-2.1753948 * h_12)+ (-1.3116426 * h_13)+ (0.282484 * h_14) + 1.4396598
    o[8] = (-0.16161667 * h_0)+ (0.43831876 * h_1)+ (1.2334727 * h_2)+ (-0.5098443 * h_3)+ (-0.040750537 * h_4)+ (0.40708122 * h_5)+ (0.36050063 * h_6)+ (0.83207667 * h_7)+ (0.61957276 * h_8)+ (-0.97329855 * h_9)+ (-0.3022037 * h_10)+ (0.14099087 * h_11)+ (-1.4816796 * h_12)+ (-0.9057636 * h_13)+ (-0.5099258 * h_14) + 0.73981345
    o[9] = (-0.07931835 * h_0)+ (-0.3061095 * h_1)+ (0.780625 * h_2)+ (-0.64876866 * h_3)+ (0.008889059 * h_4)+ (0.12766431 * h_5)+ (0.53964245 * h_6)+ (1.038336 * h_7)+ (1.3683448 * h_8)+ (-0.36526534 * h_9)+ (-0.23361996 * h_10)+ (0.20409423 * h_11)+ (-1.131949 * h_12)+ (-0.9438723 * h_13)+ (-0.40477678 * h_14) + 0.13121688
    o[10] = (-0.09799147 * h_0)+ (-0.53490686 * h_1)+ (-0.16639598 * h_2)+ (-0.70662695 * h_3)+ (0.70513135 * h_4)+ (0.05311072 * h_5)+ (0.57009006 * h_6)+ (0.72058505 * h_7)+ (2.2580376 * h_8)+ (0.17426126 * h_9)+ (-0.19110547 * h_10)+ (-0.19159937 * h_11)+ (-1.106565 * h_12)+ (-1.1179675 * h_13)+ (0.14122906 * h_14) + 0.4261073
    o[11] = (0.20208636 * h_0)+ (-0.47127768 * h_1)+ (-0.5141182 * h_2)+ (-0.2882125 * h_3)+ (0.5655616 * h_4)+ (-0.32664844 * h_5)+ (0.41558588 * h_6)+ (0.069305144 * h_7)+ (1.3095365 * h_8)+ (0.5237975 * h_9)+ (0.087122686 * h_10)+ (-0.53931355 * h_11)+ (-0.5759336 * h_12)+ (-0.34301376 * h_13)+ (0.047251996 * h_14) + 0.6697127
    o[12] = (0.060196366 * h_0)+ (-0.27877548 * h_1)+ (-0.6205622 * h_2)+ (-0.59475845 * h_3)+ (0.5690504 * h_4)+ (-0.26272973 * h_5)+ (0.19481191 * h_6)+ (-0.5559009 * h_7)+ (1.5666704 * h_8)+ (0.7225489 * h_9)+ (0.6205693 * h_10)+ (-0.34832618 * h_11)+ (0.04926815 * h_12)+ (-0.29772335 * h_13)+ (0.0006760149 * h_14) + 0.4503426
    o[13] = (-0.042376038 * h_0)+ (-0.118858315 * h_1)+ (-0.30411008 * h_2)+ (-0.13707578 * h_3)+ (0.5403144 * h_4)+ (-0.27336898 * h_5)+ (-0.1549011 * h_6)+ (-1.114499 * h_7)+ (0.54708165 * h_8)+ (0.8389813 * h_9)+ (0.56200355 * h_10)+ (-0.18386438 * h_11)+ (0.045002166 * h_12)+ (0.17768501 * h_13)+ (-0.040949587 * h_14) + 0.6644969
    o[14] = (0.08332803 * h_0)+ (0.04604908 * h_1)+ (0.17698221 * h_2)+ (0.078215085 * h_3)+ (0.18469228 * h_4)+ (-0.21098503 * h_5)+ (-0.36906457 * h_6)+ (-1.4722332 * h_7)+ (-1.2040615 * h_8)+ (0.73906153 * h_9)+ (0.38372514 * h_10)+ (-0.063698225 * h_11)+ (0.39259204 * h_12)+ (0.12820609 * h_13)+ (0.6028962 * h_14) + 1.2048465
    o[15] = (0.47984958 * h_0)+ (0.32249287 * h_1)+ (0.24075292 * h_2)+ (0.81756353 * h_3)+ (0.106909536 * h_4)+ (-0.06896426 * h_5)+ (-0.86800545 * h_6)+ (-2.127788 * h_7)+ (-2.2331998 * h_8)+ (0.890568 * h_9)+ (-0.09536705 * h_10)+ (-0.4131504 * h_11)+ (0.4393442 * h_12)+ (0.7669139 * h_13)+ (1.0759879 * h_14) + 2.1985514
    o[16] = (0.65837353 * h_0)+ (0.46144566 * h_1)+ (0.45746943 * h_2)+ (0.38091868 * h_3)+ (-0.6340189 * h_4)+ (0.09919503 * h_5)+ (-1.1290025 * h_6)+ (-2.7581575 * h_7)+ (-1.8842701 * h_8)+ (0.8629266 * h_9)+ (0.32305917 * h_10)+ (-0.051833168 * h_11)+ (1.4894074 * h_12)+ (0.8499113 * h_13)+ (0.6795888 * h_14) + 2.0137486
    o[17] = (1.3828609 * h_0)+ (-0.14031489 * h_1)+ (-0.16569439 * h_2)+ (-0.44091672 * h_3)+ (-1.1854075 * h_4)+ (-0.731188 * h_5)+ (-1.3317332 * h_6)+ (-2.7229776 * h_7)+ (-1.3461581 * h_8)+ (1.2023692 * h_9)+ (0.81052965 * h_10)+ (0.07375108 * h_11)+ (2.575715 * h_12)+ (0.9719315 * h_13)+ (0.34727803 * h_14) + 0.5100998

    

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
    w_h = np.array([[0.018664157018065453, -1.5544205904006958, -0.14637622237205505, 0.5762373208999634, 0.3365393877029419, 0.3919193744659424], [-0.002536391606554389, -0.021738681942224503, 0.003440062515437603, 1.08455228805542, -0.00445784255862236, -1.0666338205337524], [-1.005847692489624, -0.12834492325782776, -0.11251173168420792, -0.3792257308959961, 0.03714039921760559, 0.8014185428619385], [0.03449280560016632, 0.14370447397232056, 0.1949513554573059, 0.293802946805954, 0.3986876606941223, -0.5436694622039795], [-1.8899556398391724, 1.3380264043807983, 0.05839899182319641, 0.011034478433430195, 0.03027685172855854, 1.0879122018814087], [-0.009844771586358547, -0.043737925589084625, 0.6670607924461365, 0.014454743824899197, -0.6768211126327515, -0.024781759828329086], [-0.7726168036460876, 2.3607285022735596, 0.22781839966773987, 0.3882620334625244, -0.38202741742134094, 0.526639461517334], [0.05370383337140083, 0.2763947546482086, 0.06044905632734299, 0.4573672115802765, -0.018432097509503365, -1.4121382236480713], [-0.11078290641307831, -0.9475541114807129, -0.12738831341266632, 0.013451222330331802, 0.03985340893268585, -0.1897878348827362], [0.012855873443186283, -0.00846604909747839, 0.0061827367171645164, 1.2552459239959717, -0.004878020379692316, -1.2629355192184448], [1.6515345573425293, -0.27310502529144287, 0.3317323923110962, -0.09863682091236115, 0.05983313173055649, 0.22314822673797607], [-0.03863614797592163, 0.4237588047981262, 0.04894270747900009, 0.942554235458374, -0.14032697677612305, -0.30722683668136597], [-1.7908302545547485, 0.4614001214504242, -0.049587465822696686, -0.003685599658638239, 0.03718472644686699, 0.05760667473077774], [-0.17899839580059052, 0.07468704879283905, -0.036905497312545776, 0.006765602622181177, -0.7555331587791443, 0.22413328289985657], [0.7078187465667725, -0.9184041619300842, 0.0007880142075009644, -0.00545078469440341, -0.37307360768318176, 0.039863456040620804]])
    b_h = np.array([1.4524613618850708, 1.0138115882873535, 2.66216778755188, 2.363144636154175, 0.35702013969421387, -2.641631603240967, -0.2985572814941406, 3.541627883911133, 3.1230015754699707, -1.2829391956329346, 0.3580738306045532, 1.02384614944458, 3.088599920272827, 3.0938339233398438, 3.1036016941070557])
    w_o = np.array([[0.7623732686042786, -2.8390142917633057, -1.4146651029586792, 1.0551822185516357, -0.7191578149795532, -11.01739501953125, 0.20180144906044006, -0.5504254698753357, 0.56029212474823, -7.311575412750244, -0.5053052306175232, 1.6381757259368896, 0.7895894050598145, 0.7021690011024475, 0.12852783501148224], [0.40031811594963074, -0.3404727578163147, -1.2742787599563599, 0.01658213697373867, -0.42893797159194946, 0.3805445432662964, -0.026658115908503532, 0.2456597238779068, -0.5678435564041138, 0.31978940963745117, 0.40500345826148987, -1.0132044553756714, 0.9610769748687744, 1.0026291608810425, -9.664179801940918], [-0.5425577759742737, 0.8952494859695435, 0.43042272329330444, 0.0753912553191185, -0.6907066702842712, -1.5550318956375122, -0.06828038394451141, -0.35998713970184326, -0.17232702672481537, -0.3836909532546997, -0.21966543793678284, -0.05012285336852074, 0.42658373713493347, -0.614112377166748, -1.8769927024841309], [-0.6846424341201782, 0.32243111729621887, 1.1855652332305908, -0.16384266316890717, -0.8244730830192566, 1.3447086811065674, -0.31646329164505005, 0.33149856328964233, -0.08989132195711136, 0.11397109180688858, 0.16962337493896484, 0.07493098825216293, 0.30371928215026855, 0.6343446969985962, -11.848978996276855], [-0.37917274236679077, 0.3981393873691559, -0.39659383893013, 0.40915757417678833, -0.7620506882667542, -0.8421556353569031, -0.09290831536054611, 1.051557183265686, -1.3500880002975464, -1.3282734155654907, -1.0555460453033447, 0.11296288669109344, -0.40664613246917725, -0.05942347273230553, -0.22311030328273773], [0.12875139713287354, 0.23856937885284424, 1.2233513593673706, 0.8972706198692322, -2.4687185287475586, -0.4006924331188202, 0.5595154166221619, 0.6889991760253906, -0.568229615688324, -2.177335262298584, -0.9949600100517273, 0.8906038999557495, -3.016258955001831, 0.6052194833755493, -2.177854299545288], [-0.10758278518915176, 0.7045857310295105, 2.023042917251587, -0.002783578122034669, -0.49391868710517883, 0.11257810890674591, 0.17344053089618683, 1.1089569330215454, -0.11875490099191666, -1.351764440536499, -0.5297268033027649, 0.10024355351924896, -2.5049889087677, -0.1403282880783081, -1.226033091545105], [-0.23796318471431732, 0.3667389750480652, 1.7597968578338623, -0.16540059447288513, -0.41208788752555847, 0.3291175663471222, 0.5381253361701965, 1.0698323249816895, -0.16501368582248688, -1.036101222038269, -0.7986609935760498, 0.002914902288466692, -2.1753947734832764, -1.3116426467895508, 0.2824839949607849], [-0.16161666810512543, 0.43831875920295715, 1.2334727048873901, -0.5098443031311035, -0.04075053706765175, 0.4070812165737152, 0.36050063371658325, 0.8320766687393188, 0.6195727586746216, -0.9732985496520996, -0.3022037148475647, 0.14099086821079254, -1.4816795587539673, -0.9057636260986328, -0.5099257826805115], [-0.07931835204362869, -0.3061094880104065, 0.7806249856948853, -0.6487686634063721, 0.008889058604836464, 0.12766431272029877, 0.5396424531936646, 1.0383360385894775, 1.368344783782959, -0.3652653396129608, -0.23361995816230774, 0.20409423112869263, -1.1319489479064941, -0.9438722729682922, -0.40477678179740906], [-0.0979914665222168, -0.5349068641662598, -0.16639597713947296, -0.7066269516944885, 0.7051313519477844, 0.053110718727111816, 0.5700900554656982, 0.7205850481987, 2.258037567138672, 0.17426125705242157, -0.19110547006130219, -0.19159936904907227, -1.106564998626709, -1.1179674863815308, 0.1412290632724762], [0.20208635926246643, -0.4712776839733124, -0.5141181945800781, -0.28821250796318054, 0.5655615925788879, -0.32664844393730164, 0.41558587551116943, 0.06930514425039291, 1.3095364570617676, 0.5237975120544434, 0.08712268620729446, -0.539313554763794, -0.575933575630188, -0.3430137634277344, 0.04725199565291405], [0.06019636616110802, -0.27877548336982727, -0.6205621957778931, -0.5947584509849548, 0.5690503716468811, -0.2627297341823578, 0.19481191039085388, -0.5559008717536926, 1.5666704177856445, 0.7225489020347595, 0.6205692887306213, -0.348326176404953, 0.049268148839473724, -0.29772335290908813, 0.0006760149262845516], [-0.04237603768706322, -0.11885831505060196, -0.3041100800037384, -0.1370757818222046, 0.5403143763542175, -0.2733689844608307, -0.15490110218524933, -1.1144989728927612, 0.5470816493034363, 0.8389812707901001, 0.5620035529136658, -0.18386438488960266, 0.04500216618180275, 0.17768500745296478, -0.040949586778879166], [0.08332803100347519, 0.0460490807890892, 0.17698220908641815, 0.0782150849699974, 0.18469227850437164, -0.21098503470420837, -0.3690645694732666, -1.4722331762313843, -1.204061508178711, 0.7390615344047546, 0.3837251365184784, -0.06369822472333908, 0.39259204268455505, 0.12820608913898468, 0.6028962135314941], [0.47984957695007324, 0.3224928677082062, 0.2407529205083847, 0.817563533782959, 0.10690953582525253, -0.06896425783634186, -0.8680054545402527, -2.1277880668640137, -2.2331998348236084, 0.8905680179595947, -0.09536705166101456, -0.4131503999233246, 0.439344197511673, 0.766913890838623, 1.0759879350662231], [0.6583735346794128, 0.4614456593990326, 0.4574694335460663, 0.3809186816215515, -0.6340188980102539, 0.09919503331184387, -1.1290024518966675, -2.75815749168396, -1.8842700719833374, 0.8629266023635864, 0.32305917143821716, -0.05183316767215729, 1.4894074201583862, 0.8499112725257874, 0.679588794708252], [1.3828608989715576, -0.14031489193439484, -0.16569438576698303, -0.4409167170524597, -1.1854075193405151, -0.7311879992485046, -1.331733226776123, -2.722977638244629, -1.3461581468582153, 1.202369213104248, 0.8105296492576599, 0.07375107705593109, 2.5757150650024414, 0.971931517124176, 0.34727802872657776]])
    b_o = np.array([-0.7562647461891174, 0.8523717522621155, 0.3730098605155945, 1.2383123636245728, 0.9107965230941772, 1.0592801570892334, 0.968968391418457, 1.439659833908081, 0.7398134469985962, 0.1312168836593628, 0.4261072874069214, 0.6697127223014832, 0.45034259557724, 0.664496898651123, 1.2048465013504028, 2.1985514163970947, 2.0137486457824707, 0.5100998282432556])

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
        model_cap = 210
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

