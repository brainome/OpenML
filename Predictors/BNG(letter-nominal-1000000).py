#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1722/BayesianNetworkGenerator_letter_small.arff -o Predictors/BNG(letter-nominal-1000000)_NN.py -target class -stopat 59.79 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 2 days, 5:56:39.78. Finished on: Jun-12-2020 22:08:13.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        26-way classifier
Best-guess accuracy:                5.34%
Model accuracy:                     57.34% (573404/999999 correct)
Improvement over best guess:        52.00% (of possible 94.66%)
Model capacity (MEC):               2652 bits
Generalization ratio:               216.21 bits/bit
Confusion Matrix:
 [2.94% 0.03% 0.01% 0.04% 0.04% 0.10% 0.03% 0.04% 0.01% 0.14% 0.01% 0.12%
  0.01% 0.02% 0.05% 0.00% 0.01% 0.07% 0.11% 0.02% 0.04% 0.01% 0.03% 0.05%
  0.10% 0.01%]
 [0.02% 1.62% 0.01% 0.01% 0.09% 0.39% 0.07% 0.08% 0.01% 0.00% 0.13% 0.02%
  0.11% 0.01% 0.24% 0.01% 0.01% 0.21% 0.02% 0.10% 0.08% 0.04% 0.01% 0.04%
  0.43% 0.10%]
 [0.00% 0.00% 2.91% 0.01% 0.01% 0.04% 0.01% 0.03% 0.39% 0.01% 0.00% 0.01%
  0.01% 0.20% 0.00% 0.04% 0.01% 0.01% 0.01% 0.02% 0.03% 0.06% 0.07% 0.07%
  0.00% 0.02%]
 [0.08% 0.01% 0.02% 2.60% 0.02% 0.04% 0.07% 0.07% 0.04% 0.04% 0.05% 0.07%
  0.02% 0.07% 0.02% 0.02% 0.01% 0.06% 0.34% 0.02% 0.02% 0.03% 0.12% 0.02%
  0.08% 0.07%]
 [0.02% 0.15% 0.01% 0.01% 1.37% 0.48% 0.24% 0.09% 0.01% 0.01% 0.22% 0.03%
  0.04% 0.03% 0.07% 0.03% 0.04% 0.16% 0.03% 0.05% 0.05% 0.16% 0.03% 0.04%
  0.29% 0.08%]
 [0.01% 0.04% 0.06% 0.05% 0.14% 1.84% 0.18% 0.22% 0.04% 0.01% 0.06% 0.01%
  0.01% 0.05% 0.10% 0.01% 0.01% 0.18% 0.03% 0.03% 0.02% 0.46% 0.04% 0.06%
  0.15% 0.07%]
 [0.01% 0.02% 0.03% 0.10% 0.09% 0.19% 1.97% 0.10% 0.01% 0.00% 0.17% 0.01%
  0.03% 0.07% 0.06% 0.02% 0.03% 0.07% 0.05% 0.04% 0.13% 0.21% 0.19% 0.20%
  0.16% 0.06%]
 [0.01% 0.04% 0.04% 0.03% 0.05% 0.30% 0.06% 1.10% 0.06% 0.01% 0.02% 0.02%
  0.35% 0.05% 0.02% 0.05% 0.04% 0.11% 0.04% 0.05% 0.09% 0.10% 0.56% 0.15%
  0.23% 0.28%]
 [0.03% 0.00% 0.34% 0.01% 0.00% 0.04% 0.01% 0.05% 2.79% 0.07% 0.00% 0.05%
  0.01% 0.18% 0.00% 0.00% 0.01% 0.01% 0.02% 0.01% 0.03% 0.03% 0.02% 0.02%
  0.01% 0.02%]
 [0.08% 0.01% 0.03% 0.15% 0.02% 0.04% 0.01% 0.08% 0.09% 2.63% 0.01% 0.23%
  0.01% 0.04% 0.01% 0.01% 0.01% 0.05% 0.13% 0.01% 0.02% 0.02% 0.01% 0.01%
  0.08% 0.03%]
 [0.02% 0.07% 0.01% 0.07% 0.06% 0.08% 0.06% 0.03% 0.00% 0.01% 2.53% 0.01%
  0.02% 0.03% 0.12% 0.04% 0.09% 0.06% 0.14% 0.01% 0.03% 0.06% 0.01% 0.01%
  0.14% 0.05%]
 [0.32% 0.02% 0.02% 0.05% 0.03% 0.02% 0.01% 0.02% 0.02% 0.40% 0.01% 2.56%
  0.02% 0.03% 0.02% 0.03% 0.02% 0.05% 0.06% 0.01% 0.09% 0.01% 0.01% 0.01%
  0.01% 0.06%]
 [0.01% 0.12% 0.02% 0.01% 0.02% 0.03% 0.03% 0.14% 0.02% 0.01% 0.02% 0.02%
  2.70% 0.01% 0.02% 0.01% 0.01% 0.03% 0.04% 0.15% 0.12% 0.01% 0.03% 0.03%
  0.04% 0.05%]
 [0.01% 0.01% 0.38% 0.06% 0.04% 0.04% 0.12% 0.17% 0.37% 0.03% 0.03% 0.02%
  0.01% 1.65% 0.01% 0.02% 0.00% 0.09% 0.04% 0.03% 0.05% 0.17% 0.19% 0.20%
  0.11% 0.03%]
 [0.03% 0.10% 0.01% 0.01% 0.08% 0.09% 0.06% 0.01% 0.00% 0.01% 0.16% 0.01%
  0.01% 0.01% 2.42% 0.03% 0.02% 0.17% 0.03% 0.01% 0.02% 0.07% 0.01% 0.02%
  0.21% 0.05%]
 [0.01% 0.01% 0.08% 0.02% 0.05% 0.01% 0.02% 0.04% 0.02% 0.01% 0.12% 0.02%
  0.01% 0.05% 0.04% 3.00% 0.07% 0.02% 0.02% 0.01% 0.02% 0.15% 0.01% 0.02%
  0.05% 0.06%]
 [0.02% 0.04% 0.01% 0.01% 0.06% 0.04% 0.03% 0.04% 0.02% 0.01% 0.04% 0.03%
  0.04% 0.01% 0.03% 0.04% 3.06% 0.04% 0.01% 0.03% 0.04% 0.01% 0.01% 0.02%
  0.06% 0.04%]
 [0.02% 0.11% 0.01% 0.02% 0.25% 0.29% 0.22% 0.24% 0.01% 0.01% 0.08% 0.08%
  0.02% 0.10% 0.33% 0.01% 0.02% 0.97% 0.02% 0.10% 0.10% 0.05% 0.02% 0.11%
  0.73% 0.02%]
 [0.11% 0.05% 0.01% 0.29% 0.06% 0.08% 0.05% 0.10% 0.02% 0.04% 0.08% 0.07%
  0.02% 0.07% 0.04% 0.02% 0.01% 0.08% 2.35% 0.02% 0.04% 0.08% 0.01% 0.02%
  0.09% 0.04%]
 [0.01% 0.10% 0.04% 0.03% 0.05% 0.13% 0.21% 0.05% 0.02% 0.00% 0.02% 0.02%
  0.09% 0.04% 0.03% 0.01% 0.04% 0.19% 0.03% 1.69% 0.19% 0.24% 0.01% 0.26%
  0.20% 0.03%]
 [0.01% 0.04% 0.07% 0.02% 0.06% 0.05% 0.09% 0.17% 0.11% 0.02% 0.02% 0.04%
  0.10% 0.07% 0.02% 0.02% 0.04% 0.14% 0.04% 0.10% 2.20% 0.03% 0.18% 0.20%
  0.17% 0.10%]
 [0.02% 0.01% 0.08% 0.03% 0.08% 0.21% 0.28% 0.12% 0.03% 0.01% 0.06% 0.01%
  0.01% 0.06% 0.03% 0.01% 0.01% 0.04% 0.03% 0.10% 0.03% 2.14% 0.19% 0.08%
  0.05% 0.04%]
 [0.01% 0.01% 0.13% 0.02% 0.03% 0.04% 0.06% 0.14% 0.02% 0.00% 0.01% 0.01%
  0.03% 0.05% 0.02% 0.07% 0.02% 0.04% 0.01% 0.02% 0.03% 0.13% 2.61% 0.10%
  0.03% 0.13%]
 [0.03% 0.02% 0.14% 0.02% 0.05% 0.18% 0.20% 0.17% 0.06% 0.01% 0.02% 0.02%
  0.03% 0.15% 0.03% 0.02% 0.02% 0.14% 0.02% 0.18% 0.08% 0.18% 0.17% 1.52%
  0.18% 0.03%]
 [0.01% 0.14% 0.01% 0.07% 0.05% 0.06% 0.05% 0.26% 0.01% 0.01% 0.20% 0.01%
  0.02% 0.03% 0.08% 0.02% 0.03% 0.14% 0.09% 0.02% 0.02% 0.02% 0.00% 0.02%
  2.36% 0.03%]
 [0.02% 0.04% 0.03% 0.01% 0.09% 0.17% 0.03% 0.22% 0.03% 0.01% 0.03% 0.03%
  0.09% 0.02% 0.09% 0.14% 0.05% 0.05% 0.01% 0.01% 0.05% 0.02% 0.73% 0.03%
  0.07% 1.81%]

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
TRAINFILE = "BayesianNetworkGenerator_letter_small.csv"


#Number of output logits
num_output_logits = 26

#Number of attributes
num_attr = 16
n_classes = 26

mappings = [{1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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
    clean.mapping={'T': 0, 'E': 1, 'M': 2, 'P': 3, 'S': 4, 'B': 5, 'D': 6, 'G': 7, 'W': 8, 'V': 9, 'J': 10, 'Y': 11, 'C': 12, 'N': 13, 'Z': 14, 'A': 15, 'L': 16, 'X': 17, 'F': 18, 'K': 19, 'U': 20, 'R': 21, 'O': 22, 'H': 23, 'I': 24, 'Q': 25}

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
    h_0 = max((((0.03208806 * float(x[0]))+ (0.0007330638 * float(x[1]))+ (0.00352266 * float(x[2]))+ (0.001218545 * float(x[3]))+ (-0.01661984 * float(x[4]))+ (0.025218138 * float(x[5]))+ (-0.005449615 * float(x[6]))+ (0.24137174 * float(x[7]))+ (0.09471927 * float(x[8]))+ (-0.33803713 * float(x[9]))+ (-0.08226314 * float(x[10]))+ (-0.10104948 * float(x[11]))+ (0.0012438286 * float(x[12]))+ (0.009759361 * float(x[13]))+ (0.07173891 * float(x[14]))+ (-0.12474221 * float(x[15]))) + -0.24791302), 0)
    h_1 = max((((0.001096956 * float(x[0]))+ (-4.2670184e-05 * float(x[1]))+ (-0.0044434234 * float(x[2]))+ (-0.0008313134 * float(x[3]))+ (0.02482122 * float(x[4]))+ (0.06526141 * float(x[5]))+ (-0.00394873 * float(x[6]))+ (-0.11931686 * float(x[7]))+ (0.03782134 * float(x[8]))+ (-0.040199947 * float(x[9]))+ (0.037901342 * float(x[10]))+ (0.009230476 * float(x[11]))+ (0.002084834 * float(x[12]))+ (-0.020239728 * float(x[13]))+ (0.4467246 * float(x[14]))+ (-0.39926374 * float(x[15]))) + -0.3221333), 0)
    h_2 = max((((0.9986851 * float(x[0]))+ (0.17981647 * float(x[1]))+ (0.013567521 * float(x[2]))+ (0.26691267 * float(x[3]))+ (0.12729535 * float(x[4]))+ (-1.3973613 * float(x[5]))+ (-0.049596258 * float(x[6]))+ (-0.10973323 * float(x[7]))+ (-0.67081875 * float(x[8]))+ (-0.14780788 * float(x[9]))+ (-0.9013944 * float(x[10]))+ (-0.31863546 * float(x[11]))+ (0.015465661 * float(x[12]))+ (-0.26533955 * float(x[13]))+ (0.18508627 * float(x[14]))+ (-0.06814446 * float(x[15]))) + -1.5719594), 0)
    h_3 = max((((0.0033750725 * float(x[0]))+ (-0.0027597677 * float(x[1]))+ (0.013645684 * float(x[2]))+ (-0.0065507405 * float(x[3]))+ (0.0395041 * float(x[4]))+ (-0.10538855 * float(x[5]))+ (0.0123029575 * float(x[6]))+ (0.16915305 * float(x[7]))+ (0.24208802 * float(x[8]))+ (-0.15307462 * float(x[9]))+ (0.008534191 * float(x[10]))+ (-0.0014522186 * float(x[11]))+ (0.024316948 * float(x[12]))+ (0.0077714976 * float(x[13]))+ (0.3424396 * float(x[14]))+ (-0.004284144 * float(x[15]))) + -1.1651717), 0)
    h_4 = max((((0.017895235 * float(x[0]))+ (0.0029651287 * float(x[1]))+ (-0.0029207284 * float(x[2]))+ (-0.0027525378 * float(x[3]))+ (0.016795361 * float(x[4]))+ (0.06272453 * float(x[5]))+ (-0.010562462 * float(x[6]))+ (0.56924486 * float(x[7]))+ (0.061412904 * float(x[8]))+ (-0.45723632 * float(x[9]))+ (-0.28356352 * float(x[10]))+ (-0.48928618 * float(x[11]))+ (-0.008668089 * float(x[12]))+ (0.029549725 * float(x[13]))+ (0.028971003 * float(x[14]))+ (-0.09485308 * float(x[15]))) + -0.51848656), 0)
    h_5 = max((((-0.10314116 * float(x[0]))+ (-0.0037375812 * float(x[1]))+ (-0.0094062295 * float(x[2]))+ (-0.0098572895 * float(x[3]))+ (0.05210848 * float(x[4]))+ (-0.08872864 * float(x[5]))+ (-0.0015165607 * float(x[6]))+ (0.19999951 * float(x[7]))+ (0.20989642 * float(x[8]))+ (-0.12529273 * float(x[9]))+ (-0.008737991 * float(x[10]))+ (0.0038383899 * float(x[11]))+ (0.028331216 * float(x[12]))+ (0.0002791229 * float(x[13]))+ (0.32664958 * float(x[14]))+ (-0.010580861 * float(x[15]))) + -1.0803586), 0)
    h_6 = max((((0.32582825 * float(x[0]))+ (-0.13445155 * float(x[1]))+ (-0.018065814 * float(x[2]))+ (-0.44033092 * float(x[3]))+ (0.19591787 * float(x[4]))+ (-0.79496396 * float(x[5]))+ (0.7194142 * float(x[6]))+ (0.18135758 * float(x[7]))+ (-0.34056652 * float(x[8]))+ (-1.2314904 * float(x[9]))+ (0.15732832 * float(x[10]))+ (-0.49400017 * float(x[11]))+ (0.31208995 * float(x[12]))+ (-0.27341065 * float(x[13]))+ (-0.02032241 * float(x[14]))+ (-1.476556 * float(x[15]))) + -1.3592397), 0)
    h_7 = max((((-0.025786621 * float(x[0]))+ (0.018084323 * float(x[1]))+ (0.043911908 * float(x[2]))+ (-0.028901588 * float(x[3]))+ (-0.2250977 * float(x[4]))+ (-0.05718475 * float(x[5]))+ (0.015556897 * float(x[6]))+ (-0.084072776 * float(x[7]))+ (-0.029389495 * float(x[8]))+ (0.032016322 * float(x[9]))+ (-0.03324769 * float(x[10]))+ (0.0077875964 * float(x[11]))+ (-0.14122549 * float(x[12]))+ (0.0062746815 * float(x[13]))+ (-0.060878973 * float(x[14]))+ (0.003156704 * float(x[15]))) + 0.95786256), 0)
    h_8 = max((((0.0054561943 * float(x[0]))+ (0.037251 * float(x[1]))+ (0.3149797 * float(x[2]))+ (-0.038468488 * float(x[3]))+ (-0.041925136 * float(x[4]))+ (0.27104795 * float(x[5]))+ (0.12975453 * float(x[6]))+ (-1.6321911 * float(x[7]))+ (0.37595013 * float(x[8]))+ (0.022756059 * float(x[9]))+ (-0.024197057 * float(x[10]))+ (-0.40024534 * float(x[11]))+ (0.07873211 * float(x[12]))+ (-0.08016149 * float(x[13]))+ (0.1728093 * float(x[14]))+ (-0.3317825 * float(x[15]))) + 0.2341956), 0)
    h_9 = max((((0.57565945 * float(x[0]))+ (-1.5987949 * float(x[1]))+ (-2.0460558 * float(x[2]))+ (1.7181907 * float(x[3]))+ (1.22895 * float(x[4]))+ (-2.3738642 * float(x[5]))+ (-5.162594 * float(x[6]))+ (0.85475725 * float(x[7]))+ (-3.5754752 * float(x[8]))+ (-4.9918447 * float(x[9]))+ (1.4683683 * float(x[10]))+ (-1.555289 * float(x[11]))+ (-0.4920956 * float(x[12]))+ (-2.872229 * float(x[13]))+ (-0.13596365 * float(x[14]))+ (-0.9316797 * float(x[15]))) + -0.16634357), 0)
    h_10 = max((((-0.10711475 * float(x[0]))+ (-0.0067365714 * float(x[1]))+ (-0.0046871235 * float(x[2]))+ (-0.001945117 * float(x[3]))+ (0.13730635 * float(x[4]))+ (0.0054345853 * float(x[5]))+ (0.0076632667 * float(x[6]))+ (0.039945763 * float(x[7]))+ (0.013918754 * float(x[8]))+ (-0.07198263 * float(x[9]))+ (0.008673919 * float(x[10]))+ (-0.008081138 * float(x[11]))+ (0.061810248 * float(x[12]))+ (-0.028454922 * float(x[13]))+ (0.3123949 * float(x[14]))+ (-0.0035130666 * float(x[15]))) + -0.69675034), 0)
    h_11 = max((((0.016554894 * float(x[0]))+ (0.0014513297 * float(x[1]))+ (-0.0017914217 * float(x[2]))+ (-0.0008813263 * float(x[3]))+ (0.015644055 * float(x[4]))+ (0.18585265 * float(x[5]))+ (0.15629673 * float(x[6]))+ (0.19865367 * float(x[7]))+ (-0.020461159 * float(x[8]))+ (-0.021831792 * float(x[9]))+ (-0.6054315 * float(x[10]))+ (0.31606275 * float(x[11]))+ (-0.0017559137 * float(x[12]))+ (0.027175093 * float(x[13]))+ (0.049305018 * float(x[14]))+ (0.06535793 * float(x[15]))) + -1.2422888), 0)
    h_12 = max((((-0.05699622 * float(x[0]))+ (-0.014131262 * float(x[1]))+ (-0.014042738 * float(x[2]))+ (-0.05061099 * float(x[3]))+ (0.026084904 * float(x[4]))+ (0.19602151 * float(x[5]))+ (0.09815344 * float(x[6]))+ (-0.5234299 * float(x[7]))+ (0.11393816 * float(x[8]))+ (-0.20421249 * float(x[9]))+ (0.17844918 * float(x[10]))+ (0.004228187 * float(x[11]))+ (-0.011852603 * float(x[12]))+ (-0.16885065 * float(x[13]))+ (1.7225891 * float(x[14]))+ (-2.087177 * float(x[15]))) + -0.98756903), 0)
    h_13 = max((((1.1236025 * float(x[0]))+ (-1.9826014 * float(x[1]))+ (0.13389532 * float(x[2]))+ (-1.4198003 * float(x[3]))+ (0.7948133 * float(x[4]))+ (-0.2260094 * float(x[5]))+ (1.1815735 * float(x[6]))+ (-0.3043388 * float(x[7]))+ (0.6680688 * float(x[8]))+ (-0.784386 * float(x[9]))+ (-0.34594223 * float(x[10]))+ (-0.72820896 * float(x[11]))+ (-0.42844802 * float(x[12]))+ (-1.4637041 * float(x[13]))+ (-0.3307236 * float(x[14]))+ (-1.2247962 * float(x[15]))) + -1.6236224), 0)
    h_14 = max((((0.03812653 * float(x[0]))+ (0.005834429 * float(x[1]))+ (-0.0054655694 * float(x[2]))+ (0.023252629 * float(x[3]))+ (0.078380175 * float(x[4]))+ (0.0251883 * float(x[5]))+ (-0.01243104 * float(x[6]))+ (0.055579625 * float(x[7]))+ (-1.3154217 * float(x[8]))+ (-0.036063954 * float(x[9]))+ (0.008132148 * float(x[10]))+ (0.033026416 * float(x[11]))+ (-0.0121371895 * float(x[12]))+ (0.024095831 * float(x[13]))+ (0.05291167 * float(x[14]))+ (-0.003549327 * float(x[15]))) + -0.20883329), 0)
    h_15 = max((((-0.01904029 * float(x[0]))+ (-0.00075954816 * float(x[1]))+ (-0.014132951 * float(x[2]))+ (-0.013589161 * float(x[3]))+ (0.035593342 * float(x[4]))+ (0.11187178 * float(x[5]))+ (-0.008587156 * float(x[6]))+ (-0.012819292 * float(x[7]))+ (-0.007198024 * float(x[8]))+ (0.044882994 * float(x[9]))+ (-0.0038936487 * float(x[10]))+ (-0.2823748 * float(x[11]))+ (0.00013014616 * float(x[12]))+ (0.004121577 * float(x[13]))+ (-0.0018248453 * float(x[14]))+ (-0.10924368 * float(x[15]))) + 0.29590952), 0)
    h_16 = max((((0.002234824 * float(x[0]))+ (0.004302683 * float(x[1]))+ (-0.032200623 * float(x[2]))+ (0.0011114813 * float(x[3]))+ (-0.03927764 * float(x[4]))+ (-0.063419096 * float(x[5]))+ (0.33867368 * float(x[6]))+ (0.131603 * float(x[7]))+ (0.23067586 * float(x[8]))+ (-0.07130137 * float(x[9]))+ (0.24774344 * float(x[10]))+ (0.035795905 * float(x[11]))+ (-0.0035215379 * float(x[12]))+ (0.0073236353 * float(x[13]))+ (-0.012037236 * float(x[14]))+ (-0.013666372 * float(x[15]))) + -1.4029604), 0)
    h_17 = max((((0.046358738 * float(x[0]))+ (-0.0064600105 * float(x[1]))+ (0.00071073684 * float(x[2]))+ (-0.07568358 * float(x[3]))+ (0.1568708 * float(x[4]))+ (-0.31026787 * float(x[5]))+ (0.0011003292 * float(x[6]))+ (-0.19600146 * float(x[7]))+ (0.38194466 * float(x[8]))+ (-0.0007818078 * float(x[9]))+ (0.02770257 * float(x[10]))+ (0.027340865 * float(x[11]))+ (0.00015845412 * float(x[12]))+ (0.028320111 * float(x[13]))+ (-0.025617823 * float(x[14]))+ (0.0074053057 * float(x[15]))) + -0.24074428), 0)
    h_18 = max((((0.5103723 * float(x[0]))+ (0.0034608094 * float(x[1]))+ (0.49196622 * float(x[2]))+ (-0.03680718 * float(x[3]))+ (0.45505765 * float(x[4]))+ (-0.37241605 * float(x[5]))+ (0.01514512 * float(x[6]))+ (-0.11020195 * float(x[7]))+ (0.01291649 * float(x[8]))+ (-0.023861509 * float(x[9]))+ (0.0034850973 * float(x[10]))+ (-0.028393522 * float(x[11]))+ (0.015399035 * float(x[12]))+ (-0.029744018 * float(x[13]))+ (3.7377948e-05 * float(x[14]))+ (-0.07690106 * float(x[15]))) + -2.2818956), 0)
    h_19 = max((((0.04447743 * float(x[0]))+ (-0.0039144894 * float(x[1]))+ (0.0053096996 * float(x[2]))+ (-0.015597409 * float(x[3]))+ (0.14876266 * float(x[4]))+ (0.14612386 * float(x[5]))+ (0.44049 * float(x[6]))+ (0.054344647 * float(x[7]))+ (0.039219126 * float(x[8]))+ (-0.03846666 * float(x[9]))+ (0.08266534 * float(x[10]))+ (-0.0010627218 * float(x[11]))+ (0.036029115 * float(x[12]))+ (0.062581226 * float(x[13]))+ (0.029689794 * float(x[14]))+ (-0.008609401 * float(x[15]))) + -1.5142306), 0)
    h_20 = max((((0.022572765 * float(x[0]))+ (0.005280738 * float(x[1]))+ (-0.0025471745 * float(x[2]))+ (-0.011840373 * float(x[3]))+ (0.018318769 * float(x[4]))+ (-0.0045215935 * float(x[5]))+ (0.005441153 * float(x[6]))+ (0.07022341 * float(x[7]))+ (0.017692648 * float(x[8]))+ (-0.47197172 * float(x[9]))+ (0.0001461176 * float(x[10]))+ (-0.004328141 * float(x[11]))+ (-0.009701408 * float(x[12]))+ (-0.0064797006 * float(x[13]))+ (0.03988647 * float(x[14]))+ (-0.49006575 * float(x[15]))) + -0.049272224), 0)
    h_21 = max((((0.056200873 * float(x[0]))+ (-0.40864322 * float(x[1]))+ (-0.08075022 * float(x[2]))+ (-0.5012057 * float(x[3]))+ (0.43316844 * float(x[4]))+ (-0.38339686 * float(x[5]))+ (0.30840394 * float(x[6]))+ (0.31769988 * float(x[7]))+ (0.11659582 * float(x[8]))+ (-0.9090367 * float(x[9]))+ (0.15490143 * float(x[10]))+ (-0.5024395 * float(x[11]))+ (0.075401075 * float(x[12]))+ (-0.049406495 * float(x[13]))+ (-0.2793815 * float(x[14]))+ (-0.7507791 * float(x[15]))) + -1.0614606), 0)
    h_22 = max((((-0.076418675 * float(x[0]))+ (-0.032656997 * float(x[1]))+ (0.5685874 * float(x[2]))+ (-0.031102534 * float(x[3]))+ (0.04481503 * float(x[4]))+ (-0.020168956 * float(x[5]))+ (-0.019999987 * float(x[6]))+ (0.0359845 * float(x[7]))+ (0.01599539 * float(x[8]))+ (0.011691427 * float(x[9]))+ (0.009189793 * float(x[10]))+ (0.0092098415 * float(x[11]))+ (0.5326909 * float(x[12]))+ (-0.011843509 * float(x[13]))+ (-0.003522913 * float(x[14]))+ (0.006834072 * float(x[15]))) + -2.1339207), 0)
    h_23 = max((((-0.012240457 * float(x[0]))+ (0.027437264 * float(x[1]))+ (-0.028184032 * float(x[2]))+ (-0.015173861 * float(x[3]))+ (0.031552006 * float(x[4]))+ (0.00591629 * float(x[5]))+ (-0.048618767 * float(x[6]))+ (0.12782045 * float(x[7]))+ (0.008171523 * float(x[8]))+ (-0.64795685 * float(x[9]))+ (-0.004375734 * float(x[10]))+ (-0.6127349 * float(x[11]))+ (-0.014994208 * float(x[12]))+ (0.018152222 * float(x[13]))+ (0.0023229315 * float(x[14]))+ (-0.6263616 * float(x[15]))) + 0.048709027), 0)
    h_24 = max((((-0.37395513 * float(x[0]))+ (-0.4755893 * float(x[1]))+ (-0.17367202 * float(x[2]))+ (-0.019452536 * float(x[3]))+ (-0.023021292 * float(x[4]))+ (-0.010987747 * float(x[5]))+ (-0.02767141 * float(x[6]))+ (0.008609994 * float(x[7]))+ (0.006575885 * float(x[8]))+ (-0.027410114 * float(x[9]))+ (-0.0007182316 * float(x[10]))+ (-0.02469363 * float(x[11]))+ (0.039045185 * float(x[12]))+ (-0.049388457 * float(x[13]))+ (-0.00372413 * float(x[14]))+ (-0.009249664 * float(x[15]))) + 0.68183845), 0)
    h_25 = max((((-0.015479093 * float(x[0]))+ (-0.022964887 * float(x[1]))+ (0.0030937295 * float(x[2]))+ (-1.226379 * float(x[3]))+ (0.0058021997 * float(x[4]))+ (0.043954417 * float(x[5]))+ (-0.0045021786 * float(x[6]))+ (0.05925246 * float(x[7]))+ (0.4661135 * float(x[8]))+ (-0.00790319 * float(x[9]))+ (0.0016144442 * float(x[10]))+ (-0.0073303147 * float(x[11]))+ (0.018688483 * float(x[12]))+ (-0.018369637 * float(x[13]))+ (0.022649202 * float(x[14]))+ (-0.005143181 * float(x[15]))) + -0.89134926), 0)
    h_26 = max((((-0.023716727 * float(x[0]))+ (0.0021141176 * float(x[1]))+ (0.0102662165 * float(x[2]))+ (-0.023909599 * float(x[3]))+ (-0.06786622 * float(x[4]))+ (-0.103580505 * float(x[5]))+ (-0.0066364845 * float(x[6]))+ (0.34562615 * float(x[7]))+ (0.49294508 * float(x[8]))+ (0.0026830777 * float(x[9]))+ (-0.0055297064 * float(x[10]))+ (0.0025760026 * float(x[11]))+ (-0.02192392 * float(x[12]))+ (-0.01894905 * float(x[13]))+ (0.40826502 * float(x[14]))+ (-0.006228509 * float(x[15]))) + -1.670618), 0)
    h_27 = max((((-0.024006113 * float(x[0]))+ (-0.0048950133 * float(x[1]))+ (0.073716514 * float(x[2]))+ (0.01036344 * float(x[3]))+ (0.21677865 * float(x[4]))+ (-0.24684374 * float(x[5]))+ (0.18376583 * float(x[6]))+ (0.031266317 * float(x[7]))+ (0.0034964413 * float(x[8]))+ (0.0096743675 * float(x[9]))+ (0.0010547177 * float(x[10]))+ (-0.0024307992 * float(x[11]))+ (-0.030006232 * float(x[12]))+ (0.007489552 * float(x[13]))+ (0.00550537 * float(x[14]))+ (-0.0060890866 * float(x[15]))) + -0.6664517), 0)
    h_28 = max((((-0.023550218 * float(x[0]))+ (-0.00072861224 * float(x[1]))+ (-0.020129267 * float(x[2]))+ (0.007301169 * float(x[3]))+ (-0.019046286 * float(x[4]))+ (0.15305303 * float(x[5]))+ (0.06921419 * float(x[6]))+ (-0.011029409 * float(x[7]))+ (0.10966214 * float(x[8]))+ (0.34471965 * float(x[9]))+ (-0.038591474 * float(x[10]))+ (0.09323553 * float(x[11]))+ (-0.011959202 * float(x[12]))+ (-0.0081384005 * float(x[13]))+ (0.035786733 * float(x[14]))+ (0.06409871 * float(x[15]))) + -1.0569516), 0)
    h_29 = max((((-0.124594495 * float(x[0]))+ (-0.017254453 * float(x[1]))+ (0.025814107 * float(x[2]))+ (0.005332424 * float(x[3]))+ (0.17182118 * float(x[4]))+ (0.007551041 * float(x[5]))+ (0.016227538 * float(x[6]))+ (0.0037294675 * float(x[7]))+ (0.0004455223 * float(x[8]))+ (-0.0066471323 * float(x[9]))+ (-0.004584739 * float(x[10]))+ (-0.0124719655 * float(x[11]))+ (-0.23652002 * float(x[12]))+ (0.06314551 * float(x[13]))+ (-0.0081281755 * float(x[14]))+ (0.0027906643 * float(x[15]))) + 0.1455378), 0)
    h_30 = max((((0.0045719245 * float(x[0]))+ (0.0040361118 * float(x[1]))+ (0.0009401535 * float(x[2]))+ (-0.041702736 * float(x[3]))+ (0.06428679 * float(x[4]))+ (0.025152927 * float(x[5]))+ (-0.0009905983 * float(x[6]))+ (0.031042531 * float(x[7]))+ (0.16746424 * float(x[8]))+ (0.0062188664 * float(x[9]))+ (0.10231875 * float(x[10]))+ (0.08346291 * float(x[11]))+ (0.011050752 * float(x[12]))+ (0.0076813516 * float(x[13]))+ (0.013260899 * float(x[14]))+ (-0.019863224 * float(x[15]))) + -0.4746683), 0)
    h_31 = max((((0.057054617 * float(x[0]))+ (0.0017244844 * float(x[1]))+ (-0.007873919 * float(x[2]))+ (-0.0014001768 * float(x[3]))+ (0.040286757 * float(x[4]))+ (-0.049649257 * float(x[5]))+ (-0.1265557 * float(x[6]))+ (-0.007741437 * float(x[7]))+ (0.009451998 * float(x[8]))+ (0.059212778 * float(x[9]))+ (0.00653245 * float(x[10]))+ (-0.017294874 * float(x[11]))+ (0.24810758 * float(x[12]))+ (-0.48219886 * float(x[13]))+ (0.02648804 * float(x[14]))+ (-0.007663363 * float(x[15]))) + -0.07101058), 0)
    h_32 = max((((-0.03695744 * float(x[0]))+ (-0.1287766 * float(x[1]))+ (-0.038557548 * float(x[2]))+ (-0.71181995 * float(x[3]))+ (0.27971104 * float(x[4]))+ (0.4060573 * float(x[5]))+ (-1.4850055 * float(x[6]))+ (1.319819 * float(x[7]))+ (0.021016829 * float(x[8]))+ (-2.4248228 * float(x[9]))+ (-0.860455 * float(x[10]))+ (-0.73989296 * float(x[11]))+ (-0.2717749 * float(x[12]))+ (-0.57746303 * float(x[13]))+ (-1.0626075 * float(x[14]))+ (0.028272208 * float(x[15]))) + -0.4141287), 0)
    h_33 = max((((0.118900545 * float(x[0]))+ (-0.13926049 * float(x[1]))+ (-0.018608939 * float(x[2]))+ (-0.26715282 * float(x[3]))+ (0.08707709 * float(x[4]))+ (-0.45319524 * float(x[5]))+ (0.37627342 * float(x[6]))+ (0.1774371 * float(x[7]))+ (-0.25136676 * float(x[8]))+ (-1.3931106 * float(x[9]))+ (0.16012333 * float(x[10]))+ (-0.3657891 * float(x[11]))+ (0.39416185 * float(x[12]))+ (-0.17799819 * float(x[13]))+ (0.09902628 * float(x[14]))+ (-0.9614018 * float(x[15]))) + -0.9912624), 0)
    h_34 = max((((0.0005709422 * float(x[0]))+ (0.002194964 * float(x[1]))+ (-0.0073786923 * float(x[2]))+ (0.00025066687 * float(x[3]))+ (0.0070334473 * float(x[4]))+ (-0.037260287 * float(x[5]))+ (-0.014585 * float(x[6]))+ (-0.08086692 * float(x[7]))+ (-0.013529591 * float(x[8]))+ (0.22790594 * float(x[9]))+ (0.1498491 * float(x[10]))+ (-0.2526799 * float(x[11]))+ (0.001865881 * float(x[12]))+ (0.0067215743 * float(x[13]))+ (0.066087015 * float(x[14]))+ (0.11941556 * float(x[15]))) + -0.30920893), 0)
    h_35 = max((((-0.0104574375 * float(x[0]))+ (0.00028414468 * float(x[1]))+ (0.019448685 * float(x[2]))+ (-0.010125909 * float(x[3]))+ (-0.010767407 * float(x[4]))+ (0.019224662 * float(x[5]))+ (0.055943716 * float(x[6]))+ (-0.0964397 * float(x[7]))+ (0.09451067 * float(x[8]))+ (-0.09750718 * float(x[9]))+ (0.05989436 * float(x[10]))+ (-0.07655338 * float(x[11]))+ (0.0020441788 * float(x[12]))+ (-0.0005566199 * float(x[13]))+ (0.0010019088 * float(x[14]))+ (-0.011833619 * float(x[15]))) + -0.02270733), 0)
    h_36 = max((((0.027845759 * float(x[0]))+ (0.00053366966 * float(x[1]))+ (-0.017335938 * float(x[2]))+ (0.0016948112 * float(x[3]))+ (0.043889295 * float(x[4]))+ (-0.009004647 * float(x[5]))+ (0.07470332 * float(x[6]))+ (0.12394156 * float(x[7]))+ (0.39035952 * float(x[8]))+ (-0.5430764 * float(x[9]))+ (0.25478867 * float(x[10]))+ (-0.04670117 * float(x[11]))+ (0.0005334792 * float(x[12]))+ (-0.00037920338 * float(x[13]))+ (0.016016675 * float(x[14]))+ (-0.00015073219 * float(x[15]))) + -1.3590816), 0)
    h_37 = max((((-0.13184772 * float(x[0]))+ (-0.006872754 * float(x[1]))+ (0.00017176462 * float(x[2]))+ (-0.0020637566 * float(x[3]))+ (0.38843784 * float(x[4]))+ (-0.10525158 * float(x[5]))+ (0.39149576 * float(x[6]))+ (0.053084433 * float(x[7]))+ (0.0085233 * float(x[8]))+ (-0.027591925 * float(x[9]))+ (-0.002876069 * float(x[10]))+ (0.0042869765 * float(x[11]))+ (0.015290593 * float(x[12]))+ (-0.057823148 * float(x[13]))+ (0.0042174165 * float(x[14]))+ (0.008180384 * float(x[15]))) + -1.2076089), 0)
    h_38 = max((((1.2168845 * float(x[0]))+ (-1.685461 * float(x[1]))+ (-0.42858154 * float(x[2]))+ (-2.7299693 * float(x[3]))+ (0.42778933 * float(x[4]))+ (-0.7806877 * float(x[5]))+ (1.2639425 * float(x[6]))+ (-0.27903703 * float(x[7]))+ (1.5201857 * float(x[8]))+ (-2.0321915 * float(x[9]))+ (-0.8548626 * float(x[10]))+ (-0.7258723 * float(x[11]))+ (-0.25778925 * float(x[12]))+ (-0.72952616 * float(x[13]))+ (-1.3625197 * float(x[14]))+ (-2.640115 * float(x[15]))) + -1.0519685), 0)
    h_39 = max((((-0.03578712 * float(x[0]))+ (0.0030916305 * float(x[1]))+ (0.02432042 * float(x[2]))+ (-0.007022465 * float(x[3]))+ (0.11916441 * float(x[4]))+ (0.008476381 * float(x[5]))+ (0.015807018 * float(x[6]))+ (-0.10582012 * float(x[7]))+ (-0.20463617 * float(x[8]))+ (0.0002149915 * float(x[9]))+ (0.0061766626 * float(x[10]))+ (-0.01647866 * float(x[11]))+ (0.007717246 * float(x[12]))+ (0.028683508 * float(x[13]))+ (0.3841173 * float(x[14]))+ (-0.04999125 * float(x[15]))) + -0.5680969), 0)
    h_40 = max((((-0.009138712 * float(x[0]))+ (0.0014206036 * float(x[1]))+ (0.0018825167 * float(x[2]))+ (0.010309449 * float(x[3]))+ (-0.0017863457 * float(x[4]))+ (0.08175639 * float(x[5]))+ (-0.02861385 * float(x[6]))+ (-0.06644828 * float(x[7]))+ (0.3677691 * float(x[8]))+ (-0.12088753 * float(x[9]))+ (0.3697467 * float(x[10]))+ (0.0724112 * float(x[11]))+ (-0.0012710424 * float(x[12]))+ (-0.00020826467 * float(x[13]))+ (-0.0042976667 * float(x[14]))+ (0.034989472 * float(x[15]))) + -1.1000348), 0)
    h_41 = max((((0.075046904 * float(x[0]))+ (0.003201502 * float(x[1]))+ (0.034269966 * float(x[2]))+ (-0.013680132 * float(x[3]))+ (0.20921935 * float(x[4]))+ (0.03411596 * float(x[5]))+ (0.0648998 * float(x[6]))+ (0.15983665 * float(x[7]))+ (0.07453223 * float(x[8]))+ (-0.03520735 * float(x[9]))+ (0.0064259665 * float(x[10]))+ (0.01018331 * float(x[11]))+ (0.012479456 * float(x[12]))+ (0.0013116391 * float(x[13]))+ (-0.03556236 * float(x[14]))+ (-0.010806411 * float(x[15]))) + -1.0358703), 0)
    h_42 = max((((0.03216456 * float(x[0]))+ (-0.57806647 * float(x[1]))+ (-0.15856601 * float(x[2]))+ (-0.2555761 * float(x[3]))+ (0.0023742735 * float(x[4]))+ (-0.41865572 * float(x[5]))+ (0.33401343 * float(x[6]))+ (-0.36671042 * float(x[7]))+ (0.24208018 * float(x[8]))+ (0.09885185 * float(x[9]))+ (0.013285229 * float(x[10]))+ (-0.56252074 * float(x[11]))+ (0.045050375 * float(x[12]))+ (-0.34056142 * float(x[13]))+ (-0.39467666 * float(x[14]))+ (-0.5646484 * float(x[15]))) + -0.4866836), 0)
    h_43 = max((((-0.0006315825 * float(x[0]))+ (-0.00034823714 * float(x[1]))+ (-0.034957536 * float(x[2]))+ (0.015545545 * float(x[3]))+ (-0.020052789 * float(x[4]))+ (0.07819489 * float(x[5]))+ (0.014000858 * float(x[6]))+ (-0.060241036 * float(x[7]))+ (-0.004414494 * float(x[8]))+ (0.23884681 * float(x[9]))+ (-0.07374232 * float(x[10]))+ (-0.27815267 * float(x[11]))+ (0.0069089183 * float(x[12]))+ (-0.011360755 * float(x[13]))+ (0.008475729 * float(x[14]))+ (-0.00941927 * float(x[15]))) + -0.123126805), 0)
    h_44 = max((((-0.013985541 * float(x[0]))+ (0.015383309 * float(x[1]))+ (-0.011783724 * float(x[2]))+ (-0.011931393 * float(x[3]))+ (0.035766147 * float(x[4]))+ (0.18444791 * float(x[5]))+ (0.084594026 * float(x[6]))+ (0.054061636 * float(x[7]))+ (0.0759356 * float(x[8]))+ (0.12511574 * float(x[9]))+ (-0.08727156 * float(x[10]))+ (-0.04328153 * float(x[11]))+ (-0.01411368 * float(x[12]))+ (0.001965514 * float(x[13]))+ (0.027843578 * float(x[14]))+ (0.015591206 * float(x[15]))) + -0.41213867), 0)
    h_45 = max((((0.23024094 * float(x[0]))+ (-0.19965163 * float(x[1]))+ (0.040941473 * float(x[2]))+ (-0.5047523 * float(x[3]))+ (0.3157115 * float(x[4]))+ (-1.3981453 * float(x[5]))+ (-3.12331 * float(x[6]))+ (-0.69749314 * float(x[7]))+ (1.1473191 * float(x[8]))+ (-3.0075448 * float(x[9]))+ (0.19063176 * float(x[10]))+ (-1.0003272 * float(x[11]))+ (0.9768888 * float(x[12]))+ (0.29257232 * float(x[13]))+ (-1.3459072 * float(x[14]))+ (-1.7114975 * float(x[15]))) + -1.6588801), 0)
    h_46 = max((((-0.00147695 * float(x[0]))+ (-0.010183811 * float(x[1]))+ (-0.0074554747 * float(x[2]))+ (-0.018830314 * float(x[3]))+ (0.059304263 * float(x[4]))+ (-0.010649631 * float(x[5]))+ (0.061225872 * float(x[6]))+ (-0.08131706 * float(x[7]))+ (0.22049534 * float(x[8]))+ (-0.40319148 * float(x[9]))+ (0.060809795 * float(x[10]))+ (-0.06930513 * float(x[11]))+ (-0.01442126 * float(x[12]))+ (0.08324933 * float(x[13]))+ (0.04769355 * float(x[14]))+ (-0.09844031 * float(x[15]))) + 0.15041465), 0)
    h_47 = max((((-0.007058528 * float(x[0]))+ (0.000589869 * float(x[1]))+ (0.008103455 * float(x[2]))+ (-0.0072419285 * float(x[3]))+ (-0.0011304352 * float(x[4]))+ (0.09810718 * float(x[5]))+ (-0.09766911 * float(x[6]))+ (0.17542912 * float(x[7]))+ (-0.029938119 * float(x[8]))+ (-0.0057677045 * float(x[9]))+ (0.19482684 * float(x[10]))+ (-0.119271316 * float(x[11]))+ (0.0011867192 * float(x[12]))+ (0.005139199 * float(x[13]))+ (-0.00671303 * float(x[14]))+ (0.009382326 * float(x[15]))) + -0.3867864), 0)
    h_48 = max((((-0.076407075 * float(x[0]))+ (0.00011446821 * float(x[1]))+ (9.962465e-05 * float(x[2]))+ (-0.09720396 * float(x[3]))+ (-0.0027216736 * float(x[4]))+ (-0.47545913 * float(x[5]))+ (-0.10466952 * float(x[6]))+ (-0.2284147 * float(x[7]))+ (0.20225139 * float(x[8]))+ (0.18206616 * float(x[9]))+ (0.002245081 * float(x[10]))+ (-0.050152864 * float(x[11]))+ (0.03143664 * float(x[12]))+ (-0.02855054 * float(x[13]))+ (0.085307755 * float(x[14]))+ (0.08162236 * float(x[15]))) + 0.23893577), 0)
    h_49 = max((((-0.0733294 * float(x[0]))+ (-0.009859599 * float(x[1]))+ (0.0054325587 * float(x[2]))+ (-0.04850911 * float(x[3]))+ (-0.0146924155 * float(x[4]))+ (0.07549279 * float(x[5]))+ (0.13802971 * float(x[6]))+ (-0.034941383 * float(x[7]))+ (0.26930287 * float(x[8]))+ (-0.786386 * float(x[9]))+ (0.115329005 * float(x[10]))+ (-0.3271739 * float(x[11]))+ (-0.011316623 * float(x[12]))+ (0.090858504 * float(x[13]))+ (-0.013239294 * float(x[14]))+ (-0.1724237 * float(x[15]))) + 0.40236717), 0)
    h_50 = max((((0.050202254 * float(x[0]))+ (-0.0029094457 * float(x[1]))+ (-0.036547337 * float(x[2]))+ (-0.06146796 * float(x[3]))+ (0.04674239 * float(x[4]))+ (-0.16815217 * float(x[5]))+ (0.0698947 * float(x[6]))+ (0.030845167 * float(x[7]))+ (0.27878442 * float(x[8]))+ (-0.0492036 * float(x[9]))+ (0.04991542 * float(x[10]))+ (0.01853959 * float(x[11]))+ (-0.010954354 * float(x[12]))+ (-0.044085555 * float(x[13]))+ (0.005141039 * float(x[14]))+ (-0.04104953 * float(x[15]))) + -0.4032734), 0)
    h_51 = max((((0.15037507 * float(x[0]))+ (-0.39415917 * float(x[1]))+ (-0.009324071 * float(x[2]))+ (-0.15144709 * float(x[3]))+ (0.017881425 * float(x[4]))+ (-0.20136224 * float(x[5]))+ (-0.11435324 * float(x[6]))+ (-0.10561389 * float(x[7]))+ (-0.1551405 * float(x[8]))+ (-0.25710446 * float(x[9]))+ (-0.16217324 * float(x[10]))+ (-0.27191472 * float(x[11]))+ (0.2658004 * float(x[12]))+ (-0.29256782 * float(x[13]))+ (-0.1938994 * float(x[14]))+ (-0.23777457 * float(x[15]))) + -0.28603035), 0)
    h_52 = max((((0.040152565 * float(x[0]))+ (0.005463421 * float(x[1]))+ (-0.016688624 * float(x[2]))+ (0.043596916 * float(x[3]))+ (0.17343035 * float(x[4]))+ (0.0140313655 * float(x[5]))+ (-0.021643896 * float(x[6]))+ (-0.023841608 * float(x[7]))+ (-0.003307418 * float(x[8]))+ (-0.0026438117 * float(x[9]))+ (-0.009051268 * float(x[10]))+ (-0.016838748 * float(x[11]))+ (-0.9613836 * float(x[12]))+ (0.022532633 * float(x[13]))+ (0.006469413 * float(x[14]))+ (0.004437337 * float(x[15]))) + 0.16071796), 0)
    h_53 = max((((0.8348907 * float(x[0]))+ (-0.3898466 * float(x[1]))+ (0.2365755 * float(x[2]))+ (-0.72570425 * float(x[3]))+ (-0.39312023 * float(x[4]))+ (-0.3334505 * float(x[5]))+ (0.43410516 * float(x[6]))+ (0.00727004 * float(x[7]))+ (-0.21870905 * float(x[8]))+ (-1.0259688 * float(x[9]))+ (0.12196478 * float(x[10]))+ (-0.28300574 * float(x[11]))+ (0.9523741 * float(x[12]))+ (-1.177539 * float(x[13]))+ (-0.10517692 * float(x[14]))+ (-0.2834239 * float(x[15]))) + -1.7111464), 0)
    h_54 = max((((6.714593e-05 * float(x[0]))+ (-0.0041292314 * float(x[1]))+ (-0.029317735 * float(x[2]))+ (0.0006435165 * float(x[3]))+ (0.032595884 * float(x[4]))+ (-0.034051932 * float(x[5]))+ (-0.030170621 * float(x[6]))+ (-0.027695384 * float(x[7]))+ (0.027304444 * float(x[8]))+ (0.19579622 * float(x[9]))+ (0.07760673 * float(x[10]))+ (-0.10358664 * float(x[11]))+ (0.0025949236 * float(x[12]))+ (0.006939926 * float(x[13]))+ (0.097471245 * float(x[14]))+ (0.20819363 * float(x[15]))) + -0.4270586), 0)
    h_55 = max((((0.020821288 * float(x[0]))+ (-0.38891864 * float(x[1]))+ (-0.11014416 * float(x[2]))+ (-0.2012317 * float(x[3]))+ (2.3339462e-06 * float(x[4]))+ (-0.4373659 * float(x[5]))+ (0.27881804 * float(x[6]))+ (-0.052991144 * float(x[7]))+ (-0.047048774 * float(x[8]))+ (-0.40998882 * float(x[9]))+ (0.09070544 * float(x[10]))+ (-0.40366593 * float(x[11]))+ (0.091298394 * float(x[12]))+ (-0.28148493 * float(x[13]))+ (-0.032143105 * float(x[14]))+ (-0.4492773 * float(x[15]))) + -0.31905696), 0)
    h_56 = max((((-0.004855707 * float(x[0]))+ (0.0027881146 * float(x[1]))+ (0.041607194 * float(x[2]))+ (0.0021665639 * float(x[3]))+ (0.135294 * float(x[4]))+ (0.019449675 * float(x[5]))+ (0.009208399 * float(x[6]))+ (-0.080773845 * float(x[7]))+ (-0.14304978 * float(x[8]))+ (0.00877393 * float(x[9]))+ (-0.0014296278 * float(x[10]))+ (-0.0037246 * float(x[11]))+ (0.003655342 * float(x[12]))+ (0.007398446 * float(x[13]))+ (0.37386698 * float(x[14]))+ (-0.03937285 * float(x[15]))) + -0.6731065), 0)
    h_57 = max((((-0.016375143 * float(x[0]))+ (0.0025681143 * float(x[1]))+ (0.0015879276 * float(x[2]))+ (-0.013124501 * float(x[3]))+ (0.017443955 * float(x[4]))+ (0.07807065 * float(x[5]))+ (-0.006616635 * float(x[6]))+ (0.12900557 * float(x[7]))+ (-0.02056516 * float(x[8]))+ (0.2110693 * float(x[9]))+ (0.083514236 * float(x[10]))+ (-0.08781353 * float(x[11]))+ (0.0050065517 * float(x[12]))+ (-0.016183665 * float(x[13]))+ (0.014897664 * float(x[14]))+ (-0.039343894 * float(x[15]))) + -0.6462901), 0)
    h_58 = max((((0.024314063 * float(x[0]))+ (-0.010407279 * float(x[1]))+ (-0.0038693899 * float(x[2]))+ (0.028979087 * float(x[3]))+ (0.017672965 * float(x[4]))+ (0.09630162 * float(x[5]))+ (-0.030504322 * float(x[6]))+ (-0.11267092 * float(x[7]))+ (0.9057353 * float(x[8]))+ (-0.30648416 * float(x[9]))+ (0.7630372 * float(x[10]))+ (0.087452166 * float(x[11]))+ (-0.019186832 * float(x[12]))+ (0.011925237 * float(x[13]))+ (0.007989113 * float(x[14]))+ (0.067728624 * float(x[15]))) + -2.328599), 0)
    h_59 = max((((0.12233683 * float(x[0]))+ (-0.01963287 * float(x[1]))+ (0.012828412 * float(x[2]))+ (0.022547266 * float(x[3]))+ (0.24589165 * float(x[4]))+ (-0.010575174 * float(x[5]))+ (-0.0029954978 * float(x[6]))+ (0.11497107 * float(x[7]))+ (-0.1443887 * float(x[8]))+ (-0.061802916 * float(x[9]))+ (0.03566259 * float(x[10]))+ (-0.0011216601 * float(x[11]))+ (-0.053221352 * float(x[12]))+ (-0.016893817 * float(x[13]))+ (-1.2618986 * float(x[14]))+ (0.008551499 * float(x[15]))) + -0.09821573), 0)
    h_60 = max((((1.3013675 * float(x[0]))+ (-4.817006 * float(x[1]))+ (-1.1061187 * float(x[2]))+ (-1.2097453 * float(x[3]))+ (-1.4268256 * float(x[4]))+ (-0.30999345 * float(x[5]))+ (-1.8152577 * float(x[6]))+ (-2.699166 * float(x[7]))+ (2.0243518 * float(x[8]))+ (-1.042558 * float(x[9]))+ (0.43989763 * float(x[10]))+ (-1.5270615 * float(x[11]))+ (0.99700963 * float(x[12]))+ (-0.9002964 * float(x[13]))+ (-1.9067087 * float(x[14]))+ (-3.5978022 * float(x[15]))) + -0.34626973), 0)
    h_61 = max((((0.13549449 * float(x[0]))+ (0.0011204189 * float(x[1]))+ (0.022958806 * float(x[2]))+ (-0.009078963 * float(x[3]))+ (0.42384136 * float(x[4]))+ (0.11951225 * float(x[5]))+ (-0.30131197 * float(x[6]))+ (-0.02179198 * float(x[7]))+ (0.06384845 * float(x[8]))+ (0.047733035 * float(x[9]))+ (0.019194873 * float(x[10]))+ (-0.011238966 * float(x[11]))+ (0.03297246 * float(x[12]))+ (-0.0025741244 * float(x[13]))+ (0.02479294 * float(x[14]))+ (-0.047748245 * float(x[15]))) + -1.1798713), 0)
    h_62 = max((((0.028150989 * float(x[0]))+ (-0.02024315 * float(x[1]))+ (0.0133613385 * float(x[2]))+ (-0.004100667 * float(x[3]))+ (0.2532496 * float(x[4]))+ (-1.7888217 * float(x[5]))+ (0.06947634 * float(x[6]))+ (0.13448033 * float(x[7]))+ (-0.027942872 * float(x[8]))+ (-0.022591665 * float(x[9]))+ (-0.00051472336 * float(x[10]))+ (-0.037924703 * float(x[11]))+ (-0.0024876602 * float(x[12]))+ (0.009874438 * float(x[13]))+ (0.012672903 * float(x[14]))+ (-0.027939849 * float(x[15]))) + -0.3876835), 0)
    h_63 = max((((0.076057814 * float(x[0]))+ (0.017924342 * float(x[1]))+ (-0.06855637 * float(x[2]))+ (0.03290112 * float(x[3]))+ (-0.03959135 * float(x[4]))+ (0.8132063 * float(x[5]))+ (0.119875535 * float(x[6]))+ (0.060322333 * float(x[7]))+ (0.08828753 * float(x[8]))+ (0.65237975 * float(x[9]))+ (-0.57411593 * float(x[10]))+ (-1.3915708 * float(x[11]))+ (0.015115481 * float(x[12]))+ (-0.013332383 * float(x[13]))+ (0.038411897 * float(x[14]))+ (-0.12026071 * float(x[15]))) + -0.501578), 0)
    h_64 = max((((0.182352 * float(x[0]))+ (0.0062959394 * float(x[1]))+ (0.002855354 * float(x[2]))+ (0.00095243123 * float(x[3]))+ (0.19027998 * float(x[4]))+ (0.03685983 * float(x[5]))+ (0.048101734 * float(x[6]))+ (0.018716017 * float(x[7]))+ (0.00839441 * float(x[8]))+ (0.008679332 * float(x[9]))+ (0.01381739 * float(x[10]))+ (-0.0047597247 * float(x[11]))+ (-0.012328834 * float(x[12]))+ (-0.08704711 * float(x[13]))+ (0.0119204875 * float(x[14]))+ (-0.008529388 * float(x[15]))) + -0.81490165), 0)
    h_65 = max((((0.010628513 * float(x[0]))+ (-0.002700176 * float(x[1]))+ (-0.020302137 * float(x[2]))+ (0.037718125 * float(x[3]))+ (0.01187211 * float(x[4]))+ (0.022727372 * float(x[5]))+ (-0.036009252 * float(x[6]))+ (-0.07711413 * float(x[7]))+ (-0.033892367 * float(x[8]))+ (-0.00890962 * float(x[9]))+ (-0.0033813568 * float(x[10]))+ (-0.01068536 * float(x[11]))+ (0.009305559 * float(x[12]))+ (-0.0004895087 * float(x[13]))+ (-0.00049839047 * float(x[14]))+ (-0.003368492 * float(x[15]))) + 0.045013588), 0)
    h_66 = max((((-0.0233292 * float(x[0]))+ (0.009129533 * float(x[1]))+ (-0.061484776 * float(x[2]))+ (-0.03607568 * float(x[3]))+ (0.0014639427 * float(x[4]))+ (0.2507192 * float(x[5]))+ (0.0063390355 * float(x[6]))+ (0.15883887 * float(x[7]))+ (0.060452644 * float(x[8]))+ (-0.03498064 * float(x[9]))+ (0.0019917446 * float(x[10]))+ (-0.06785136 * float(x[11]))+ (0.0077566737 * float(x[12]))+ (0.0044097113 * float(x[13]))+ (-0.01834221 * float(x[14]))+ (0.005659793 * float(x[15]))) + -0.40798542), 0)
    h_67 = max((((0.026164247 * float(x[0]))+ (-0.022045702 * float(x[1]))+ (-0.03496102 * float(x[2]))+ (-0.0021838848 * float(x[3]))+ (0.10273955 * float(x[4]))+ (0.29971942 * float(x[5]))+ (0.046410117 * float(x[6]))+ (0.3520247 * float(x[7]))+ (0.15928707 * float(x[8]))+ (-0.06594786 * float(x[9]))+ (-0.91028696 * float(x[10]))+ (-0.18277492 * float(x[11]))+ (-0.0382377 * float(x[12]))+ (0.02982549 * float(x[13]))+ (0.06030141 * float(x[14]))+ (-0.030101761 * float(x[15]))) + -0.599354), 0)
    h_68 = max((((0.0069697183 * float(x[0]))+ (0.008172414 * float(x[1]))+ (0.009468687 * float(x[2]))+ (-0.001499253 * float(x[3]))+ (0.124821976 * float(x[4]))+ (0.028755743 * float(x[5]))+ (0.004956278 * float(x[6]))+ (0.05888123 * float(x[7]))+ (-0.10688903 * float(x[8]))+ (-0.05670542 * float(x[9]))+ (0.012674108 * float(x[10]))+ (-0.008360507 * float(x[11]))+ (-0.0022764606 * float(x[12]))+ (0.0058226692 * float(x[13]))+ (-0.13270384 * float(x[14]))+ (-0.00070903474 * float(x[15]))) + -0.032195173), 0)
    h_69 = max((((0.011831078 * float(x[0]))+ (-0.013552402 * float(x[1]))+ (0.032584306 * float(x[2]))+ (0.0071845166 * float(x[3]))+ (0.060833666 * float(x[4]))+ (0.022426596 * float(x[5]))+ (-0.0069197817 * float(x[6]))+ (-0.034537878 * float(x[7]))+ (0.046620138 * float(x[8]))+ (-0.0028867873 * float(x[9]))+ (0.015800493 * float(x[10]))+ (-0.0111062955 * float(x[11]))+ (-0.0181149 * float(x[12]))+ (-0.030917566 * float(x[13]))+ (0.0054821074 * float(x[14]))+ (0.010181797 * float(x[15]))) + -0.04774874), 0)
    h_70 = max((((0.012298336 * float(x[0]))+ (0.0114795985 * float(x[1]))+ (0.015484483 * float(x[2]))+ (-0.16062292 * float(x[3]))+ (0.232652 * float(x[4]))+ (-0.4017595 * float(x[5]))+ (-0.008887565 * float(x[6]))+ (-0.2855538 * float(x[7]))+ (0.5563935 * float(x[8]))+ (0.047137376 * float(x[9]))+ (0.029685458 * float(x[10]))+ (-0.035765313 * float(x[11]))+ (0.026225606 * float(x[12]))+ (-0.03216859 * float(x[13]))+ (0.044290468 * float(x[14]))+ (0.00956522 * float(x[15]))) + -0.1227263), 0)
    h_71 = max((((-0.016322246 * float(x[0]))+ (0.0116935335 * float(x[1]))+ (-0.03710247 * float(x[2]))+ (0.06312863 * float(x[3]))+ (0.29569042 * float(x[4]))+ (0.01582504 * float(x[5]))+ (-0.0044477205 * float(x[6]))+ (-0.029183248 * float(x[7]))+ (0.0453205 * float(x[8]))+ (-0.0011052588 * float(x[9]))+ (-0.003820082 * float(x[10]))+ (-0.020520382 * float(x[11]))+ (-0.3925321 * float(x[12]))+ (0.012771702 * float(x[13]))+ (0.0071000094 * float(x[14]))+ (0.017327748 * float(x[15]))) + 0.18753356), 0)
    h_72 = max((((-0.01646618 * float(x[0]))+ (-0.0056372075 * float(x[1]))+ (0.2589778 * float(x[2]))+ (-0.12837961 * float(x[3]))+ (0.1628224 * float(x[4]))+ (0.40244487 * float(x[5]))+ (-0.24208963 * float(x[6]))+ (0.922463 * float(x[7]))+ (0.08203027 * float(x[8]))+ (-0.25287077 * float(x[9]))+ (-0.11969713 * float(x[10]))+ (0.08595094 * float(x[11]))+ (0.019058254 * float(x[12]))+ (-0.07027003 * float(x[13]))+ (0.04668903 * float(x[14]))+ (0.016612304 * float(x[15]))) + -2.1224985), 0)
    h_73 = max((((0.15677327 * float(x[0]))+ (-0.07597259 * float(x[1]))+ (0.20902231 * float(x[2]))+ (-0.14379804 * float(x[3]))+ (0.100676 * float(x[4]))+ (0.95160633 * float(x[5]))+ (-0.38701406 * float(x[6]))+ (1.3039378 * float(x[7]))+ (0.058248945 * float(x[8]))+ (0.11316226 * float(x[9]))+ (0.91958123 * float(x[10]))+ (-0.8866733 * float(x[11]))+ (0.018627537 * float(x[12]))+ (0.027088782 * float(x[13]))+ (0.072931446 * float(x[14]))+ (-0.1407296 * float(x[15]))) + -3.1070433), 0)
    h_74 = max((((0.010782227 * float(x[0]))+ (0.04187179 * float(x[1]))+ (0.0059220265 * float(x[2]))+ (0.041651927 * float(x[3]))+ (0.011889523 * float(x[4]))+ (0.1780558 * float(x[5]))+ (0.009064431 * float(x[6]))+ (-0.11169037 * float(x[7]))+ (0.011470057 * float(x[8]))+ (0.2617449 * float(x[9]))+ (0.15474343 * float(x[10]))+ (-0.42995209 * float(x[11]))+ (-0.011414481 * float(x[12]))+ (0.04491139 * float(x[13]))+ (0.21645428 * float(x[14]))+ (-0.30227283 * float(x[15]))) + 0.13944761), 0)
    h_75 = max((((0.1003717 * float(x[0]))+ (-0.0047167456 * float(x[1]))+ (0.123564325 * float(x[2]))+ (-0.030971706 * float(x[3]))+ (0.27041784 * float(x[4]))+ (-0.07911708 * float(x[5]))+ (0.028778395 * float(x[6]))+ (0.20345974 * float(x[7]))+ (-0.053614028 * float(x[8]))+ (-0.0076143593 * float(x[9]))+ (0.03967137 * float(x[10]))+ (-0.06918758 * float(x[11]))+ (0.0035127376 * float(x[12]))+ (-0.034517884 * float(x[13]))+ (-0.21190782 * float(x[14]))+ (-0.012721767 * float(x[15]))) + -0.32547668), 0)
    h_76 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))) + 0.0), 0)
    h_77 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))) + 0.0), 0)
    o[0] = (-3.3857844 * h_0)+ (1.6670532 * h_1)+ (-66.70655 * h_2)+ (23.640896 * h_3)+ (10.694703 * h_4)+ (13.849205 * h_5)+ (-32.479763 * h_6)+ (1.6979362 * h_7)+ (3.3877451 * h_8)+ (-54.51956 * h_9)+ (15.0305605 * h_10)+ (7.8121877 * h_11)+ (4.1263146 * h_12)+ (-26.018032 * h_13)+ (12.775791 * h_14)+ (9.189016 * h_15)+ (9.834666 * h_16)+ (5.1039095 * h_17)+ (13.85121 * h_18)+ (10.409742 * h_19)+ (31.802303 * h_20)+ (-36.43945 * h_21)+ (14.803088 * h_22)+ (17.22103 * h_23)+ (12.95558 * h_24)+ (24.97996 * h_25)+ (5.8724394 * h_26)+ (28.91715 * h_27)+ (28.73117 * h_28)+ (13.852306 * h_29)+ (-3.269037 * h_30)+ (12.223116 * h_31)+ (-26.641338 * h_32)+ (15.868569 * h_33)+ (17.106016 * h_34)+ (11.118667 * h_35)+ (22.14617 * h_36)+ (-4.844286 * h_37)+ (-30.790268 * h_38)+ (27.568932 * h_39)+ (16.288645 * h_40)+ (-4.035344 * h_41)+ (-13.12741 * h_42)+ (16.033043 * h_43)+ (4.3364058 * h_44)+ (-68.760605 * h_45)+ (-0.25336254 * h_46)+ (11.158744 * h_47)+ (6.673077 * h_48)+ (6.9465213 * h_49)+ (12.049123 * h_50)+ (-10.258276 * h_51)+ (19.137177 * h_52)+ (6.8017282 * h_53)+ (3.5952575 * h_54)+ (-12.481865 * h_55)+ (1.4441173 * h_56)+ (20.86039 * h_57)+ (12.461857 * h_58)+ (8.773164 * h_59)+ (-37.22669 * h_60)+ (18.586985 * h_61)+ (13.830215 * h_62)+ (3.8187587 * h_63)+ (15.584364 * h_64)+ (1.9012941 * h_65)+ (12.0181265 * h_66)+ (6.5538673 * h_67)+ (0.12813348 * h_68)+ (5.513799 * h_69)+ (3.6354704 * h_70)+ (4.2556844 * h_71)+ (10.645859 * h_72)+ (0.6607999 * h_73)+ (1.2131858 * h_74)+ (4.1667223 * h_75)+ (-13.0 * h_76)+ (0.0 * h_77) + 5.967509
    o[1] = (30.849216 * h_0)+ (-5.629671 * h_1)+ (18.854893 * h_2)+ (15.602013 * h_3)+ (1.7356695 * h_4)+ (12.556167 * h_5)+ (16.906345 * h_6)+ (4.9460316 * h_7)+ (2.3606472 * h_8)+ (18.327389 * h_9)+ (14.996312 * h_10)+ (20.447727 * h_11)+ (6.1583405 * h_12)+ (-64.88078 * h_13)+ (15.600199 * h_14)+ (1.6628667 * h_15)+ (13.476903 * h_16)+ (6.00832 * h_17)+ (10.844827 * h_18)+ (2.0443385 * h_19)+ (12.579536 * h_20)+ (-10.459713 * h_21)+ (30.76871 * h_22)+ (9.698319 * h_23)+ (4.9296436 * h_24)+ (15.571552 * h_25)+ (-2.5777342 * h_26)+ (9.62747 * h_27)+ (20.494364 * h_28)+ (16.93477 * h_29)+ (1.89641 * h_30)+ (6.9520884 * h_31)+ (22.875177 * h_32)+ (13.16471 * h_33)+ (21.546854 * h_34)+ (13.054583 * h_35)+ (24.056477 * h_36)+ (4.389917 * h_37)+ (10.855435 * h_38)+ (27.38467 * h_39)+ (26.689758 * h_40)+ (12.279336 * h_41)+ (14.099153 * h_42)+ (7.2022243 * h_43)+ (10.55438 * h_44)+ (-50.253094 * h_45)+ (0.78795147 * h_46)+ (4.242657 * h_47)+ (7.1272492 * h_48)+ (3.9269183 * h_49)+ (21.939436 * h_50)+ (11.15882 * h_51)+ (17.40637 * h_52)+ (9.034894 * h_53)+ (-1.5489404 * h_54)+ (14.161869 * h_55)+ (0.5601303 * h_56)+ (1.4530803 * h_57)+ (5.778481 * h_58)+ (3.1277912 * h_59)+ (15.66804 * h_60)+ (1.8093047 * h_61)+ (17.568382 * h_62)+ (6.244483 * h_63)+ (28.053137 * h_64)+ (15.5799465 * h_65)+ (11.50228 * h_66)+ (4.481326 * h_67)+ (15.669687 * h_68)+ (24.753881 * h_69)+ (5.134022 * h_70)+ (1.2951043 * h_71)+ (3.6319754 * h_72)+ (3.432455 * h_73)+ (2.9121253 * h_74)+ (3.6733308 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 3.4328485
    o[2] = (7.8725114 * h_0)+ (15.279812 * h_1)+ (13.910998 * h_2)+ (13.12444 * h_3)+ (18.192352 * h_4)+ (19.549185 * h_5)+ (8.721635 * h_6)+ (15.902435 * h_7)+ (4.342549 * h_8)+ (18.420446 * h_9)+ (-0.5313327 * h_10)+ (23.49417 * h_11)+ (0.8006689 * h_12)+ (11.327874 * h_13)+ (29.766827 * h_14)+ (9.524606 * h_15)+ (19.740896 * h_16)+ (14.324648 * h_17)+ (13.366125 * h_18)+ (7.731494 * h_19)+ (22.936773 * h_20)+ (17.873865 * h_21)+ (-4.0421767 * h_22)+ (9.2598715 * h_23)+ (24.590178 * h_24)+ (3.8697832 * h_25)+ (14.534098 * h_26)+ (6.9754653 * h_27)+ (13.161534 * h_28)+ (3.794145 * h_29)+ (18.795486 * h_30)+ (12.816234 * h_31)+ (20.204199 * h_32)+ (21.066587 * h_33)+ (9.737033 * h_34)+ (13.780411 * h_35)+ (23.934214 * h_36)+ (5.103565 * h_37)+ (15.272922 * h_38)+ (13.261212 * h_39)+ (-2.8646157 * h_40)+ (19.104748 * h_41)+ (11.765532 * h_42)+ (19.327915 * h_43)+ (9.6192665 * h_44)+ (13.932171 * h_45)+ (8.118841 * h_46)+ (15.249919 * h_47)+ (5.446141 * h_48)+ (0.8885101 * h_49)+ (-0.46002862 * h_50)+ (15.849216 * h_51)+ (-0.88617545 * h_52)+ (3.3319924 * h_53)+ (14.094884 * h_54)+ (5.3460894 * h_55)+ (16.3769 * h_56)+ (17.970848 * h_57)+ (17.172441 * h_58)+ (7.2881913 * h_59)+ (-10.132892 * h_60)+ (8.729535 * h_61)+ (25.741537 * h_62)+ (3.1356487 * h_63)+ (-3.2487235 * h_64)+ (-0.207587 * h_65)+ (24.968939 * h_66)+ (7.7056355 * h_67)+ (13.733105 * h_68)+ (-3.0489686 * h_69)+ (0.014489152 * h_70)+ (25.230047 * h_71)+ (1.2135878 * h_72)+ (0.65856934 * h_73)+ (2.8455229 * h_74)+ (4.820348 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -4.1943374
    o[3] = (8.835768 * h_0)+ (20.287159 * h_1)+ (12.836228 * h_2)+ (14.888142 * h_3)+ (17.569569 * h_4)+ (12.205701 * h_5)+ (-5.2842107 * h_6)+ (9.859244 * h_7)+ (3.139791 * h_8)+ (18.888853 * h_9)+ (9.627562 * h_10)+ (16.643227 * h_11)+ (0.06975779 * h_12)+ (-6.008081 * h_13)+ (28.668077 * h_14)+ (16.250414 * h_15)+ (6.267247 * h_16)+ (5.840308 * h_17)+ (15.068847 * h_18)+ (8.002918 * h_19)+ (14.818714 * h_20)+ (-11.3143425 * h_21)+ (12.724468 * h_22)+ (14.908283 * h_23)+ (22.818577 * h_24)+ (24.331003 * h_25)+ (5.949748 * h_26)+ (15.355208 * h_27)+ (8.904095 * h_28)+ (11.28052 * h_29)+ (19.764952 * h_30)+ (9.365788 * h_31)+ (18.772621 * h_32)+ (33.35921 * h_33)+ (18.556107 * h_34)+ (7.285591 * h_35)+ (33.600372 * h_36)+ (0.7728908 * h_37)+ (19.270763 * h_38)+ (21.21956 * h_39)+ (19.475637 * h_40)+ (-1.616848 * h_41)+ (14.164682 * h_42)+ (14.413771 * h_43)+ (21.2613 * h_44)+ (21.334476 * h_45)+ (4.1488357 * h_46)+ (11.416028 * h_47)+ (8.098849 * h_48)+ (4.41663 * h_49)+ (11.169074 * h_50)+ (12.162909 * h_51)+ (13.896313 * h_52)+ (8.969817 * h_53)+ (6.9768767 * h_54)+ (14.092534 * h_55)+ (2.0845315 * h_56)+ (26.18477 * h_57)+ (9.515519 * h_58)+ (8.77634 * h_59)+ (-52.26367 * h_60)+ (14.613903 * h_61)+ (24.337715 * h_62)+ (4.0610895 * h_63)+ (13.818094 * h_64)+ (8.747799 * h_65)+ (-3.1259983 * h_66)+ (4.2477245 * h_67)+ (9.494358 * h_68)+ (17.176027 * h_69)+ (2.0051367 * h_70)+ (4.8451085 * h_71)+ (8.860669 * h_72)+ (0.57530946 * h_73)+ (-2.9017918 * h_74)+ (1.609985 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 0.8015677
    o[4] = (30.904081 * h_0)+ (-2.0848067 * h_1)+ (16.019077 * h_2)+ (5.128392 * h_3)+ (4.824321 * h_4)+ (21.490532 * h_5)+ (3.774908 * h_6)+ (8.832664 * h_7)+ (3.6171026 * h_8)+ (20.058863 * h_9)+ (22.026382 * h_10)+ (21.634378 * h_11)+ (5.6169386 * h_12)+ (9.232651 * h_13)+ (24.611137 * h_14)+ (2.1435764 * h_15)+ (19.204111 * h_16)+ (7.9040527 * h_17)+ (2.5989296 * h_18)+ (2.5448225 * h_19)+ (3.2883198 * h_20)+ (22.337397 * h_21)+ (13.068413 * h_22)+ (13.313779 * h_23)+ (13.880528 * h_24)+ (15.184951 * h_25)+ (8.111266 * h_26)+ (15.862621 * h_27)+ (15.352009 * h_28)+ (24.136625 * h_29)+ (8.067544 * h_30)+ (7.513375 * h_31)+ (20.685247 * h_32)+ (24.99067 * h_33)+ (6.892582 * h_34)+ (8.4817915 * h_35)+ (13.805558 * h_36)+ (14.866277 * h_37)+ (13.330908 * h_38)+ (20.12309 * h_39)+ (11.5413265 * h_40)+ (6.295949 * h_41)+ (14.676779 * h_42)+ (24.004173 * h_43)+ (11.481412 * h_44)+ (22.645313 * h_45)+ (6.4785504 * h_46)+ (12.564141 * h_47)+ (9.184148 * h_48)+ (3.3242455 * h_49)+ (9.956676 * h_50)+ (11.588092 * h_51)+ (25.833258 * h_52)+ (9.099845 * h_53)+ (11.817939 * h_54)+ (14.661815 * h_55)+ (1.3281932 * h_56)+ (16.131397 * h_57)+ (13.063833 * h_58)+ (5.728459 * h_59)+ (-57.44388 * h_60)+ (-1.4410893 * h_61)+ (11.7431135 * h_62)+ (3.2621768 * h_63)+ (9.892559 * h_64)+ (28.012817 * h_65)+ (22.611887 * h_66)+ (5.019576 * h_67)+ (-10.342744 * h_68)+ (24.678986 * h_69)+ (0.9501489 * h_70)+ (-9.167653 * h_71)+ (0.86915374 * h_72)+ (0.40545654 * h_73)+ (4.415265 * h_74)+ (15.967607 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 0.23531844
    o[5] = (11.982144 * h_0)+ (3.646425 * h_1)+ (16.693583 * h_2)+ (13.314359 * h_3)+ (14.7304125 * h_4)+ (8.215763 * h_5)+ (1.1975775 * h_6)+ (7.3486767 * h_7)+ (2.1018767 * h_8)+ (-30.679596 * h_9)+ (10.125356 * h_10)+ (21.151777 * h_11)+ (3.9090447 * h_12)+ (15.777777 * h_13)+ (17.223577 * h_14)+ (-4.858177 * h_15)+ (14.294834 * h_16)+ (5.831868 * h_17)+ (17.626759 * h_18)+ (1.005305 * h_19)+ (25.150011 * h_20)+ (16.817415 * h_21)+ (-4.4406013 * h_22)+ (11.347497 * h_23)+ (17.418827 * h_24)+ (18.381435 * h_25)+ (11.998873 * h_26)+ (-15.285044 * h_27)+ (22.73196 * h_28)+ (26.775816 * h_29)+ (9.749499 * h_30)+ (4.328678 * h_31)+ (21.924154 * h_32)+ (31.526527 * h_33)+ (20.131771 * h_34)+ (23.082748 * h_35)+ (20.678785 * h_36)+ (16.450638 * h_37)+ (-37.72043 * h_38)+ (21.205362 * h_39)+ (8.588549 * h_40)+ (21.120348 * h_41)+ (14.776132 * h_42)+ (24.662403 * h_43)+ (-0.9353337 * h_44)+ (25.031593 * h_45)+ (-2.1064801 * h_46)+ (14.616273 * h_47)+ (8.191538 * h_48)+ (5.8652773 * h_49)+ (9.880597 * h_50)+ (12.664233 * h_51)+ (2.509833 * h_52)+ (6.6825175 * h_53)+ (-3.44211 * h_54)+ (15.053873 * h_55)+ (0.0078065945 * h_56)+ (10.89836 * h_57)+ (13.304769 * h_58)+ (3.053628 * h_59)+ (10.238356 * h_60)+ (4.4784694 * h_61)+ (24.451988 * h_62)+ (5.618859 * h_63)+ (15.799563 * h_64)+ (11.157806 * h_65)+ (17.873503 * h_66)+ (7.1056156 * h_67)+ (-0.20413403 * h_68)+ (26.126934 * h_69)+ (2.6621242 * h_70)+ (6.0911856 * h_71)+ (0.7291556 * h_72)+ (1.2351182 * h_73)+ (3.026299 * h_74)+ (8.216231 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 4.026748
    o[6] = (13.474652 * h_0)+ (25.114689 * h_1)+ (12.948739 * h_2)+ (14.925369 * h_3)+ (13.434636 * h_4)+ (20.586092 * h_5)+ (3.7852287 * h_6)+ (8.088285 * h_7)+ (3.5701962 * h_8)+ (-23.402393 * h_9)+ (1.6617374 * h_10)+ (18.040518 * h_11)+ (-0.99953955 * h_12)+ (15.276108 * h_13)+ (18.036152 * h_14)+ (12.045217 * h_15)+ (16.74847 * h_16)+ (9.1002 * h_17)+ (20.205406 * h_18)+ (2.956862 * h_19)+ (16.92621 * h_20)+ (7.196778 * h_21)+ (8.532075 * h_22)+ (9.025251 * h_23)+ (10.836964 * h_24)+ (22.2408 * h_25)+ (6.1830716 * h_26)+ (-8.175423 * h_27)+ (15.694598 * h_28)+ (20.454138 * h_29)+ (16.51306 * h_30)+ (1.7042099 * h_31)+ (22.231544 * h_32)+ (29.146166 * h_33)+ (15.01703 * h_34)+ (18.35482 * h_35)+ (19.313234 * h_36)+ (22.332054 * h_37)+ (8.28767 * h_38)+ (13.320913 * h_39)+ (12.0339985 * h_40)+ (14.954259 * h_41)+ (13.588395 * h_42)+ (27.263453 * h_43)+ (7.8095922 * h_44)+ (24.094944 * h_45)+ (-5.39249 * h_46)+ (11.793477 * h_47)+ (8.654023 * h_48)+ (8.901094 * h_49)+ (12.841858 * h_50)+ (10.663429 * h_51)+ (7.7648253 * h_52)+ (2.74896 * h_53)+ (7.2802653 * h_54)+ (13.389862 * h_55)+ (16.190931 * h_56)+ (13.451935 * h_57)+ (11.761933 * h_58)+ (5.4523234 * h_59)+ (17.093752 * h_60)+ (4.6933107 * h_61)+ (29.533823 * h_62)+ (2.9070885 * h_63)+ (21.81266 * h_64)+ (6.789757 * h_65)+ (4.703827 * h_66)+ (8.547823 * h_67)+ (10.398714 * h_68)+ (27.929237 * h_69)+ (-1.3891652 * h_70)+ (4.7021604 * h_71)+ (1.9099365 * h_72)+ (2.3838794 * h_73)+ (-2.6163437 * h_74)+ (2.8733304 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 2.9770107
    o[7] = (9.141141 * h_0)+ (13.869079 * h_1)+ (14.379904 * h_2)+ (5.3088984 * h_3)+ (13.73437 * h_4)+ (18.608946 * h_5)+ (13.282547 * h_6)+ (9.574619 * h_7)+ (0.80604076 * h_8)+ (-22.876446 * h_9)+ (12.405782 * h_10)+ (23.44479 * h_11)+ (1.3415034 * h_12)+ (12.576128 * h_13)+ (22.865328 * h_14)+ (14.2921295 * h_15)+ (22.223654 * h_16)+ (-3.3327923 * h_17)+ (19.82823 * h_18)+ (6.135602 * h_19)+ (18.039389 * h_20)+ (12.071718 * h_21)+ (9.132388 * h_22)+ (9.21165 * h_23)+ (12.848889 * h_24)+ (15.457726 * h_25)+ (9.297397 * h_26)+ (11.629912 * h_27)+ (21.207884 * h_28)+ (3.7228878 * h_29)+ (8.591281 * h_30)+ (6.5121064 * h_31)+ (20.841253 * h_32)+ (16.184145 * h_33)+ (16.481064 * h_34)+ (16.133802 * h_35)+ (17.863197 * h_36)+ (14.128163 * h_37)+ (-41.259365 * h_38)+ (18.29639 * h_39)+ (24.683897 * h_40)+ (25.846176 * h_41)+ (14.202685 * h_42)+ (16.129309 * h_43)+ (6.755312 * h_44)+ (-19.313698 * h_45)+ (3.2809124 * h_46)+ (8.722068 * h_47)+ (8.025768 * h_48)+ (4.0654798 * h_49)+ (15.423238 * h_50)+ (11.210822 * h_51)+ (10.788186 * h_52)+ (8.487023 * h_53)+ (4.3332357 * h_54)+ (14.340809 * h_55)+ (7.2124715 * h_56)+ (10.841476 * h_57)+ (6.2823467 * h_58)+ (5.7731943 * h_59)+ (-34.141804 * h_60)+ (3.0378706 * h_61)+ (18.094488 * h_62)+ (3.0703714 * h_63)+ (9.301202 * h_64)+ (1.4910973 * h_65)+ (17.542597 * h_66)+ (4.832642 * h_67)+ (7.763307 * h_68)+ (34.252586 * h_69)+ (7.192912 * h_70)+ (9.392342 * h_71)+ (4.241923 * h_72)+ (0.81417805 * h_73)+ (2.7940912 * h_74)+ (4.5487466 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 0.7147861
    o[8] = (6.4587436 * h_0)+ (13.487616 * h_1)+ (14.566809 * h_2)+ (6.7261233 * h_3)+ (14.286833 * h_4)+ (16.037754 * h_5)+ (-66.06142 * h_6)+ (19.203218 * h_7)+ (2.7425385 * h_8)+ (18.270414 * h_9)+ (8.281094 * h_10)+ (21.241304 * h_11)+ (1.9026339 * h_12)+ (13.088243 * h_13)+ (32.81339 * h_14)+ (9.287724 * h_15)+ (12.561413 * h_16)+ (13.5146055 * h_17)+ (17.87452 * h_18)+ (7.0932217 * h_19)+ (21.393305 * h_20)+ (-36.783817 * h_21)+ (6.987525 * h_22)+ (11.548588 * h_23)+ (25.017405 * h_24)+ (5.771164 * h_25)+ (17.52634 * h_26)+ (-1.0411551 * h_27)+ (11.563475 * h_28)+ (2.8802433 * h_29)+ (13.469556 * h_30)+ (15.794574 * h_31)+ (19.191319 * h_32)+ (-29.036495 * h_33)+ (12.7871475 * h_34)+ (16.250818 * h_35)+ (24.207586 * h_36)+ (0.8311823 * h_37)+ (3.069923 * h_38)+ (25.159138 * h_39)+ (13.411866 * h_40)+ (5.2583556 * h_41)+ (23.738108 * h_42)+ (27.446774 * h_43)+ (12.461645 * h_44)+ (-10.00306 * h_45)+ (3.3301525 * h_46)+ (10.548966 * h_47)+ (1.3021516 * h_48)+ (5.117716 * h_49)+ (11.16112 * h_50)+ (11.640335 * h_51)+ (-3.9802608 * h_52)+ (-58.16492 * h_53)+ (11.549322 * h_54)+ (14.648039 * h_55)+ (2.3124099 * h_56)+ (6.1326 * h_57)+ (9.612921 * h_58)+ (8.372766 * h_59)+ (-47.94358 * h_60)+ (10.484964 * h_61)+ (31.321129 * h_62)+ (1.258141 * h_63)+ (16.867949 * h_64)+ (4.577655 * h_65)+ (22.242943 * h_66)+ (7.53424 * h_67)+ (11.129545 * h_68)+ (15.763914 * h_69)+ (0.48678112 * h_70)+ (24.508648 * h_71)+ (6.2067223 * h_72)+ (1.3738748 * h_73)+ (2.2998831 * h_74)+ (4.3523755 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -5.8150854
    o[9] = (10.673399 * h_0)+ (13.599363 * h_1)+ (17.49849 * h_2)+ (4.7649155 * h_3)+ (8.679568 * h_4)+ (8.811628 * h_5)+ (8.78453 * h_6)+ (13.570303 * h_7)+ (3.0705438 * h_8)+ (-52.974316 * h_9)+ (21.998764 * h_10)+ (1.8667897 * h_11)+ (1.5302992 * h_12)+ (11.481596 * h_13)+ (32.824604 * h_14)+ (17.043016 * h_15)+ (14.773633 * h_16)+ (14.111781 * h_17)+ (15.828348 * h_18)+ (11.9866085 * h_19)+ (18.217806 * h_20)+ (4.8086667 * h_21)+ (18.426632 * h_22)+ (19.74866 * h_23)+ (19.670925 * h_24)+ (10.102963 * h_25)+ (15.666676 * h_26)+ (22.044464 * h_27)+ (12.72581 * h_28)+ (11.399649 * h_29)+ (17.118965 * h_30)+ (13.307955 * h_31)+ (20.880516 * h_32)+ (21.091106 * h_33)+ (18.590002 * h_34)+ (18.82076 * h_35)+ (29.666952 * h_36)+ (-5.361272 * h_37)+ (14.114373 * h_38)+ (36.462994 * h_39)+ (17.307453 * h_40)+ (8.517452 * h_41)+ (23.428478 * h_42)+ (24.416325 * h_43)+ (17.521986 * h_44)+ (20.784204 * h_45)+ (1.8483027 * h_46)+ (21.24042 * h_47)+ (3.2391036 * h_48)+ (4.2116466 * h_49)+ (14.3370495 * h_50)+ (11.437744 * h_51)+ (6.0945983 * h_52)+ (8.456557 * h_53)+ (6.3216114 * h_54)+ (14.76672 * h_55)+ (-10.910324 * h_56)+ (16.644697 * h_57)+ (7.619972 * h_58)+ (7.576758 * h_59)+ (11.56798 * h_60)+ (17.056698 * h_61)+ (16.883465 * h_62)+ (0.512544 * h_63)+ (-5.5436044 * h_64)+ (8.920818 * h_65)+ (8.280149 * h_66)+ (12.365704 * h_67)+ (7.1602807 * h_68)+ (19.035246 * h_69)+ (0.37016594 * h_70)+ (10.609722 * h_71)+ (6.5238614 * h_72)+ (0.12988546 * h_73)+ (-1.327897 * h_74)+ (6.0148106 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -2.325949
    o[10] = (18.639776 * h_0)+ (13.316567 * h_1)+ (12.978133 * h_2)+ (22.074654 * h_3)+ (13.684276 * h_4)+ (3.8705368 * h_5)+ (16.020617 * h_6)+ (5.83172 * h_7)+ (3.6602247 * h_8)+ (-16.952559 * h_9)+ (13.960031 * h_10)+ (20.961681 * h_11)+ (1.8155762 * h_12)+ (15.83915 * h_13)+ (18.169056 * h_14)+ (13.657578 * h_15)+ (5.1482844 * h_16)+ (1.2359157 * h_17)+ (24.954847 * h_18)+ (13.652579 * h_19)+ (5.331447 * h_20)+ (21.36659 * h_21)+ (28.532066 * h_22)+ (10.978574 * h_23)+ (10.6367035 * h_24)+ (27.760143 * h_25)+ (2.5486124 * h_26)+ (19.442163 * h_27)+ (20.760767 * h_28)+ (10.727908 * h_29)+ (8.263296 * h_30)+ (5.744011 * h_31)+ (-38.221973 * h_32)+ (10.282729 * h_33)+ (17.389706 * h_34)+ (6.7061453 * h_35)+ (22.320107 * h_36)+ (0.61543536 * h_37)+ (6.2471867 * h_38)+ (21.070591 * h_39)+ (12.728093 * h_40)+ (18.879282 * h_41)+ (13.769755 * h_42)+ (22.122702 * h_43)+ (13.748041 * h_44)+ (15.567985 * h_45)+ (-0.27409717 * h_46)+ (13.262079 * h_47)+ (8.47306 * h_48)+ (7.76592 * h_49)+ (19.83137 * h_50)+ (11.800089 * h_51)+ (21.633715 * h_52)+ (10.127531 * h_53)+ (5.592832 * h_54)+ (5.6776443 * h_55)+ (9.781175 * h_56)+ (15.474561 * h_57)+ (13.718599 * h_58)+ (3.7897644 * h_59)+ (-24.896381 * h_60)+ (10.970963 * h_61)+ (17.899355 * h_62)+ (3.551038 * h_63)+ (3.9277804 * h_64)+ (26.516314 * h_65)+ (4.5779033 * h_66)+ (4.0026917 * h_67)+ (7.597347 * h_68)+ (6.1275105 * h_69)+ (6.8054757 * h_70)+ (3.7561707 * h_71)+ (4.366917 * h_72)+ (1.8270626 * h_73)+ (-3.2683773 * h_74)+ (9.208002 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 1.8209504
    o[11] = (16.212387 * h_0)+ (-2.6448398 * h_1)+ (14.352641 * h_2)+ (6.2181573 * h_3)+ (8.020131 * h_4)+ (12.665826 * h_5)+ (9.820102 * h_6)+ (11.234469 * h_7)+ (2.692967 * h_8)+ (18.815155 * h_9)+ (21.217638 * h_10)+ (10.7749 * h_11)+ (5.7878456 * h_12)+ (-21.980469 * h_13)+ (17.480078 * h_14)+ (16.419964 * h_15)+ (16.013994 * h_16)+ (5.0759745 * h_17)+ (14.82773 * h_18)+ (12.755836 * h_19)+ (16.85799 * h_20)+ (-54.465347 * h_21)+ (5.9753065 * h_22)+ (20.749924 * h_23)+ (9.990537 * h_24)+ (28.736385 * h_25)+ (11.640588 * h_26)+ (19.71592 * h_27)+ (9.407302 * h_28)+ (12.305618 * h_29)+ (11.117618 * h_30)+ (13.787195 * h_31)+ (20.080084 * h_32)+ (21.476189 * h_33)+ (15.786335 * h_34)+ (20.247997 * h_35)+ (26.69127 * h_36)+ (-3.538878 * h_37)+ (-49.98018 * h_38)+ (17.17778 * h_39)+ (10.030909 * h_40)+ (-6.197864 * h_41)+ (14.469167 * h_42)+ (22.072262 * h_43)+ (24.365679 * h_44)+ (22.262228 * h_45)+ (-0.6061574 * h_46)+ (24.58275 * h_47)+ (4.397782 * h_48)+ (4.6935487 * h_49)+ (11.435693 * h_50)+ (11.431271 * h_51)+ (14.456746 * h_52)+ (10.51108 * h_53)+ (7.446973 * h_54)+ (14.70305 * h_55)+ (16.206987 * h_56)+ (25.422026 * h_57)+ (13.382881 * h_58)+ (4.9373164 * h_59)+ (-0.44601586 * h_60)+ (11.886433 * h_61)+ (18.264332 * h_62)+ (1.0057498 * h_63)+ (33.62829 * h_64)+ (24.274801 * h_65)+ (9.392603 * h_66)+ (8.48169 * h_67)+ (13.971192 * h_68)+ (8.854611 * h_69)+ (6.523503 * h_70)+ (3.9677863 * h_71)+ (9.540923 * h_72)+ (-1.177594 * h_73)+ (-1.8235073 * h_74)+ (6.044589 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -0.019551566
    o[12] = (8.058688 * h_0)+ (17.650251 * h_1)+ (15.137214 * h_2)+ (-1.2879289 * h_3)+ (15.023431 * h_4)+ (16.137518 * h_5)+ (-44.13841 * h_6)+ (10.585775 * h_7)+ (1.9284154 * h_8)+ (-12.203305 * h_9)+ (10.06876 * h_10)+ (24.44158 * h_11)+ (0.102156535 * h_12)+ (-8.403881 * h_13)+ (25.997517 * h_14)+ (17.83726 * h_15)+ (21.48762 * h_16)+ (6.703702 * h_17)+ (9.609578 * h_18)+ (1.7055695 * h_19)+ (12.7822 * h_20)+ (-32.03307 * h_21)+ (36.498676 * h_22)+ (4.722189 * h_23)+ (12.153954 * h_24)+ (15.28889 * h_25)+ (13.653128 * h_26)+ (14.497369 * h_27)+ (23.997122 * h_28)+ (9.668743 * h_29)+ (1.156426 * h_30)+ (10.75165 * h_31)+ (19.50779 * h_32)+ (25.951456 * h_33)+ (26.031195 * h_34)+ (-3.8861113 * h_35)+ (15.462866 * h_36)+ (2.5878534 * h_37)+ (18.304905 * h_38)+ (10.246404 * h_39)+ (41.606133 * h_40)+ (34.65217 * h_41)+ (14.89723 * h_42)+ (9.064973 * h_43)+ (13.284437 * h_44)+ (24.078444 * h_45)+ (11.040884 * h_46)+ (6.170431 * h_47)+ (7.870979 * h_48)+ (5.504243 * h_49)+ (20.360357 * h_50)+ (11.746962 * h_51)+ (10.642099 * h_52)+ (4.9227667 * h_53)+ (1.6585144 * h_54)+ (15.225843 * h_55)+ (22.197058 * h_56)+ (7.261971 * h_57)+ (0.57496345 * h_58)+ (7.0663223 * h_59)+ (-38.728004 * h_60)+ (0.9048077 * h_61)+ (18.676126 * h_62)+ (4.6617117 * h_63)+ (0.30630106 * h_64)+ (-4.6448708 * h_65)+ (18.192434 * h_66)+ (1.5628042 * h_67)+ (4.984982 * h_68)+ (29.211231 * h_69)+ (1.3198814 * h_70)+ (6.834172 * h_71)+ (4.613202 * h_72)+ (1.292476 * h_73)+ (-0.4581539 * h_74)+ (5.9314823 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -0.80752146
    o[13] = (11.814629 * h_0)+ (20.228123 * h_1)+ (14.316292 * h_2)+ (12.80287 * h_3)+ (9.884466 * h_4)+ (16.745012 * h_5)+ (9.358621 * h_6)+ (10.939442 * h_7)+ (6.0595284 * h_8)+ (-6.8028126 * h_9)+ (13.041096 * h_10)+ (20.435513 * h_11)+ (-0.23457743 * h_12)+ (15.379908 * h_13)+ (32.21219 * h_14)+ (13.716281 * h_15)+ (15.921692 * h_16)+ (7.2658205 * h_17)+ (18.044344 * h_18)+ (-3.9056997 * h_19)+ (17.719198 * h_20)+ (15.029115 * h_21)+ (-4.7395434 * h_22)+ (11.284324 * h_23)+ (22.361273 * h_24)+ (11.414859 * h_25)+ (9.853216 * h_26)+ (-5.283238 * h_27)+ (11.89309 * h_28)+ (7.052741 * h_29)+ (20.642832 * h_30)+ (5.522009 * h_31)+ (19.54565 * h_32)+ (18.129759 * h_33)+ (9.868722 * h_34)+ (-1.3083469 * h_35)+ (24.485273 * h_36)+ (12.981945 * h_37)+ (2.8155715 * h_38)+ (8.485559 * h_39)+ (12.840252 * h_40)+ (24.04458 * h_41)+ (11.581774 * h_42)+ (22.433575 * h_43)+ (4.771404 * h_44)+ (-21.249998 * h_45)+ (-2.5628111 * h_46)+ (2.9859533 * h_47)+ (5.388272 * h_48)+ (9.492931 * h_49)+ (10.940262 * h_50)+ (11.663998 * h_51)+ (-0.09732228 * h_52)+ (8.955959 * h_53)+ (15.831864 * h_54)+ (13.410976 * h_55)+ (14.346635 * h_56)+ (28.883675 * h_57)+ (10.786994 * h_58)+ (8.786078 * h_59)+ (-40.521187 * h_60)+ (6.294961 * h_61)+ (28.72607 * h_62)+ (2.091985 * h_63)+ (16.107004 * h_64)+ (-6.9723973 * h_65)+ (20.846851 * h_66)+ (9.469207 * h_67)+ (12.244151 * h_68)+ (17.059834 * h_69)+ (2.7405005 * h_70)+ (18.75678 * h_71)+ (2.4425256 * h_72)+ (1.900748 * h_73)+ (0.14944409 * h_74)+ (2.7720478 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -1.3946979
    o[14] = (24.861582 * h_0)+ (0.059486 * h_1)+ (16.038069 * h_2)+ (14.040918 * h_3)+ (4.7544184 * h_4)+ (6.5717034 * h_5)+ (10.759092 * h_6)+ (2.169209 * h_7)+ (3.8424516 * h_8)+ (19.481783 * h_9)+ (18.134249 * h_10)+ (19.665201 * h_11)+ (5.912853 * h_12)+ (11.106138 * h_13)+ (11.568702 * h_14)+ (-0.0762044 * h_15)+ (17.181011 * h_16)+ (8.787625 * h_17)+ (16.805326 * h_18)+ (8.589556 * h_19)+ (2.8295064 * h_20)+ (17.396744 * h_21)+ (14.302474 * h_22)+ (18.432041 * h_23)+ (15.009387 * h_24)+ (38.18556 * h_25)+ (9.217025 * h_26)+ (16.084538 * h_27)+ (32.993824 * h_28)+ (15.739981 * h_29)+ (-6.2721806 * h_30)+ (6.0934134 * h_31)+ (21.182243 * h_32)+ (20.764715 * h_33)+ (18.840809 * h_34)+ (6.1595516 * h_35)+ (5.9731836 * h_36)+ (6.040168 * h_37)+ (14.23621 * h_38)+ (25.839682 * h_39)+ (12.781335 * h_40)+ (17.815573 * h_41)+ (13.905086 * h_42)+ (22.7501 * h_43)+ (-0.085554615 * h_44)+ (20.428408 * h_45)+ (-2.9501035 * h_46)+ (2.27214 * h_47)+ (1.3144976 * h_48)+ (9.109838 * h_49)+ (16.16297 * h_50)+ (11.941261 * h_51)+ (24.764158 * h_52)+ (0.47140878 * h_53)+ (5.197268 * h_54)+ (5.8984656 * h_55)+ (9.341005 * h_56)+ (15.640074 * h_57)+ (16.764801 * h_58)+ (-3.7450771 * h_59)+ (-36.546795 * h_60)+ (15.705234 * h_61)+ (13.167201 * h_62)+ (5.31996 * h_63)+ (-0.6751521 * h_64)+ (23.720999 * h_65)+ (15.729255 * h_66)+ (5.783078 * h_67)+ (24.738794 * h_68)+ (-8.881609 * h_69)+ (8.518483 * h_70)+ (0.6333896 * h_71)+ (3.256804 * h_72)+ (2.4804435 * h_73)+ (-0.15824047 * h_74)+ (13.35593 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 3.740356
    o[15] = (9.309845 * h_0)+ (14.669458 * h_1)+ (16.803783 * h_2)+ (12.603026 * h_3)+ (14.693996 * h_4)+ (22.546244 * h_5)+ (10.408212 * h_6)+ (9.223115 * h_7)+ (3.635295 * h_8)+ (19.11774 * h_9)+ (-1.5009756 * h_10)+ (24.642176 * h_11)+ (1.1170249 * h_12)+ (16.87153 * h_13)+ (28.616217 * h_14)+ (13.8768425 * h_15)+ (14.255163 * h_16)+ (12.279673 * h_17)+ (15.986861 * h_18)+ (22.22585 * h_19)+ (22.80611 * h_20)+ (18.15092 * h_21)+ (17.282906 * h_22)+ (10.002376 * h_23)+ (16.48497 * h_24)+ (16.83665 * h_25)+ (11.113735 * h_26)+ (22.986906 * h_27)+ (16.845291 * h_28)+ (18.707792 * h_29)+ (2.9273052 * h_30)+ (10.346652 * h_31)+ (-41.96923 * h_32)+ (18.770687 * h_33)+ (8.107635 * h_34)+ (-0.5763704 * h_35)+ (19.292454 * h_36)+ (1.1128238 * h_37)+ (5.5443377 * h_38)+ (39.748928 * h_39)+ (-0.7198531 * h_40)+ (13.766195 * h_41)+ (14.807795 * h_42)+ (25.08313 * h_43)+ (9.32498 * h_44)+ (-18.679834 * h_45)+ (1.5584551 * h_46)+ (-0.6349214 * h_47)+ (0.60359734 * h_48)+ (6.4150405 * h_49)+ (6.204858 * h_50)+ (11.667548 * h_51)+ (12.475112 * h_52)+ (6.6494284 * h_53)+ (14.687715 * h_54)+ (14.879608 * h_55)+ (-12.290105 * h_56)+ (11.559964 * h_57)+ (19.311737 * h_58)+ (9.12119 * h_59)+ (1.9481257 * h_60)+ (5.8065658 * h_61)+ (20.723145 * h_62)+ (2.4737453 * h_63)+ (19.380209 * h_64)+ (19.781479 * h_65)+ (11.380534 * h_66)+ (7.206562 * h_67)+ (3.6935472 * h_68)+ (8.437363 * h_69)+ (5.468293 * h_70)+ (3.9781518 * h_71)+ (3.521169 * h_72)+ (3.513401 * h_73)+ (0.040733386 * h_74)+ (4.5960116 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 0.121689335
    o[16] = (19.349789 * h_0)+ (14.290332 * h_1)+ (13.78317 * h_2)+ (-7.609772 * h_3)+ (12.7089815 * h_4)+ (38.141953 * h_5)+ (-3.1190598 * h_6)+ (11.910177 * h_7)+ (3.1137166 * h_8)+ (-30.6786 * h_9)+ (-7.2492876 * h_10)+ (22.822208 * h_11)+ (1.9960588 * h_12)+ (11.290587 * h_13)+ (22.896578 * h_14)+ (16.789291 * h_15)+ (3.0490358 * h_16)+ (11.127709 * h_17)+ (13.998775 * h_18)+ (24.844109 * h_19)+ (4.4145794 * h_20)+ (19.045456 * h_21)+ (18.840607 * h_22)+ (6.7317653 * h_23)+ (20.029934 * h_24)+ (19.583118 * h_25)+ (11.90612 * h_26)+ (7.4983997 * h_27)+ (21.377289 * h_28)+ (10.516179 * h_29)+ (1.9698508 * h_30)+ (11.084198 * h_31)+ (18.14934 * h_32)+ (29.6226 * h_33)+ (8.772761 * h_34)+ (4.0636635 * h_35)+ (31.17477 * h_36)+ (13.466725 * h_37)+ (13.9640255 * h_38)+ (3.76395 * h_39)+ (21.071455 * h_40)+ (15.703041 * h_41)+ (14.017907 * h_42)+ (25.111197 * h_43)+ (8.371775 * h_44)+ (19.588837 * h_45)+ (3.3651044 * h_46)+ (-3.0395174 * h_47)+ (4.6539207 * h_48)+ (4.824609 * h_49)+ (18.511816 * h_50)+ (12.026021 * h_51)+ (12.880323 * h_52)+ (3.833595 * h_53)+ (13.115339 * h_54)+ (6.137951 * h_55)+ (21.925745 * h_56)+ (7.474778 * h_57)+ (8.011276 * h_58)+ (7.3696976 * h_59)+ (8.594207 * h_60)+ (2.3150234 * h_61)+ (20.868956 * h_62)+ (1.5862098 * h_63)+ (22.640028 * h_64)+ (16.359978 * h_65)+ (8.341459 * h_66)+ (6.615198 * h_67)+ (1.0927972 * h_68)+ (23.526257 * h_69)+ (1.141658 * h_70)+ (5.2268715 * h_71)+ (3.9578462 * h_72)+ (4.2962804 * h_73)+ (-0.6993837 * h_74)+ (6.6225286 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -0.448809
    o[17] = (13.42025 * h_0)+ (6.9513726 * h_1)+ (16.621212 * h_2)+ (15.300176 * h_3)+ (10.540639 * h_4)+ (8.875565 * h_5)+ (6.154733 * h_6)+ (4.926076 * h_7)+ (5.1625147 * h_8)+ (16.025724 * h_9)+ (26.457367 * h_10)+ (23.561337 * h_11)+ (2.7472029 * h_12)+ (16.321938 * h_13)+ (11.31854 * h_14)+ (-4.899853 * h_15)+ (13.213668 * h_16)+ (14.194796 * h_17)+ (12.219333 * h_18)+ (3.324052 * h_19)+ (15.393679 * h_20)+ (22.012804 * h_21)+ (6.7182574 * h_22)+ (16.793484 * h_23)+ (23.845982 * h_24)+ (14.981152 * h_25)+ (7.9571605 * h_26)+ (11.888738 * h_27)+ (14.321695 * h_28)+ (11.258539 * h_29)+ (8.689636 * h_30)+ (10.918517 * h_31)+ (20.938786 * h_32)+ (23.582577 * h_33)+ (19.790924 * h_34)+ (19.07856 * h_35)+ (21.645155 * h_36)+ (-3.036529 * h_37)+ (-42.60829 * h_38)+ (24.923006 * h_39)+ (30.47176 * h_40)+ (10.231538 * h_41)+ (14.412117 * h_42)+ (22.838919 * h_43)+ (13.190122 * h_44)+ (22.060846 * h_45)+ (0.027916314 * h_46)+ (-3.9779375 * h_47)+ (5.6409206 * h_48)+ (6.6979766 * h_49)+ (11.074645 * h_50)+ (12.414083 * h_51)+ (17.972176 * h_52)+ (7.9939055 * h_53)+ (2.3761206 * h_54)+ (14.564443 * h_55)+ (5.3677425 * h_56)+ (7.3304167 * h_57)+ (3.4745886 * h_58)+ (1.0548878 * h_59)+ (12.709491 * h_60)+ (3.6344697 * h_61)+ (22.419676 * h_62)+ (3.9436076 * h_63)+ (2.8283088 * h_64)+ (39.237747 * h_65)+ (20.387463 * h_66)+ (7.559988 * h_67)+ (13.333475 * h_68)+ (-2.0508587 * h_69)+ (2.2037191 * h_70)+ (6.654043 * h_71)+ (1.0172845 * h_72)+ (4.2644815 * h_73)+ (1.4356081 * h_74)+ (12.142375 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 1.4262958
    o[18] = (17.722473 * h_0)+ (8.692465 * h_1)+ (16.007858 * h_2)+ (13.946533 * h_3)+ (12.355133 * h_4)+ (9.3177185 * h_5)+ (0.35760045 * h_6)+ (3.512391 * h_7)+ (3.4718091 * h_8)+ (-26.758734 * h_9)+ (7.4837427 * h_10)+ (12.925334 * h_11)+ (2.672729 * h_12)+ (16.537449 * h_13)+ (26.691553 * h_14)+ (18.871231 * h_15)+ (8.138228 * h_16)+ (6.0482874 * h_17)+ (19.57855 * h_18)+ (1.462822 * h_19)+ (6.4025908 * h_20)+ (18.662794 * h_21)+ (16.121458 * h_22)+ (16.19758 * h_23)+ (15.912301 * h_24)+ (10.841591 * h_25)+ (7.463431 * h_26)+ (5.918142 * h_27)+ (17.987154 * h_28)+ (12.765411 * h_29)+ (13.432477 * h_30)+ (6.8419166 * h_31)+ (-45.95705 * h_32)+ (32.829426 * h_33)+ (14.487023 * h_34)+ (15.532221 * h_35)+ (22.05821 * h_36)+ (5.012765 * h_37)+ (9.481242 * h_38)+ (16.817146 * h_39)+ (5.28524 * h_40)+ (7.472748 * h_41)+ (14.183644 * h_42)+ (26.435205 * h_43)+ (13.741286 * h_44)+ (-59.68378 * h_45)+ (7.21917 * h_46)+ (12.6273365 * h_47)+ (10.551606 * h_48)+ (0.24031027 * h_49)+ (19.879177 * h_50)+ (11.179001 * h_51)+ (17.851326 * h_52)+ (10.449204 * h_53)+ (7.8906794 * h_54)+ (13.930178 * h_55)+ (7.9792 * h_56)+ (20.09715 * h_57)+ (15.157592 * h_58)+ (9.302885 * h_59)+ (15.303486 * h_60)+ (15.697606 * h_61)+ (21.271093 * h_62)+ (1.8987831 * h_63)+ (21.153584 * h_64)+ (10.728971 * h_65)+ (8.376379 * h_66)+ (5.925711 * h_67)+ (0.25647515 * h_68)+ (12.040132 * h_69)+ (0.07247352 * h_70)+ (3.422297 * h_71)+ (8.132171 * h_72)+ (0.7968171 * h_73)+ (-1.9823005 * h_74)+ (3.7195058 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 3.2877977
    o[19] = (6.4209 * h_0)+ (10.874384 * h_1)+ (14.078663 * h_2)+ (-2.8063555 * h_3)+ (18.623768 * h_4)+ (29.102716 * h_5)+ (13.523007 * h_6)+ (5.8662076 * h_7)+ (4.0736017 * h_8)+ (15.566772 * h_9)+ (19.904366 * h_10)+ (24.355536 * h_11)+ (2.0337873 * h_12)+ (14.649365 * h_13)+ (22.672241 * h_14)+ (3.7499769 * h_15)+ (11.178032 * h_16)+ (3.1633408 * h_17)+ (21.385462 * h_18)+ (5.837539 * h_19)+ (20.125803 * h_20)+ (-45.792282 * h_21)+ (17.34145 * h_22)+ (7.4266825 * h_23)+ (22.60097 * h_24)+ (15.129695 * h_25)+ (5.5957775 * h_26)+ (11.185631 * h_27)+ (18.777874 * h_28)+ (14.241328 * h_29)+ (9.408469 * h_30)+ (7.2336607 * h_31)+ (21.344606 * h_32)+ (17.790419 * h_33)+ (14.88555 * h_34)+ (18.007217 * h_35)+ (22.212887 * h_36)+ (0.37836725 * h_37)+ (10.274522 * h_38)+ (3.2423995 * h_39)+ (15.385099 * h_40)+ (20.021433 * h_41)+ (14.445737 * h_42)+ (16.078615 * h_43)+ (13.890759 * h_44)+ (-17.6028 * h_45)+ (3.4459407 * h_46)+ (0.390496 * h_47)+ (11.565553 * h_48)+ (4.2921467 * h_49)+ (17.852652 * h_50)+ (11.39091 * h_51)+ (6.179779 * h_52)+ (8.474044 * h_53)+ (4.748127 * h_54)+ (14.250168 * h_55)+ (15.68833 * h_56)+ (-5.36901 * h_57)+ (11.830196 * h_58)+ (7.627881 * h_59)+ (-39.136776 * h_60)+ (8.941629 * h_61)+ (25.22784 * h_62)+ (5.024463 * h_63)+ (16.16342 * h_64)+ (17.952349 * h_65)+ (11.508127 * h_66)+ (4.784108 * h_67)+ (9.682814 * h_68)+ (-1.0054545 * h_69)+ (2.9051967 * h_70)+ (12.527814 * h_71)+ (1.994866 * h_72)+ (4.386981 * h_73)+ (2.0015838 * h_74)+ (3.750923 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 1.8918736
    o[20] = (14.084813 * h_0)+ (13.858028 * h_1)+ (12.635431 * h_2)+ (0.41253048 * h_3)+ (11.516034 * h_4)+ (18.281792 * h_5)+ (1.2630144 * h_6)+ (11.466712 * h_7)+ (3.3874867 * h_8)+ (19.348402 * h_9)+ (14.190837 * h_10)+ (16.605553 * h_11)+ (1.1841823 * h_12)+ (10.960504 * h_13)+ (18.917717 * h_14)+ (14.182626 * h_15)+ (11.936274 * h_16)+ (14.6495285 * h_17)+ (18.328152 * h_18)+ (8.785611 * h_19)+ (19.815432 * h_20)+ (12.61904 * h_21)+ (9.738217 * h_22)+ (10.276751 * h_23)+ (18.157944 * h_24)+ (14.051972 * h_25)+ (16.16401 * h_26)+ (-7.6415796 * h_27)+ (10.507857 * h_28)+ (14.822336 * h_29)+ (-1.8963721 * h_30)+ (6.4063487 * h_31)+ (18.393763 * h_32)+ (30.477352 * h_33)+ (11.391033 * h_34)+ (5.6385875 * h_35)+ (29.205494 * h_36)+ (10.843312 * h_37)+ (14.119656 * h_38)+ (1.4188561 * h_39)+ (40.172592 * h_40)+ (25.142912 * h_41)+ (14.512089 * h_42)+ (31.216232 * h_43)+ (20.915956 * h_44)+ (22.282333 * h_45)+ (4.27663 * h_46)+ (5.216222 * h_47)+ (0.5038708 * h_48)+ (4.689237 * h_49)+ (11.635569 * h_50)+ (11.45536 * h_51)+ (8.586473 * h_52)+ (6.7132154 * h_53)+ (12.048423 * h_54)+ (14.449654 * h_55)+ (29.904118 * h_56)+ (18.001875 * h_57)+ (1.6215551 * h_58)+ (7.129042 * h_59)+ (-36.90956 * h_60)+ (-0.22849514 * h_61)+ (23.77077 * h_62)+ (-0.5355494 * h_63)+ (10.902057 * h_64)+ (-2.2299757 * h_65)+ (19.766169 * h_66)+ (5.917413 * h_67)+ (15.556444 * h_68)+ (21.783365 * h_69)+ (2.1222372 * h_70)+ (7.387311 * h_71)+ (3.1975844 * h_72)+ (1.6520805 * h_73)+ (0.97470236 * h_74)+ (3.4847105 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -0.04936377
    o[21] = (14.058102 * h_0)+ (8.948098 * h_1)+ (14.62698 * h_2)+ (15.555763 * h_3)+ (13.83839 * h_4)+ (16.986942 * h_5)+ (-1.7020458 * h_6)+ (10.366737 * h_7)+ (2.0347614 * h_8)+ (18.793816 * h_9)+ (3.2842646 * h_10)+ (23.392307 * h_11)+ (3.2749927 * h_12)+ (16.814617 * h_13)+ (26.596498 * h_14)+ (4.5155177 * h_15)+ (13.283425 * h_16)+ (8.861073 * h_17)+ (20.845308 * h_18)+ (0.2207596 * h_19)+ (9.850833 * h_20)+ (22.855606 * h_21)+ (6.482996 * h_22)+ (7.1510415 * h_23)+ (17.903915 * h_24)+ (16.957443 * h_25)+ (4.5397096 * h_26)+ (-5.533227 * h_27)+ (18.984787 * h_28)+ (25.799974 * h_29)+ (13.564608 * h_30)+ (6.243954 * h_31)+ (13.9879 * h_32)+ (28.413568 * h_33)+ (4.097073 * h_34)+ (21.722925 * h_35)+ (21.232319 * h_36)+ (16.731209 * h_37)+ (-40.171448 * h_38)+ (19.05557 * h_39)+ (18.258781 * h_40)+ (7.4591413 * h_41)+ (14.333329 * h_42)+ (35.159122 * h_43)+ (6.094737 * h_44)+ (21.85713 * h_45)+ (7.3201594 * h_46)+ (7.0738344 * h_47)+ (7.43521 * h_48)+ (1.1038934 * h_49)+ (9.562217 * h_50)+ (11.30576 * h_51)+ (2.9351149 * h_52)+ (7.833931 * h_53)+ (13.916707 * h_54)+ (14.484743 * h_55)+ (3.038294 * h_56)+ (2.0865467 * h_57)+ (9.905346 * h_58)+ (7.6027536 * h_59)+ (14.804188 * h_60)+ (12.420443 * h_61)+ (27.475239 * h_62)+ (1.773297 * h_63)+ (17.970058 * h_64)+ (18.638098 * h_65)+ (12.905453 * h_66)+ (10.193897 * h_67)+ (4.585013 * h_68)+ (19.579224 * h_69)+ (0.8470362 * h_70)+ (6.2195525 * h_71)+ (2.0662537 * h_72)+ (3.316077 * h_73)+ (4.1140018 * h_74)+ (6.1364594 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -0.47865307
    o[22] = (11.887729 * h_0)+ (22.672287 * h_1)+ (15.474639 * h_2)+ (5.8256326 * h_3)+ (16.802382 * h_4)+ (13.385832 * h_5)+ (-8.945136 * h_6)+ (12.850637 * h_7)+ (1.4894217 * h_8)+ (-38.032806 * h_9)+ (16.413721 * h_10)+ (25.991375 * h_11)+ (-1.0007079 * h_12)+ (18.366568 * h_13)+ (24.02438 * h_14)+ (10.041078 * h_15)+ (19.915216 * h_16)+ (-3.1723704 * h_17)+ (11.296225 * h_18)+ (9.406404 * h_19)+ (25.583054 * h_20)+ (12.389421 * h_21)+ (25.995787 * h_22)+ (7.9660664 * h_23)+ (14.574 * h_24)+ (13.390469 * h_25)+ (10.3084545 * h_26)+ (14.312911 * h_27)+ (16.108713 * h_28)+ (11.808031 * h_29)+ (10.100435 * h_30)+ (4.7869935 * h_31)+ (21.671967 * h_32)+ (29.695864 * h_33)+ (8.18728 * h_34)+ (17.719992 * h_35)+ (20.34962 * h_36)+ (-1.2027427 * h_37)+ (-52.847572 * h_38)+ (14.113335 * h_39)+ (13.045285 * h_40)+ (25.18513 * h_41)+ (14.253071 * h_42)+ (30.73973 * h_43)+ (5.434293 * h_44)+ (26.175587 * h_45)+ (4.583923 * h_46)+ (12.554777 * h_47)+ (3.4580102 * h_48)+ (1.382364 * h_49)+ (11.088642 * h_50)+ (11.251644 * h_51)+ (13.18238 * h_52)+ (8.706816 * h_53)+ (13.65923 * h_54)+ (14.1722555 * h_55)+ (13.609689 * h_56)+ (17.390516 * h_57)+ (11.933809 * h_58)+ (6.7352533 * h_59)+ (-33.28024 * h_60)+ (-1.0450225 * h_61)+ (19.835957 * h_62)+ (2.0409927 * h_63)+ (-10.923644 * h_64)+ (19.633184 * h_65)+ (24.622667 * h_66)+ (7.5960484 * h_67)+ (15.689927 * h_68)+ (24.485048 * h_69)+ (10.117927 * h_70)+ (6.8431926 * h_71)+ (-0.15578642 * h_72)+ (0.67111605 * h_73)+ (2.2528496 * h_74)+ (6.7087674 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -2.1511908
    o[23] = (12.299655 * h_0)+ (15.391688 * h_1)+ (15.069421 * h_2)+ (-1.2899134 * h_3)+ (14.493777 * h_4)+ (32.78128 * h_5)+ (10.228662 * h_6)+ (4.023411 * h_7)+ (4.303375 * h_8)+ (-22.75026 * h_9)+ (12.114963 * h_10)+ (21.699371 * h_11)+ (0.7501328 * h_12)+ (-37.722885 * h_13)+ (21.616163 * h_14)+ (-0.9902508 * h_15)+ (15.220409 * h_16)+ (10.853783 * h_17)+ (18.207048 * h_18)+ (4.249735 * h_19)+ (24.113516 * h_20)+ (-60.235107 * h_21)+ (13.676828 * h_22)+ (15.641216 * h_23)+ (21.912855 * h_24)+ (17.219585 * h_25)+ (10.549645 * h_26)+ (5.5923986 * h_27)+ (17.014355 * h_28)+ (14.698356 * h_29)+ (6.171306 * h_30)+ (8.574603 * h_31)+ (21.966639 * h_32)+ (22.081493 * h_33)+ (10.267156 * h_34)+ (32.44267 * h_35)+ (21.765236 * h_36)+ (-1.1510295 * h_37)+ (17.667334 * h_38)+ (7.3081727 * h_39)+ (12.78443 * h_40)+ (34.866814 * h_41)+ (14.449248 * h_42)+ (27.289497 * h_43)+ (2.5541177 * h_44)+ (21.382715 * h_45)+ (-0.1638078 * h_46)+ (5.517803 * h_47)+ (6.1346245 * h_48)+ (2.0394351 * h_49)+ (13.670397 * h_50)+ (11.388443 * h_51)+ (4.1601086 * h_52)+ (-56.303352 * h_53)+ (7.881973 * h_54)+ (14.257779 * h_55)+ (14.89114 * h_56)+ (13.242075 * h_57)+ (12.089062 * h_58)+ (9.106621 * h_59)+ (-54.332287 * h_60)+ (8.090147 * h_61)+ (23.902521 * h_62)+ (3.831561 * h_63)+ (8.565878 * h_64)+ (6.8712673 * h_65)+ (26.168161 * h_66)+ (6.2114835 * h_67)+ (2.7064872 * h_68)+ (5.34959 * h_69)+ (1.0543152 * h_70)+ (14.11038 * h_71)+ (1.6738396 * h_72)+ (1.9983617 * h_73)+ (3.305286 * h_74)+ (4.135538 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 4.0969334
    o[24] = (14.703642 * h_0)+ (7.1263933 * h_1)+ (16.918089 * h_2)+ (21.216696 * h_3)+ (7.677622 * h_4)+ (-1.0208416 * h_5)+ (11.145886 * h_6)+ (6.5322437 * h_7)+ (3.8882565 * h_8)+ (19.6234 * h_9)+ (15.861206 * h_10)+ (18.612762 * h_11)+ (2.9513364 * h_12)+ (14.720576 * h_13)+ (25.615767 * h_14)+ (5.247887 * h_15)+ (11.182844 * h_16)+ (11.270356 * h_17)+ (17.248922 * h_18)+ (8.999535 * h_19)+ (19.827408 * h_20)+ (17.251654 * h_21)+ (29.55021 * h_22)+ (10.947073 * h_23)+ (18.14681 * h_24)+ (14.401242 * h_25)+ (13.642352 * h_26)+ (8.553999 * h_27)+ (22.597313 * h_28)+ (16.972017 * h_29)+ (6.5401115 * h_30)+ (5.903169 * h_31)+ (22.147581 * h_32)+ (20.213871 * h_33)+ (24.821638 * h_34)+ (8.298085 * h_35)+ (20.045301 * h_36)+ (12.065532 * h_37)+ (-33.088528 * h_38)+ (0.65230745 * h_39)+ (39.95426 * h_40)+ (15.2384 * h_41)+ (13.785612 * h_42)+ (13.720684 * h_43)+ (7.6451235 * h_44)+ (-26.690773 * h_45)+ (-2.6458697 * h_46)+ (-5.402947 * h_47)+ (6.84762 * h_48)+ (11.031567 * h_49)+ (12.15068 * h_50)+ (10.836885 * h_51)+ (20.779507 * h_52)+ (5.0212097 * h_53)+ (-0.4888488 * h_54)+ (13.77255 * h_55)+ (26.320805 * h_56)+ (17.942 * h_57)+ (2.8698714 * h_58)+ (6.3304596 * h_59)+ (14.428867 * h_60)+ (7.5304155 * h_61)+ (18.839985 * h_62)+ (5.727867 * h_63)+ (-7.5357337 * h_64)+ (28.958422 * h_65)+ (13.02674 * h_66)+ (4.7752814 * h_67)+ (-1.356228 * h_68)+ (10.39248 * h_69)+ (2.6270063 * h_70)+ (1.1871922 * h_71)+ (5.253771 * h_72)+ (2.7115662 * h_73)+ (-0.82657146 * h_74)+ (10.987911 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + 1.5110277
    o[25] = (21.999975 * h_0)+ (10.907715 * h_1)+ (14.354741 * h_2)+ (7.5123425 * h_3)+ (14.796068 * h_4)+ (17.896437 * h_5)+ (1.7967321 * h_6)+ (11.37831 * h_7)+ (0.2959112 * h_8)+ (-48.201607 * h_9)+ (2.7582693 * h_10)+ (27.364178 * h_11)+ (1.2899281 * h_12)+ (-23.925373 * h_13)+ (10.5288515 * h_14)+ (-0.68197095 * h_15)+ (30.130901 * h_16)+ (-1.4385849 * h_17)+ (16.663803 * h_18)+ (13.200875 * h_19)+ (21.143227 * h_20)+ (15.090088 * h_21)+ (8.380124 * h_22)+ (4.645639 * h_23)+ (15.763214 * h_24)+ (30.505013 * h_25)+ (8.853854 * h_26)+ (17.378803 * h_27)+ (11.860477 * h_28)+ (8.13557 * h_29)+ (11.456454 * h_30)+ (11.189583 * h_31)+ (22.749191 * h_32)+ (30.546005 * h_33)+ (15.536247 * h_34)+ (21.825546 * h_35)+ (15.0086775 * h_36)+ (8.255868 * h_37)+ (-59.94408 * h_38)+ (24.033937 * h_39)+ (25.919487 * h_40)+ (15.695132 * h_41)+ (13.930216 * h_42)+ (13.059608 * h_43)+ (15.163684 * h_44)+ (23.159117 * h_45)+ (3.080729 * h_46)+ (18.724592 * h_47)+ (0.7408195 * h_48)+ (4.0664525 * h_49)+ (0.075759776 * h_50)+ (10.981539 * h_51)+ (19.317934 * h_52)+ (6.031955 * h_53)+ (6.0035057 * h_54)+ (14.193722 * h_55)+ (14.518063 * h_56)+ (12.601884 * h_57)+ (4.7677536 * h_58)+ (0.2683717 * h_59)+ (-23.6437 * h_60)+ (6.678664 * h_61)+ (18.714464 * h_62)+ (5.585188 * h_63)+ (-0.36611056 * h_64)+ (33.26546 * h_65)+ (6.912748 * h_66)+ (2.1620715 * h_67)+ (29.849749 * h_68)+ (6.966547 * h_69)+ (16.005062 * h_70)+ (3.2989228 * h_71)+ (4.5980854 * h_72)+ (1.0776417 * h_73)+ (4.420276 * h_74)+ (6.996999 * h_75)+ (13.0 * h_76)+ (0.0 * h_77) + -1.4562547

    

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
        model_cap = 2652
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
