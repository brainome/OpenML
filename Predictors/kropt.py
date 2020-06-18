#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/3621/dataset_188_kropt.arff -o Predictors/kropt_NN.py -target game -stopat 88.47 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 3:40:50.86. Finished on: Jun-08-2020 21:10:49.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        18-way classifier
Best-guess accuracy:                25.19%
Model accuracy:                     39.23% (11008/28056 correct)
Improvement over best guess:        14.04% (of possible 74.81%)
Model capacity (MEC):               448 bits
Generalization ratio:               24.57 bits/bit
Confusion Matrix:
 [7.02% 0.00% 0.01% 0.03% 0.00% 0.10% 0.04% 0.21% 0.00% 0.11% 0.29% 0.00%
  0.22% 0.04% 0.18% 1.43% 0.27% 0.00%]
 [0.01% 0.02% 0.01% 0.02% 0.00% 0.00% 0.00% 0.03% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.12% 0.04% 0.00% 0.00% 0.05% 0.00% 0.01% 0.04% 0.02% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.02% 0.00% 0.00% 0.65% 0.00% 0.01% 0.09% 0.02% 0.00% 0.02% 0.03% 0.01%
  0.01% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.03% 0.00% 0.00% 0.06% 0.00% 0.05% 0.06% 0.05% 0.00% 0.02% 0.00% 0.01%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.10% 0.00% 0.00% 0.00% 0.00% 0.30% 0.12% 0.14% 0.00% 0.04% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.21% 0.00% 0.00% 0.01% 0.00% 0.25% 0.58% 0.24% 0.01% 0.25% 0.05% 0.02%
  0.04% 0.01% 0.01% 0.00% 0.00% 0.00%]
 [0.37% 0.00% 0.00% 0.00% 0.00% 0.09% 0.29% 0.67% 0.00% 0.35% 0.08% 0.09%
  0.11% 0.02% 0.04% 0.00% 0.00% 0.00%]
 [0.32% 0.00% 0.00% 0.02% 0.00% 0.02% 0.15% 0.16% 0.05% 0.76% 0.21% 0.15%
  0.32% 0.08% 0.10% 0.06% 0.02% 0.00%]
 [0.67% 0.00% 0.00% 0.02% 0.02% 0.01% 0.15% 0.19% 0.06% 2.47% 0.53% 0.13%
  0.40% 0.11% 0.11% 0.16% 0.07% 0.00%]
 [0.91% 0.00% 0.00% 0.09% 0.00% 0.00% 0.03% 0.14% 0.03% 1.14% 1.24% 0.57%
  1.18% 0.37% 0.15% 0.18% 0.09% 0.00%]
 [1.28% 0.00% 0.00% 0.03% 0.00% 0.00% 0.04% 0.12% 0.01% 0.68% 0.71% 1.03%
  1.37% 0.85% 0.38% 0.42% 0.14% 0.01%]
 [1.83% 0.00% 0.00% 0.02% 0.00% 0.00% 0.00% 0.04% 0.02% 0.22% 0.47% 0.52%
  2.71% 2.17% 1.11% 0.89% 0.14% 0.03%]
 [2.38% 0.00% 0.00% 0.01% 0.00% 0.00% 0.01% 0.02% 0.02% 0.10% 0.16% 0.38%
  1.30% 4.44% 2.03% 1.81% 0.15% 0.00%]
 [2.71% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01% 0.00% 0.06% 0.04% 0.16%
  0.35% 1.91% 5.18% 4.12% 0.36% 0.04%]
 [2.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.02% 0.03% 0.16%
  0.11% 0.47% 2.17% 9.87% 1.32% 0.06%]
 [0.82% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01% 0.01% 0.03%
  0.02% 0.06% 0.19% 3.89% 2.54% 0.15%]
 [0.16% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01%
  0.00% 0.00% 0.00% 0.53% 0.34% 0.34%]

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
TRAINFILE = "dataset_188_kropt.csv"


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
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="game"


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
    h_0 = max((((-5.4383416 * float(x[0]))+ (-2.3751092 * float(x[1]))+ (-0.00016159378 * float(x[2]))+ (0.8273832 * float(x[3]))+ (-0.05191514 * float(x[4]))+ (-1.5330273 * float(x[5]))) + 2.4080877), 0)
    h_1 = max((((0.030399367 * float(x[0]))+ (0.1870197 * float(x[1]))+ (0.016716165 * float(x[2]))+ (0.21960585 * float(x[3]))+ (0.008024881 * float(x[4]))+ (-0.5155685 * float(x[5]))) + 0.79822254), 0)
    h_2 = max((((-0.4500272 * float(x[0]))+ (-1.3126651 * float(x[1]))+ (0.47082183 * float(x[2]))+ (-0.66584903 * float(x[3]))+ (-0.47097716 * float(x[4]))+ (0.66603893 * float(x[5]))) + -1.2387958), 0)
    h_3 = max((((0.002447926 * float(x[0]))+ (-0.0006866755 * float(x[1]))+ (0.08910594 * float(x[2]))+ (1.4464507 * float(x[3]))+ (-0.1224822 * float(x[4]))+ (-1.4462209 * float(x[5]))) + -1.8113295), 0)
    h_4 = max((((-1.5903125 * float(x[0]))+ (-3.4630039 * float(x[1]))+ (0.044423424 * float(x[2]))+ (0.32430595 * float(x[3]))+ (1.3107618 * float(x[4]))+ (-2.7460022 * float(x[5]))) + -4.5643005), 0)
    h_5 = max((((-2.1096253 * float(x[0]))+ (-2.8391643 * float(x[1]))+ (1.2713808 * float(x[2]))+ (0.17725314 * float(x[3]))+ (0.05107836 * float(x[4]))+ (-2.15677 * float(x[5]))) + -4.3590946), 0)
    h_6 = max((((0.6393881 * float(x[0]))+ (-1.365707 * float(x[1]))+ (0.00035635525 * float(x[2]))+ (0.65482706 * float(x[3]))+ (-0.0026375442 * float(x[4]))+ (-0.6953398 * float(x[5]))) + -1.0124768), 0)
    h_7 = max((((0.120848216 * float(x[0]))+ (-1.279101 * float(x[1]))+ (0.005145123 * float(x[2]))+ (0.8307891 * float(x[3]))+ (-0.002157 * float(x[4]))+ (-1.0355695 * float(x[5]))) + -0.8219294), 0)
    h_8 = max((((-1.8434628 * float(x[0]))+ (-2.845664 * float(x[1]))+ (-0.719937 * float(x[2]))+ (0.877445 * float(x[3]))+ (-0.72036964 * float(x[4]))+ (-0.6315792 * float(x[5]))) + -0.036671713), 0)
    h_9 = max((((-4.5416064 * float(x[0]))+ (0.07846285 * float(x[1]))+ (0.03349808 * float(x[2]))+ (1.7432721 * float(x[3]))+ (0.011032507 * float(x[4]))+ (-2.2745943 * float(x[5]))) + -2.8359697), 0)
    h_10 = max((((0.08438311 * float(x[0]))+ (-1.5175023 * float(x[1]))+ (0.0034299481 * float(x[2]))+ (0.635277 * float(x[3]))+ (-0.013153296 * float(x[4]))+ (-1.0381638 * float(x[5]))) + 0.77151966), 0)
    h_11 = max((((-3.5337896 * float(x[0]))+ (-4.4021254 * float(x[1]))+ (-2.7953265 * float(x[2]))+ (1.4801883 * float(x[3]))+ (0.27341154 * float(x[4]))+ (-3.9433537 * float(x[5]))) + -1.3611404), 0)
    h_12 = max((((-1.3636005 * float(x[0]))+ (-0.9763895 * float(x[1]))+ (-0.093027115 * float(x[2]))+ (0.40635848 * float(x[3]))+ (-0.20393477 * float(x[4]))+ (-0.41907367 * float(x[5]))) + -1.8565418), 0)
    h_13 = max((((-0.35397694 * float(x[0]))+ (-0.020245697 * float(x[1]))+ (0.005976237 * float(x[2]))+ (0.84990907 * float(x[3]))+ (0.004615876 * float(x[4]))+ (-0.563616 * float(x[5]))) + -1.4445964), 0)
    h_14 = max((((-1.9643492 * float(x[0]))+ (-2.5838847 * float(x[1]))+ (-5.339273 * float(x[2]))+ (1.3305576 * float(x[3]))+ (-0.84262794 * float(x[4]))+ (-4.0078964 * float(x[5]))) + 1.2822913), 0)
    h_15 = max((((0.00902665 * float(x[0]))+ (-0.06650374 * float(x[1]))+ (-0.008475303 * float(x[2]))+ (0.012528497 * float(x[3]))+ (0.8857356 * float(x[4]))+ (0.05241719 * float(x[5]))) + -5.4534698), 0)
    h_16 = max((((-2.0574985 * float(x[0]))+ (-3.3605535 * float(x[1]))+ (-1.1543418 * float(x[2]))+ (1.2304368 * float(x[3]))+ (-0.28057474 * float(x[4]))+ (-2.3078635 * float(x[5]))) + -0.7188179), 0)
    h_17 = max((((-1.9179118 * float(x[0]))+ (-1.5986754 * float(x[1]))+ (-0.79994583 * float(x[2]))+ (0.45105663 * float(x[3]))+ (0.88610756 * float(x[4]))+ (-0.68326116 * float(x[5]))) + -5.2623053), 0)
    h_18 = max((((0.00028138093 * float(x[0]))+ (6.812937e-05 * float(x[1]))+ (-0.008653825 * float(x[2]))+ (0.97173804 * float(x[3]))+ (0.008659206 * float(x[4]))+ (-0.5510307 * float(x[5]))) + -3.9534948), 0)
    h_19 = max((((-6.1962843 * float(x[0]))+ (0.06751537 * float(x[1]))+ (0.027041903 * float(x[2]))+ (1.766492 * float(x[3]))+ (-0.081417926 * float(x[4]))+ (-3.0094604 * float(x[5]))) + -0.64870685), 0)
    h_20 = max((((-2.5446165 * float(x[0]))+ (-0.19073062 * float(x[1]))+ (-0.0039349943 * float(x[2]))+ (0.48241502 * float(x[3]))+ (0.31788132 * float(x[4]))+ (-0.39777088 * float(x[5]))) + -1.6268482), 0)
    h_21 = max((((0.0041499827 * float(x[0]))+ (-0.016855586 * float(x[1]))+ (0.0004543657 * float(x[2]))+ (0.0034867732 * float(x[3]))+ (0.256703 * float(x[4]))+ (0.025676906 * float(x[5]))) + -1.4971527), 0)
    h_22 = max((((0.0003387359 * float(x[0]))+ (-1.9959899 * float(x[1]))+ (-0.022003809 * float(x[2]))+ (0.56244624 * float(x[3]))+ (0.022141797 * float(x[4]))+ (-0.5617071 * float(x[5]))) + 1.3397205), 0)
    h_23 = max((((-4.202087 * float(x[0]))+ (-0.062036373 * float(x[1]))+ (-0.004154619 * float(x[2]))+ (0.010052439 * float(x[3]))+ (0.42681634 * float(x[4]))+ (0.064779855 * float(x[5]))) + -2.0398667), 0)
    h_24 = max((((-0.3784026 * float(x[0]))+ (-0.6382655 * float(x[1]))+ (-3.7510602 * float(x[2]))+ (0.59581447 * float(x[3]))+ (-0.001122647 * float(x[4]))+ (-3.10019 * float(x[5]))) + 2.7066655), 0)
    h_25 = max((((-0.54992336 * float(x[0]))+ (-0.7404856 * float(x[1]))+ (-0.52731407 * float(x[2]))+ (-0.16516177 * float(x[3]))+ (0.34236065 * float(x[4]))+ (-0.3534531 * float(x[5]))) + -1.1391416), 0)
    h_26 = max((((0.32773772 * float(x[0]))+ (-0.32596937 * float(x[1]))+ (0.035876893 * float(x[2]))+ (1.5690975 * float(x[3]))+ (-0.044774503 * float(x[4]))+ (-1.5641352 * float(x[5]))) + -2.3246703), 0)
    h_27 = max((((-3.136434 * float(x[0]))+ (0.015278082 * float(x[1]))+ (-3.1975873 * float(x[2]))+ (0.51342493 * float(x[3]))+ (-0.021591382 * float(x[4]))+ (-1.5568773 * float(x[5]))) + 0.65896946), 0)
    h_28 = max((((-0.00052811956 * float(x[0]))+ (-0.0015301355 * float(x[1]))+ (-0.07366195 * float(x[2]))+ (-1.4873537 * float(x[3]))+ (0.071771085 * float(x[4]))+ (1.493274 * float(x[5]))) + -1.8296596), 0)
    h_29 = max((((0.004522826 * float(x[0]))+ (0.39402193 * float(x[1]))+ (0.021639254 * float(x[2]))+ (-0.030124933 * float(x[3]))+ (0.0053689205 * float(x[4]))+ (0.15364765 * float(x[5]))) + -1.3245906), 0)
    h_30 = max((((-0.16427882 * float(x[0]))+ (-0.83735853 * float(x[1]))+ (-0.34332952 * float(x[2]))+ (0.027361618 * float(x[3]))+ (0.021071345 * float(x[4]))+ (0.042655602 * float(x[5]))) + 2.410236), 0)
    h_31 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))) + 0.0), 0)
    o[0] = (10.969662 * h_0)+ (5.138938 * h_1)+ (-269.56262 * h_2)+ (-259.64993 * h_3)+ (-11.143427 * h_4)+ (55.694775 * h_5)+ (-1.3712833 * h_6)+ (-7.668498 * h_7)+ (-9.905409 * h_8)+ (-17.303234 * h_9)+ (-12.7867365 * h_10)+ (-12.805586 * h_11)+ (-9.252719 * h_12)+ (9.646574 * h_13)+ (-12.467041 * h_14)+ (5.0872717 * h_15)+ (-6.423334 * h_16)+ (-8.186686 * h_17)+ (-279.67194 * h_18)+ (40.074524 * h_19)+ (3.6824286 * h_20)+ (2.4972603 * h_21)+ (-189.49887 * h_22)+ (7.1787276 * h_23)+ (94.99832 * h_24)+ (-6.949244 * h_25)+ (122.51807 * h_26)+ (22.219181 * h_27)+ (-255.4333 * h_28)+ (10.419388 * h_29)+ (4.6535645 * h_30)+ (0.0 * h_31) + 3.620971
    o[1] = (12.096622 * h_0)+ (12.659002 * h_1)+ (-80.1939 * h_2)+ (10.074524 * h_3)+ (-16.133623 * h_4)+ (-57.737953 * h_5)+ (5.51962 * h_6)+ (8.255545 * h_7)+ (7.26082 * h_8)+ (14.018795 * h_9)+ (4.9008293 * h_10)+ (2.4644318 * h_11)+ (15.670833 * h_12)+ (3.3316638 * h_13)+ (7.660233 * h_14)+ (-113.81373 * h_15)+ (-10.80915 * h_16)+ (-16.106493 * h_17)+ (13.308075 * h_18)+ (17.142326 * h_19)+ (15.151203 * h_20)+ (-51.093678 * h_21)+ (6.6456575 * h_22)+ (15.081373 * h_23)+ (-17.284035 * h_24)+ (7.9700484 * h_25)+ (8.9744425 * h_26)+ (-71.834175 * h_27)+ (-88.9708 * h_28)+ (6.713444 * h_29)+ (-116.94257 * h_30)+ (0.0 * h_31) + -7.3510237
    o[2] = (8.576501 * h_0)+ (13.935918 * h_1)+ (-99.991104 * h_2)+ (12.315681 * h_3)+ (-18.562466 * h_4)+ (-104.99235 * h_5)+ (6.3462186 * h_6)+ (7.763891 * h_7)+ (-0.76155937 * h_8)+ (9.548626 * h_9)+ (11.21601 * h_10)+ (1.5790831 * h_11)+ (16.195381 * h_12)+ (1.919332 * h_13)+ (-36.21217 * h_14)+ (-19.977346 * h_15)+ (-29.799698 * h_16)+ (-121.06547 * h_17)+ (14.0428505 * h_18)+ (21.14854 * h_19)+ (7.1069584 * h_20)+ (14.270534 * h_21)+ (2.936353 * h_22)+ (19.244545 * h_23)+ (17.956139 * h_24)+ (9.26172 * h_25)+ (10.2686615 * h_26)+ (5.174284 * h_27)+ (-146.21 * h_28)+ (-1.0307997 * h_29)+ (4.7890973 * h_30)+ (0.0 * h_31) + -10.248235
    o[3] = (8.114291 * h_0)+ (18.37919 * h_1)+ (-108.49651 * h_2)+ (10.817759 * h_3)+ (-46.200035 * h_4)+ (17.053452 * h_5)+ (5.9589567 * h_6)+ (9.153381 * h_7)+ (-20.398104 * h_8)+ (13.484145 * h_9)+ (2.8184137 * h_10)+ (63.463512 * h_11)+ (12.361779 * h_12)+ (7.5408792 * h_13)+ (-53.158062 * h_14)+ (-151.76807 * h_15)+ (-54.812695 * h_16)+ (14.327819 * h_17)+ (13.974319 * h_18)+ (18.329927 * h_19)+ (1.1574503 * h_20)+ (-29.469566 * h_21)+ (12.826009 * h_22)+ (31.240911 * h_23)+ (17.221552 * h_24)+ (8.945344 * h_25)+ (8.376357 * h_26)+ (5.9837055 * h_27)+ (-6.9152765 * h_28)+ (-8.272526 * h_29)+ (4.009871 * h_30)+ (0.0 * h_31) + -13.951436
    o[4] = (2.707816 * h_0)+ (18.178467 * h_1)+ (-94.28094 * h_2)+ (6.582086 * h_3)+ (-10.535049 * h_4)+ (-90.10991 * h_5)+ (0.22304963 * h_6)+ (7.545707 * h_7)+ (0.44261694 * h_8)+ (7.7590966 * h_9)+ (18.169802 * h_10)+ (6.481681 * h_11)+ (11.364775 * h_12)+ (4.366532 * h_13)+ (5.560153 * h_14)+ (-141.05164 * h_15)+ (6.852643 * h_16)+ (-9.733165 * h_17)+ (14.691002 * h_18)+ (23.40224 * h_19)+ (2.9437335 * h_20)+ (-10.094598 * h_21)+ (6.2272024 * h_22)+ (22.239323 * h_23)+ (-142.58267 * h_24)+ (8.866596 * h_25)+ (13.702064 * h_26)+ (4.332243 * h_27)+ (-75.34887 * h_28)+ (2.6935594 * h_29)+ (4.4571853 * h_30)+ (0.0 * h_31) + -12.96542
    o[5] = (-125.51231 * h_0)+ (13.932995 * h_1)+ (-145.8877 * h_2)+ (9.061997 * h_3)+ (11.263051 * h_4)+ (-56.65471 * h_5)+ (-5.325275 * h_6)+ (11.812132 * h_7)+ (5.603808 * h_8)+ (9.150931 * h_9)+ (23.226536 * h_10)+ (10.835111 * h_11)+ (10.863873 * h_12)+ (-1.088675 * h_13)+ (-29.745579 * h_14)+ (-23.502604 * h_15)+ (10.232785 * h_16)+ (7.738918 * h_17)+ (15.24281 * h_18)+ (16.87374 * h_19)+ (24.70857 * h_20)+ (50.670074 * h_21)+ (1.9864873 * h_22)+ (-18.030462 * h_23)+ (17.669157 * h_24)+ (8.560122 * h_25)+ (13.976504 * h_26)+ (-120.49709 * h_27)+ (-146.36162 * h_28)+ (5.949381 * h_29)+ (4.9602256 * h_30)+ (0.0 * h_31) + -7.009919
    o[6] = (22.346434 * h_0)+ (11.269215 * h_1)+ (2.6011 * h_2)+ (10.836292 * h_3)+ (-140.63022 * h_4)+ (-80.79479 * h_5)+ (-2.0506613 * h_6)+ (13.595918 * h_7)+ (-18.429823 * h_8)+ (9.221067 * h_9)+ (11.000954 * h_10)+ (13.911133 * h_11)+ (11.29929 * h_12)+ (1.9290122 * h_13)+ (-71.54023 * h_14)+ (-17.755392 * h_15)+ (11.080348 * h_16)+ (-16.943218 * h_17)+ (13.406444 * h_18)+ (10.500447 * h_19)+ (11.8369255 * h_20)+ (36.38463 * h_21)+ (10.141204 * h_22)+ (3.466494 * h_23)+ (17.848364 * h_24)+ (8.209672 * h_25)+ (12.237576 * h_26)+ (-160.97214 * h_27)+ (5.9441376 * h_28)+ (9.864873 * h_29)+ (4.218239 * h_30)+ (0.0 * h_31) + -3.7035832
    o[7] = (-66.607346 * h_0)+ (9.915949 * h_1)+ (-123.01598 * h_2)+ (11.03367 * h_3)+ (-2.8963206 * h_4)+ (40.228245 * h_5)+ (-2.312648 * h_6)+ (9.266659 * h_7)+ (-33.879314 * h_8)+ (7.057776 * h_9)+ (19.073639 * h_10)+ (16.011007 * h_11)+ (12.097259 * h_12)+ (2.4786608 * h_13)+ (5.446292 * h_14)+ (-7.426208 * h_15)+ (12.872307 * h_16)+ (10.175884 * h_17)+ (14.028507 * h_18)+ (21.286737 * h_19)+ (6.62922 * h_20)+ (17.802364 * h_21)+ (6.533765 * h_22)+ (7.9486647 * h_23)+ (-45.9255 * h_24)+ (8.661179 * h_25)+ (11.942702 * h_26)+ (3.8643744 * h_27)+ (5.806166 * h_28)+ (11.081963 * h_29)+ (4.487074 * h_30)+ (0.0 * h_31) + -1.8440837
    o[8] = (6.458209 * h_0)+ (9.091059 * h_1)+ (6.6659427 * h_2)+ (13.591427 * h_3)+ (-243.83105 * h_4)+ (14.53595 * h_5)+ (3.3702085 * h_6)+ (8.961524 * h_7)+ (-48.86913 * h_8)+ (9.90116 * h_9)+ (11.514993 * h_10)+ (80.349724 * h_11)+ (15.845263 * h_12)+ (4.956849 * h_13)+ (-177.55475 * h_14)+ (-9.373134 * h_15)+ (-148.51654 * h_16)+ (-211.07076 * h_17)+ (14.312783 * h_18)+ (20.015974 * h_19)+ (7.0057635 * h_20)+ (25.928118 * h_21)+ (8.909296 * h_22)+ (6.6868734 * h_23)+ (17.078789 * h_24)+ (9.09474 * h_25)+ (8.226442 * h_26)+ (6.6794825 * h_27)+ (5.6728334 * h_28)+ (12.203809 * h_29)+ (5.132166 * h_30)+ (0.0 * h_31) + -1.6174256
    o[9] = (6.4045935 * h_0)+ (8.7800455 * h_1)+ (-70.8067 * h_2)+ (13.647349 * h_3)+ (-13.049249 * h_4)+ (14.466831 * h_5)+ (4.4213147 * h_6)+ (9.274824 * h_7)+ (-31.417088 * h_8)+ (10.9350815 * h_9)+ (7.828654 * h_10)+ (47.50004 * h_11)+ (16.794832 * h_12)+ (5.9507866 * h_13)+ (-28.807516 * h_14)+ (-5.4383 * h_15)+ (15.072526 * h_16)+ (-49.965385 * h_17)+ (13.431661 * h_18)+ (19.612047 * h_19)+ (5.9659796 * h_20)+ (18.853607 * h_21)+ (11.3687935 * h_22)+ (6.7995377 * h_23)+ (17.586803 * h_24)+ (9.068339 * h_25)+ (8.154216 * h_26)+ (5.081358 * h_27)+ (5.547486 * h_28)+ (13.140225 * h_29)+ (5.26955 * h_30)+ (0.0 * h_31) + -1.1426237
    o[10] = (5.174482 * h_0)+ (7.9276013 * h_1)+ (11.687481 * h_2)+ (13.918301 * h_3)+ (20.837803 * h_4)+ (-10.234342 * h_5)+ (3.4833286 * h_6)+ (5.5809674 * h_7)+ (17.916607 * h_8)+ (13.465171 * h_9)+ (14.17802 * h_10)+ (-1.9446176 * h_11)+ (11.908984 * h_12)+ (7.1066427 * h_13)+ (-80.961555 * h_14)+ (-5.258414 * h_15)+ (29.728815 * h_16)+ (15.811065 * h_17)+ (13.391826 * h_18)+ (17.89408 * h_19)+ (4.7965555 * h_20)+ (19.939611 * h_21)+ (10.790567 * h_22)+ (7.134615 * h_23)+ (17.163458 * h_24)+ (8.708976 * h_25)+ (7.3309054 * h_26)+ (5.3146973 * h_27)+ (5.480328 * h_28)+ (14.373335 * h_29)+ (5.242394 * h_30)+ (0.0 * h_31) + -0.7396612
    o[11] = (5.8663383 * h_0)+ (6.686847 * h_1)+ (12.000407 * h_2)+ (13.329612 * h_3)+ (-12.991983 * h_4)+ (-109.16213 * h_5)+ (5.8349123 * h_6)+ (4.544638 * h_7)+ (18.928783 * h_8)+ (14.842132 * h_9)+ (13.027385 * h_10)+ (-48.90837 * h_11)+ (16.39784 * h_12)+ (8.154806 * h_13)+ (-72.223145 * h_14)+ (-2.9970593 * h_15)+ (-91.55938 * h_16)+ (-138.7709 * h_17)+ (12.76831 * h_18)+ (16.798662 * h_19)+ (4.1938696 * h_20)+ (16.875875 * h_21)+ (10.509665 * h_22)+ (7.2996855 * h_23)+ (17.28877 * h_24)+ (8.892373 * h_25)+ (7.8926587 * h_26)+ (-241.4974 * h_27)+ (5.5025125 * h_28)+ (12.828739 * h_29)+ (4.6831884 * h_30)+ (0.0 * h_31) + 0.68307555
    o[12] = (6.850918 * h_0)+ (4.90201 * h_1)+ (12.387225 * h_2)+ (14.531642 * h_3)+ (5.0992184 * h_4)+ (15.674025 * h_5)+ (7.83787 * h_6)+ (4.5560856 * h_7)+ (15.967844 * h_8)+ (16.719217 * h_9)+ (10.582096 * h_10)+ (-64.371605 * h_11)+ (15.883255 * h_12)+ (8.077789 * h_13)+ (21.685753 * h_14)+ (-1.197359 * h_15)+ (-229.65497 * h_16)+ (18.135822 * h_17)+ (12.251922 * h_18)+ (14.978142 * h_19)+ (3.7938635 * h_20)+ (13.043587 * h_21)+ (11.391724 * h_22)+ (7.2469025 * h_23)+ (17.568756 * h_24)+ (8.853728 * h_25)+ (7.1707516 * h_26)+ (7.394085 * h_27)+ (5.5312223 * h_28)+ (12.3450365 * h_29)+ (4.2532544 * h_30)+ (0.0 * h_31) + 1.83716
    o[13] = (6.8663535 * h_0)+ (3.0635579 * h_1)+ (12.308556 * h_2)+ (14.32101 * h_3)+ (20.008944 * h_4)+ (-130.20929 * h_5)+ (8.11931 * h_6)+ (5.007527 * h_7)+ (-29.875715 * h_8)+ (17.260506 * h_9)+ (9.822697 * h_10)+ (16.506937 * h_11)+ (12.185591 * h_12)+ (8.899994 * h_13)+ (23.11849 * h_14)+ (2.3527713 * h_15)+ (-1.305676 * h_16)+ (4.8583965 * h_17)+ (11.195474 * h_18)+ (14.627361 * h_19)+ (3.4921236 * h_20)+ (6.820171 * h_21)+ (11.415084 * h_22)+ (7.5355263 * h_23)+ (18.625093 * h_24)+ (8.825079 * h_25)+ (7.653624 * h_26)+ (5.1958785 * h_27)+ (5.578021 * h_28)+ (10.743235 * h_29)+ (3.465925 * h_30)+ (0.0 * h_31) + 3.06
    o[14] = (7.58649 * h_0)+ (1.3739249 * h_1)+ (12.240843 * h_2)+ (13.911388 * h_3)+ (19.262432 * h_4)+ (-123.66034 * h_5)+ (8.596493 * h_6)+ (5.0421 * h_7)+ (15.445069 * h_8)+ (17.941393 * h_9)+ (10.771032 * h_10)+ (8.700178 * h_11)+ (15.868739 * h_12)+ (9.670601 * h_13)+ (-13.269779 * h_14)+ (7.6640806 * h_15)+ (-77.38762 * h_16)+ (-110.59456 * h_17)+ (10.195483 * h_18)+ (13.826825 * h_19)+ (3.1000984 * h_20)+ (-4.1345205 * h_21)+ (10.227451 * h_22)+ (8.249723 * h_23)+ (-131.04625 * h_24)+ (8.78415 * h_25)+ (8.24865 * h_26)+ (9.07345 * h_27)+ (5.699071 * h_28)+ (8.044676 * h_29)+ (3.6529558 * h_30)+ (0.0 * h_31) + 3.9865003
    o[15] = (7.0165796 * h_0)+ (-0.022468515 * h_1)+ (12.252286 * h_2)+ (13.419605 * h_3)+ (-195.32173 * h_4)+ (-51.756172 * h_5)+ (8.824888 * h_6)+ (8.254172 * h_7)+ (18.508024 * h_8)+ (17.522415 * h_9)+ (7.0433064 * h_10)+ (-14.876976 * h_11)+ (14.2676735 * h_12)+ (10.335767 * h_13)+ (-72.81131 * h_14)+ (12.645046 * h_15)+ (26.253284 * h_16)+ (20.1762 * h_17)+ (8.7512045 * h_18)+ (14.313478 * h_19)+ (3.2967298 * h_20)+ (-13.656039 * h_21)+ (10.534669 * h_22)+ (8.15176 * h_23)+ (17.990166 * h_24)+ (16.503174 * h_25)+ (9.0883255 * h_26)+ (5.842355 * h_27)+ (5.9042697 * h_28)+ (2.5300179 * h_29)+ (3.8141453 * h_30)+ (0.0 * h_31) + 4.3301907
    o[16] = (9.722958 * h_0)+ (-0.8323024 * h_1)+ (11.297134 * h_2)+ (12.1012125 * h_3)+ (-136.70201 * h_4)+ (-33.4353 * h_5)+ (8.064348 * h_6)+ (11.01767 * h_7)+ (16.519753 * h_8)+ (17.661837 * h_9)+ (4.4427924 * h_10)+ (0.8297278 * h_11)+ (14.020972 * h_12)+ (10.799414 * h_13)+ (-8.038619 * h_14)+ (21.194479 * h_15)+ (-43.939316 * h_16)+ (17.066956 * h_17)+ (6.94106 * h_18)+ (13.028757 * h_19)+ (3.2293267 * h_20)+ (-31.905336 * h_21)+ (11.186146 * h_22)+ (7.767453 * h_23)+ (-212.0704 * h_24)+ (16.321325 * h_25)+ (10.732751 * h_26)+ (10.91073 * h_27)+ (6.021362 * h_28)+ (-10.2644 * h_29)+ (4.08113 * h_30)+ (0.0 * h_31) + 3.7025049
    o[17] = (-61.877575 * h_0)+ (-3.4068024 * h_1)+ (-85.869446 * h_2)+ (11.412305 * h_3)+ (12.02264 * h_4)+ (9.098438 * h_5)+ (10.570935 * h_6)+ (15.931639 * h_7)+ (-41.946476 * h_8)+ (-141.03474 * h_9)+ (-0.7569162 * h_10)+ (8.545741 * h_11)+ (14.586383 * h_12)+ (10.032577 * h_13)+ (13.981398 * h_14)+ (19.1627 * h_15)+ (11.782029 * h_16)+ (14.763794 * h_17)+ (4.40988 * h_18)+ (10.947718 * h_19)+ (-33.069935 * h_20)+ (-26.908752 * h_21)+ (14.319061 * h_22)+ (-188.21542 * h_23)+ (-92.45705 * h_24)+ (8.600572 * h_25)+ (10.8485985 * h_26)+ (2.8836074 * h_27)+ (5.8366714 * h_28)+ (-36.074562 * h_29)+ (4.1341457 * h_30)+ (0.0 * h_31) + 2.4632897

    

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
        model_cap = 448
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
