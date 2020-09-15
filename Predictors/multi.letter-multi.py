#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class letter-multi.csv -o letter-multi_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 1:34:56.49. Finished on: Sep-08-2020 16:22:59.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         26-way classifier
Best-guess accuracy:                 4.06%
Overall Model accuracy:              85.02% (17005/20000 correct)
Overall Improvement over best guess: 80.96% (of possible 95.94%)
Model capacity (MEC):                510 bits
Generalization ratio:                33.34 bits/bit
Model efficiency:                    0.15%/parameter
Confusion Matrix:
 [3.31% 0.00% 0.07% 0.00% 0.01% 0.00% 0.02% 0.00% 0.00% 0.00% 0.01% 0.00%
  0.06% 0.00% 0.04% 0.02% 0.03% 0.00% 0.01% 0.00% 0.00% 0.02% 0.00% 0.00%
  0.00% 0.08%]
 [0.00% 3.45% 0.01% 0.02% 0.19% 0.02% 0.01% 0.00% 0.04% 0.01% 0.00% 0.00%
  0.01% 0.01% 0.02% 0.01% 0.00% 0.07% 0.01% 0.01% 0.00% 0.00% 0.00% 0.07%
  0.06% 0.00%]
 [0.12% 0.00% 2.98% 0.00% 0.08% 0.00% 0.04% 0.00% 0.01% 0.01% 0.01% 0.02%
  0.14% 0.00% 0.03% 0.07% 0.03% 0.03% 0.01% 0.00% 0.00% 0.01% 0.00% 0.02%
  0.10% 0.03%]
 [0.01% 0.01% 0.01% 2.48% 0.06% 0.04% 0.25% 0.02% 0.18% 0.08% 0.00% 0.14%
  0.00% 0.06% 0.03% 0.02% 0.03% 0.01% 0.00% 0.01% 0.08% 0.01% 0.06% 0.01%
  0.07% 0.00%]
 [0.01% 0.10% 0.04% 0.07% 3.26% 0.02% 0.01% 0.00% 0.02% 0.00% 0.01% 0.01%
  0.03% 0.00% 0.00% 0.00% 0.01% 0.02% 0.03% 0.01% 0.00% 0.08% 0.03% 0.03%
  0.06% 0.03%]
 [0.00% 0.01% 0.00% 0.08% 0.01% 3.40% 0.04% 0.06% 0.04% 0.03% 0.01% 0.01%
  0.00% 0.11% 0.00% 0.00% 0.00% 0.01% 0.00% 0.03% 0.04% 0.00% 0.00% 0.00%
  0.01% 0.00%]
 [0.00% 0.01% 0.00% 0.11% 0.01% 0.04% 3.05% 0.03% 0.11% 0.03% 0.00% 0.10%
  0.01% 0.03% 0.03% 0.00% 0.01% 0.00% 0.00% 0.00% 0.00% 0.01% 0.01% 0.04%
  0.14% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.04% 0.09% 3.66% 0.00% 0.00% 0.03% 0.00%
  0.00% 0.01% 0.00% 0.00% 0.00% 0.00% 0.00% 0.06% 0.03% 0.00% 0.01% 0.00%
  0.03% 0.00%]
 [0.01% 0.01% 0.01% 0.11% 0.01% 0.04% 0.06% 0.01% 3.31% 0.00% 0.01% 0.02%
  0.00% 0.06% 0.01% 0.00% 0.01% 0.01% 0.00% 0.00% 0.00% 0.03% 0.00% 0.01%
  0.24% 0.03%]
 [0.00% 0.06% 0.01% 0.01% 0.02% 0.01% 0.03% 0.01% 0.00% 3.40% 0.00% 0.00%
  0.00% 0.02% 0.00% 0.00% 0.00% 0.08% 0.00% 0.06% 0.02% 0.03% 0.00% 0.03%
  0.06% 0.00%]
 [0.01% 0.01% 0.01% 0.03% 0.00% 0.01% 0.03% 0.01% 0.01% 0.00% 3.65% 0.03%
  0.00% 0.01% 0.01% 0.01% 0.01% 0.04% 0.01% 0.00% 0.03% 0.01% 0.00% 0.00%
  0.00% 0.01%]
 [0.00% 0.00% 0.01% 0.08% 0.00% 0.01% 0.18% 0.01% 0.01% 0.02% 0.01% 3.06%
  0.04% 0.00% 0.00% 0.04% 0.14% 0.00% 0.00% 0.00% 0.05% 0.00% 0.01% 0.00%
  0.02% 0.00%]
 [0.08% 0.00% 0.03% 0.01% 0.03% 0.00% 0.04% 0.00% 0.02% 0.00% 0.00% 0.03%
  3.11% 0.00% 0.08% 0.03% 0.03% 0.00% 0.01% 0.00% 0.00% 0.04% 0.02% 0.20%
  0.06% 0.00%]
 [0.00% 0.03% 0.00% 0.04% 0.00% 0.02% 0.04% 0.01% 0.06% 0.00% 0.05% 0.00%
  0.00% 3.19% 0.10% 0.00% 0.01% 0.00% 0.01% 0.10% 0.01% 0.01% 0.01% 0.07%
  0.01% 0.00%]
 [0.05% 0.01% 0.01% 0.01% 0.00% 0.00% 0.01% 0.01% 0.04% 0.01% 0.09% 0.00%
  0.01% 0.17% 3.31% 0.03% 0.01% 0.01% 0.01% 0.01% 0.01% 0.00% 0.03% 0.07%
  0.03% 0.01%]
 [0.00% 0.00% 0.03% 0.04% 0.00% 0.00% 0.01% 0.00% 0.01% 0.00% 0.02% 0.03%
  0.07% 0.01% 0.06% 3.24% 0.10% 0.03% 0.01% 0.00% 0.00% 0.06% 0.01% 0.08%
  0.02% 0.00%]
 [0.02% 0.00% 0.03% 0.01% 0.03% 0.00% 0.00% 0.00% 0.03% 0.01% 0.01% 0.10%
  0.06% 0.00% 0.03% 0.03% 3.50% 0.01% 0.03% 0.00% 0.01% 0.00% 0.00% 0.02%
  0.01% 0.00%]
 [0.01% 0.05% 0.01% 0.01% 0.07% 0.00% 0.00% 0.01% 0.03% 0.10% 0.03% 0.01%
  0.01% 0.00% 0.03% 0.01% 0.03% 3.40% 0.00% 0.01% 0.01% 0.10% 0.00% 0.01%
  0.00% 0.01%]
 [0.05% 0.06% 0.08% 0.00% 0.04% 0.00% 0.00% 0.00% 0.03% 0.00% 0.01% 0.00%
  0.01% 0.00% 0.01% 0.01% 0.05% 0.01% 3.31% 0.00% 0.00% 0.01% 0.02% 0.00%
  0.03% 0.07%]
 [0.00% 0.01% 0.00% 0.01% 0.00% 0.01% 0.01% 0.11% 0.00% 0.01% 0.00% 0.00%
  0.00% 0.03% 0.01% 0.00% 0.00% 0.00% 0.00% 3.48% 0.04% 0.01% 0.00% 0.02%
  0.01% 0.00%]
 [0.00% 0.00% 0.00% 0.06% 0.00% 0.07% 0.01% 0.09% 0.01% 0.01% 0.03% 0.04%
  0.00% 0.07% 0.02% 0.00% 0.00% 0.00% 0.00% 0.09% 3.55% 0.00% 0.01% 0.00%
  0.00% 0.00%]
 [0.04% 0.00% 0.04% 0.07% 0.06% 0.00% 0.01% 0.00% 0.04% 0.03% 0.00% 0.03%
  0.06% 0.00% 0.01% 0.02% 0.04% 0.06% 0.00% 0.00% 0.01% 3.38% 0.00% 0.04%
  0.01% 0.03%]
 [0.00% 0.01% 0.00% 0.02% 0.03% 0.00% 0.00% 0.01% 0.00% 0.00% 0.00% 0.02%
  0.09% 0.07% 0.01% 0.03% 0.01% 0.00% 0.01% 0.00% 0.04% 0.01% 3.22% 0.10%
  0.00% 0.00%]
 [0.00% 0.06% 0.06% 0.01% 0.01% 0.00% 0.04% 0.01% 0.02% 0.04% 0.00% 0.05%
  0.01% 0.07% 0.12% 0.03% 0.02% 0.00% 0.02% 0.01% 0.00% 0.00% 0.18% 3.03%
  0.06% 0.02%]
 [0.00% 0.01% 0.11% 0.03% 0.03% 0.01% 0.19% 0.00% 0.12% 0.03% 0.00% 0.02%
  0.03% 0.00% 0.00% 0.01% 0.00% 0.04% 0.03% 0.00% 0.01% 0.01% 0.00% 0.02%
  3.15% 0.00%]
 [0.10% 0.00% 0.03% 0.03% 0.02% 0.00% 0.01% 0.00% 0.03% 0.00% 0.07% 0.00%
  0.01% 0.03% 0.01% 0.01% 0.08% 0.00% 0.14% 0.00% 0.01% 0.01% 0.00% 0.00%
  0.01% 3.15%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to 'Z'=0, 'P'=1, 'S'=2, 'H'=3, 'F'=4, 'N'=5, 'R'=6, 'M'=7, 'D'=8, 'V'=9, 'A'=10, 'K'=11, 'E'=12, 'O'=13, 'Q'=14, 'L'=15, 'X'=16, 'Y'=17, 'I'=18, 'W'=19, 'U'=20, 'T'=21, 'C'=22, 'G'=23, 'B'=24, 'J'=25.
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


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "letter-multi.csv"


#Number of output logits
num_output_logits = 26

#Number of attributes
num_attr = 16
n_classes = 26


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
    clean.mapping={'Z': 0, 'P': 1, 'S': 2, 'H': 3, 'F': 4, 'N': 5, 'R': 6, 'M': 7, 'D': 8, 'V': 9, 'A': 10, 'K': 11, 'E': 12, 'O': 13, 'Q': 14, 'L': 15, 'X': 16, 'Y': 17, 'I': 18, 'W': 19, 'U': 20, 'T': 21, 'C': 22, 'G': 23, 'B': 24, 'J': 25}

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
    x = row
    o = [0] * num_output_logits
    h_0 = max((((0.09793622 * float(x[0]))+ (-0.06735979 * float(x[1]))+ (-0.52026993 * float(x[2]))+ (0.28699428 * float(x[3]))+ (0.19801016 * float(x[4]))+ (0.35479996 * float(x[5]))+ (-0.6302399 * float(x[6]))+ (1.0899372 * float(x[7]))+ (0.44819355 * float(x[8]))+ (0.90224314 * float(x[9]))+ (-0.0145574035 * float(x[10]))+ (-0.41580343 * float(x[11]))+ (-0.88187784 * float(x[12]))+ (0.060788374 * float(x[13]))+ (0.27212292 * float(x[14]))+ (-0.7257857 * float(x[15]))) + 1.323824), 0)
    h_1 = max((((-0.058642633 * float(x[0]))+ (-0.0038956853 * float(x[1]))+ (0.27397764 * float(x[2]))+ (-0.21577357 * float(x[3]))+ (0.28143528 * float(x[4]))+ (-0.47086143 * float(x[5]))+ (0.8903598 * float(x[6]))+ (0.9749646 * float(x[7]))+ (-0.2692911 * float(x[8]))+ (0.53774905 * float(x[9]))+ (-0.54149836 * float(x[10]))+ (-0.2905106 * float(x[11]))+ (0.6676953 * float(x[12]))+ (-0.19697289 * float(x[13]))+ (-0.9477805 * float(x[14]))+ (0.50631416 * float(x[15]))) + -0.39049762), 0)
    h_2 = max((((0.016908394 * float(x[0]))+ (0.20357169 * float(x[1]))+ (0.027075166 * float(x[2]))+ (-0.5218375 * float(x[3]))+ (0.1872053 * float(x[4]))+ (-0.002962965 * float(x[5]))+ (-0.20347334 * float(x[6]))+ (0.76057553 * float(x[7]))+ (1.3923627 * float(x[8]))+ (-0.03140361 * float(x[9]))+ (0.039201528 * float(x[10]))+ (0.220235 * float(x[11]))+ (0.49070877 * float(x[12]))+ (-0.040462956 * float(x[13]))+ (-1.4786536 * float(x[14]))+ (-0.5247859 * float(x[15]))) + 1.1571441), 0)
    h_3 = max((((-0.48453605 * float(x[0]))+ (-0.047060568 * float(x[1]))+ (-0.24252284 * float(x[2]))+ (-0.03427854 * float(x[3]))+ (0.85658944 * float(x[4]))+ (0.19234642 * float(x[5]))+ (-0.15310194 * float(x[6]))+ (-0.46509928 * float(x[7]))+ (0.83304405 * float(x[8]))+ (-0.29585195 * float(x[9]))+ (0.78726363 * float(x[10]))+ (-0.5712398 * float(x[11]))+ (-0.32226112 * float(x[12]))+ (0.41165298 * float(x[13]))+ (0.15791844 * float(x[14]))+ (-0.42253175 * float(x[15]))) + 2.537034), 0)
    h_4 = max((((0.15941858 * float(x[0]))+ (0.14164148 * float(x[1]))+ (-0.62894064 * float(x[2]))+ (0.4472249 * float(x[3]))+ (-0.4348632 * float(x[4]))+ (0.19778438 * float(x[5]))+ (0.30785426 * float(x[6]))+ (0.082238466 * float(x[7]))+ (1.4696794 * float(x[8]))+ (-0.064994045 * float(x[9]))+ (-0.1875036 * float(x[10]))+ (0.19548601 * float(x[11]))+ (-0.5514891 * float(x[12]))+ (-0.27779406 * float(x[13]))+ (-0.23336153 * float(x[14]))+ (-0.049508654 * float(x[15]))) + 1.613091), 0)
    h_5 = max((((0.75909257 * float(x[0]))+ (-0.14001527 * float(x[1]))+ (-0.8701853 * float(x[2]))+ (0.34626314 * float(x[3]))+ (-0.003036859 * float(x[4]))+ (-0.32945797 * float(x[5]))+ (-0.2570645 * float(x[6]))+ (0.21574305 * float(x[7]))+ (0.4896044 * float(x[8]))+ (0.41517287 * float(x[9]))+ (0.6743113 * float(x[10]))+ (-0.83946806 * float(x[11]))+ (-0.3344408 * float(x[12]))+ (0.78936255 * float(x[13]))+ (0.059172988 * float(x[14]))+ (0.4740281 * float(x[15]))) + -1.463601), 0)
    h_6 = max((((-0.008469452 * float(x[0]))+ (-0.41697887 * float(x[1]))+ (0.7105127 * float(x[2]))+ (0.3545181 * float(x[3]))+ (0.06226111 * float(x[4]))+ (0.78672063 * float(x[5]))+ (0.6274386 * float(x[6]))+ (0.66180503 * float(x[7]))+ (-0.74653405 * float(x[8]))+ (-0.9945521 * float(x[9]))+ (0.23319735 * float(x[10]))+ (0.12227544 * float(x[11]))+ (0.6096295 * float(x[12]))+ (0.07228634 * float(x[13]))+ (-1.0877477 * float(x[14]))+ (-0.15564053 * float(x[15]))) + 1.0334257), 0)
    h_7 = max((((-0.13355887 * float(x[0]))+ (0.17577884 * float(x[1]))+ (0.2509618 * float(x[2]))+ (0.25629494 * float(x[3]))+ (-0.44335893 * float(x[4]))+ (-0.7195018 * float(x[5]))+ (-0.44240543 * float(x[6]))+ (0.43203583 * float(x[7]))+ (-0.36092752 * float(x[8]))+ (0.25400582 * float(x[9]))+ (-0.22841719 * float(x[10]))+ (0.44607863 * float(x[11]))+ (0.16585726 * float(x[12]))+ (0.0004379735 * float(x[13]))+ (-0.33105195 * float(x[14]))+ (0.9044097 * float(x[15]))) + 3.287738), 0)
    h_8 = max((((-0.42234534 * float(x[0]))+ (0.1865626 * float(x[1]))+ (-0.22121887 * float(x[2]))+ (-0.68027675 * float(x[3]))+ (0.9555312 * float(x[4]))+ (-0.20856903 * float(x[5]))+ (-0.732635 * float(x[6]))+ (0.7157789 * float(x[7]))+ (0.26754966 * float(x[8]))+ (0.1834897 * float(x[9]))+ (-0.12110228 * float(x[10]))+ (0.5254832 * float(x[11]))+ (0.50506157 * float(x[12]))+ (-0.3822635 * float(x[13]))+ (0.45044678 * float(x[14]))+ (0.50614595 * float(x[15]))) + -1.3837354), 0)
    h_9 = max((((-0.07770641 * float(x[0]))+ (-0.08321381 * float(x[1]))+ (0.25345024 * float(x[2]))+ (-0.41762388 * float(x[3]))+ (0.54028153 * float(x[4]))+ (0.56242526 * float(x[5]))+ (-0.13418043 * float(x[6]))+ (1.253808 * float(x[7]))+ (-0.13672443 * float(x[8]))+ (-0.45727405 * float(x[9]))+ (-0.077973075 * float(x[10]))+ (-0.2020327 * float(x[11]))+ (0.7462373 * float(x[12]))+ (-0.14834332 * float(x[13]))+ (-0.24948825 * float(x[14]))+ (0.05770888 * float(x[15]))) + 4.6412134), 0)
    h_10 = max((((0.8133791 * float(x[0]))+ (0.22550996 * float(x[1]))+ (-0.4070819 * float(x[2]))+ (0.2892084 * float(x[3]))+ (-0.9349221 * float(x[4]))+ (0.37858865 * float(x[5]))+ (0.18528011 * float(x[6]))+ (0.5735024 * float(x[7]))+ (-0.034955464 * float(x[8]))+ (0.23559174 * float(x[9]))+ (-0.15707564 * float(x[10]))+ (0.07243272 * float(x[11]))+ (-2.9856386 * float(x[12]))+ (0.19765086 * float(x[13]))+ (-0.77521586 * float(x[14]))+ (-0.2949874 * float(x[15]))) + 0.18160053), 0)
    h_11 = max((((0.044785712 * float(x[0]))+ (-0.0829273 * float(x[1]))+ (0.5745895 * float(x[2]))+ (0.30989286 * float(x[3]))+ (-0.08218093 * float(x[4]))+ (0.22440831 * float(x[5]))+ (0.34748745 * float(x[6]))+ (0.17389739 * float(x[7]))+ (-0.7499996 * float(x[8]))+ (-0.812547 * float(x[9]))+ (0.9218099 * float(x[10]))+ (0.5694684 * float(x[11]))+ (0.5159434 * float(x[12]))+ (0.67892355 * float(x[13]))+ (-0.66963905 * float(x[14]))+ (-0.059599176 * float(x[15]))) + -1.241066), 0)
    h_12 = max((((-0.31131053 * float(x[0]))+ (-0.15409163 * float(x[1]))+ (0.40618712 * float(x[2]))+ (0.5236203 * float(x[3]))+ (-0.43018436 * float(x[4]))+ (0.5407163 * float(x[5]))+ (-0.042046014 * float(x[6]))+ (0.26350784 * float(x[7]))+ (-0.5152184 * float(x[8]))+ (0.15151057 * float(x[9]))+ (-0.5040125 * float(x[10]))+ (0.51353985 * float(x[11]))+ (-0.6634115 * float(x[12]))+ (-0.42593178 * float(x[13]))+ (1.5344822 * float(x[14]))+ (0.7565374 * float(x[15]))) + 0.42219406), 0)
    h_13 = max((((0.5623633 * float(x[0]))+ (0.0332069 * float(x[1]))+ (0.52777475 * float(x[2]))+ (0.69503975 * float(x[3]))+ (-2.1386125 * float(x[4]))+ (-0.0029264586 * float(x[5]))+ (-0.06048409 * float(x[6]))+ (-0.5633931 * float(x[7]))+ (-0.57834244 * float(x[8]))+ (-0.23906216 * float(x[9]))+ (0.71511763 * float(x[10]))+ (0.30477825 * float(x[11]))+ (0.31332767 * float(x[12]))+ (0.20207946 * float(x[13]))+ (-1.1907626 * float(x[14]))+ (-0.01161695 * float(x[15]))) + 2.142528), 0)
    h_14 = max((((0.532394 * float(x[0]))+ (0.15882652 * float(x[1]))+ (-0.36935097 * float(x[2]))+ (0.21772222 * float(x[3]))+ (-0.86888534 * float(x[4]))+ (-0.14626612 * float(x[5]))+ (-0.14396647 * float(x[6]))+ (0.6790971 * float(x[7]))+ (1.5808316 * float(x[8]))+ (0.18314661 * float(x[9]))+ (0.0058880984 * float(x[10]))+ (0.18098043 * float(x[11]))+ (-1.1747817 * float(x[12]))+ (0.10204216 * float(x[13]))+ (-0.668642 * float(x[14]))+ (-0.2358057 * float(x[15]))) + 0.63535523), 0)
    o[0] = (0.8017084 * h_0)+ (-1.2670017 * h_1)+ (0.5181204 * h_2)+ (-0.0060072057 * h_3)+ (0.23525302 * h_4)+ (-0.0868261 * h_5)+ (1.1685555 * h_6)+ (-0.7372505 * h_7)+ (-0.14057364 * h_8)+ (-1.0809065 * h_9)+ (-2.1742373 * h_10)+ (-0.5989333 * h_11)+ (0.919921 * h_12)+ (-0.37916824 * h_13)+ (0.08030461 * h_14) + -1.3719171
    o[1] = (0.26892304 * h_0)+ (0.48894113 * h_1)+ (-1.8558027 * h_2)+ (0.80846363 * h_3)+ (-0.9549403 * h_4)+ (0.64032125 * h_5)+ (-0.093998164 * h_6)+ (0.8951045 * h_7)+ (-1.0644026 * h_8)+ (0.4414896 * h_9)+ (1.1689129 * h_10)+ (-0.30051687 * h_11)+ (-0.34744218 * h_12)+ (-1.6996001 * h_13)+ (-0.025026232 * h_14) + 0.12536994
    o[2] = (0.050367408 * h_0)+ (0.071130335 * h_1)+ (-0.83111405 * h_2)+ (0.50937146 * h_3)+ (-0.79281366 * h_4)+ (-0.5057267 * h_5)+ (-1.0380954 * h_6)+ (-0.26127893 * h_7)+ (-0.6455279 * h_8)+ (0.11856074 * h_9)+ (-0.19764687 * h_10)+ (-0.0455136 * h_11)+ (0.74537677 * h_12)+ (0.5306879 * h_13)+ (1.1833012 * h_14) + 0.40205365
    o[3] = (-0.30694205 * h_0)+ (0.35355216 * h_1)+ (1.1805453 * h_2)+ (0.08242564 * h_3)+ (-0.59310997 * h_4)+ (-0.099463314 * h_5)+ (0.025324348 * h_6)+ (0.0055427225 * h_7)+ (-0.3213998 * h_8)+ (-0.20053041 * h_9)+ (0.36479965 * h_10)+ (-0.10957667 * h_11)+ (0.46473438 * h_12)+ (-0.12678042 * h_13)+ (-0.32854107 * h_14) + 0.43311208
    o[4] = (-0.09269177 * h_0)+ (0.6760652 * h_1)+ (-0.7894236 * h_2)+ (0.5275813 * h_3)+ (-0.21333937 * h_4)+ (0.32191828 * h_5)+ (-0.5004618 * h_6)+ (-0.18902227 * h_7)+ (-0.3494741 * h_8)+ (-0.24751894 * h_9)+ (1.0714641 * h_10)+ (0.1328254 * h_11)+ (0.054828167 * h_12)+ (-0.53941643 * h_13)+ (-0.24240887 * h_14) + 0.98519355
    o[5] = (0.26038375 * h_0)+ (0.8948275 * h_1)+ (0.23018673 * h_2)+ (-0.022384234 * h_3)+ (-0.2536457 * h_4)+ (-1.0107554 * h_5)+ (-0.525936 * h_6)+ (-0.10195026 * h_7)+ (-0.6618341 * h_8)+ (0.79895365 * h_9)+ (-4.1149044 * h_10)+ (0.1986807 * h_11)+ (-0.39460152 * h_12)+ (0.7075925 * h_13)+ (-0.1791828 * h_14) + 1.259321
    o[6] = (-0.5814531 * h_0)+ (0.68192464 * h_1)+ (-1.3661888 * h_2)+ (0.26036587 * h_3)+ (0.4400039 * h_4)+ (-0.6665949 * h_5)+ (-0.5895287 * h_6)+ (0.36490783 * h_7)+ (-0.567925 * h_8)+ (0.8625211 * h_9)+ (-0.60129774 * h_10)+ (-0.060860835 * h_11)+ (-0.24355204 * h_12)+ (-3.963834 * h_13)+ (0.31801018 * h_14) + 1.6057613
    o[7] = (-0.0043513896 * h_0)+ (0.73374265 * h_1)+ (1.2182015 * h_2)+ (-0.5303439 * h_3)+ (-0.71976405 * h_4)+ (-0.903272 * h_5)+ (-0.079416424 * h_6)+ (-0.46863687 * h_7)+ (0.48364702 * h_8)+ (0.23677316 * h_9)+ (-8.836261 * h_10)+ (-0.08649065 * h_11)+ (-0.200557 * h_12)+ (1.1539385 * h_13)+ (-1.9638855 * h_14) + 0.910928
    o[8] = (0.48225993 * h_0)+ (0.23147342 * h_1)+ (0.2774493 * h_2)+ (0.49764967 * h_3)+ (0.24364859 * h_4)+ (0.16915125 * h_5)+ (-0.5479312 * h_6)+ (0.12351347 * h_7)+ (-0.058912687 * h_8)+ (0.8431305 * h_9)+ (-0.5000801 * h_10)+ (-0.7221121 * h_11)+ (-0.40472507 * h_12)+ (0.042160727 * h_13)+ (-0.806762 * h_14) + 1.965732
    o[9] = (-0.08800557 * h_0)+ (0.080322996 * h_1)+ (-1.0301408 * h_2)+ (-0.16161439 * h_3)+ (-0.16428794 * h_4)+ (0.36828202 * h_5)+ (0.36204135 * h_6)+ (-0.23724012 * h_7)+ (0.4558333 * h_8)+ (0.032539774 * h_9)+ (-0.6116923 * h_10)+ (0.15995346 * h_11)+ (-0.5545738 * h_12)+ (0.862265 * h_13)+ (-0.1430186 * h_14) + 0.04791633
    o[10] = (1.0293424 * h_0)+ (-0.6921306 * h_1)+ (-0.16022702 * h_2)+ (0.92843986 * h_3)+ (0.20275709 * h_4)+ (-1.6923958 * h_5)+ (0.3730117 * h_6)+ (1.569913 * h_7)+ (-0.9084574 * h_8)+ (0.04262944 * h_9)+ (0.2810743 * h_10)+ (0.042049132 * h_11)+ (-0.17098601 * h_12)+ (0.14240703 * h_13)+ (-1.1536905 * h_14) + 2.652452
    o[11] = (-2.5559964 * h_0)+ (0.30862164 * h_1)+ (0.31125873 * h_2)+ (0.27396747 * h_3)+ (-0.16660213 * h_4)+ (-0.55332077 * h_5)+ (-0.75054884 * h_6)+ (0.3033883 * h_7)+ (0.026523855 * h_8)+ (-0.057910062 * h_9)+ (-5.1892414 * h_10)+ (0.0006754587 * h_11)+ (0.32095525 * h_12)+ (0.46950963 * h_13)+ (0.62335014 * h_14) + 0.57102585
    o[12] = (-0.44416276 * h_0)+ (0.16509634 * h_1)+ (-0.076638326 * h_2)+ (-0.25621033 * h_3)+ (0.051179264 * h_4)+ (0.2585506 * h_5)+ (0.2098516 * h_6)+ (-0.5710384 * h_7)+ (1.0549878 * h_8)+ (-1.3003466 * h_9)+ (0.23183203 * h_10)+ (0.29090658 * h_11)+ (0.31771517 * h_12)+ (-2.2786071 * h_13)+ (0.40177017 * h_14) + -1.157176
    o[13] = (0.54773194 * h_0)+ (0.17274636 * h_1)+ (-0.7390344 * h_2)+ (-2.0358334 * h_3)+ (0.17313196 * h_4)+ (-0.1748232 * h_5)+ (-0.14354831 * h_6)+ (-0.29339805 * h_7)+ (0.14207901 * h_8)+ (0.18854745 * h_9)+ (-0.83636457 * h_10)+ (0.6410925 * h_11)+ (-0.5479258 * h_12)+ (-0.68279684 * h_13)+ (0.41723177 * h_14) + 2.572383
    o[14] = (0.4314962 * h_0)+ (-1.4982872 * h_1)+ (-0.36750537 * h_2)+ (-1.6670512 * h_3)+ (0.58001673 * h_4)+ (0.6274395 * h_5)+ (2.0465808 * h_6)+ (0.32842416 * h_7)+ (0.60173523 * h_8)+ (-0.50599504 * h_9)+ (-0.6237639 * h_10)+ (-0.40441135 * h_11)+ (-0.105823226 * h_12)+ (-0.5966051 * h_13)+ (-0.5324448 * h_14) + 0.38510308
    o[15] = (-0.21477255 * h_0)+ (-1.5522953 * h_1)+ (0.5946458 * h_2)+ (0.089443415 * h_3)+ (0.11387361 * h_4)+ (0.0762511 * h_5)+ (0.8804951 * h_6)+ (1.2031202 * h_7)+ (0.27452317 * h_8)+ (-0.47731203 * h_9)+ (0.6864858 * h_10)+ (-0.3137784 * h_11)+ (-0.13243157 * h_12)+ (-0.58506984 * h_13)+ (-0.41026473 * h_14) + 2.5866876
    o[16] = (-1.3110406 * h_0)+ (-0.41758 * h_1)+ (-0.047944233 * h_2)+ (0.60928 * h_3)+ (0.07488779 * h_4)+ (-0.5641056 * h_5)+ (-1.1594526 * h_6)+ (0.03256625 * h_7)+ (-1.4079701 * h_8)+ (0.6487579 * h_9)+ (-5.8815236 * h_10)+ (-0.080391936 * h_11)+ (0.62008935 * h_12)+ (0.8129872 * h_13)+ (0.4348233 * h_14) + 3.288226
    o[17] = (0.7834944 * h_0)+ (-0.7750562 * h_1)+ (-0.55803025 * h_2)+ (0.2094963 * h_3)+ (-0.88130283 * h_4)+ (0.7392601 * h_5)+ (0.5606178 * h_6)+ (-0.71801054 * h_7)+ (0.60133725 * h_8)+ (-0.7834024 * h_9)+ (0.5824258 * h_10)+ (0.2270211 * h_11)+ (-0.18163691 * h_12)+ (0.6247374 * h_13)+ (0.1525784 * h_14) + 1.3410163
    o[18] = (-0.13013795 * h_0)+ (-0.377488 * h_1)+ (-0.6534501 * h_2)+ (1.0627205 * h_3)+ (0.106697 * h_4)+ (0.096643634 * h_5)+ (0.51624477 * h_6)+ (0.862561 * h_7)+ (-0.90216124 * h_8)+ (-0.01626651 * h_9)+ (1.6497402 * h_10)+ (-0.77196157 * h_11)+ (0.23132916 * h_12)+ (-0.119817995 * h_13)+ (-0.49159342 * h_14) + 1.6272155
    o[19] = (-0.20296016 * h_0)+ (0.51819205 * h_1)+ (-0.28444043 * h_2)+ (-0.23926824 * h_3)+ (-1.0635337 * h_4)+ (0.0025573275 * h_5)+ (0.037486017 * h_6)+ (-0.25398558 * h_7)+ (0.25397578 * h_8)+ (0.6914117 * h_9)+ (-11.86741 * h_10)+ (0.009311077 * h_11)+ (-0.93953174 * h_12)+ (1.2363206 * h_13)+ (-2.8849854 * h_14) + 1.2019612
    o[20] = (0.28269437 * h_0)+ (-0.2961585 * h_1)+ (0.24026623 * h_2)+ (-0.29502448 * h_3)+ (-0.038711756 * h_4)+ (-0.41352785 * h_5)+ (0.35384458 * h_6)+ (-0.038142342 * h_7)+ (0.8432703 * h_8)+ (0.037820466 * h_9)+ (-1.6485561 * h_10)+ (0.261521 * h_11)+ (-0.88162506 * h_12)+ (0.49093902 * h_13)+ (0.23966968 * h_14) + 1.7325287
    o[21] = (0.2563771 * h_0)+ (-0.04740843 * h_1)+ (-0.31980067 * h_2)+ (-0.7339891 * h_3)+ (1.1549653 * h_4)+ (0.8067878 * h_5)+ (0.7664764 * h_6)+ (-1.3879532 * h_7)+ (1.6097207 * h_8)+ (-1.0757511 * h_9)+ (0.72535896 * h_10)+ (0.35546505 * h_11)+ (-0.45630997 * h_12)+ (-0.29618627 * h_13)+ (-0.67152905 * h_14) + -0.6609352
    o[22] = (-0.680736 * h_0)+ (-0.5139506 * h_1)+ (0.6686525 * h_2)+ (-2.0911 * h_3)+ (-0.0051088985 * h_4)+ (0.5115766 * h_5)+ (-0.2187608 * h_6)+ (-0.3315247 * h_7)+ (0.66896653 * h_8)+ (-0.8313613 * h_9)+ (1.3833365 * h_10)+ (1.1381851 * h_11)+ (-0.13984825 * h_12)+ (-1.5888008 * h_13)+ (0.10620318 * h_14) + -1.6782384
    o[23] = (-0.37954006 * h_0)+ (-0.61707044 * h_1)+ (-0.8548516 * h_2)+ (-1.5529941 * h_3)+ (-0.56652594 * h_4)+ (-0.10024474 * h_5)+ (-0.604571 * h_6)+ (0.27586162 * h_7)+ (-0.24893159 * h_8)+ (0.3613051 * h_9)+ (-0.3371406 * h_10)+ (0.82469517 * h_11)+ (-0.09924926 * h_12)+ (-1.8421353 * h_13)+ (1.3875837 * h_14) + 1.0041662
    o[24] = (-0.29622677 * h_0)+ (0.4394119 * h_1)+ (-2.296392 * h_2)+ (0.5761811 * h_3)+ (0.18380989 * h_4)+ (0.27692553 * h_5)+ (-0.93521744 * h_6)+ (-0.021967912 * h_7)+ (-0.12388043 * h_8)+ (0.9722753 * h_9)+ (-0.35634464 * h_10)+ (-0.37433565 * h_11)+ (-0.35678345 * h_12)+ (-2.5246027 * h_13)+ (0.220872 * h_14) + 2.0988107
    o[25] = (0.79508805 * h_0)+ (-0.32700118 * h_1)+ (0.9973899 * h_2)+ (-0.0105389645 * h_3)+ (0.6664509 * h_4)+ (-0.21330765 * h_5)+ (0.38825673 * h_6)+ (0.11385653 * h_7)+ (-0.61629784 * h_8)+ (-0.7380193 * h_9)+ (1.3813434 * h_10)+ (-0.11836147 * h_11)+ (0.39061424 * h_12)+ (-0.1248156 * h_13)+ (-1.5611292 * h_14) + 0.95721775

    if num_output_logits == 1:
        return o[0] >= 0
    else:
        return argmax(o)


#for classifying batches
def classify(arr):
    outputs = []
    for row in arr:
        outputs.append(single_classify(row))
    return outputs

def Validate(cleanvalfile):
    #Binary
    if n_classes == 2:
        with open(cleanvalfile, 'r') as valcsvfile:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
            valcsvreader = csv.reader(valcsvfile)
            preds = []
            y_trues = []
            for valrow in valcsvreader:
                if len(valrow) == 0:
                    continue
                y_true = int(float(valrow[-1]))
                pred = int(single_classify(valrow[:-1]))
                y_trues.append(y_true)
                preds.append(pred)
                if pred == y_true:
                    correct_count += 1
                    if int(float(valrow[-1])) == 1:
                        num_class_1 += 1
                        num_TP += 1
                    else:
                        num_class_0 += 1
                        num_TN += 1
                else:
                    if int(float(valrow[-1])) == 1:
                        num_class_1 += 1
                        num_FN += 1
                    else:
                        num_class_0 += 1
                        num_FP += 1
                count += 1
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, y_trues

    #Multiclass
    else:
        with open(cleanvalfile, 'r') as valcsvfile:
            count, correct_count = 0, 0
            valcsvreader = csv.reader(valcsvfile)
            numeachclass = {}
            preds = []
            y_trues = []
            for i, valrow in enumerate(valcsvreader):
                pred = int(single_classify(valrow[:-1]))
                preds.append(pred)
                y_true = int(float(valrow[-1]))
                y_trues.append(y_true)
                if len(valrow) == 0:
                    continue
                if pred == y_true:
                    correct_count += 1
                #if class seen, add to its counter
                if y_true in numeachclass.keys():
                    numeachclass[y_true] += 1
                #initialize a new counter
                else:
                    numeachclass[y_true] = 1
                count += 1
        return count, correct_count, numeachclass, preds,  y_trues



def Predict(cleanfile, preprocessedfile, headerless, get_key, classmapping):
    with open(cleanfile,'r') as cleancsvfile, open(preprocessedfile,'r') as dirtycsvfile:
        cleancsvreader = csv.reader(cleancsvfile)
        dirtycsvreader = csv.reader(dirtycsvfile)
        if (not headerless):
            print(','.join(next(dirtycsvreader, None) + ["Prediction"]))
        for cleanrow, dirtyrow in zip(cleancsvreader, dirtycsvreader):
            if len(cleanrow) == 0:
                continue
            print(str(','.join(str(j) for j in ([i for i in dirtyrow]))) + ',' + str(get_key(int(single_classify(cleanrow)), classmapping)))



# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    args = parser.parse_args()
    faulthandler.enable()
    
    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}


    #Predict
    if not args.validate:
        Predict(cleanfile, preprocessedfile, args.headerless, get_key, classmapping)


    #Validate
    else: 
        classifier_type = 'NN'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap=510
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


