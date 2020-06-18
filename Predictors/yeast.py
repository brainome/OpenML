#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/3616/dataset_185_yeast.arff -o Predictors/yeast_NN.py -target class_protein_localization -stopat 61.55 -f NN -e 20 --yes --runlocalonly
# Total compiler execution time: 0:23:06.18. Finished on: May-29-2020 23:16:10.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        10-way classifier
Best-guess accuracy:                31.34%
Model accuracy:                     69.94% (1038/1484 correct)
Improvement over best guess:        38.60% (of possible 68.66%)
Model capacity (MEC):               864 bits
Generalization ratio:               1.20 bits/bit
Confusion Matrix:
 [10.78% 1.28% 3.57% 0.13% 0.07% 0.13% 0.34% 0.07% 0.07% 0.00%]
 [1.62% 18.60% 7.82% 0.00% 0.07% 0.13% 0.67% 0.00% 0.00% 0.00%]
 [2.22% 6.13% 22.30% 0.00% 0.07% 0.00% 0.34% 0.07% 0.07% 0.00%]
 [0.00% 0.00% 0.00% 2.76% 0.20% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.13% 0.07% 0.20% 0.07% 1.75% 0.13% 0.00% 0.00% 0.00% 0.00%]
 [0.27% 0.20% 0.20% 0.07% 0.13% 2.56% 0.00% 0.00% 0.00% 0.00%]
 [0.40% 0.88% 0.27% 0.00% 0.00% 0.00% 9.43% 0.00% 0.00% 0.00%]
 [0.20% 0.34% 0.74% 0.00% 0.13% 0.00% 0.07% 0.54% 0.00% 0.00%]
 [0.07% 0.00% 0.34% 0.00% 0.00% 0.07% 0.00% 0.00% 0.88% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.34%]

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
TRAINFILE = "dataset_185_yeast.csv"


#Number of output logits
num_output_logits = 10

#Number of attributes
num_attr = 8
n_classes = 10


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="class_protein_localization"


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
    clean.mapping={'MIT': 0, 'NUC': 1, 'CYT': 2, 'ME1': 3, 'EXC': 4, 'ME2': 5, 'ME3': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}

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
    x = row
    o = [0] * num_output_logits
    h_0 = max((((-0.52099144 * float(x[0]))+ (0.12900849 * float(x[1]))+ (0.36584437 * float(x[2]))+ (-0.21984461 * float(x[3]))+ (0.9416025 * float(x[4]))+ (0.25588986 * float(x[5]))+ (-1.0026923 * float(x[6]))+ (-1.4476961 * float(x[7]))) + 2.369355), 0)
    h_1 = max((((0.020867001 * float(x[0]))+ (0.07769437 * float(x[1]))+ (-0.17045976 * float(x[2]))+ (-0.16612099 * float(x[3]))+ (0.5798303 * float(x[4]))+ (0.24176341 * float(x[5]))+ (-0.2191039 * float(x[6]))+ (-0.4874269 * float(x[7]))) + 1.913059), 0)
    h_2 = max((((1.2404845 * float(x[0]))+ (0.70094216 * float(x[1]))+ (-0.18540636 * float(x[2]))+ (0.5312519 * float(x[3]))+ (1.0516948 * float(x[4]))+ (0.10150142 * float(x[5]))+ (0.22036208 * float(x[6]))+ (-0.6899262 * float(x[7]))) + 0.74774396), 0)
    h_3 = max((((0.2744255 * float(x[0]))+ (-0.036579408 * float(x[1]))+ (0.2880777 * float(x[2]))+ (-0.19296953 * float(x[3]))+ (-0.18444653 * float(x[4]))+ (0.16841224 * float(x[5]))+ (0.33992243 * float(x[6]))+ (-0.14054485 * float(x[7]))) + 1.2219832), 0)
    h_4 = max((((-1.5463898 * float(x[0]))+ (-0.035091575 * float(x[1]))+ (-0.16915469 * float(x[2]))+ (-2.1933088 * float(x[3]))+ (0.9054705 * float(x[4]))+ (-1.2707163 * float(x[5]))+ (-0.076809995 * float(x[6]))+ (-0.0036436815 * float(x[7]))) + 2.2115889), 0)
    h_5 = max((((-0.5577702 * float(x[0]))+ (-0.5548053 * float(x[1]))+ (-1.2676121 * float(x[2]))+ (-3.561314 * float(x[3]))+ (0.22872113 * float(x[4]))+ (-0.87814534 * float(x[5]))+ (-0.64039725 * float(x[6]))+ (-0.22051015 * float(x[7]))) + 1.4875753), 0)
    h_6 = max((((0.1986288 * float(x[0]))+ (0.037936587 * float(x[1]))+ (-2.4280999 * float(x[2]))+ (-0.12552115 * float(x[3]))+ (0.046955034 * float(x[4]))+ (-0.64647377 * float(x[5]))+ (0.72858053 * float(x[6]))+ (2.010347 * float(x[7]))) + 1.1934309), 0)
    h_7 = max((((0.56785405 * float(x[0]))+ (-0.62409294 * float(x[1]))+ (-0.33168286 * float(x[2]))+ (-1.2162995 * float(x[3]))+ (-0.013456784 * float(x[4]))+ (0.81187195 * float(x[5]))+ (1.9268172 * float(x[6]))+ (0.25510386 * float(x[7]))) + 0.46295017), 0)
    h_8 = max((((0.8733745 * float(x[0]))+ (1.1386561 * float(x[1]))+ (-0.57842255 * float(x[2]))+ (0.07879888 * float(x[3]))+ (-0.6304528 * float(x[4]))+ (-2.4235692 * float(x[5]))+ (1.1639556 * float(x[6]))+ (2.0302753 * float(x[7]))) + 0.4466227), 0)
    h_9 = max((((-0.096175686 * float(x[0]))+ (0.2863054 * float(x[1]))+ (-1.6315273 * float(x[2]))+ (0.25261202 * float(x[3]))+ (-0.1992391 * float(x[4]))+ (0.9197842 * float(x[5]))+ (0.50154763 * float(x[6]))+ (1.6725082 * float(x[7]))) + 1.2246603), 0)
    h_10 = max((((1.7909772 * float(x[0]))+ (1.5351117 * float(x[1]))+ (0.57751906 * float(x[2]))+ (-8.630718 * float(x[3]))+ (0.29779837 * float(x[4]))+ (-0.71502 * float(x[5]))+ (-1.9845089 * float(x[6]))+ (-1.9194738 * float(x[7]))) + 0.1923826), 0)
    h_11 = max((((1.4549748 * float(x[0]))+ (0.96130896 * float(x[1]))+ (0.43286654 * float(x[2]))+ (1.995265 * float(x[3]))+ (1.1311203 * float(x[4]))+ (0.6833369 * float(x[5]))+ (1.659436 * float(x[6]))+ (-0.8232886 * float(x[7]))) + 1.9472283), 0)
    h_12 = max((((2.8444233 * float(x[0]))+ (1.4395443 * float(x[1]))+ (1.043046 * float(x[2]))+ (1.2129979 * float(x[3]))+ (-3.2093773 * float(x[4]))+ (-0.9362245 * float(x[5]))+ (0.7386711 * float(x[6]))+ (0.28941566 * float(x[7]))) + -2.183706), 0)
    h_13 = max((((0.30077985 * float(x[0]))+ (0.44462553 * float(x[1]))+ (-1.8765432 * float(x[2]))+ (-0.020886391 * float(x[3]))+ (2.0724335 * float(x[4]))+ (-0.5919864 * float(x[5]))+ (-1.3571246 * float(x[6]))+ (-0.58533835 * float(x[7]))) + 1.0280262), 0)
    h_14 = max((((1.5893458 * float(x[0]))+ (0.71641845 * float(x[1]))+ (2.1571708 * float(x[2]))+ (-1.3040084 * float(x[3]))+ (1.2035053 * float(x[4]))+ (0.9576388 * float(x[5]))+ (0.3524425 * float(x[6]))+ (-1.8936497 * float(x[7]))) + 2.0566075), 0)
    h_15 = max((((-1.2615436 * float(x[0]))+ (-0.030603591 * float(x[1]))+ (0.66851854 * float(x[2]))+ (-1.8899748 * float(x[3]))+ (1.2448987 * float(x[4]))+ (-0.7962721 * float(x[5]))+ (-0.8173042 * float(x[6]))+ (-0.68950284 * float(x[7]))) + 0.5709021), 0)
    h_16 = max((((-0.11277863 * float(x[0]))+ (-0.13885505 * float(x[1]))+ (-0.1298636 * float(x[2]))+ (-0.6627842 * float(x[3]))+ (-0.1520969 * float(x[4]))+ (-0.60656416 * float(x[5]))+ (-0.31257483 * float(x[6]))+ (-0.40885875 * float(x[7]))) + 0.44745225), 0)
    h_17 = max((((0.97404325 * float(x[0]))+ (1.0498706 * float(x[1]))+ (-2.3695583 * float(x[2]))+ (-2.4976254 * float(x[3]))+ (0.198914 * float(x[4]))+ (-0.6031115 * float(x[5]))+ (-1.4230416 * float(x[6]))+ (-2.5220656 * float(x[7]))) + 1.2933618), 0)
    h_18 = max((((0.88369757 * float(x[0]))+ (-1.8838243 * float(x[1]))+ (0.55779094 * float(x[2]))+ (-3.7593143 * float(x[3]))+ (-0.19287902 * float(x[4]))+ (-0.7332641 * float(x[5]))+ (-0.6265743 * float(x[6]))+ (0.8546305 * float(x[7]))) + 0.9183544), 0)
    h_19 = max((((1.7225304 * float(x[0]))+ (1.4537 * float(x[1]))+ (2.5008833 * float(x[2]))+ (0.34839112 * float(x[3]))+ (1.4231625 * float(x[4]))+ (0.30324075 * float(x[5]))+ (2.0769572 * float(x[6]))+ (1.4837537 * float(x[7]))) + 2.743578), 0)
    h_20 = max((((-0.3731795 * float(x[0]))+ (-0.8911915 * float(x[1]))+ (-0.97327477 * float(x[2]))+ (0.3063723 * float(x[3]))+ (0.30130836 * float(x[4]))+ (-0.5013902 * float(x[5]))+ (-1.3305323 * float(x[6]))+ (-0.8268219 * float(x[7]))) + 1.535554), 0)
    h_21 = max((((-1.4781544 * float(x[0]))+ (-0.6970114 * float(x[1]))+ (0.16868243 * float(x[2]))+ (-1.5385084 * float(x[3]))+ (1.6303877 * float(x[4]))+ (0.0917048 * float(x[5]))+ (0.08689171 * float(x[6]))+ (1.3578398 * float(x[7]))) + 0.7871616), 0)
    h_22 = max((((0.35616073 * float(x[0]))+ (-0.39567217 * float(x[1]))+ (-4.456509 * float(x[2]))+ (-2.6277623 * float(x[3]))+ (0.88268536 * float(x[4]))+ (-0.568601 * float(x[5]))+ (-0.3757528 * float(x[6]))+ (-0.04369655 * float(x[7]))) + 1.9511248), 0)
    h_23 = max((((2.0365362 * float(x[0]))+ (-1.9499502 * float(x[1]))+ (-2.0900908 * float(x[2]))+ (-1.544273 * float(x[3]))+ (0.024443785 * float(x[4]))+ (-0.54894036 * float(x[5]))+ (-1.2040873 * float(x[6]))+ (0.57378376 * float(x[7]))) + 1.4010088), 0)
    h_24 = max((((0.869894 * float(x[0]))+ (-1.8516272 * float(x[1]))+ (0.13143349 * float(x[2]))+ (-2.7653859 * float(x[3]))+ (-0.33742487 * float(x[4]))+ (-0.48507082 * float(x[5]))+ (-0.9207059 * float(x[6]))+ (1.4312689 * float(x[7]))) + 0.9761181), 0)
    h_25 = max((((-0.017009167 * float(x[0]))+ (0.46631062 * float(x[1]))+ (-0.29195622 * float(x[2]))+ (-7.0121098 * float(x[3]))+ (0.8201632 * float(x[4]))+ (-0.46748814 * float(x[5]))+ (-0.36912608 * float(x[6]))+ (0.5836606 * float(x[7]))) + 0.41068256), 0)
    h_26 = max((((0.8145935 * float(x[0]))+ (-1.047149 * float(x[1]))+ (-2.0068944 * float(x[2]))+ (0.052962445 * float(x[3]))+ (1.4114649 * float(x[4]))+ (-0.39755097 * float(x[5]))+ (0.32242036 * float(x[6]))+ (-2.8875709 * float(x[7]))) + 0.73174053), 0)
    h_27 = max((((1.5202111 * float(x[0]))+ (1.2831644 * float(x[1]))+ (2.3933768 * float(x[2]))+ (0.2675532 * float(x[3]))+ (1.1880527 * float(x[4]))+ (0.27730206 * float(x[5]))+ (1.8151159 * float(x[6]))+ (1.3365544 * float(x[7]))) + 2.3576012), 0)
    h_28 = max((((-0.12567443 * float(x[0]))+ (1.9417673 * float(x[1]))+ (-2.732208 * float(x[2]))+ (-6.1293616 * float(x[3]))+ (0.21992011 * float(x[4]))+ (-0.37896284 * float(x[5]))+ (-1.4123263 * float(x[6]))+ (-2.5954928 * float(x[7]))) + 1.9983175), 0)
    h_29 = max((((0.07857649 * float(x[0]))+ (-0.73112637 * float(x[1]))+ (-3.8368251 * float(x[2]))+ (-2.0621848 * float(x[3]))+ (0.22466247 * float(x[4]))+ (-0.37303954 * float(x[5]))+ (0.62124765 * float(x[6]))+ (1.072387 * float(x[7]))) + 1.0707512), 0)
    h_30 = max((((1.3665311 * float(x[0]))+ (1.489064 * float(x[1]))+ (3.76642 * float(x[2]))+ (0.8148398 * float(x[3]))+ (0.5103793 * float(x[4]))+ (-2.8122256 * float(x[5]))+ (-0.0560707 * float(x[6]))+ (-0.5155003 * float(x[7]))) + 0.6332151), 0)
    h_31 = max((((-0.07675578 * float(x[0]))+ (-0.057797335 * float(x[1]))+ (-0.11103545 * float(x[2]))+ (-0.33263192 * float(x[3]))+ (-0.112977274 * float(x[4]))+ (-0.36247602 * float(x[5]))+ (-0.22777289 * float(x[6]))+ (-0.29266283 * float(x[7]))) + 0.29160964), 0)
    h_32 = max((((-0.020612419 * float(x[0]))+ (-0.10276332 * float(x[1]))+ (-0.08099734 * float(x[2]))+ (-0.36075103 * float(x[3]))+ (-0.11690584 * float(x[4]))+ (-0.37361333 * float(x[5]))+ (-0.29378724 * float(x[6]))+ (-0.3297534 * float(x[7]))) + 0.2787988), 0)
    h_33 = max((((1.0522995 * float(x[0]))+ (-1.0731677 * float(x[1]))+ (-1.3922164 * float(x[2]))+ (-0.8032841 * float(x[3]))+ (0.55474484 * float(x[4]))+ (-0.33917415 * float(x[5]))+ (-1.0590866 * float(x[6]))+ (-1.982238 * float(x[7]))) + 1.319393), 0)
    h_34 = max((((0.9485914 * float(x[0]))+ (-0.008529822 * float(x[1]))+ (-0.95845306 * float(x[2]))+ (-3.3049843 * float(x[3]))+ (-0.037909783 * float(x[4]))+ (-0.3300236 * float(x[5]))+ (-1.0095665 * float(x[6]))+ (-0.5605385 * float(x[7]))) + 1.0034914), 0)
    h_35 = max((((-0.08093335 * float(x[0]))+ (0.4829242 * float(x[1]))+ (-0.3105383 * float(x[2]))+ (-2.0032568 * float(x[3]))+ (0.11954205 * float(x[4]))+ (-0.29077882 * float(x[5]))+ (-0.42845413 * float(x[6]))+ (-3.674245 * float(x[7]))) + 1.1130971), 0)
    h_36 = max((((0.6919957 * float(x[0]))+ (-0.54997313 * float(x[1]))+ (-1.4842434 * float(x[2]))+ (-1.9006871 * float(x[3]))+ (0.84773165 * float(x[4]))+ (-0.2951479 * float(x[5]))+ (-1.5110859 * float(x[6]))+ (-2.573105 * float(x[7]))) + 1.7094727), 0)
    h_37 = max((((-0.02305995 * float(x[0]))+ (-0.04270994 * float(x[1]))+ (-0.12584138 * float(x[2]))+ (-0.246968 * float(x[3]))+ (-0.09401186 * float(x[4]))+ (-0.26150295 * float(x[5]))+ (-0.08714581 * float(x[6]))+ (-0.24312449 * float(x[7]))) + 0.21981177), 0)
    h_38 = max((((1.3559484 * float(x[0]))+ (1.1646726 * float(x[1]))+ (2.366679 * float(x[2]))+ (0.24772826 * float(x[3]))+ (1.0845343 * float(x[4]))+ (0.26264673 * float(x[5]))+ (1.6067058 * float(x[6]))+ (1.1867585 * float(x[7]))) + 2.1091614), 0)
    h_39 = max((((-0.71352917 * float(x[0]))+ (-0.67355466 * float(x[1]))+ (0.47184533 * float(x[2]))+ (1.3214054 * float(x[3]))+ (1.2804674 * float(x[4]))+ (-0.2271809 * float(x[5]))+ (-2.6985366 * float(x[6]))+ (-1.3866932 * float(x[7]))) + 0.7947683), 0)
    h_40 = max((((-0.24245563 * float(x[0]))+ (0.2510398 * float(x[1]))+ (-2.740558 * float(x[2]))+ (-0.84649676 * float(x[3]))+ (0.019855298 * float(x[4]))+ (-0.14020392 * float(x[5]))+ (-0.15469578 * float(x[6]))+ (-0.008070784 * float(x[7]))) + 1.5073333), 0)
    h_41 = max((((-0.04620113 * float(x[0]))+ (0.0018598865 * float(x[1]))+ (-0.075668864 * float(x[2]))+ (-0.15410636 * float(x[3]))+ (-0.049648497 * float(x[4]))+ (-0.14605175 * float(x[5]))+ (-0.10273915 * float(x[6]))+ (-0.12642594 * float(x[7]))) + 0.11228287), 0)
    h_42 = max((((-0.022870867 * float(x[0]))+ (0.013112317 * float(x[1]))+ (-0.032168508 * float(x[2]))+ (-0.123294234 * float(x[3]))+ (-0.054841448 * float(x[4]))+ (-0.1370179 * float(x[5]))+ (-0.05689264 * float(x[6]))+ (-0.22935055 * float(x[7]))) + 0.08942562), 0)
    h_43 = max((((2.1035972 * float(x[0]))+ (-2.4116836 * float(x[1]))+ (-1.3004792 * float(x[2]))+ (-0.43828166 * float(x[3]))+ (0.29917854 * float(x[4]))+ (-0.1456321 * float(x[5]))+ (-0.64463896 * float(x[6]))+ (0.6679186 * float(x[7]))) + 0.5420739), 0)
    h_44 = max((((-1.2223369 * float(x[0]))+ (0.24458984 * float(x[1]))+ (0.65915877 * float(x[2]))+ (-0.66534674 * float(x[3]))+ (0.20049915 * float(x[4]))+ (-0.10140728 * float(x[5]))+ (-1.4041495 * float(x[6]))+ (0.28151563 * float(x[7]))) + 0.32031098), 0)
    h_45 = max((((-0.85051924 * float(x[0]))+ (0.17939381 * float(x[1]))+ (0.46662915 * float(x[2]))+ (-0.46893266 * float(x[3]))+ (0.13228463 * float(x[4]))+ (-0.08658782 * float(x[5]))+ (-0.98457307 * float(x[6]))+ (0.1862478 * float(x[7]))) + 0.22843301), 0)
    h_46 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))) + 0.0), 0)
    h_47 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))) + 0.0), 0)
    o[0] = (0.63116676 * h_0)+ (0.6036917 * h_1)+ (-0.6665166 * h_2)+ (-0.3709836 * h_3)+ (1.4008796 * h_4)+ (-6.348959 * h_5)+ (1.0611253 * h_6)+ (0.4652892 * h_7)+ (-0.80511844 * h_8)+ (1.1163881 * h_9)+ (8.128773 * h_10)+ (1.6598815 * h_11)+ (-3.0992007 * h_12)+ (0.6224091 * h_13)+ (1.4495666 * h_14)+ (7.064472 * h_15)+ (-6.4789534 * h_16)+ (-33.51038 * h_17)+ (14.077294 * h_18)+ (1.7179041 * h_19)+ (11.93642 * h_20)+ (-2.024667 * h_21)+ (9.497714 * h_22)+ (-9.991077 * h_23)+ (-14.597242 * h_24)+ (-137.53117 * h_25)+ (5.996654 * h_26)+ (1.7562793 * h_27)+ (-125.63232 * h_28)+ (-63.92067 * h_29)+ (0.47699964 * h_30)+ (-6.689901 * h_31)+ (-3.6792665 * h_32)+ (-122.36308 * h_33)+ (45.214157 * h_34)+ (-26.235891 * h_35)+ (-134.06946 * h_36)+ (-0.7479014 * h_37)+ (1.7920452 * h_38)+ (-1.2872317 * h_39)+ (14.281065 * h_40)+ (-5.8716927 * h_41)+ (-6.886583 * h_42)+ (-63.37474 * h_43)+ (-16.049786 * h_44)+ (-17.812445 * h_45)+ (-7.0 * h_46)+ (0.0 * h_47) + 1.3807646
    o[1] = (1.0257772 * h_0)+ (1.0823497 * h_1)+ (2.5844514 * h_2)+ (2.1904697 * h_3)+ (0.6470605 * h_4)+ (-6.1994147 * h_5)+ (0.70291305 * h_6)+ (0.901699 * h_7)+ (2.7488863 * h_8)+ (0.7052501 * h_9)+ (8.317313 * h_10)+ (0.3581434 * h_11)+ (-0.400876 * h_12)+ (0.8751543 * h_13)+ (0.6688719 * h_14)+ (-4.112113 * h_15)+ (7.448862 * h_16)+ (-76.729355 * h_17)+ (8.075575 * h_18)+ (0.70926654 * h_19)+ (17.82036 * h_20)+ (8.328328 * h_21)+ (-125.058624 * h_22)+ (-15.149982 * h_23)+ (-1.7997938 * h_24)+ (9.002026 * h_25)+ (4.587117 * h_26)+ (0.7567081 * h_27)+ (28.936752 * h_28)+ (-1.6440711 * h_29)+ (2.2046432 * h_30)+ (7.442734 * h_31)+ (4.4746976 * h_32)+ (8.761135 * h_33)+ (-16.09044 * h_34)+ (-80.26871 * h_35)+ (21.582674 * h_36)+ (1.2584571 * h_37)+ (0.8028099 * h_38)+ (0.985893 * h_39)+ (10.0531 * h_40)+ (6.2425737 * h_41)+ (7.14038 * h_42)+ (18.795664 * h_43)+ (34.59524 * h_44)+ (37.08634 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + 1.2529978
    o[2] = (0.34435815 * h_0)+ (0.46782425 * h_1)+ (2.5167072 * h_2)+ (2.4945223 * h_3)+ (1.4078331 * h_4)+ (47.204338 * h_5)+ (1.372398 * h_6)+ (5.1682487 * h_7)+ (0.5538965 * h_8)+ (1.333186 * h_9)+ (-13.011699 * h_10)+ (0.75238866 * h_11)+ (-3.4190652 * h_12)+ (0.77928805 * h_13)+ (1.6598197 * h_14)+ (4.108498 * h_15)+ (7.665735 * h_16)+ (-56.973675 * h_17)+ (-8.369227 * h_18)+ (0.7063629 * h_19)+ (-0.346054 * h_20)+ (3.596869 * h_21)+ (-94.6329 * h_22)+ (3.4647615 * h_23)+ (14.363074 * h_24)+ (9.916258 * h_25)+ (9.989834 * h_26)+ (0.72979254 * h_27)+ (-113.720695 * h_28)+ (15.079124 * h_29)+ (1.653286 * h_30)+ (7.261159 * h_31)+ (4.2533283 * h_32)+ (51.994453 * h_33)+ (26.294699 * h_34)+ (140.45894 * h_35)+ (-11.754011 * h_36)+ (1.258845 * h_37)+ (0.75578797 * h_38)+ (9.224776 * h_39)+ (-8.393152 * h_40)+ (6.0563583 * h_41)+ (7.1199355 * h_42)+ (-0.5193429 * h_43)+ (-25.087622 * h_44)+ (-20.809313 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + 1.2427907
    o[3] = (1.597883 * h_0)+ (2.7115986 * h_1)+ (2.6772196 * h_2)+ (2.987034 * h_3)+ (4.895391 * h_4)+ (4.701021 * h_5)+ (8.749175 * h_6)+ (6.746918 * h_7)+ (6.7346554 * h_8)+ (3.3042364 * h_9)+ (39.16814 * h_10)+ (-0.15816611 * h_11)+ (6.539357 * h_12)+ (7.4155183 * h_13)+ (2.1981454 * h_14)+ (-28.84429 * h_15)+ (7.027684 * h_16)+ (29.865124 * h_17)+ (-54.002617 * h_18)+ (-0.4860074 * h_19)+ (-80.29737 * h_20)+ (-34.118137 * h_21)+ (27.820957 * h_22)+ (18.866985 * h_23)+ (-62.56812 * h_24)+ (8.2553005 * h_25)+ (-39.33719 * h_26)+ (-0.53135103 * h_27)+ (-72.7009 * h_28)+ (-53.816364 * h_29)+ (-1.0534351 * h_30)+ (7.210356 * h_31)+ (4.1780214 * h_32)+ (-2.4030325 * h_33)+ (-21.763903 * h_34)+ (22.226837 * h_35)+ (-72.617424 * h_36)+ (1.2387676 * h_37)+ (-0.61366034 * h_38)+ (20.437355 * h_39)+ (6.0848913 * h_40)+ (7.1305075 * h_41)+ (7.108104 * h_42)+ (1.8925309 * h_43)+ (2.9031556 * h_44)+ (6.2325306 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + -1.0410886
    o[4] = (5.9289103 * h_0)+ (3.6161118 * h_1)+ (2.1931458 * h_2)+ (3.0932684 * h_3)+ (5.1874127 * h_4)+ (4.8426375 * h_5)+ (1.5503488 * h_6)+ (1.1932526 * h_7)+ (-0.19237618 * h_8)+ (-0.6943782 * h_9)+ (20.683928 * h_10)+ (-0.3431906 * h_11)+ (8.233705 * h_12)+ (8.009494 * h_13)+ (3.6641397 * h_14)+ (5.541033 * h_15)+ (7.0170527 * h_16)+ (17.687872 * h_17)+ (-141.44777 * h_18)+ (-0.40362 * h_19)+ (-115.68555 * h_20)+ (-3.4426308 * h_21)+ (-61.74128 * h_22)+ (17.630928 * h_23)+ (-70.699615 * h_24)+ (-120.96592 * h_25)+ (0.8076723 * h_26)+ (-0.36688387 * h_27)+ (-64.2397 * h_28)+ (5.196489 * h_29)+ (1.3886845 * h_30)+ (7.2201853 * h_31)+ (4.158741 * h_32)+ (47.799324 * h_33)+ (-108.8439 * h_34)+ (-116.521805 * h_35)+ (-112.104294 * h_36)+ (1.2305295 * h_37)+ (-0.30910516 * h_38)+ (-89.63819 * h_39)+ (-16.688753 * h_40)+ (7.120607 * h_41)+ (7.081916 * h_42)+ (-23.027119 * h_43)+ (-6.4927917 * h_44)+ (-0.9737219 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + 2.246852
    o[5] = (2.15337 * h_0)+ (2.132955 * h_1)+ (2.733019 * h_2)+ (2.0831017 * h_3)+ (0.56830776 * h_4)+ (-4.965386 * h_5)+ (3.7249167 * h_6)+ (2.7340736 * h_7)+ (3.5058353 * h_8)+ (0.118926995 * h_9)+ (33.420456 * h_10)+ (4.3004246 * h_11)+ (8.049157 * h_12)+ (4.659251 * h_13)+ (2.14813 * h_14)+ (4.205746 * h_15)+ (6.7137322 * h_16)+ (-37.292747 * h_17)+ (-31.39907 * h_18)+ (-0.88337487 * h_19)+ (-125.86245 * h_20)+ (9.712554 * h_21)+ (-23.817715 * h_22)+ (28.455324 * h_23)+ (-8.4627495 * h_24)+ (-123.44651 * h_25)+ (23.291441 * h_26)+ (-0.9054139 * h_27)+ (30.173107 * h_28)+ (62.326042 * h_29)+ (0.38726133 * h_30)+ (7.041241 * h_31)+ (4.083069 * h_32)+ (-32.05327 * h_33)+ (-18.512552 * h_34)+ (7.9581347 * h_35)+ (74.557365 * h_36)+ (1.1158764 * h_37)+ (-0.91927433 * h_38)+ (-78.759674 * h_39)+ (1.3886486 * h_40)+ (7.1302724 * h_41)+ (7.1826525 * h_42)+ (-3.4392505 * h_43)+ (-8.629077 * h_44)+ (-2.1760945 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + -0.14496322
    o[6] = (5.943498 * h_0)+ (5.71696 * h_1)+ (5.487675 * h_2)+ (5.4838347 * h_3)+ (6.0696664 * h_4)+ (2.1691523 * h_5)+ (5.7926455 * h_6)+ (4.5132556 * h_7)+ (5.390436 * h_8)+ (5.8936 * h_9)+ (-52.562283 * h_10)+ (-0.65911275 * h_11)+ (-6.262021 * h_12)+ (6.80061 * h_13)+ (-0.8676345 * h_14)+ (-6.7480826 * h_15)+ (6.587554 * h_16)+ (-6.231866 * h_17)+ (2.2019074 * h_18)+ (-0.8212206 * h_19)+ (-11.21162 * h_20)+ (0.7074901 * h_21)+ (-8.354684 * h_22)+ (-29.795427 * h_23)+ (-4.1354036 * h_24)+ (6.4743705 * h_25)+ (11.446501 * h_26)+ (-0.87972105 * h_27)+ (-4.5463166 * h_28)+ (81.26249 * h_29)+ (-0.8450327 * h_30)+ (6.694983 * h_31)+ (3.681393 * h_32)+ (3.3456786 * h_33)+ (27.864393 * h_34)+ (23.859253 * h_35)+ (-4.7178636 * h_36)+ (5.7538114 * h_37)+ (-0.9299411 * h_38)+ (-123.28466 * h_39)+ (2.7296183 * h_40)+ (5.859743 * h_41)+ (6.8729315 * h_42)+ (31.096195 * h_43)+ (1.7300959 * h_44)+ (5.037312 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + 1.2002681
    o[7] = (5.484283 * h_0)+ (5.5886765 * h_1)+ (4.7920604 * h_2)+ (5.829095 * h_3)+ (1.3594186 * h_4)+ (-32.7773 * h_5)+ (0.9924284 * h_6)+ (6.7178254 * h_7)+ (6.0889544 * h_8)+ (0.749598 * h_9)+ (-131.33232 * h_10)+ (-0.31835037 * h_11)+ (3.4621212 * h_12)+ (-2.1520026 * h_13)+ (-0.027913637 * h_14)+ (0.6913012 * h_15)+ (6.287134 * h_16)+ (35.849125 * h_17)+ (12.292749 * h_18)+ (-0.30443996 * h_19)+ (6.0276504 * h_20)+ (1.4740306 * h_21)+ (38.477604 * h_22)+ (-2.538749 * h_23)+ (-10.107818 * h_24)+ (13.826809 * h_25)+ (-15.757207 * h_26)+ (-0.29251385 * h_27)+ (27.1295 * h_28)+ (-80.53138 * h_29)+ (0.47949374 * h_30)+ (7.0258455 * h_31)+ (4.2914047 * h_32)+ (-25.24307 * h_33)+ (6.8853984 * h_34)+ (-101.20115 * h_35)+ (1.0036724 * h_36)+ (1.2058189 * h_37)+ (-0.29802248 * h_38)+ (-147.67018 * h_39)+ (-11.627944 * h_40)+ (6.1916237 * h_41)+ (7.203096 * h_42)+ (3.4048529 * h_43)+ (-9.635301 * h_44)+ (-4.672712 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + -0.38677526
    o[8] = (7.7478633 * h_0)+ (7.7082186 * h_1)+ (7.994797 * h_2)+ (7.7808146 * h_3)+ (-4.3782134 * h_4)+ (2.7589014 * h_5)+ (0.71302605 * h_6)+ (3.860636 * h_7)+ (0.73558706 * h_8)+ (0.7160925 * h_9)+ (3.93644 * h_10)+ (-0.04252622 * h_11)+ (4.5305476 * h_12)+ (2.16727 * h_13)+ (-0.13658032 * h_14)+ (-10.392002 * h_15)+ (6.745165 * h_16)+ (11.748511 * h_17)+ (21.531199 * h_18)+ (-0.64477587 * h_19)+ (-60.454655 * h_20)+ (8.352461 * h_21)+ (-72.777466 * h_22)+ (53.996857 * h_23)+ (-131.993 * h_24)+ (27.261183 * h_25)+ (-48.951275 * h_26)+ (-0.66387194 * h_27)+ (-62.835842 * h_28)+ (1.6254513 * h_29)+ (0.4264318 * h_30)+ (7.046271 * h_31)+ (3.948763 * h_32)+ (74.86572 * h_33)+ (24.536417 * h_34)+ (-116.45114 * h_35)+ (-121.76622 * h_36)+ (1.1215794 * h_37)+ (-0.6757712 * h_38)+ (-149.52238 * h_39)+ (-7.786474 * h_40)+ (6.0795426 * h_41)+ (7.0659404 * h_42)+ (-67.55968 * h_43)+ (-2.5672934 * h_44)+ (1.6593287 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + 0.91636074
    o[9] = (1.7268797 * h_0)+ (1.5816184 * h_1)+ (17.42156 * h_2)+ (-10.215654 * h_3)+ (-5.950348 * h_4)+ (4.6560907 * h_5)+ (-5.3027163 * h_6)+ (4.059688 * h_7)+ (2.7498958 * h_8)+ (-15.5925865 * h_9)+ (-20.099916 * h_10)+ (1.5553886 * h_11)+ (-26.32278 * h_12)+ (31.474005 * h_13)+ (3.972164 * h_14)+ (-21.716831 * h_15)+ (6.7495027 * h_16)+ (-5.1777253 * h_17)+ (-12.420664 * h_18)+ (-0.45039693 * h_19)+ (-5.651546 * h_20)+ (-5.207754 * h_21)+ (-6.8896146 * h_22)+ (-2.7986617 * h_23)+ (-6.1052785 * h_24)+ (-16.136585 * h_25)+ (15.777576 * h_26)+ (-0.6435624 * h_27)+ (-11.119215 * h_28)+ (4.8959846 * h_29)+ (0.8006639 * h_30)+ (7.0251865 * h_31)+ (3.9787574 * h_32)+ (10.737999 * h_33)+ (-4.7977967 * h_34)+ (-3.7665157 * h_35)+ (-2.0230615 * h_36)+ (1.1172861 * h_37)+ (-0.7273122 * h_38)+ (-27.527214 * h_39)+ (-5.009186 * h_40)+ (6.0541854 * h_41)+ (7.015518 * h_42)+ (18.735523 * h_43)+ (3.7541912 * h_44)+ (6.8680267 * h_45)+ (7.0 * h_46)+ (0.0 * h_47) + -14.5636015

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
            for valrow in valcsvreader:
                if len(valrow) == 0:
                    continue
                if int(single_classify(valrow[:-1])) == int(float(valrow[-1])):
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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0

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
                    numeachclass[y_true] = 0
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
        print("Classifier Type: Neural Network")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap=864
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

