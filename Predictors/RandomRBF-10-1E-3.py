#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/2425/RandomRBF_10_1E-3.arff -o Predictors/RandomRBF-10-1E-3_NN.py -target class -stopat 85.51 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 5:44:01.14. Finished on: Jun-06-2020 04:53:30.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        5-way classifier
Best-guess accuracy:                30.01%
Model accuracy:                     81.29% (812909/999999 correct)
Improvement over best guess:        51.28% (of possible 69.99%)
Model capacity (MEC):               725 bits
Generalization ratio:               1121.25 bits/bit
Confusion Matrix:
 [25.31% 1.52% 1.26% 1.24% 0.68%]
 [2.00% 14.18% 0.73% 0.55% 0.51%]
 [1.19% 0.54% 23.25% 0.66% 0.47%]
 [1.30% 0.45% 0.69% 13.74% 0.45%]
 [1.87% 0.96% 0.89% 0.74% 4.80%]

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
TRAINFILE = "RandomRBF_10_1E-3.csv"


#Number of output logits
num_output_logits = 5

#Number of attributes
num_attr = 10
n_classes = 5


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
    clean.mapping={'class5': 0, 'class3': 1, 'class1': 2, 'class2': 3, 'class4': 4}

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
    h_0 = max((((0.24897428 * float(x[0]))+ (0.8034107 * float(x[1]))+ (0.9922967 * float(x[2]))+ (1.0824105 * float(x[3]))+ (0.91325665 * float(x[4]))+ (1.6579732 * float(x[5]))+ (0.76348484 * float(x[6]))+ (0.6335229 * float(x[7]))+ (0.8405756 * float(x[8]))+ (0.47589168 * float(x[9]))) + 5.504533), 0)
    h_1 = max((((0.21803425 * float(x[0]))+ (1.4907628 * float(x[1]))+ (1.1029164 * float(x[2]))+ (0.9562938 * float(x[3]))+ (0.5384897 * float(x[4]))+ (1.5578781 * float(x[5]))+ (1.305517 * float(x[6]))+ (0.89549255 * float(x[7]))+ (1.0554744 * float(x[8]))+ (0.92917717 * float(x[9]))) + 6.742019), 0)
    h_2 = max((((0.24860911 * float(x[0]))+ (1.5905818 * float(x[1]))+ (0.8945899 * float(x[2]))+ (1.2035218 * float(x[3]))+ (0.8287441 * float(x[4]))+ (1.8955113 * float(x[5]))+ (1.1166348 * float(x[6]))+ (0.8652709 * float(x[7]))+ (0.8049938 * float(x[8]))+ (0.7065832 * float(x[9]))) + 7.056301), 0)
    h_3 = max((((1.5092806 * float(x[0]))+ (2.7363753 * float(x[1]))+ (3.106684 * float(x[2]))+ (2.485577 * float(x[3]))+ (3.162168 * float(x[4]))+ (3.5743954 * float(x[5]))+ (2.760955 * float(x[6]))+ (1.5979527 * float(x[7]))+ (2.925856 * float(x[8]))+ (1.9604186 * float(x[9]))) + 10.477948), 0)
    h_4 = max((((0.32025176 * float(x[0]))+ (0.7270952 * float(x[1]))+ (0.9583573 * float(x[2]))+ (1.0382576 * float(x[3]))+ (0.9565698 * float(x[4]))+ (1.4135793 * float(x[5]))+ (0.7211603 * float(x[6]))+ (0.42719376 * float(x[7]))+ (0.93493587 * float(x[8]))+ (0.4396327 * float(x[9]))) + 4.980432), 0)
    h_5 = max((((0.5186845 * float(x[0]))+ (0.520746 * float(x[1]))+ (0.5187444 * float(x[2]))+ (0.7158468 * float(x[3]))+ (1.0272363 * float(x[4]))+ (0.86009395 * float(x[5]))+ (0.4344403 * float(x[6]))+ (0.23422594 * float(x[7]))+ (0.62209034 * float(x[8]))+ (0.36682546 * float(x[9]))) + 3.9342067), 0)
    h_6 = max((((0.6852419 * float(x[0]))+ (0.9659389 * float(x[1]))+ (0.88953406 * float(x[2]))+ (1.2469639 * float(x[3]))+ (1.1999509 * float(x[4]))+ (0.7641891 * float(x[5]))+ (0.9696431 * float(x[6]))+ (0.5345433 * float(x[7]))+ (1.2915132 * float(x[8]))+ (0.4994226 * float(x[9]))) + 4.190643), 0)
    h_7 = max((((0.3599822 * float(x[0]))+ (0.69988817 * float(x[1]))+ (0.6642578 * float(x[2]))+ (0.65139633 * float(x[3]))+ (0.8729752 * float(x[4]))+ (0.96742 * float(x[5]))+ (0.66457814 * float(x[6]))+ (0.5092551 * float(x[7]))+ (0.97393835 * float(x[8]))+ (0.48823944 * float(x[9]))) + 3.1406121), 0)
    h_8 = max((((0.2659823 * float(x[0]))+ (0.69703627 * float(x[1]))+ (0.7418161 * float(x[2]))+ (0.79249865 * float(x[3]))+ (0.8701144 * float(x[4]))+ (1.0063126 * float(x[5]))+ (0.4525652 * float(x[6]))+ (0.5192463 * float(x[7]))+ (1.1265099 * float(x[8]))+ (0.5144208 * float(x[9]))) + 3.1907642), 0)
    h_9 = max((((0.4943057 * float(x[0]))+ (0.1705109 * float(x[1]))+ (0.71258014 * float(x[2]))+ (0.7905496 * float(x[3]))+ (1.1794604 * float(x[4]))+ (0.27386466 * float(x[5]))+ (0.4620106 * float(x[6]))+ (0.1565744 * float(x[7]))+ (1.4227831 * float(x[8]))+ (0.10319123 * float(x[9]))) + 3.7417562), 0)
    h_10 = max((((0.22878943 * float(x[0]))+ (0.08436735 * float(x[1]))+ (0.32491753 * float(x[2]))+ (0.7181819 * float(x[3]))+ (0.49012467 * float(x[4]))+ (0.3921572 * float(x[5]))+ (0.24134956 * float(x[6]))+ (0.17395797 * float(x[7]))+ (0.94913507 * float(x[8]))+ (-0.031370316 * float(x[9]))) + 4.002415), 0)
    h_11 = max((((0.27758005 * float(x[0]))+ (0.29845655 * float(x[1]))+ (0.1585103 * float(x[2]))+ (0.6549239 * float(x[3]))+ (0.85804814 * float(x[4]))+ (0.63708866 * float(x[5]))+ (0.13561843 * float(x[6]))+ (0.074579544 * float(x[7]))+ (0.54274046 * float(x[8]))+ (0.18761352 * float(x[9]))) + 1.9949746), 0)
    h_12 = max((((0.26576138 * float(x[0]))+ (0.0916755 * float(x[1]))+ (0.075830616 * float(x[2]))+ (0.772403 * float(x[3]))+ (1.0265927 * float(x[4]))+ (0.3734581 * float(x[5]))+ (0.056220736 * float(x[6]))+ (0.26088956 * float(x[7]))+ (0.6391224 * float(x[8]))+ (0.09914195 * float(x[9]))) + 2.6633048), 0)
    h_13 = max((((0.31531295 * float(x[0]))+ (0.3352607 * float(x[1]))+ (0.40932354 * float(x[2]))+ (0.51790947 * float(x[3]))+ (0.7873538 * float(x[4]))+ (0.6341505 * float(x[5]))+ (0.10258813 * float(x[6]))+ (0.3079028 * float(x[7]))+ (0.6486709 * float(x[8]))+ (0.29874167 * float(x[9]))) + 2.107032), 0)
    h_14 = max((((0.1886148 * float(x[0]))+ (0.76015216 * float(x[1]))+ (0.018772023 * float(x[2]))+ (0.64792037 * float(x[3]))+ (0.9750886 * float(x[4]))+ (0.4549841 * float(x[5]))+ (0.61483383 * float(x[6]))+ (0.18839514 * float(x[7]))+ (0.25275397 * float(x[8]))+ (0.52893907 * float(x[9]))) + 2.0786052), 0)
    h_15 = max((((0.050205104 * float(x[0]))+ (0.25049305 * float(x[1]))+ (0.92439127 * float(x[2]))+ (0.82642543 * float(x[3]))+ (1.0331516 * float(x[4]))+ (0.92705786 * float(x[5]))+ (0.365502 * float(x[6]))+ (0.5922217 * float(x[7]))+ (0.97718084 * float(x[8]))+ (0.5907273 * float(x[9]))) + 4.4535637), 0)
    h_16 = max((((-0.5513661 * float(x[0]))+ (0.28225383 * float(x[1]))+ (0.3507346 * float(x[2]))+ (0.84531814 * float(x[3]))+ (1.6891078 * float(x[4]))+ (0.9119656 * float(x[5]))+ (-0.20760094 * float(x[6]))+ (-0.21823564 * float(x[7]))+ (1.328833 * float(x[8]))+ (-0.5773475 * float(x[9]))) + 6.647362), 0)
    h_17 = max((((1.9584008 * float(x[0]))+ (2.252957 * float(x[1]))+ (2.4413626 * float(x[2]))+ (2.3648324 * float(x[3]))+ (2.4079368 * float(x[4]))+ (2.164831 * float(x[5]))+ (2.5192904 * float(x[6]))+ (1.7112976 * float(x[7]))+ (2.875751 * float(x[8]))+ (1.8680371 * float(x[9]))) + 6.9485846), 0)
    h_18 = max((((-0.33002406 * float(x[0]))+ (0.6010566 * float(x[1]))+ (0.42530507 * float(x[2]))+ (0.11067787 * float(x[3]))+ (-0.6051468 * float(x[4]))+ (0.0045815944 * float(x[5]))+ (-0.056513425 * float(x[6]))+ (-0.57099736 * float(x[7]))+ (1.5712663 * float(x[8]))+ (-0.6700432 * float(x[9]))) + -0.33773544), 0)
    h_19 = max((((1.1347641 * float(x[0]))+ (-0.54736644 * float(x[1]))+ (-0.79497254 * float(x[2]))+ (0.4421804 * float(x[3]))+ (0.37476343 * float(x[4]))+ (-0.9477638 * float(x[5]))+ (1.2096366 * float(x[6]))+ (-0.1535802 * float(x[7]))+ (0.3752562 * float(x[8]))+ (-0.51638925 * float(x[9]))) + 0.0004628246), 0)
    h_20 = max((((-0.2743488 * float(x[0]))+ (1.7224264 * float(x[1]))+ (0.743694 * float(x[2]))+ (0.7450911 * float(x[3]))+ (0.15018298 * float(x[4]))+ (1.8464754 * float(x[5]))+ (0.24947987 * float(x[6]))+ (0.892386 * float(x[7]))+ (1.1370721 * float(x[8]))+ (0.56051713 * float(x[9]))) + 5.5271316), 0)
    h_21 = max((((-0.119486324 * float(x[0]))+ (-0.52216405 * float(x[1]))+ (0.9516915 * float(x[2]))+ (-0.09855065 * float(x[3]))+ (-0.4163633 * float(x[4]))+ (0.031717066 * float(x[5]))+ (-0.33881068 * float(x[6]))+ (0.021637276 * float(x[7]))+ (1.2329192 * float(x[8]))+ (-0.909448 * float(x[9]))) + 0.16883071), 0)
    h_22 = max((((-0.78464293 * float(x[0]))+ (-0.49528265 * float(x[1]))+ (0.028174562 * float(x[2]))+ (-0.40756646 * float(x[3]))+ (-0.94022286 * float(x[4]))+ (0.55709225 * float(x[5]))+ (0.97345424 * float(x[6]))+ (-0.47221744 * float(x[7]))+ (-1.1682338 * float(x[8]))+ (-0.2993202 * float(x[9]))) + 0.93222725), 0)
    h_23 = max((((-0.722492 * float(x[0]))+ (0.8305595 * float(x[1]))+ (-1.4207655 * float(x[2]))+ (0.116050564 * float(x[3]))+ (-0.5909926 * float(x[4]))+ (-0.45512033 * float(x[5]))+ (0.4438898 * float(x[6]))+ (0.21661505 * float(x[7]))+ (0.10456645 * float(x[8]))+ (-0.041404508 * float(x[9]))) + 0.037733406), 0)
    h_24 = max((((-0.57081443 * float(x[0]))+ (2.2766094 * float(x[1]))+ (-0.81364554 * float(x[2]))+ (-2.0492153 * float(x[3]))+ (-2.914295 * float(x[4]))+ (-0.53599876 * float(x[5]))+ (2.9328744 * float(x[6]))+ (2.1927955 * float(x[7]))+ (-0.8782664 * float(x[8]))+ (-0.28298602 * float(x[9]))) + 0.21082465), 0)
    h_25 = max((((-1.1377997 * float(x[0]))+ (-1.3873212 * float(x[1]))+ (-0.23042749 * float(x[2]))+ (-0.04818442 * float(x[3]))+ (-0.6489347 * float(x[4]))+ (-0.56416696 * float(x[5]))+ (-1.0306817 * float(x[6]))+ (-2.350605 * float(x[7]))+ (-0.58772045 * float(x[8]))+ (-0.3014342 * float(x[9]))) + -2.178446), 0)
    h_26 = max((((-0.5542707 * float(x[0]))+ (0.6608981 * float(x[1]))+ (0.0684259 * float(x[2]))+ (1.1908935 * float(x[3]))+ (-0.518861 * float(x[4]))+ (0.36062196 * float(x[5]))+ (1.0780605 * float(x[6]))+ (0.6841318 * float(x[7]))+ (0.78519416 * float(x[8]))+ (-0.036603507 * float(x[9]))) + -2.5904741), 0)
    h_27 = max((((0.107107766 * float(x[0]))+ (0.13171059 * float(x[1]))+ (0.34732118 * float(x[2]))+ (0.16511236 * float(x[3]))+ (-0.065555036 * float(x[4]))+ (0.19385704 * float(x[5]))+ (0.29886878 * float(x[6]))+ (0.26119974 * float(x[7]))+ (0.36235097 * float(x[8]))+ (0.44234753 * float(x[9]))) + 0.90538335), 0)
    h_28 = max((((-0.004727901 * float(x[0]))+ (-0.37849927 * float(x[1]))+ (-0.14200641 * float(x[2]))+ (0.42964825 * float(x[3]))+ (0.49807262 * float(x[4]))+ (0.12278797 * float(x[5]))+ (-0.32781413 * float(x[6]))+ (-0.43358386 * float(x[7]))+ (0.07592659 * float(x[8]))+ (-0.097692125 * float(x[9]))) + -0.1341539), 0)
    h_29 = max((((-0.9370747 * float(x[0]))+ (-0.55080587 * float(x[1]))+ (-0.18602012 * float(x[2]))+ (-0.05746479 * float(x[3]))+ (-0.5669907 * float(x[4]))+ (0.36906478 * float(x[5]))+ (0.6523266 * float(x[6]))+ (-0.74755794 * float(x[7]))+ (-0.7493937 * float(x[8]))+ (-0.16527335 * float(x[9]))) + 1.2961049), 0)
    h_30 = max((((0.24935713 * float(x[0]))+ (0.28930244 * float(x[1]))+ (0.2659271 * float(x[2]))+ (-0.029689377 * float(x[3]))+ (-0.46043554 * float(x[4]))+ (-0.75688 * float(x[5]))+ (0.4670325 * float(x[6]))+ (0.5951932 * float(x[7]))+ (-0.091591254 * float(x[8]))+ (-0.31549427 * float(x[9]))) + -0.5340513), 0)
    h_31 = max((((-0.61964613 * float(x[0]))+ (0.86214757 * float(x[1]))+ (0.11362644 * float(x[2]))+ (0.66225344 * float(x[3]))+ (0.00955098 * float(x[4]))+ (-0.16148105 * float(x[5]))+ (-0.28304783 * float(x[6]))+ (-0.067462966 * float(x[7]))+ (0.35436785 * float(x[8]))+ (0.47364792 * float(x[9]))) + -1.1040777), 0)
    h_32 = max((((-0.63070655 * float(x[0]))+ (1.4674592 * float(x[1]))+ (0.06660747 * float(x[2]))+ (0.7439347 * float(x[3]))+ (-0.21731329 * float(x[4]))+ (-1.4664606 * float(x[5]))+ (-0.0077678175 * float(x[6]))+ (-1.1411895 * float(x[7]))+ (0.522132 * float(x[8]))+ (-0.8487652 * float(x[9]))) + 0.508415), 0)
    h_33 = max((((-0.25055057 * float(x[0]))+ (-0.11844808 * float(x[1]))+ (-0.22737724 * float(x[2]))+ (0.13672887 * float(x[3]))+ (0.2571655 * float(x[4]))+ (-0.1664414 * float(x[5]))+ (-0.0073179784 * float(x[6]))+ (-0.17109205 * float(x[7]))+ (0.30051577 * float(x[8]))+ (-0.5285578 * float(x[9]))) + 3.392632), 0)
    h_34 = max((((0.5130302 * float(x[0]))+ (-1.4860275 * float(x[1]))+ (-0.07427903 * float(x[2]))+ (-1.1266412 * float(x[3]))+ (0.2861114 * float(x[4]))+ (0.2699504 * float(x[5]))+ (1.0480386 * float(x[6]))+ (0.35931212 * float(x[7]))+ (-0.17330222 * float(x[8]))+ (-0.769365 * float(x[9]))) + 0.30645725), 0)
    h_35 = max((((0.033426754 * float(x[0]))+ (0.21667087 * float(x[1]))+ (-0.5144529 * float(x[2]))+ (0.038912293 * float(x[3]))+ (-0.05684424 * float(x[4]))+ (1.2843585 * float(x[5]))+ (0.2966767 * float(x[6]))+ (-1.4979513 * float(x[7]))+ (0.56932425 * float(x[8]))+ (0.25167578 * float(x[9]))) + -0.814876), 0)
    h_36 = max((((-1.1736555 * float(x[0]))+ (0.060472462 * float(x[1]))+ (1.1051488 * float(x[2]))+ (0.4411952 * float(x[3]))+ (-0.67732674 * float(x[4]))+ (1.0950286 * float(x[5]))+ (-0.55302405 * float(x[6]))+ (0.5822632 * float(x[7]))+ (0.008567608 * float(x[8]))+ (0.15362117 * float(x[9]))) + -1.3259293), 0)
    h_37 = max((((-1.402838 * float(x[0]))+ (-1.0862514 * float(x[1]))+ (0.9036138 * float(x[2]))+ (-0.3905458 * float(x[3]))+ (1.138078 * float(x[4]))+ (0.48970094 * float(x[5]))+ (-0.76634246 * float(x[6]))+ (-0.8655771 * float(x[7]))+ (0.9485932 * float(x[8]))+ (-0.3165834 * float(x[9]))) + -0.24898829), 0)
    h_38 = max((((-0.30184156 * float(x[0]))+ (0.17736538 * float(x[1]))+ (0.18592405 * float(x[2]))+ (-0.22586374 * float(x[3]))+ (-0.71844596 * float(x[4]))+ (-0.7142156 * float(x[5]))+ (-0.23411383 * float(x[6]))+ (-0.3337317 * float(x[7]))+ (0.8673929 * float(x[8]))+ (-1.108408 * float(x[9]))) + 0.491978), 0)
    h_39 = max((((-1.7983917 * float(x[0]))+ (-1.6036837 * float(x[1]))+ (-0.3897253 * float(x[2]))+ (-1.359707 * float(x[3]))+ (-1.0581914 * float(x[4]))+ (-1.3694738 * float(x[5]))+ (-0.23889175 * float(x[6]))+ (-1.8418823 * float(x[7]))+ (-1.3597028 * float(x[8]))+ (-0.5610691 * float(x[9]))) + -0.98768866), 0)
    h_40 = max((((1.8648599 * float(x[0]))+ (-0.18009216 * float(x[1]))+ (-0.29202 * float(x[2]))+ (1.492236 * float(x[3]))+ (-0.7429842 * float(x[4]))+ (-0.92120135 * float(x[5]))+ (0.99749875 * float(x[6]))+ (0.09955764 * float(x[7]))+ (0.59809864 * float(x[8]))+ (-0.3324452 * float(x[9]))) + -1.3036562), 0)
    h_41 = max((((-0.818392 * float(x[0]))+ (0.7424275 * float(x[1]))+ (0.032550044 * float(x[2]))+ (0.70237005 * float(x[3]))+ (0.57050824 * float(x[4]))+ (-0.5689395 * float(x[5]))+ (0.29360262 * float(x[6]))+ (-0.5345384 * float(x[7]))+ (-0.62903744 * float(x[8]))+ (0.1411727 * float(x[9]))) + -0.6663884), 0)
    h_42 = max((((0.27500913 * float(x[0]))+ (0.6905889 * float(x[1]))+ (-0.7914219 * float(x[2]))+ (-1.0042666 * float(x[3]))+ (-0.3602209 * float(x[4]))+ (0.5461325 * float(x[5]))+ (0.7510164 * float(x[6]))+ (-0.88646686 * float(x[7]))+ (0.83158684 * float(x[8]))+ (-0.45133835 * float(x[9]))) + -0.4943159), 0)
    h_43 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))) + 0.0), 0)
    h_44 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))) + 0.0), 0)
    o[0] = (-1.9070383 * h_0)+ (1.1483097 * h_1)+ (2.2307978 * h_2)+ (2.2646995 * h_3)+ (-1.9464021 * h_4)+ (0.016573286 * h_5)+ (-0.016587848 * h_6)+ (-2.0950024 * h_7)+ (-1.1047252 * h_8)+ (-0.058806095 * h_9)+ (1.0343585 * h_10)+ (-0.17993315 * h_11)+ (0.94032687 * h_12)+ (-2.1876273 * h_13)+ (0.9274976 * h_14)+ (1.0689499 * h_15)+ (2.4368122 * h_16)+ (2.2100399 * h_17)+ (2.1135101 * h_18)+ (-1.9316018 * h_19)+ (2.1451337 * h_20)+ (-2.2403843 * h_21)+ (3.3555295 * h_22)+ (3.9184546 * h_23)+ (-0.1223756 * h_24)+ (5.771108 * h_25)+ (0.3379078 * h_26)+ (-0.7108265 * h_27)+ (5.5135894 * h_28)+ (-1.5513058 * h_29)+ (-2.0337172 * h_30)+ (-2.6692636 * h_31)+ (-0.093163595 * h_32)+ (2.301857 * h_33)+ (3.2258656 * h_34)+ (1.1378068 * h_35)+ (2.7238414 * h_36)+ (0.19446056 * h_37)+ (0.6863827 * h_38)+ (0.39955783 * h_39)+ (3.0727773 * h_40)+ (1.6568141 * h_41)+ (-0.6228893 * h_42)+ (-1.0 * h_43)+ (0.0 * h_44) + 2.0699782
    o[1] = (3.2381866 * h_0)+ (0.45176393 * h_1)+ (-0.63991433 * h_2)+ (-0.67065257 * h_3)+ (3.291387 * h_4)+ (1.3006887 * h_5)+ (1.4280992 * h_6)+ (3.3417478 * h_7)+ (2.3633943 * h_8)+ (1.5323248 * h_9)+ (0.5862439 * h_10)+ (1.3283072 * h_11)+ (0.44198513 * h_12)+ (3.365283 * h_13)+ (0.31816667 * h_14)+ (0.4685016 * h_15)+ (-0.29630476 * h_16)+ (-0.28412947 * h_17)+ (-0.2850795 * h_18)+ (0.22867975 * h_19)+ (-0.33912703 * h_20)+ (-0.36365682 * h_21)+ (-2.299589 * h_22)+ (-1.3498329 * h_23)+ (2.6843617 * h_24)+ (-19.67712 * h_25)+ (-0.46343485 * h_26)+ (1.4215057 * h_27)+ (-6.918799 * h_28)+ (3.2436721 * h_29)+ (-6.60642 * h_30)+ (-0.457126 * h_31)+ (1.4109912 * h_32)+ (0.5288069 * h_33)+ (-0.22845452 * h_34)+ (2.1043563 * h_35)+ (0.46966687 * h_36)+ (-0.61852366 * h_37)+ (4.247834 * h_38)+ (-21.308678 * h_39)+ (1.1164167 * h_40)+ (4.2837095 * h_41)+ (-1.0364723 * h_42)+ (1.0 * h_43)+ (0.0 * h_44) + -0.29305944
    o[2] = (3.146916 * h_0)+ (1.1215624 * h_1)+ (1.0647132 * h_2)+ (-1.0482546 * h_3)+ (3.144503 * h_4)+ (1.201647 * h_5)+ (0.18376297 * h_6)+ (3.2677276 * h_7)+ (2.274729 * h_8)+ (0.15786262 * h_9)+ (0.17874958 * h_10)+ (1.339557 * h_11)+ (0.19769382 * h_12)+ (3.297338 * h_13)+ (0.31835505 * h_14)+ (0.15425448 * h_15)+ (-1.1515738 * h_16)+ (-0.9604837 * h_17)+ (-3.3959203 * h_18)+ (5.6141863 * h_19)+ (3.2406476 * h_20)+ (2.1757178 * h_21)+ (0.15066741 * h_22)+ (-1.4421428 * h_23)+ (-0.32229316 * h_24)+ (-24.247124 * h_25)+ (4.6556935 * h_26)+ (1.763044 * h_27)+ (0.052292965 * h_28)+ (0.7667421 * h_29)+ (4.3100457 * h_30)+ (-1.175144 * h_31)+ (4.503588 * h_32)+ (-1.0228932 * h_33)+ (-0.41042697 * h_34)+ (-2.50375 * h_35)+ (-3.0881822 * h_36)+ (3.6844628 * h_37)+ (-2.6830518 * h_38)+ (6.33786 * h_39)+ (-1.9592086 * h_40)+ (-2.0557797 * h_41)+ (3.4720213 * h_42)+ (1.0 * h_43)+ (0.0 * h_44) + -0.84671706
    o[3] = (1.7128239 * h_0)+ (-1.4162399 * h_1)+ (1.4570297 * h_2)+ (2.5099354 * h_3)+ (-1.2330838 * h_4)+ (2.7951937 * h_5)+ (-1.1723006 * h_6)+ (0.9469404 * h_7)+ (-0.03929501 * h_8)+ (-0.12548399 * h_9)+ (1.673255 * h_10)+ (3.045548 * h_11)+ (2.8484013 * h_12)+ (0.075657316 * h_13)+ (1.8748847 * h_14)+ (2.67213 * h_15)+ (2.113318 * h_16)+ (-1.4280002 * h_17)+ (0.81385463 * h_18)+ (-0.32356292 * h_19)+ (0.5427682 * h_20)+ (2.9251359 * h_21)+ (-2.4642284 * h_22)+ (0.41408852 * h_23)+ (1.9008723 * h_24)+ (7.374252 * h_25)+ (-0.64431804 * h_26)+ (-0.19399714 * h_27)+ (-3.130039 * h_28)+ (5.868698 * h_29)+ (0.71427935 * h_30)+ (5.6580267 * h_31)+ (0.95584756 * h_32)+ (-1.8557394 * h_33)+ (-0.24362078 * h_34)+ (-2.182114 * h_35)+ (0.9513648 * h_36)+ (-1.7647885 * h_37)+ (-1.378871 * h_38)+ (2.3608744 * h_39)+ (2.9513814 * h_40)+ (-1.5736616 * h_41)+ (4.7063446 * h_42)+ (1.0 * h_43)+ (0.0 * h_44) + -1.6060208
    o[4] = (2.852042 * h_0)+ (1.7210531 * h_1)+ (1.6891726 * h_2)+ (-1.5127023 * h_3)+ (2.8655186 * h_4)+ (0.9608189 * h_5)+ (2.8761966 * h_6)+ (2.9933512 * h_7)+ (1.9789864 * h_8)+ (0.93424475 * h_9)+ (-0.07196269 * h_10)+ (1.1299541 * h_11)+ (0.022211008 * h_12)+ (3.0721593 * h_13)+ (3.011599 * h_14)+ (-1.21037 * h_15)+ (-0.5507737 * h_16)+ (-0.5077109 * h_17)+ (1.9195976 * h_18)+ (0.8816156 * h_19)+ (-0.37316474 * h_20)+ (1.8598326 * h_21)+ (0.9336379 * h_22)+ (-0.20121536 * h_23)+ (1.0712538 * h_24)+ (-12.89998 * h_25)+ (-0.66151154 * h_26)+ (-0.63560283 * h_27)+ (-3.7917871 * h_28)+ (-0.55381024 * h_29)+ (-1.4812766 * h_30)+ (0.21970975 * h_31)+ (-0.2258498 * h_32)+ (1.565037 * h_33)+ (0.16349745 * h_34)+ (-0.98817474 * h_35)+ (2.097371 * h_36)+ (1.7251656 * h_37)+ (-1.5195483 * h_38)+ (-1.7779394 * h_39)+ (0.8373644 * h_40)+ (1.1165614 * h_41)+ (0.19233842 * h_42)+ (1.0 * h_43)+ (0.0 * h_44) + -0.9671126

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
        model_cap=725
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

