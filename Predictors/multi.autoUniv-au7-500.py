#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Class autoUniv-au7-500.csv -o autoUniv-au7-500.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:11.03. Finished on: Sep-04-2020 10:20:00.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         5-way classifier
Best-guess accuracy:                 38.40%
Overall Model accuracy:              100.00% (500/500 correct)
Overall Improvement over best guess: 61.60% (of possible 61.6%)
Model capacity (MEC):                330 bits
Generalization ratio:                1.51 bits/bit
Model efficiency:                    0.18%/parameter
Confusion Matrix:
 [14.20% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 21.80% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 38.40% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 17.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 8.60%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to 'class2'=0, 'class4'=1, 'class5'=2, 'class3'=3, 'class1'=4.
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
TRAINFILE = "autoUniv-au7-500.csv"


#Number of attributes
num_attr = 12
n_classes = 5


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="Class"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="Class"
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
    clean.mapping={'class2': 0, 'class4': 1, 'class5': 2, 'class3': 3, 'class1': 4}

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


# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([7072336904.184999, 7072336922.85, 7072337521.855, 7072337809.145, 7072337977.91, 7072338035.58, 7072340389.094999, 7072340615.225, 7072340801.115, 7072340883.975, 7072340956.865, 7324320932.84, 7576300686.24, 7576301270.41, 7576301339.75, 7576301621.33, 7576301922.754999, 7576302401.264999, 7576302666.21, 7576302914.4, 7576303432.425, 7576304076.955, 7576304161.66, 7576304628.275, 7576304672.24, 7576304739.095, 7576304770.485001, 7576304772.34, 7576304892.3, 7576305144.59, 7576305188.5, 7828284827.055, 8080264436.37, 8080264892.059999, 8080265835.82, 8709052372.375, 9337838431.619999, 9337838458.265, 9337838538.935, 9337838568.7, 9337838613.46, 9337838646.365, 9337838662.985, 9337838672.724998, 9337838702.14, 9337838710.61, 9337838745.4, 9337838787.785, 9337838889.895, 9337838913.69, 9337839191.470001, 9337839244.57, 9337839261.765, 9337839305.3, 9337839341.150002, 9337839354.89, 9337839381.25, 9337839448.81, 9337839564.895, 9337839654.445, 9337839669.41, 9337839688.039999, 9337839702.044998, 9337839717.914999, 9337839764.68, 9337839828.015, 9337839850.63, 9337839858.395, 9337839872.575, 9337839883.615, 9337839884.64, 9337839902.035, 9337839920.485, 9337839921.724998, 9337839940.715, 9337839956.29, 9337839968.869999, 9337839981.27, 9337840032.939999, 9337840082.794998, 9337840093.875, 9337840215.670002, 9337840229.77, 9337840235.195, 9337840243.015, 9337840251.23, 9337840263.925, 9337840275.66, 9337840284.32, 9337840285.285, 9337840286.585001, 9337840335.045002, 9337840403.66, 9337840511.849998, 9337840553.08, 9337840693.810001, 9337840800.71, 9337841178.435001, 9337841215.35, 9337841242.765, 9337841279.489998, 9337841306.785, 9337841386.43, 9337841532.115, 9337841700.25, 9337841738.7, 9337841830.375, 9337842049.8, 9337842080.675, 9337842093.895, 9337842115.465, 9337842141.11, 9337842170.69, 9337842204.68, 9337842245.99, 9337842288.314999, 9337842325.465, 9337842341.635002, 9337842347.68, 9337842363.555, 9337842454.935001, 9337842471.305, 9337842484.81, 9337842507.35, 9337842534.77, 9337842563.779999, 9337842581.085, 9337842622.720001, 9337842634.245, 9337842640.29, 9337842679.420002, 9337842705.755001, 9337842732.105, 9337842747.494999, 9337842811.815, 9337842866.695, 9337842881.09, 9337842905.220001, 9337842926.060001, 9337842970.785, 9589822702.375, 9841802255.525, 9841802450.195, 9841802581.85, 9841802674.820002, 9841802686.075, 9841802726.82, 9841802789.965, 9841802810.605, 9841802845.075, 9841802887.54, 9841802920.36, 9841802922.895, 9841802930.195, 9841803025.675, 9841803123.14, 9841803151.494999, 9841803217.255, 9841803244.41, 9841803293.285, 9841803330.39, 9841803395.645, 9841803661.465, 9841803675.525, 9841803772.21, 9841803835.68, 9841803842.68, 9841803908.010002, 9841803971.295, 9841804002.474998, 9841804062.615, 9841804102.385, 9841804126.900002, 9841804210.3, 9841804245.560001, 9841804925.025002, 9841805071.810001, 9841805868.825, 9841806205.37, 9841806255.895, 9841806356.91, 9841806369.355, 9841806379.59, 9841806394.215, 9841806420.965, 9841806442.600002, 9841806592.32, 9841806681.335, 9841806750.585, 9841806807.73, 9841806939.21, 9841807048.055, 10093786682.29, 10345766695.305, 10345767275.63, 10345768434.055, 10974554473.715, 11603340182.925, 11603340192.965, 11603340239.125, 11603340281.55, 11603340293.77, 11603340340.24, 11603340378.68, 11603340385.205, 11603340399.010002, 11603340452.420002, 11603340521.375, 11603340566.16, 11603340586.429998, 11603340690.884998, 11603340742.8, 11603340816.545002, 11603340850.115002, 11603340870.21, 11603340957.195, 11603341028.905, 11603341066.255001, 11603341202.6, 11603341342.994999, 11603341410.404999, 11603341496.905, 11603341615.795002, 11603341643.14, 11603341684.5, 11603341726.235, 11603341751.130001, 11603341765.525, 11603341807.41, 11603341822.255, 11603341912.575, 11603341917.439999, 11603341960.64, 11603342090.605, 11603342211.855, 11603342262.48, 11603342394.25, 11603342485.485, 11603342528.925, 11603342708.675, 11603342889.895, 11603342921.305, 11603342949.720001, 11603342957.87, 11603342974.135, 11603343103.52, 11603343471.580002, 11603343499.435001, 11603343505.4, 11603343511.289999, 11603343589.915, 11603343630.005001, 11603343726.654999, 11603343818.515, 11603343866.244999, 11603343892.7, 11603343954.745, 11603344016.34, 11603344048.785, 11603344094.49, 11603344184.9, 11603344292.315, 11603344338.21, 11603344373.875, 11603344411.2, 11603344428.98, 11603344440.785, 11603344456.25, 11603344468.985, 11603344583.615, 11603344700.990002, 11603344719.03, 11603344785.965, 11855324454.965, 12107304135.0, 12107304240.82, 12107304325.955002, 12107304386.3, 12107304666.505001, 12107304732.115, 12107304772.744999, 12107305504.39, 12107305563.125, 12107305594.115, 12107305648.349998, 12107305733.505, 12107305811.26, 12107305881.579998, 12107305933.739998, 12107305976.134998, 12107306048.545, 12107307020.93, 12107307071.695, 12107307316.45, 12107307642.165, 12107307784.774998, 12107308048.535, 12107308239.825, 12107308440.885, 12107308579.44, 12107308638.07, 12107308698.91, 12359289748.279999, 12611271167.404999, 12611271654.294998, 13240056839.505, 13868842271.51, 13868842328.58, 13868842455.29, 13868842517.895, 13868842575.914999, 13868842633.675, 13868843192.905, 13868843429.060001, 13868843452.939999, 13868843461.274998, 13868843799.309998, 13868844396.099998, 13868844915.04, 13868845229.635, 13868845618.91, 13868846006.1, 13868846177.105, 13868846369.029999, 14120826244.635, 14372806047.6, 14372806606.87, 14372807321.100002, 15253576929.655, 16134345228.335001])
labels = array([3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 1.0, 2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 3.0, 0.0, 1.0, 4.0, 2.0, 0.0, 4.0, 2.0, 1.0, 4.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 0.0, 2.0, 1.0, 4.0, 2.0, 3.0, 0.0, 2.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 4.0, 2.0, 3.0, 2.0, 1.0, 4.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 4.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 4.0, 1.0, 4.0, 0.0, 1.0, 0.0, 4.0, 3.0, 0.0, 2.0, 0.0, 3.0, 4.0, 1.0, 2.0, 0.0, 1.0, 3.0, 2.0, 3.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 3.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 4.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 4.0, 1.0, 2.0, 1.0, 0.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0, 1.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 0.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 3.0, 0.0, 4.0, 3.0, 1.0, 4.0, 0.0, 2.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0, 4.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 1.0, 4.0])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)

    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = labels[numers[indys]]
        outputs[defaultindys] = 2.0
        return outputs
    return thresh_search(energys)

numthresholds = 330



# Main method
model_cap = numthresholds


def Validate(file):
    #Load Array
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')


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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs, cleanarr[:, -1]


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
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    cleanarr = cleanarr.reshape(cleanarr.shape[0], -1)
    with open(preprocessedfile, 'r') as csvinput:
        dirtyreader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(dirtyreader, None) + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            print(str(','.join(str(j) for j in (['"' + i + '"' if ',' in i else i for i in row]))) + ',' + str(get_key(int(outputs[k]), classmapping)))



#Main
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
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}

    #Predict or Validate?
    if not args.validate:
        Predict(cleanfile, get_key, args.headerless, preprocessedfile, classmapping)


    else:
        classifier_type = 'DT'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
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


    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)



