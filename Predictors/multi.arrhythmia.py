#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target class arrhythmia.csv -o arrhythmia.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:45.13. Finished on: Sep-04-2020 10:19:47.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         13-way classifier
Best-guess accuracy:                 54.20%
Overall Model accuracy:              99.77% (451/452 correct)
Overall Improvement over best guess: 45.57% (of possible 45.8%)
Model capacity (MEC):                284 bits
Generalization ratio:                1.58 bits/bit
Model efficiency:                    0.16%/parameter
Confusion Matrix:
 [0.44% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 5.53% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 11.06% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00%]
 [0.00% 0.00% 0.00% 54.20% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.66% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.88% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 3.32% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 4.87% 0.00% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.22% 0.00% 0.00% 0.00% 0.00% 9.51% 0.00% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 3.32% 0.00% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 2.88% 0.00%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 1.99%
  0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  1.11%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to '8'=0, '6'=1, '10'=2, '1'=3, '7'=4, '14'=5, '3'=6, '16'=7, '2'=8, '4'=9, '5'=10, '9'=11, '15'=12.
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
TRAINFILE = "arrhythmia.csv"


#Number of attributes
num_attr = 279
n_classes = 13


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
    clean.mapping={'8': 0, '6': 1, '10': 2, '1': 3, '7': 4, '14': 5, '3': 6, '16': 7, '2': 8, '4': 9, '5': 10, '9': 11, '15': 12}

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
energy_thresholds = array([2135.15, 2200.3, 2235.2, 2265.0, 2338.8999999999996, 2420.6499999999996, 2481.6000000000004, 2565.8500000000004, 2621.2, 2664.55, 2700.7, 2732.6000000000004, 2764.3500000000004, 2780.3, 2789.0, 2794.45, 2803.7, 2819.45, 2833.3, 2844.5, 2868.4, 2895.55, 2913.05, 2921.8, 2924.2, 2930.75, 2941.5, 2959.0, 2987.15, 3010.6000000000004, 3025.1000000000004, 3044.95, 3076.15, 3101.3, 3126.8999999999996, 3153.35, 3186.3, 3220.4, 3242.6000000000004, 3253.9, 3280.3500000000004, 3311.7, 3394.3, 3407.5, 3439.45, 3483.9, 3512.6000000000004, 3548.1000000000004, 3581.8500000000004, 3668.0, 3766.0, 3838.35, 3936.4, 4064.65, 4164.25, 842166031.15, 1684326864.05, 1684327010.65, 1684327123.6999998, 1684327180.9, 1684327208.4, 1684327241.4499998, 1684327279.9499998, 1684327286.6999998, 1684327298.6999998, 1684327307.75, 1684327322.85, 1684327369.15, 1684327382.0, 1684327387.3000002, 1684327398.0, 1684327417.6999998, 1684327449.4499998, 1684327465.4499998, 1684327487.5, 1684327499.8000002, 1684327568.65, 1684327569.8, 1684327582.9, 1684327586.8000002, 1684327595.0, 1684327598.1, 1684327598.9499998, 1684327618.6, 1684327626.1, 1684327646.9499998, 1684327651.75, 1684327659.25, 1684327669.2, 1684327674.35, 1684327678.05, 1684327682.0, 1684327685.75, 1684327688.65, 1684327692.9499998, 1684327700.1, 1684327701.75, 1684327705.2, 1684327713.25, 1684327723.0500002, 1684327726.0, 1684327730.35, 1684327738.1999998, 1684327742.0, 1684327743.35, 1684327744.25, 1684327745.55, 1684327746.75, 1684327748.45, 1684327766.85, 1684327767.8, 1684327769.35, 1684327770.45, 1684327771.0, 1684327772.6, 1684327790.25, 1684327792.25, 1684327795.25, 1684327801.2, 1684327806.4, 1684327811.0500002, 1684327815.95, 1684327827.35, 1684327833.1, 1684327843.35, 1684327844.6, 1684327857.15, 1684327862.6, 1684327871.3, 1684327874.7, 1684327878.25, 1684327881.25, 1684327883.6, 1684327885.05, 1684327897.8000002, 1684327902.15, 1684327926.7, 1684327929.65, 1684327934.1, 1684327937.35, 1684327940.95, 1684327942.95, 1684327945.0, 1684327948.45, 1684327950.95, 1684327952.5, 1684327954.45, 1684327955.0, 1684327958.6999998, 1684327963.65, 1684327966.1, 1684327967.8, 1684327970.85, 1684327974.2, 1684327980.75, 1684327981.4, 1684327981.6, 1684327982.2, 1684327983.55, 1684327984.7, 1684327986.8000002, 1684327988.5, 1684327990.85, 1684328011.2, 1684328015.8, 1684328018.9, 1684328021.5, 1684328026.4, 1684328027.95, 1684328032.0500002, 1684328034.1, 1684328045.5, 1684328046.6, 1684328056.5, 1684328056.65, 1684328056.85, 1684328061.1, 1684328069.4, 1684328072.55, 1684328086.65, 1684328090.55, 1684328104.3000002, 1684328108.0, 1684328113.5, 1684328114.95, 1684328118.35, 1684328120.45, 1684328122.9, 1684328129.75, 1684328133.4499998, 1684328136.1, 1684328139.3, 1684328147.35, 1684328153.0500002, 1684328160.15, 1684328161.05, 1684328162.0, 1684328171.5, 1684328174.15, 1684328176.15, 1684328192.9, 1684328194.05, 1684328195.6999998, 1684328200.5, 1684328205.45, 1684328210.95, 1684328216.45, 1684328228.45, 1684328233.85, 1684328238.75, 1684328242.6999998, 1684328245.4, 1684328248.5500002, 1684328251.0500002, 1684328251.45, 1684328254.0, 1684328266.3, 1684328271.05, 1684328272.3, 1684328273.35, 1684328274.4, 1684328279.0500002, 1684328299.2, 1684328307.5, 1684328335.35, 1684328343.9, 1684328355.85, 1684328358.15, 1684328359.8, 1684328361.85, 1684328375.6, 1684328409.2, 1684328410.55, 1684328415.5, 1684328421.5, 1684328440.0, 1684328475.35, 1684328483.35, 1684328486.25, 1684328489.8, 1684328497.4499998, 1684328510.75, 1684328551.9499998, 1684328557.1999998, 1684328562.25, 1684328566.35, 1684328572.05, 1684328591.1999998, 1684328595.35, 1684328605.15, 1684328610.75, 1684328622.55, 1684328637.4499998, 1684328643.0, 1684328649.1, 1684328657.25, 1684328674.65, 1684328707.5500002, 1684328738.4, 1684328754.75, 1684328784.5, 1684328835.3000002, 1684328866.7, 1684328888.05, 1684329011.1999998, 2526490667.95, 3368652376.5, 3368652536.8500004, 3368652545.2, 3368652616.95, 3368652712.3, 3368652813.1, 3368652881.1, 3368652910.65, 3368652944.6000004, 3368652962.2, 3368652963.4, 3368653025.05, 3368653069.1499996, 3368653099.75, 3368653171.95, 3368653251.15, 3368653320.25, 3368653384.45])
labels = array([1.0, 8.0, 3.0, 6.0, 10.0, 3.0, 8.0, 3.0, 0.0, 3.0, 8.0, 3.0, 8.0, 10.0, 8.0, 3.0, 8.0, 6.0, 7.0, 11.0, 6.0, 11.0, 2.0, 6.0, 11.0, 3.0, 7.0, 2.0, 3.0, 8.0, 3.0, 5.0, 3.0, 8.0, 3.0, 2.0, 3.0, 7.0, 3.0, 2.0, 7.0, 8.0, 3.0, 11.0, 8.0, 9.0, 11.0, 3.0, 8.0, 11.0, 2.0, 11.0, 2.0, 11.0, 2.0, 12.0, 6.0, 3.0, 10.0, 3.0, 6.0, 8.0, 7.0, 6.0, 3.0, 9.0, 3.0, 8.0, 12.0, 3.0, 8.0, 3.0, 7.0, 3.0, 9.0, 3.0, 6.0, 3.0, 6.0, 8.0, 10.0, 8.0, 3.0, 6.0, 3.0, 7.0, 3.0, 1.0, 3.0, 8.0, 2.0, 8.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 8.0, 1.0, 3.0, 8.0, 3.0, 2.0, 3.0, 4.0, 1.0, 3.0, 2.0, 6.0, 9.0, 3.0, 7.0, 3.0, 2.0, 8.0, 3.0, 2.0, 7.0, 3.0, 8.0, 10.0, 3.0, 8.0, 3.0, 1.0, 3.0, 2.0, 10.0, 6.0, 7.0, 9.0, 3.0, 10.0, 3.0, 8.0, 3.0, 6.0, 9.0, 3.0, 2.0, 3.0, 7.0, 6.0, 10.0, 8.0, 3.0, 1.0, 3.0, 12.0, 3.0, 2.0, 3.0, 8.0, 3.0, 1.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 3.0, 10.0, 2.0, 3.0, 1.0, 3.0, 7.0, 3.0, 1.0, 3.0, 2.0, 3.0, 9.0, 3.0, 1.0, 3.0, 2.0, 3.0, 10.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 7.0, 3.0, 9.0, 3.0, 2.0, 3.0, 10.0, 2.0, 3.0, 9.0, 2.0, 3.0, 2.0, 3.0, 2.0, 8.0, 3.0, 8.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0, 7.0, 2.0, 9.0, 3.0, 9.0, 3.0, 2.0, 3.0, 1.0, 9.0, 3.0, 2.0, 3.0, 7.0, 3.0, 2.0, 3.0, 2.0, 3.0, 10.0, 7.0, 3.0, 2.0, 3.0, 10.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 7.0, 3.0, 1.0, 9.0, 2.0, 3.0, 5.0, 3.0, 9.0, 2.0, 7.0, 2.0, 3.0, 12.0, 6.0, 1.0, 12.0, 7.0, 8.0, 3.0, 8.0, 7.0, 9.0, 2.0, 3.0, 4.0, 8.0, 2.0, 7.0, 8.0, 3.0, 7.0])
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
        outputs[defaultindys] = 8.0
        return outputs
    return thresh_search(energys)

numthresholds = 284



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



