#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target type BNG-zoo-nominal-1000000.csv -o BNG-zoo-nominal-1000000_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 3:22:51.83. Finished on: Sep-05-2020 03:45:15.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         7-way classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 39.62%
Overall Model accuracy:              94.78% (947859/1000000 correct)
Overall Improvement over best guess: 55.16% (of possible 60.38%)
Model capacity (MEC):                332 bits
Generalization ratio:                2854.99 bits/bit
Model efficiency:                    0.16%/parameter
Confusion Matrix:
 [12.23% 0.03% 0.09% 0.04% 0.01% 0.41% 0.12%]
 [0.03% 39.47% 0.00% 0.01% 0.01% 0.06% 0.05%]
 [0.05% 0.01% 9.08% 0.02% 0.69% 0.15% 0.10%]
 [0.01% 0.01% 0.02% 19.42% 0.02% 0.13% 0.07%]
 [0.01% 0.02% 0.76% 0.03% 7.23% 0.04% 0.04%]
 [0.49% 0.11% 0.16% 0.23% 0.04% 3.98% 0.26%]
 [0.25% 0.09% 0.15% 0.06% 0.04% 0.34% 3.37%]
Overfitting:                         No
Note: Labels have been remapped to 'fish'=0, 'mammal'=1, 'invertebrate'=2, 'bird'=3, 'insect'=4, 'reptile'=5, 'amphibian'=6.
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
TRAINFILE = "BNG-zoo-nominal-1000000.csv"


#Number of output logits
num_output_logits = 7

#Number of attributes
num_attr = 17
n_classes = 7

mappings = [{13120529.0: 0, 22038877.0: 1, 117437110.0: 2, 133782293.0: 3, 149060979.0: 4, 189113590.0: 5, 197980915.0: 6, 214846638.0: 7, 293482766.0: 8, 298438918.0: 9, 373135080.0: 10, 584252530.0: 11, 704615212.0: 12, 744212000.0: 13, 752743942.0: 14, 773143157.0: 15, 774942256.0: 16, 822945407.0: 17, 844703930.0: 18, 869147049.0: 19, 878725936.0: 20, 1016767401.0: 21, 1125635993.0: 22, 1125972319.0: 23, 1205411905.0: 24, 1351015120.0: 25, 1401591111.0: 26, 1406814162.0: 27, 1440416042.0: 28, 1458066827.0: 29, 1466808295.0: 30, 1489220549.0: 31, 1506756718.0: 32, 1559989163.0: 33, 1574132893.0: 34, 1585185686.0: 35, 1595133347.0: 36, 1608342591.0: 37, 1744256722.0: 38, 1777040477.0: 39, 1832803961.0: 40, 1852629197.0: 41, 1940301880.0: 42, 1950037758.0: 43, 1957767378.0: 44, 2107455857.0: 45, 2163761501.0: 46, 2203004802.0: 47, 2209811699.0: 48, 2223629247.0: 49, 2233680476.0: 50, 2284240540.0: 51, 2343511109.0: 52, 2383478420.0: 53, 2412105095.0: 54, 2453043442.0: 55, 2468141291.0: 56, 2560465762.0: 57, 2571044658.0: 58, 2579420686.0: 59, 2746820935.0: 60, 2795630676.0: 61, 2922675039.0: 62, 2964036461.0: 63, 3042426822.0: 64, 3054645060.0: 65, 3064768660.0: 66, 3112416541.0: 67, 3123563957.0: 68, 3157691388.0: 69, 3190505002.0: 70, 3198437122.0: 71, 3201480873.0: 72, 3219624110.0: 73, 3256768255.0: 74, 3398724132.0: 75, 3400449319.0: 76, 3461413561.0: 77, 3624607037.0: 78, 3654688985.0: 79, 3679943724.0: 80, 3732236668.0: 81, 3737687316.0: 82, 3751589890.0: 83, 3812981318.0: 84, 3854672160.0: 85, 3890034724.0: 86, 3914382451.0: 87, 3942618020.0: 88, 3956451473.0: 89, 4000454991.0: 90, 4021318520.0: 91, 4023774479.0: 92, 4026676655.0: 93, 4070099558.0: 94, 4071067935.0: 95, 4073228349.0: 96, 4143483097.0: 97, 4184026786.0: 98, 4189989858.0: 99}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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
target="type"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="type"
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
    clean.mapping={'fish': 0, 'mammal': 1, 'invertebrate': 2, 'bird': 3, 'insect': 4, 'reptile': 5, 'amphibian': 6}

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
    h_0 = max((((0.015112827 * float(x[0]))+ (-0.66854244 * float(x[1]))+ (-1.5237334 * float(x[2]))+ (1.0658032 * float(x[3]))+ (0.7528866 * float(x[4]))+ (-1.7573813 * float(x[5]))+ (-0.5310456 * float(x[6]))+ (0.09518318 * float(x[7]))+ (1.6325492 * float(x[8]))+ (0.82384896 * float(x[9]))+ (-0.24029359 * float(x[10]))+ (-0.36213768 * float(x[11]))+ (-0.23612884 * float(x[12]))+ (-0.13743131 * float(x[13]))+ (0.028852778 * float(x[14]))+ (-0.21921483 * float(x[15]))+ (0.4694592 * float(x[16]))) + 2.1632738), 0)
    h_1 = max((((0.073510304 * float(x[0]))+ (-1.1760643 * float(x[1]))+ (1.206259 * float(x[2]))+ (0.39085054 * float(x[3]))+ (-0.9852408 * float(x[4]))+ (0.073621675 * float(x[5]))+ (-0.0934213 * float(x[6]))+ (-0.016330568 * float(x[7]))+ (-0.47550336 * float(x[8]))+ (-0.57740533 * float(x[9]))+ (-0.51902765 * float(x[10]))+ (-0.13544746 * float(x[11]))+ (0.62208825 * float(x[12]))+ (0.86493814 * float(x[13]))+ (0.78818536 * float(x[14]))+ (0.087480746 * float(x[15]))+ (0.60462683 * float(x[16]))) + 2.0262053), 0)
    h_2 = max((((0.0025019594 * float(x[0]))+ (0.015668605 * float(x[1]))+ (-0.20063208 * float(x[2]))+ (-0.07381988 * float(x[3]))+ (-0.0037445154 * float(x[4]))+ (-0.25028643 * float(x[5]))+ (-0.27272448 * float(x[6]))+ (0.06608505 * float(x[7]))+ (0.3621667 * float(x[8]))+ (0.32736647 * float(x[9]))+ (0.060955413 * float(x[10]))+ (0.024513358 * float(x[11]))+ (-1.7280109 * float(x[12]))+ (-2.322339 * float(x[13]))+ (0.11272506 * float(x[14]))+ (0.048077762 * float(x[15]))+ (0.21485291 * float(x[16]))) + 1.9362928), 0)
    h_3 = max((((0.10269175 * float(x[0]))+ (0.15518299 * float(x[1]))+ (1.4108652 * float(x[2]))+ (-0.83888197 * float(x[3]))+ (0.7601142 * float(x[4]))+ (0.14205994 * float(x[5]))+ (0.85963273 * float(x[6]))+ (0.3610523 * float(x[7]))+ (0.4329108 * float(x[8]))+ (1.8474435 * float(x[9]))+ (1.0524708 * float(x[10]))+ (0.07517288 * float(x[11]))+ (-0.82303035 * float(x[12]))+ (-0.10721853 * float(x[13]))+ (0.7856637 * float(x[14]))+ (-0.068393104 * float(x[15]))+ (0.38839385 * float(x[16]))) + 1.2353334), 0)
    h_4 = max((((0.026434937 * float(x[0]))+ (-0.22330508 * float(x[1]))+ (0.0922692 * float(x[2]))+ (0.80095595 * float(x[3]))+ (-0.77958906 * float(x[4]))+ (0.47820714 * float(x[5]))+ (-0.21279402 * float(x[6]))+ (0.0026784998 * float(x[7]))+ (0.18609759 * float(x[8]))+ (-1.1627388 * float(x[9]))+ (0.18059145 * float(x[10]))+ (0.85839295 * float(x[11]))+ (-0.19574614 * float(x[12]))+ (-0.5866054 * float(x[13]))+ (-0.752714 * float(x[14]))+ (-0.05688037 * float(x[15]))+ (-0.3181596 * float(x[16]))) + 3.2097723), 0)
    h_5 = max((((0.04460187 * float(x[0]))+ (1.7809873 * float(x[1]))+ (-0.9633918 * float(x[2]))+ (-1.5652062 * float(x[3]))+ (1.6013465 * float(x[4]))+ (0.027944425 * float(x[5]))+ (0.18529607 * float(x[6]))+ (-0.025781568 * float(x[7]))+ (0.31257433 * float(x[8]))+ (-0.5459657 * float(x[9]))+ (0.049739633 * float(x[10]))+ (0.08275003 * float(x[11]))+ (0.33899218 * float(x[12]))+ (-0.059177905 * float(x[13]))+ (0.09022313 * float(x[14]))+ (0.23161133 * float(x[15]))+ (0.5933033 * float(x[16]))) + 1.7670971), 0)
    h_6 = max((((0.14973792 * float(x[0]))+ (0.0750507 * float(x[1]))+ (-1.1822147 * float(x[2]))+ (0.5717055 * float(x[3]))+ (-1.1575363 * float(x[4]))+ (-0.3491634 * float(x[5]))+ (0.5447445 * float(x[6]))+ (0.17883864 * float(x[7]))+ (0.77962214 * float(x[8]))+ (0.33198673 * float(x[9]))+ (-1.7116307 * float(x[10]))+ (1.1251156 * float(x[11]))+ (1.1980544 * float(x[12]))+ (0.41030183 * float(x[13]))+ (0.44065884 * float(x[14]))+ (-0.05671113 * float(x[15]))+ (-1.0959233 * float(x[16]))) + 1.5232316), 0)
    h_7 = max((((0.03642353 * float(x[0]))+ (-0.26475805 * float(x[1]))+ (0.8941582 * float(x[2]))+ (-0.29074854 * float(x[3]))+ (-0.23190914 * float(x[4]))+ (1.3314593 * float(x[5]))+ (0.27708524 * float(x[6]))+ (-0.052172586 * float(x[7]))+ (-1.2334391 * float(x[8]))+ (-0.7289211 * float(x[9]))+ (0.044396814 * float(x[10]))+ (-0.7382652 * float(x[11]))+ (-0.05831989 * float(x[12]))+ (0.5373366 * float(x[13]))+ (-0.44165903 * float(x[14]))+ (-0.18862757 * float(x[15]))+ (-0.43952364 * float(x[16]))) + 1.0225617), 0)
    h_8 = max((((0.12656547 * float(x[0]))+ (0.82115614 * float(x[1]))+ (0.6924854 * float(x[2]))+ (0.45073986 * float(x[3]))+ (1.5927594 * float(x[4]))+ (1.0089762 * float(x[5]))+ (0.3114069 * float(x[6]))+ (0.019591402 * float(x[7]))+ (-0.3709515 * float(x[8]))+ (1.0122201 * float(x[9]))+ (1.6824473 * float(x[10]))+ (-0.773656 * float(x[11]))+ (0.26865175 * float(x[12]))+ (0.05624874 * float(x[13]))+ (-0.040861197 * float(x[14]))+ (0.9317133 * float(x[15]))+ (-0.004903724 * float(x[16]))) + 0.36980045), 0)
    h_9 = max((((-0.00091918214 * float(x[0]))+ (-0.7841938 * float(x[1]))+ (-0.14235674 * float(x[2]))+ (0.8007795 * float(x[3]))+ (0.015144125 * float(x[4]))+ (-0.5159695 * float(x[5]))+ (-0.24502063 * float(x[6]))+ (0.19828615 * float(x[7]))+ (-1.1373436 * float(x[8]))+ (-0.8659638 * float(x[9]))+ (-0.09447432 * float(x[10]))+ (-0.26780686 * float(x[11]))+ (-0.54167867 * float(x[12]))+ (-0.029678695 * float(x[13]))+ (-0.9570045 * float(x[14]))+ (-0.20507327 * float(x[15]))+ (-0.6153754 * float(x[16]))) + 2.1984463), 0)
    h_10 = max((((0.16183718 * float(x[0]))+ (0.087944016 * float(x[1]))+ (-1.0757797 * float(x[2]))+ (0.14949767 * float(x[3]))+ (1.0498663 * float(x[4]))+ (-0.8100558 * float(x[5]))+ (-0.2902089 * float(x[6]))+ (-0.0033796648 * float(x[7]))+ (-0.04098799 * float(x[8]))+ (-0.1395393 * float(x[9]))+ (0.51480067 * float(x[10]))+ (-0.72665715 * float(x[11]))+ (-0.6104057 * float(x[12]))+ (-0.068235755 * float(x[13]))+ (-0.54980165 * float(x[14]))+ (-0.3480383 * float(x[15]))+ (0.025593013 * float(x[16]))) + 0.7401619), 0)
    h_11 = max((((0.0130439075 * float(x[0]))+ (0.66446745 * float(x[1]))+ (-0.23900454 * float(x[2]))+ (1.0956651 * float(x[3]))+ (0.12460331 * float(x[4]))+ (0.9396287 * float(x[5]))+ (-0.004415919 * float(x[6]))+ (0.046251718 * float(x[7]))+ (0.48265564 * float(x[8]))+ (-0.40643126 * float(x[9]))+ (0.30857587 * float(x[10]))+ (-0.35232767 * float(x[11]))+ (1.3121082 * float(x[12]))+ (0.4626043 * float(x[13]))+ (-0.56720674 * float(x[14]))+ (0.40271524 * float(x[15]))+ (0.21775594 * float(x[16]))) + -0.33206832), 0)
    h_12 = max((((-0.00041295408 * float(x[0]))+ (-1.1248546 * float(x[1]))+ (-0.11089907 * float(x[2]))+ (2.341981 * float(x[3]))+ (0.01498898 * float(x[4]))+ (0.03296069 * float(x[5]))+ (-2.3321433 * float(x[6]))+ (-0.0042292336 * float(x[7]))+ (-0.47148392 * float(x[8]))+ (-0.3203445 * float(x[9]))+ (0.7832199 * float(x[10]))+ (-0.18851258 * float(x[11]))+ (-0.8033554 * float(x[12]))+ (0.014039581 * float(x[13]))+ (0.2798211 * float(x[14]))+ (-0.03745047 * float(x[15]))+ (-0.038320966 * float(x[16]))) + -0.51328254), 0)
    o[0] = (0.74365836 * h_0)+ (0.5388104 * h_1)+ (0.92557055 * h_2)+ (-0.014057041 * h_3)+ (-1.3443111 * h_4)+ (-0.765311 * h_5)+ (0.49701694 * h_6)+ (0.21818648 * h_7)+ (-0.20981617 * h_8)+ (-1.1452987 * h_9)+ (-0.09810216 * h_10)+ (0.35590804 * h_11)+ (-1.0854266 * h_12) + -1.2646188
    o[1] = (0.07824389 * h_0)+ (-0.9287599 * h_1)+ (-1.300318 * h_2)+ (0.48284417 * h_3)+ (-1.9432282 * h_4)+ (1.0553191 * h_5)+ (-0.59342176 * h_6)+ (-0.116091944 * h_7)+ (0.45427036 * h_8)+ (-0.8092782 * h_9)+ (0.48614144 * h_10)+ (-0.059301145 * h_11)+ (-0.5473138 * h_12) + -0.03243558
    o[2] = (-0.108195245 * h_0)+ (0.4968549 * h_1)+ (-0.06832169 * h_2)+ (-0.23478258 * h_3)+ (0.28638816 * h_4)+ (0.38797164 * h_5)+ (-0.21093081 * h_6)+ (0.52739847 * h_7)+ (-0.5292325 * h_8)+ (1.6369168 * h_9)+ (0.4605699 * h_10)+ (-0.49962997 * h_11)+ (-0.8939314 * h_12) + 1.5964227
    o[3] = (-1.1759164 * h_0)+ (0.5753225 * h_1)+ (-0.23916242 * h_2)+ (0.7702487 * h_3)+ (-1.1206273 * h_4)+ (-1.5370767 * h_5)+ (-0.47551933 * h_6)+ (0.58268726 * h_7)+ (0.41220486 * h_8)+ (-1.2007303 * h_9)+ (0.10607974 * h_10)+ (-0.20609446 * h_11)+ (0.5699279 * h_12) + 0.8372696
    o[4] = (-1.0797288 * h_0)+ (-0.39174947 * h_1)+ (0.9231692 * h_2)+ (-0.48808572 * h_3)+ (0.7275169 * h_4)+ (0.39885715 * h_5)+ (-0.024699636 * h_6)+ (0.5935611 * h_7)+ (0.06504165 * h_8)+ (0.27217028 * h_9)+ (0.30596718 * h_10)+ (0.48779932 * h_11)+ (0.32789078 * h_12) + 1.0471737
    o[5] = (0.13141958 * h_0)+ (-0.20348981 * h_1)+ (-0.9427718 * h_2)+ (0.5119239 * h_3)+ (-0.29592338 * h_4)+ (-0.18582521 * h_5)+ (0.24136941 * h_6)+ (0.11883901 * h_7)+ (-0.21357314 * h_8)+ (-1.0898571 * h_9)+ (-0.028200388 * h_10)+ (-0.564164 * h_11)+ (1.1338029 * h_12) + 1.4588275
    o[6] = (0.5955674 * h_0)+ (-0.5559516 * h_1)+ (-1.4648478 * h_2)+ (0.36246988 * h_3)+ (0.77644473 * h_4)+ (-1.0317918 * h_5)+ (-0.2807357 * h_6)+ (-0.0024999466 * h_7)+ (0.2626638 * h_8)+ (-0.49595195 * h_9)+ (0.33935928 * h_10)+ (-0.14124005 * h_11)+ (-2.2389545 * h_12) + 1.1846758

    

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
    w_h = np.array([[0.015112826600670815, -0.6685424447059631, -1.523733377456665, 1.0658031702041626, 0.7528865933418274, -1.7573813199996948, -0.5310456156730652, 0.09518317878246307, 1.6325491666793823, 0.8238489627838135, -0.24029359221458435, -0.36213767528533936, -0.23612883687019348, -0.1374313086271286, 0.028852777555584908, -0.21921482682228088, 0.46945920586586], [0.0735103040933609, -1.1760642528533936, 1.20625901222229, 0.3908505439758301, -0.9852408170700073, 0.07362167537212372, -0.09342130273580551, -0.016330568119883537, -0.4755033552646637, -0.5774053335189819, -0.5190276503562927, -0.1354474574327469, 0.6220882534980774, 0.8649381399154663, 0.7881853580474854, 0.08748074620962143, 0.6046268343925476], [0.0025019594468176365, 0.015668604522943497, -0.20063208043575287, -0.07381988316774368, -0.0037445153575390577, -0.25028643012046814, -0.2727244794368744, 0.06608504801988602, 0.36216670274734497, 0.3273664712905884, 0.060955412685871124, 0.024513358250260353, -1.728010892868042, -2.3223390579223633, 0.11272505670785904, 0.04807776212692261, 0.21485291421413422], [0.10269174724817276, 0.15518298745155334, 1.4108651876449585, -0.8388819694519043, 0.7601141929626465, 0.14205993711948395, 0.8596327304840088, 0.3610523045063019, 0.4329107999801636, 1.8474434614181519, 1.0524708032608032, 0.07517287880182266, -0.8230303525924683, -0.10721852630376816, 0.7856637239456177, -0.06839310377836227, 0.3883938491344452], [0.026434937492012978, -0.22330507636070251, 0.09226919710636139, 0.8009559512138367, -0.779589056968689, 0.47820714116096497, -0.21279402077198029, 0.0026784997899085283, 0.18609759211540222, -1.1627388000488281, 0.18059144914150238, 0.8583929538726807, -0.19574613869190216, -0.586605429649353, -0.7527139782905579, -0.056880369782447815, -0.3181596100330353], [0.04460186883807182, 1.78098726272583, -0.9633917808532715, -1.5652061700820923, 1.601346492767334, 0.027944425120949745, 0.18529607355594635, -0.02578156813979149, 0.3125743269920349, -0.5459656715393066, 0.04973963275551796, 0.08275002986192703, 0.338992178440094, -0.059177905321121216, 0.09022313356399536, 0.23161132633686066, 0.5933033227920532], [0.14973792433738708, 0.07505069673061371, -1.1822147369384766, 0.5717055201530457, -1.157536268234253, -0.3491634130477905, 0.5447444915771484, 0.17883864045143127, 0.7796221375465393, 0.3319867253303528, -1.7116307020187378, 1.1251156330108643, 1.1980544328689575, 0.4103018343448639, 0.440658837556839, -0.05671112984418869, -1.0959233045578003], [0.03642353042960167, -0.26475805044174194, 0.8941581845283508, -0.2907485365867615, -0.2319091409444809, 1.3314592838287354, 0.27708524465560913, -0.05217258632183075, -1.2334390878677368, -0.728921115398407, 0.044396813958883286, -0.7382652163505554, -0.05831988900899887, 0.5373365879058838, -0.44165903329849243, -0.18862757086753845, -0.4395236372947693], [0.12656547129154205, 0.8211561441421509, 0.6924853920936584, 0.45073986053466797, 1.592759370803833, 1.0089762210845947, 0.3114069104194641, 0.019591402262449265, -0.37095150351524353, 1.0122201442718506, 1.6824473142623901, -0.7736560106277466, 0.2686517536640167, 0.056248739361763, -0.04086119681596756, 0.9317132830619812, -0.004903723951429129], [-0.0009191821445710957, -0.7841938138008118, -0.142356738448143, 0.8007795214653015, 0.015144124627113342, -0.5159695148468018, -0.24502062797546387, 0.19828614592552185, -1.1373436450958252, -0.8659638166427612, -0.09447432309389114, -0.26780685782432556, -0.5416786670684814, -0.029678694903850555, -0.9570044875144958, -0.2050732672214508, -0.6153753995895386], [0.16183717548847198, 0.08794401586055756, -1.075779676437378, 0.14949767291545868, 1.0498663187026978, -0.8100557923316956, -0.2902089059352875, -0.0033796648494899273, -0.04098799079656601, -0.13953930139541626, 0.5148006677627563, -0.7266571521759033, -0.610405683517456, -0.06823575496673584, -0.5498016476631165, -0.34803828597068787, 0.02559301257133484], [0.013043907471001148, 0.664467453956604, -0.23900453746318817, 1.0956650972366333, 0.12460330873727798, 0.9396287202835083, -0.0044159190729260445, 0.04625171795487404, 0.4826556444168091, -0.40643125772476196, 0.3085758686065674, -0.3523276746273041, 1.3121081590652466, 0.4626043140888214, -0.5672067403793335, 0.4027152359485626, 0.21775594353675842], [-0.0004129540757276118, -1.124854564666748, -0.11089906841516495, 2.3419809341430664, 0.014988980256021023, 0.032960690557956696, -2.3321433067321777, -0.0042292336001992226, -0.47148391604423523, -0.3203445076942444, 0.7832198739051819, -0.18851257860660553, -0.8033553957939148, 0.014039580710232258, 0.27982109785079956, -0.03745047003030777, -0.03832096606492996]])
    b_h = np.array([2.163273811340332, 2.02620530128479, 1.9362927675247192, 1.2353334426879883, 3.2097723484039307, 1.7670971155166626, 1.5232316255569458, 1.0225616693496704, 0.3698004484176636, 2.198446273803711, 0.7401618957519531, -0.3320683240890503, -0.5132825374603271])
    w_o = np.array([[0.7436583638191223, 0.5388103723526001, 0.925570547580719, -0.014057041145861149, -1.3443111181259155, -0.7653110027313232, 0.49701693654060364, 0.21818648278713226, -0.20981617271900177, -1.145298719406128, -0.0981021597981453, 0.35590803623199463, -1.0854265689849854], [0.07824388891458511, -0.9287598729133606, -1.3003180027008057, 0.48284417390823364, -1.9432282447814941, 1.05531907081604, -0.5934217572212219, -0.11609194427728653, 0.4542703628540039, -0.8092781901359558, 0.4861414432525635, -0.05930114537477493, -0.5473138093948364], [-0.1081952452659607, 0.49685490131378174, -0.06832168996334076, -0.23478257656097412, 0.2863881587982178, 0.3879716396331787, -0.21093080937862396, 0.5273984670639038, -0.5292325019836426, 1.6369167566299438, 0.4605698883533478, -0.4996299743652344, -0.8939313888549805], [-1.1759164333343506, 0.5753225088119507, -0.239162415266037, 0.7702487111091614, -1.1206272840499878, -1.537076711654663, -0.4755193293094635, 0.582687258720398, 0.4122048616409302, -1.200730323791504, 0.10607974231243134, -0.20609445869922638, 0.5699278712272644], [-1.0797288417816162, -0.39174947142601013, 0.9231691956520081, -0.48808571696281433, 0.7275168895721436, 0.39885714650154114, -0.024699635803699493, 0.5935611128807068, 0.06504164636135101, 0.2721702754497528, 0.30596718192100525, 0.4877993166446686, 0.3278907835483551], [0.1314195841550827, -0.20348981022834778, -0.9427717924118042, 0.5119239091873169, -0.2959233820438385, -0.18582521378993988, 0.24136941134929657, 0.11883901059627533, -0.2135731428861618, -1.0898571014404297, -0.028200387954711914, -0.5641639828681946, 1.133802890777588], [0.5955674052238464, -0.5559515953063965, -1.4648478031158447, 0.362469881772995, 0.7764447331428528, -1.0317918062210083, -0.28073570132255554, -0.002499946625903249, 0.2626638114452362, -0.49595195055007935, 0.3393592834472656, -0.14124004542827606, -2.238954544067383]])
    b_o = np.array([-1.2646187543869019, -0.0324355810880661, 1.5964226722717285, 0.8372696042060852, 1.0471737384796143, 1.4588274955749512, 1.1846758127212524])

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
        model_cap = 332
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


