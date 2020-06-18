#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/52417/synthetic_control.arff -o Predictors/synthetic-control_NN.py -target class -stopat 99.17 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 0:02:25.78. Finished on: Jun-07-2020 10:58:17.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        6-way classifier
Best-guess accuracy:                16.67%
Model accuracy:                     100.00% (600/600 correct)
Improvement over best guess:        83.33% (of possible 83.33%)
Model capacity (MEC):               346 bits
Generalization ratio:               1.73 bits/bit
Confusion Matrix:
 [16.67% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 16.67% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 16.67% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 16.67% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 16.67% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 16.67%]

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
TRAINFILE = "synthetic_control.csv"


#Number of output logits
num_output_logits = 6

#Number of attributes
num_attr = 61
n_classes = 6


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
    clean.mapping={'Normal': 0, 'Cyclic': 1, 'Increasing_trend': 2, 'Decreasing_trend': 3, 'Upward_shift': 4, 'Downward_shift': 5}

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
    h_0 = max((((217.66473 * float(x[0]))+ (-10.383053 * float(x[1]))+ (-8.077911 * float(x[2]))+ (-7.9927964 * float(x[3]))+ (-5.51561 * float(x[4]))+ (-5.249931 * float(x[5]))+ (-3.6152153 * float(x[6]))+ (-6.4338937 * float(x[7]))+ (-6.803392 * float(x[8]))+ (-10.656199 * float(x[9]))+ (-10.501617 * float(x[10]))+ (-11.82401 * float(x[11]))+ (-8.102022 * float(x[12]))+ (-7.0244365 * float(x[13]))+ (-7.7154646 * float(x[14]))+ (-6.6602416 * float(x[15]))+ (-6.8388977 * float(x[16]))+ (-11.215299 * float(x[17]))+ (-7.8376975 * float(x[18]))+ (-9.165726 * float(x[19]))+ (-7.4182453 * float(x[20]))+ (-12.919829 * float(x[21]))+ (-9.100409 * float(x[22]))+ (-4.4738703 * float(x[23]))+ (-8.622908 * float(x[24]))+ (-11.697132 * float(x[25]))+ (-8.246393 * float(x[26]))+ (-10.794876 * float(x[27]))+ (-8.639523 * float(x[28]))+ (-8.504484 * float(x[29]))+ (-7.7544127 * float(x[30]))+ (-8.61899 * float(x[31]))+ (-10.183303 * float(x[32]))+ (-8.8748045 * float(x[33]))+ (-8.957909 * float(x[34]))+ (-11.127968 * float(x[35]))+ (-13.332764 * float(x[36]))+ (-12.847344 * float(x[37]))+ (-9.67766 * float(x[38]))+ (-8.181923 * float(x[39]))+ (-5.791044 * float(x[40]))+ (-13.01199 * float(x[41]))+ (-9.0728035 * float(x[42]))+ (-16.624208 * float(x[43]))+ (-13.748044 * float(x[44]))+ (-13.481232 * float(x[45]))+ (-14.260984 * float(x[46]))+ (-7.762744 * float(x[47]))+ (-10.738691 * float(x[48]))+ (-9.188061 * float(x[49])))+ ((-11.129878 * float(x[50]))+ (-11.986507 * float(x[51]))+ (-9.486499 * float(x[52]))+ (-9.2683325 * float(x[53]))+ (-7.94213 * float(x[54]))+ (-8.373388 * float(x[55]))+ (-13.571588 * float(x[56]))+ (-15.074884 * float(x[57]))+ (-10.4108715 * float(x[58]))+ (-7.6920075 * float(x[59]))+ (-8.474125 * float(x[60]))) + -28.42743), 0)
    h_1 = max((((154.72311 * float(x[0]))+ (-1.1684195 * float(x[1]))+ (1.7593867 * float(x[2]))+ (5.297611 * float(x[3]))+ (6.0035324 * float(x[4]))+ (6.6029444 * float(x[5]))+ (1.3673785 * float(x[6]))+ (-1.6421154 * float(x[7]))+ (-6.4913507 * float(x[8]))+ (-5.0856996 * float(x[9]))+ (-8.405961 * float(x[10]))+ (-5.9822097 * float(x[11]))+ (-2.7560477 * float(x[12]))+ (0.039248094 * float(x[13]))+ (2.1059127 * float(x[14]))+ (3.4926753 * float(x[15]))+ (5.073155 * float(x[16]))+ (5.5303144 * float(x[17]))+ (0.5803428 * float(x[18]))+ (-0.40143955 * float(x[19]))+ (-2.635335 * float(x[20]))+ (-2.659671 * float(x[21]))+ (-0.71266437 * float(x[22]))+ (-0.7837052 * float(x[23]))+ (1.5318326 * float(x[24]))+ (3.9432027 * float(x[25]))+ (2.4830787 * float(x[26]))+ (2.5197563 * float(x[27]))+ (0.8035659 * float(x[28]))+ (1.0254261 * float(x[29]))+ (1.400139 * float(x[30]))+ (-2.3188248 * float(x[31]))+ (-1.0423548 * float(x[32]))+ (-0.14805764 * float(x[33]))+ (3.618924 * float(x[34]))+ (3.3803537 * float(x[35]))+ (5.7145557 * float(x[36]))+ (3.256871 * float(x[37]))+ (0.9181454 * float(x[38]))+ (-0.21984199 * float(x[39]))+ (1.0538836 * float(x[40]))+ (0.5490974 * float(x[41]))+ (1.1268586 * float(x[42]))+ (3.3605835 * float(x[43]))+ (4.4486456 * float(x[44]))+ (4.442219 * float(x[45]))+ (6.3621163 * float(x[46]))+ (2.0527227 * float(x[47]))+ (4.580345 * float(x[48]))+ (3.2173502 * float(x[49])))+ ((3.2819314 * float(x[50]))+ (2.8183837 * float(x[51]))+ (3.7917414 * float(x[52]))+ (1.5880021 * float(x[53]))+ (1.7436591 * float(x[54]))+ (2.6545362 * float(x[55]))+ (4.5657463 * float(x[56]))+ (4.659023 * float(x[57]))+ (4.109647 * float(x[58]))+ (2.7439666 * float(x[59]))+ (5.24297 * float(x[60]))) + -16.418018), 0)
    h_2 = max((((96.02618 * float(x[0]))+ (6.39641 * float(x[1]))+ (1.3527231 * float(x[2]))+ (-1.4505132 * float(x[3]))+ (-0.4207387 * float(x[4]))+ (0.09747873 * float(x[5]))+ (1.7661159 * float(x[6]))+ (3.5336444 * float(x[7]))+ (4.6335683 * float(x[8]))+ (8.165996 * float(x[9]))+ (7.4491377 * float(x[10]))+ (9.688237 * float(x[11]))+ (6.661086 * float(x[12]))+ (5.5383935 * float(x[13]))+ (1.9675128 * float(x[14]))+ (-0.57438153 * float(x[15]))+ (-3.3607857 * float(x[16]))+ (-0.5256494 * float(x[17]))+ (-0.9263286 * float(x[18]))+ (0.27884993 * float(x[19]))+ (0.2139358 * float(x[20]))+ (1.3060068 * float(x[21]))+ (3.6379845 * float(x[22]))+ (3.1285322 * float(x[23]))+ (1.548488 * float(x[24]))+ (0.6325705 * float(x[25]))+ (2.5102959 * float(x[26]))+ (-0.27341664 * float(x[27]))+ (0.3749716 * float(x[28]))+ (-0.06823797 * float(x[29]))+ (0.5312414 * float(x[30]))+ (-0.1285032 * float(x[31]))+ (-0.94188714 * float(x[32]))+ (-0.28225514 * float(x[33]))+ (0.57755744 * float(x[34]))+ (2.058254 * float(x[35]))+ (2.639745 * float(x[36]))+ (1.8506888 * float(x[37]))+ (4.4796824 * float(x[38]))+ (6.1108313 * float(x[39]))+ (3.5690932 * float(x[40]))+ (4.6015344 * float(x[41]))+ (3.0301006 * float(x[42]))+ (2.4683168 * float(x[43]))+ (2.640805 * float(x[44]))+ (2.721131 * float(x[45]))+ (1.713695 * float(x[46]))+ (4.2598133 * float(x[47]))+ (2.636513 * float(x[48]))+ (1.27809 * float(x[49])))+ ((0.2419057 * float(x[50]))+ (0.84951735 * float(x[51]))+ (-2.4280508 * float(x[52]))+ (0.75626993 * float(x[53]))+ (0.07924175 * float(x[54]))+ (1.5275996 * float(x[55]))+ (1.3088793 * float(x[56]))+ (3.6605449 * float(x[57]))+ (1.9718704 * float(x[58]))+ (2.005714 * float(x[59]))+ (2.1447775 * float(x[60]))) + -3.3015285), 0)
    h_3 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))+ (0.0 * float(x[16]))+ (0.0 * float(x[17]))+ (0.0 * float(x[18]))+ (0.0 * float(x[19]))+ (0.0 * float(x[20]))+ (0.0 * float(x[21]))+ (0.0 * float(x[22]))+ (0.0 * float(x[23]))+ (0.0 * float(x[24]))+ (0.0 * float(x[25]))+ (0.0 * float(x[26]))+ (0.0 * float(x[27]))+ (0.0 * float(x[28]))+ (0.0 * float(x[29]))+ (0.0 * float(x[30]))+ (0.0 * float(x[31]))+ (0.0 * float(x[32]))+ (0.0 * float(x[33]))+ (0.0 * float(x[34]))+ (0.0 * float(x[35]))+ (0.0 * float(x[36]))+ (0.0 * float(x[37]))+ (0.0 * float(x[38]))+ (0.0 * float(x[39]))+ (0.0 * float(x[40]))+ (0.0 * float(x[41]))+ (0.0 * float(x[42]))+ (0.0 * float(x[43]))+ (0.0 * float(x[44]))+ (0.0 * float(x[45]))+ (0.0 * float(x[46]))+ (0.0 * float(x[47]))+ (0.0 * float(x[48]))+ (0.0 * float(x[49])))+ ((0.0 * float(x[50]))+ (0.0 * float(x[51]))+ (0.0 * float(x[52]))+ (0.0 * float(x[53]))+ (0.0 * float(x[54]))+ (0.0 * float(x[55]))+ (0.0 * float(x[56]))+ (0.0 * float(x[57]))+ (0.0 * float(x[58]))+ (0.0 * float(x[59]))+ (0.0 * float(x[60]))) + 0.0), 0)
    h_4 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))+ (0.0 * float(x[16]))+ (0.0 * float(x[17]))+ (0.0 * float(x[18]))+ (0.0 * float(x[19]))+ (0.0 * float(x[20]))+ (0.0 * float(x[21]))+ (0.0 * float(x[22]))+ (0.0 * float(x[23]))+ (0.0 * float(x[24]))+ (0.0 * float(x[25]))+ (0.0 * float(x[26]))+ (0.0 * float(x[27]))+ (0.0 * float(x[28]))+ (0.0 * float(x[29]))+ (0.0 * float(x[30]))+ (0.0 * float(x[31]))+ (0.0 * float(x[32]))+ (0.0 * float(x[33]))+ (0.0 * float(x[34]))+ (0.0 * float(x[35]))+ (0.0 * float(x[36]))+ (0.0 * float(x[37]))+ (0.0 * float(x[38]))+ (0.0 * float(x[39]))+ (0.0 * float(x[40]))+ (0.0 * float(x[41]))+ (0.0 * float(x[42]))+ (0.0 * float(x[43]))+ (0.0 * float(x[44]))+ (0.0 * float(x[45]))+ (0.0 * float(x[46]))+ (0.0 * float(x[47]))+ (0.0 * float(x[48]))+ (0.0 * float(x[49])))+ ((0.0 * float(x[50]))+ (0.0 * float(x[51]))+ (0.0 * float(x[52]))+ (0.0 * float(x[53]))+ (0.0 * float(x[54]))+ (0.0 * float(x[55]))+ (0.0 * float(x[56]))+ (0.0 * float(x[57]))+ (0.0 * float(x[58]))+ (0.0 * float(x[59]))+ (0.0 * float(x[60]))) + 0.0), 0)
    o[0] = (2.06169 * h_0)+ (2.907949 * h_1)+ (3.9542015 * h_2)+ (-1.0 * h_3)+ (0.0 * h_4) + 6.2905374
    o[1] = (3.7930627 * h_0)+ (3.9037652 * h_1)+ (1.9538223 * h_2)+ (1.0 * h_3)+ (0.0 * h_4) + 17.273087
    o[2] = (3.8039417 * h_0)+ (3.784605 * h_1)+ (2.1223688 * h_2)+ (1.0 * h_3)+ (0.0 * h_4) + -4.3709445
    o[3] = (4.247164 * h_0)+ (3.2169385 * h_1)+ (2.284556 * h_2)+ (1.0 * h_3)+ (0.0 * h_4) + 12.051462
    o[4] = (3.948099 * h_0)+ (3.7406394 * h_1)+ (1.9802899 * h_2)+ (1.0 * h_3)+ (0.0 * h_4) + -15.620698
    o[5] = (4.245781 * h_0)+ (3.3129518 * h_1)+ (2.1388376 * h_2)+ (1.0 * h_3)+ (0.0 * h_4) + -14.140264

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
        model_cap=346
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

