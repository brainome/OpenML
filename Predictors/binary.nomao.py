#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target Class -cm {'1':0,'2':1} nomao.csv -o nomao_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 1:36:16.70. Finished on: Sep-03-2020 13:00:53.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 71.43%
Overall Model accuracy:              94.65% (32624/34465 correct)
Overall Improvement over best guess: 23.22% (of possible 28.57%)
Model capacity (MEC):                241 bits
Generalization ratio:                135.36 bits/bit
Model efficiency:                    0.09%/parameter
System behavior
True Negatives:                      25.80% (8893/34465)
True Positives:                      68.86% (23731/34465)
False Negatives:                     2.58% (890/34465)
False Positives:                     2.76% (951/34465)
True Pos. Rate/Sensitivity/Recall:   0.96
True Neg. Rate/Specificity:          0.90
Precision:                           0.96
F-1 Measure:                         0.96
False Negative Rate/Miss Rate:       0.04
Critical Success Index:              0.93
Confusion Matrix:
 [25.80% 2.76%]
 [2.58% 68.86%]
Overfitting:                         No
Note: Labels have been remapped to '1'=0, '2'=1.
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
TRAINFILE = "nomao.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 118
n_classes = 2


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
    clean.mapping={'1':0,'2':1}

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
    h_0 = max((((0.11995244 * float(x[0]))+ (-0.18403849 * float(x[1]))+ (-0.3939747 * float(x[2]))+ (-0.08612314 * float(x[3]))+ (-0.1489347 * float(x[4]))+ (-0.0044303685 * float(x[5]))+ (-0.7627397 * float(x[6]))+ (-0.9598789 * float(x[7]))+ (-0.7501503 * float(x[8]))+ (-0.8399004 * float(x[9]))+ (-0.8743673 * float(x[10]))+ (-1.0165243 * float(x[11]))+ (-0.9655153 * float(x[12]))+ (-0.8958778 * float(x[13]))+ (-0.93175197 * float(x[14]))+ (-0.8516355 * float(x[15]))+ (-0.6074086 * float(x[16]))+ (-0.8878447 * float(x[17]))+ (-0.9365287 * float(x[18]))+ (-0.7026117 * float(x[19]))+ (-0.65126115 * float(x[20]))+ (-0.82682186 * float(x[21]))+ (-0.7727076 * float(x[22]))+ (-0.97607684 * float(x[23]))+ (-0.57651037 * float(x[24]))+ (-0.33710486 * float(x[25]))+ (-0.85502267 * float(x[26]))+ (-0.7241341 * float(x[27]))+ (-0.68278944 * float(x[28]))+ (-0.6392032 * float(x[29]))+ (-0.74683875 * float(x[30]))+ (-1.0849725 * float(x[31]))+ (-0.88366 * float(x[32]))+ (-0.84158725 * float(x[33]))+ (-0.9705575 * float(x[34]))+ (-0.8623241 * float(x[35]))+ (-0.5154964 * float(x[36]))+ (-0.56259954 * float(x[37]))+ (-0.8316591 * float(x[38]))+ (-0.92827535 * float(x[39]))+ (-0.7974425 * float(x[40]))+ (-0.64497215 * float(x[41]))+ (-1.0138642 * float(x[42]))+ (-0.9164312 * float(x[43]))+ (-0.9194329 * float(x[44]))+ (-0.6137051 * float(x[45]))+ (-0.80892974 * float(x[46]))+ (-1.0392036 * float(x[47]))+ (-0.9481034 * float(x[48]))+ (-0.7970251 * float(x[49])))+ ((-0.71774024 * float(x[50]))+ (-0.85128963 * float(x[51]))+ (-0.9098275 * float(x[52]))+ (-0.7901099 * float(x[53]))+ (-2.9517658 * float(x[54]))+ (-2.7784379 * float(x[55]))+ (-0.6433812 * float(x[56]))+ (-0.62499636 * float(x[57]))+ (-0.7072299 * float(x[58]))+ (-0.38352957 * float(x[59]))+ (-0.48081 * float(x[60]))+ (-0.45344126 * float(x[61]))+ (-1.8125172 * float(x[62]))+ (-1.7573165 * float(x[63]))+ (-0.96825826 * float(x[64]))+ (-0.9314975 * float(x[65]))+ (-0.8882244 * float(x[66]))+ (-0.8013148 * float(x[67]))+ (-0.7253293 * float(x[68]))+ (-1.1123519 * float(x[69]))+ (-2.9020507 * float(x[70]))+ (-2.732226 * float(x[71]))+ (-0.67554444 * float(x[72]))+ (-0.66396487 * float(x[73]))+ (-0.7070533 * float(x[74]))+ (-0.83565676 * float(x[75]))+ (-1.100631 * float(x[76]))+ (-0.8595998 * float(x[77]))+ (-1.0799248 * float(x[78]))+ (-1.1484264 * float(x[79]))+ (-0.7413622 * float(x[80]))+ (-0.7700136 * float(x[81]))+ (-1.1630892 * float(x[82]))+ (-0.8981822 * float(x[83]))+ (-0.977704 * float(x[84]))+ (-0.97683215 * float(x[85]))+ (-2.9224632 * float(x[86]))+ (-2.734116 * float(x[87]))+ (-0.13244404 * float(x[88]))+ (-0.29404178 * float(x[89]))+ (-0.09581674 * float(x[90]))+ (-1.7199966 * float(x[91]))+ (0.083608136 * float(x[92]))+ (-0.5973256 * float(x[93]))+ (-0.35394952 * float(x[94]))+ (-1.7684201 * float(x[95]))+ (-0.6969396 * float(x[96]))+ (-1.0094117 * float(x[97]))+ (-0.90805286 * float(x[98]))+ (-1.6697124 * float(x[99])))+ ((-0.78936493 * float(x[100]))+ (-1.0375334 * float(x[101]))+ (-0.9963136 * float(x[102]))+ (-3.033113 * float(x[103]))+ (-0.9440543 * float(x[104]))+ (-1.1084243 * float(x[105]))+ (-1.1125158 * float(x[106]))+ (-2.832426 * float(x[107]))+ (-0.88641924 * float(x[108]))+ (-0.6630024 * float(x[109]))+ (-0.23972861 * float(x[110]))+ (-1.0761087 * float(x[111]))+ (-0.7207619 * float(x[112]))+ (-0.4524608 * float(x[113]))+ (-0.60034996 * float(x[114]))+ (-0.9956874 * float(x[115]))+ (-0.92994547 * float(x[116]))+ (-0.88797265 * float(x[117]))) + -0.9417093), 0)
    h_1 = max((((1.025101 * float(x[0]))+ (-0.6201807 * float(x[1]))+ (0.41258416 * float(x[2]))+ (1.5421181 * float(x[3]))+ (-0.92437613 * float(x[4]))+ (2.9631295 * float(x[5]))+ (0.6742484 * float(x[6]))+ (-1.450602 * float(x[7]))+ (-0.04420501 * float(x[8]))+ (-1.008348 * float(x[9]))+ (0.35451663 * float(x[10]))+ (0.1399384 * float(x[11]))+ (0.19647636 * float(x[12]))+ (0.5463161 * float(x[13]))+ (0.3932423 * float(x[14]))+ (-0.57698023 * float(x[15]))+ (0.861957 * float(x[16]))+ (0.5747139 * float(x[17]))+ (-0.7550201 * float(x[18]))+ (-0.06672095 * float(x[19]))+ (-0.66059625 * float(x[20]))+ (-0.18092105 * float(x[21]))+ (0.13822277 * float(x[22]))+ (-0.3431352 * float(x[23]))+ (0.28238806 * float(x[24]))+ (-0.22596021 * float(x[25]))+ (0.04557218 * float(x[26]))+ (0.27610776 * float(x[27]))+ (-0.09093972 * float(x[28]))+ (1.0642086 * float(x[29]))+ (0.12045406 * float(x[30]))+ (-0.0999403 * float(x[31]))+ (-0.06785512 * float(x[32]))+ (-0.093576655 * float(x[33]))+ (-0.057753336 * float(x[34]))+ (-0.35385445 * float(x[35]))+ (0.18233427 * float(x[36]))+ (0.3614409 * float(x[37]))+ (0.42039728 * float(x[38]))+ (0.18725047 * float(x[39]))+ (-0.5923294 * float(x[40]))+ (-0.59316313 * float(x[41]))+ (-0.06839373 * float(x[42]))+ (-0.14804806 * float(x[43]))+ (0.14072666 * float(x[44]))+ (0.047201447 * float(x[45]))+ (0.053346127 * float(x[46]))+ (0.23915985 * float(x[47]))+ (-0.45429286 * float(x[48]))+ (-0.12257353 * float(x[49])))+ ((0.21365348 * float(x[50]))+ (0.05330104 * float(x[51]))+ (0.3449059 * float(x[52]))+ (0.29849797 * float(x[53]))+ (-0.44464302 * float(x[54]))+ (-0.021738935 * float(x[55]))+ (0.49921143 * float(x[56]))+ (-0.40550873 * float(x[57]))+ (1.4558327 * float(x[58]))+ (1.3087488 * float(x[59]))+ (-3.2740176 * float(x[60]))+ (2.053313 * float(x[61]))+ (-0.25232172 * float(x[62]))+ (-1.1215887 * float(x[63]))+ (1.0570726 * float(x[64]))+ (-0.38532558 * float(x[65]))+ (-0.6289135 * float(x[66]))+ (0.08672838 * float(x[67]))+ (-1.1390886 * float(x[68]))+ (1.0122551 * float(x[69]))+ (0.3101463 * float(x[70]))+ (0.3941229 * float(x[71]))+ (-0.09376659 * float(x[72]))+ (-0.13819955 * float(x[73]))+ (-0.7762799 * float(x[74]))+ (-0.1302613 * float(x[75]))+ (-0.38377005 * float(x[76]))+ (0.103718035 * float(x[77]))+ (0.17076965 * float(x[78]))+ (0.1487899 * float(x[79]))+ (-0.4090257 * float(x[80]))+ (-0.27429125 * float(x[81]))+ (-0.58746403 * float(x[82]))+ (-0.57194406 * float(x[83]))+ (-0.54827464 * float(x[84]))+ (-0.26192358 * float(x[85]))+ (-0.17160907 * float(x[86]))+ (0.09738243 * float(x[87]))+ (1.7266268 * float(x[88]))+ (0.51909006 * float(x[89]))+ (1.2759352 * float(x[90]))+ (0.9491976 * float(x[91]))+ (0.6848528 * float(x[92]))+ (-0.07777722 * float(x[93]))+ (0.325755 * float(x[94]))+ (-0.67493075 * float(x[95]))+ (0.3578338 * float(x[96]))+ (0.75539154 * float(x[97]))+ (0.35367376 * float(x[98]))+ (1.4303898 * float(x[99])))+ ((-0.28038636 * float(x[100]))+ (0.881755 * float(x[101]))+ (-0.01974092 * float(x[102]))+ (-0.45946795 * float(x[103]))+ (-0.16044433 * float(x[104]))+ (0.93513423 * float(x[105]))+ (-0.021859393 * float(x[106]))+ (-0.57395935 * float(x[107]))+ (0.0437724 * float(x[108]))+ (0.26512626 * float(x[109]))+ (-0.18000062 * float(x[110]))+ (0.305248 * float(x[111]))+ (-0.0049774465 * float(x[112]))+ (0.6546012 * float(x[113]))+ (0.43729085 * float(x[114]))+ (-0.2542056 * float(x[115]))+ (0.16722584 * float(x[116]))+ (-0.1624828 * float(x[117]))) + -0.41464776), 0)
    o[0] = (0.36022532 * h_0)+ (1.4598427 * h_1) + -4.216574

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
        model_cap=241
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


