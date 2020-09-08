#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target binaryClass fri-c0-1000-50.csv -o fri-c0-1000-50_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:19:43.50. Finished on: Sep-03-2020 16:48:18.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 51.00%
Training accuracy:                   90.60% (453/500 correct)
Validation accuracy:                 87.60% (438/500 correct)
Overall Model accuracy:              89.10% (891/1000 correct)
Overall Improvement over best guess: 38.10% (of possible 49.0%)
Model capacity (MEC):                261 bits
Generalization ratio:                3.41 bits/bit
Model efficiency:                    0.14%/parameter
System behavior
True Negatives:                      45.30% (453/1000)
True Positives:                      43.80% (438/1000)
False Negatives:                     5.20% (52/1000)
False Positives:                     5.70% (57/1000)
True Pos. Rate/Sensitivity/Recall:   0.89
True Neg. Rate/Specificity:          0.89
Precision:                           0.88
F-1 Measure:                         0.89
False Negative Rate/Miss Rate:       0.11
Critical Success Index:              0.80
Confusion Matrix:
 [45.30% 5.70%]
 [5.20% 43.80%]
Overfitting:                         No
Note: Labels have been remapped to 'P'=0, 'N'=1.
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
TRAINFILE = "fri-c0-1000-50.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50
n_classes = 2


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="binaryClass"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"
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
    clean.mapping={'P': 0, 'N': 1}

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
    h_0 = max((((8.973863 * float(x[0]))+ (11.819556 * float(x[1]))+ (-0.0012664635 * float(x[2]))+ (22.121769 * float(x[3]))+ (3.1727583 * float(x[4]))+ (-0.5869671 * float(x[5]))+ (0.6083971 * float(x[6]))+ (0.087530464 * float(x[7]))+ (-2.1061895 * float(x[8]))+ (-2.6900716 * float(x[9]))+ (-1.8938698 * float(x[10]))+ (2.2281778 * float(x[11]))+ (-5.023972 * float(x[12]))+ (-2.9658267 * float(x[13]))+ (0.5483225 * float(x[14]))+ (2.0074477 * float(x[15]))+ (-1.8843933 * float(x[16]))+ (4.902292 * float(x[17]))+ (4.8837442 * float(x[18]))+ (-2.0981462 * float(x[19]))+ (-3.4210095 * float(x[20]))+ (-4.664538 * float(x[21]))+ (-0.8848408 * float(x[22]))+ (1.421112 * float(x[23]))+ (0.107214585 * float(x[24]))+ (-1.1429181 * float(x[25]))+ (7.970785 * float(x[26]))+ (1.2983321 * float(x[27]))+ (-0.83256406 * float(x[28]))+ (4.733686 * float(x[29]))+ (4.4247403 * float(x[30]))+ (1.3737924 * float(x[31]))+ (-1.2645694 * float(x[32]))+ (2.4162867 * float(x[33]))+ (-2.1051383 * float(x[34]))+ (-1.1857115 * float(x[35]))+ (-4.113349 * float(x[36]))+ (2.375697 * float(x[37]))+ (4.167623 * float(x[38]))+ (2.910851 * float(x[39]))+ (1.0274228 * float(x[40]))+ (3.5805154 * float(x[41]))+ (-1.7504207 * float(x[42]))+ (3.38039 * float(x[43]))+ (-2.7207139 * float(x[44]))+ (-2.380976 * float(x[45]))+ (-3.6474357 * float(x[46]))+ (2.2491071 * float(x[47]))+ (0.88129324 * float(x[48]))+ (8.137511 * float(x[49]))) + 9.283743), 0)
    h_1 = max((((-7.7143984 * float(x[0]))+ (-7.5971847 * float(x[1]))+ (-3.2781093 * float(x[2]))+ (-2.8835065 * float(x[3]))+ (1.1363257 * float(x[4]))+ (2.1800692 * float(x[5]))+ (-4.186847 * float(x[6]))+ (-1.3095325 * float(x[7]))+ (4.615683 * float(x[8]))+ (1.224311 * float(x[9]))+ (2.1193855 * float(x[10]))+ (-1.2820894 * float(x[11]))+ (-2.2358763 * float(x[12]))+ (1.7803197 * float(x[13]))+ (-2.721685 * float(x[14]))+ (-1.29392 * float(x[15]))+ (-3.3235373 * float(x[16]))+ (-0.18261096 * float(x[17]))+ (0.061404776 * float(x[18]))+ (-3.3446977 * float(x[19]))+ (-0.029314565 * float(x[20]))+ (1.5355392 * float(x[21]))+ (-1.4030479 * float(x[22]))+ (2.2294323 * float(x[23]))+ (-2.703724 * float(x[24]))+ (2.4805589 * float(x[25]))+ (-4.604972 * float(x[26]))+ (-2.2380738 * float(x[27]))+ (0.20895265 * float(x[28]))+ (-1.2981697 * float(x[29]))+ (6.192837 * float(x[30]))+ (-2.3347092 * float(x[31]))+ (-5.909197 * float(x[32]))+ (-0.7042454 * float(x[33]))+ (2.5598006 * float(x[34]))+ (4.2945147 * float(x[35]))+ (2.269195 * float(x[36]))+ (-2.0676856 * float(x[37]))+ (-7.748029 * float(x[38]))+ (0.7523452 * float(x[39]))+ (-0.5245662 * float(x[40]))+ (-0.38908616 * float(x[41]))+ (-3.018124 * float(x[42]))+ (-0.5499844 * float(x[43]))+ (-0.8180432 * float(x[44]))+ (-1.0610491 * float(x[45]))+ (2.274157 * float(x[46]))+ (-3.8062122 * float(x[47]))+ (0.16291772 * float(x[48]))+ (1.6321642 * float(x[49]))) + -6.969895), 0)
    h_2 = max((((0.16579396 * float(x[0]))+ (-1.0897166 * float(x[1]))+ (2.368882 * float(x[2]))+ (-2.688231 * float(x[3]))+ (0.048810497 * float(x[4]))+ (-0.36516714 * float(x[5]))+ (1.8747826 * float(x[6]))+ (2.242445 * float(x[7]))+ (-0.4823528 * float(x[8]))+ (-1.1026102 * float(x[9]))+ (0.49073985 * float(x[10]))+ (2.14852 * float(x[11]))+ (-2.1309657 * float(x[12]))+ (1.1948259 * float(x[13]))+ (2.2531714 * float(x[14]))+ (-0.8369482 * float(x[15]))+ (2.5296953 * float(x[16]))+ (0.6210748 * float(x[17]))+ (2.3688207 * float(x[18]))+ (-0.074056596 * float(x[19]))+ (-1.7617872 * float(x[20]))+ (1.704152 * float(x[21]))+ (0.116003424 * float(x[22]))+ (-1.7533021 * float(x[23]))+ (0.11165496 * float(x[24]))+ (1.1988348 * float(x[25]))+ (0.74449885 * float(x[26]))+ (1.3939763 * float(x[27]))+ (0.6464732 * float(x[28]))+ (1.5778286 * float(x[29]))+ (0.9796764 * float(x[30]))+ (1.4294529 * float(x[31]))+ (0.99253774 * float(x[32]))+ (0.576534 * float(x[33]))+ (-0.14314856 * float(x[34]))+ (-2.189352 * float(x[35]))+ (-0.14006509 * float(x[36]))+ (1.780342 * float(x[37]))+ (1.4613497 * float(x[38]))+ (-2.2100968 * float(x[39]))+ (0.86834365 * float(x[40]))+ (0.7268524 * float(x[41]))+ (-0.38035795 * float(x[42]))+ (-0.4976686 * float(x[43]))+ (-0.37189686 * float(x[44]))+ (0.7781666 * float(x[45]))+ (0.35859042 * float(x[46]))+ (0.8700503 * float(x[47]))+ (0.49967748 * float(x[48]))+ (-2.0880616 * float(x[49]))) + 2.5941427), 0)
    h_3 = max((((0.7438433 * float(x[0]))+ (-0.22749758 * float(x[1]))+ (1.0050969 * float(x[2]))+ (3.151022 * float(x[3]))+ (-1.0607214 * float(x[4]))+ (1.13616 * float(x[5]))+ (-0.67253196 * float(x[6]))+ (-0.4668879 * float(x[7]))+ (-0.18496475 * float(x[8]))+ (-0.460429 * float(x[9]))+ (-0.023018522 * float(x[10]))+ (-0.37742835 * float(x[11]))+ (-0.35643893 * float(x[12]))+ (-0.20153642 * float(x[13]))+ (1.2937053 * float(x[14]))+ (0.28402352 * float(x[15]))+ (0.0885473 * float(x[16]))+ (0.5158465 * float(x[17]))+ (0.7002172 * float(x[18]))+ (-1.1958466 * float(x[19]))+ (-0.0724428 * float(x[20]))+ (-0.21270616 * float(x[21]))+ (0.8356874 * float(x[22]))+ (0.44760555 * float(x[23]))+ (0.1276474 * float(x[24]))+ (0.2017789 * float(x[25]))+ (0.5878044 * float(x[26]))+ (0.6500231 * float(x[27]))+ (-0.14722103 * float(x[28]))+ (0.49607536 * float(x[29]))+ (1.2565027 * float(x[30]))+ (0.24827544 * float(x[31]))+ (0.65141714 * float(x[32]))+ (-0.79241306 * float(x[33]))+ (-0.72353154 * float(x[34]))+ (-0.34123552 * float(x[35]))+ (-2.935139 * float(x[36]))+ (0.31895253 * float(x[37]))+ (0.81082255 * float(x[38]))+ (2.418942 * float(x[39]))+ (0.1797873 * float(x[40]))+ (0.53991425 * float(x[41]))+ (-0.102812864 * float(x[42]))+ (0.8133752 * float(x[43]))+ (-0.50868803 * float(x[44]))+ (-1.054197 * float(x[45]))+ (-1.1224252 * float(x[46]))+ (1.9525542 * float(x[47]))+ (0.7922029 * float(x[48]))+ (3.483525 * float(x[49]))) + 1.0759932), 0)
    h_4 = max((((0.44824338 * float(x[0]))+ (-0.8823642 * float(x[1]))+ (1.7912197 * float(x[2]))+ (0.43680456 * float(x[3]))+ (1.0060529 * float(x[4]))+ (0.7057989 * float(x[5]))+ (-0.8768654 * float(x[6]))+ (0.39405784 * float(x[7]))+ (1.208211 * float(x[8]))+ (-0.4292768 * float(x[9]))+ (1.6716297 * float(x[10]))+ (-0.45938313 * float(x[11]))+ (-0.8311403 * float(x[12]))+ (1.4072598 * float(x[13]))+ (2.051022 * float(x[14]))+ (-1.0721774 * float(x[15]))+ (0.13381661 * float(x[16]))+ (0.48062608 * float(x[17]))+ (0.7958068 * float(x[18]))+ (-1.6034615 * float(x[19]))+ (-0.5320852 * float(x[20]))+ (2.008288 * float(x[21]))+ (0.17082141 * float(x[22]))+ (-0.2644963 * float(x[23]))+ (-0.74483687 * float(x[24]))+ (1.4102026 * float(x[25]))+ (-1.8907558 * float(x[26]))+ (0.48877448 * float(x[27]))+ (1.0423563 * float(x[28]))+ (1.367761 * float(x[29]))+ (2.5861225 * float(x[30]))+ (0.025481591 * float(x[31]))+ (-0.6994616 * float(x[32]))+ (-0.73904574 * float(x[33]))+ (0.530475 * float(x[34]))+ (-0.17758758 * float(x[35]))+ (0.0993233 * float(x[36]))+ (0.8735939 * float(x[37]))+ (-0.04306568 * float(x[38]))+ (1.1235874 * float(x[39]))+ (-0.29704234 * float(x[40]))+ (-0.25297034 * float(x[41]))+ (-0.9150426 * float(x[42]))+ (-0.34984455 * float(x[43]))+ (-0.28537306 * float(x[44]))+ (-1.325385 * float(x[45]))+ (-0.14063483 * float(x[46]))+ (1.7558662 * float(x[47]))+ (1.0286454 * float(x[48]))+ (1.4854429 * float(x[49]))) + 0.0485157), 0)
    o[0] = (0.6725097 * h_0)+ (-1.0748922 * h_1)+ (-1.3954525 * h_2)+ (-2.846869 * h_3)+ (3.1462798 * h_4) + 1.7078115

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
        model_cap=261
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

