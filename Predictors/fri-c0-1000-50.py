#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53438/fri_c0_1000_50.arff -o Predictors/fri-c0-1000-50_NN.py -target binaryClass -stopat 88.6 -f NN -e 20 --yes
# Total compiler execution time: 0:32:01.98. Finished on: Apr-21-2020 20:53:40.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                51.00%
Model accuracy:                     88.50% (885/1000 correct)
Improvement over best guess:        37.50% (of possible 49.0%)
Model capacity (MEC):               313 bits
Generalization ratio:               2.82 bits/bit
Model efficiency:                   0.11%/parameter
System behavior
True Negatives:                     46.20% (462/1000)
True Positives:                     42.30% (423/1000)
False Negatives:                    6.70% (67/1000)
False Positives:                    4.80% (48/1000)
True Pos. Rate/Sensitivity/Recall:  0.86
True Neg. Rate/Specificity:         0.91
Precision:                          0.90
F-1 Measure:                        0.88
False Negative Rate/Miss Rate:      0.14
Critical Success Index:             0.79

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
TRAINFILE = "fri_c0_1000_50.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50
n_classes = 2


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"


    if (testfile):
        target=''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless==False):
                header=next(reader, None)
                try:
                    if (target!=''): 
                        hc=header.index(target)
                    else:
                        hc=len(header)-1
                        target=header[hc]
                except:
                    raise NameError("Target '"+target+"' not found! Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=header.index(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute '"+ignorecolumns[i]+"' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '"+ignorecolumns[i]+"' not found in header. Header must be same as in file passed to btc.")
                for i in range(0,len(header)):      
                    if (i==hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i]+",", end = '', file=outputfile)
                print(header[hc],file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"'+row[name]+'"'+",",end = '', file=outputfile)
                        else:
                            print (row[name]+",",end = '', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target!=""): 
                        hc=int(target)
                    else:
                        hc=-1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=int(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute "+str(col)+" is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    if (hc==-1):
                        hc=len(row)-1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0,len(row)):
                        if (i in il):
                            continue
                        if (i==hc):
                            continue
                        if (',' in row[i]):
                            print ('"'+row[i]+'"'+",",end = '', file=outputfile)
                        else:
                            print(row[i]+",",end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
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
def classify(row):
    x = row
    o = [0] * num_output_logits
    h_0 = max((((15.363158 * float(x[0]))+ (22.43136 * float(x[1]))+ (-2.6790166 * float(x[2]))+ (15.7109585 * float(x[3]))+ (3.4418602 * float(x[4]))+ (-1.5837942 * float(x[5]))+ (-4.0121536 * float(x[6]))+ (-0.13588373 * float(x[7]))+ (0.90931267 * float(x[8]))+ (1.0898898 * float(x[9]))+ (-4.3059044 * float(x[10]))+ (1.8166447 * float(x[11]))+ (-1.0565134 * float(x[12]))+ (-1.8959967 * float(x[13]))+ (0.98761225 * float(x[14]))+ (4.7608256 * float(x[15]))+ (-1.6316974 * float(x[16]))+ (3.1891472 * float(x[17]))+ (1.1787685 * float(x[18]))+ (0.7106459 * float(x[19]))+ (-4.780088 * float(x[20]))+ (-0.5013518 * float(x[21]))+ (-2.9164884 * float(x[22]))+ (1.6246343 * float(x[23]))+ (-1.6913271 * float(x[24]))+ (-1.2375798 * float(x[25]))+ (0.9902879 * float(x[26]))+ (-0.59876776 * float(x[27]))+ (-2.761289 * float(x[28]))+ (1.9262644 * float(x[29]))+ (0.63977087 * float(x[30]))+ (4.5357547 * float(x[31]))+ (-0.68353 * float(x[32]))+ (-1.1137778 * float(x[33]))+ (0.44628033 * float(x[34]))+ (1.6781799 * float(x[35]))+ (0.35588452 * float(x[36]))+ (-4.0935946 * float(x[37]))+ (-1.4350611 * float(x[38]))+ (-0.77949244 * float(x[39]))+ (1.2633646 * float(x[40]))+ (-3.5609171 * float(x[41]))+ (-3.9336076 * float(x[42]))+ (3.5863376 * float(x[43]))+ (-7.9927893 * float(x[44]))+ (-2.020609 * float(x[45]))+ (-0.87249535 * float(x[46]))+ (5.19843 * float(x[47]))+ (3.0296972 * float(x[48]))+ (3.3120837 * float(x[49]))) + 0.8232327), 0)
    h_1 = max((((1.638557 * float(x[0]))+ (1.8878454 * float(x[1]))+ (0.39992425 * float(x[2]))+ (4.2109394 * float(x[3]))+ (0.12187495 * float(x[4]))+ (-0.023806624 * float(x[5]))+ (6.239449 * float(x[6]))+ (-2.2104073 * float(x[7]))+ (0.010491819 * float(x[8]))+ (-7.4619746 * float(x[9]))+ (-0.77211106 * float(x[10]))+ (2.0042863 * float(x[11]))+ (1.6479757 * float(x[12]))+ (-3.872217 * float(x[13]))+ (-2.4034474 * float(x[14]))+ (-1.6869432 * float(x[15]))+ (-1.052 * float(x[16]))+ (1.8696471 * float(x[17]))+ (-3.3838806 * float(x[18]))+ (1.7391671 * float(x[19]))+ (2.4003546 * float(x[20]))+ (0.39860922 * float(x[21]))+ (-2.4728198 * float(x[22]))+ (-0.6646501 * float(x[23]))+ (-0.7107552 * float(x[24]))+ (4.7874794 * float(x[25]))+ (-1.1863188 * float(x[26]))+ (0.9692109 * float(x[27]))+ (1.3594491 * float(x[28]))+ (-1.2841634 * float(x[29]))+ (-2.2019238 * float(x[30]))+ (-3.1016126 * float(x[31]))+ (-3.9870799 * float(x[32]))+ (-1.5647223 * float(x[33]))+ (1.4217566 * float(x[34]))+ (1.3506478 * float(x[35]))+ (-0.7094575 * float(x[36]))+ (-3.454223 * float(x[37]))+ (1.5396852 * float(x[38]))+ (0.45431757 * float(x[39]))+ (-2.1948335 * float(x[40]))+ (1.7966095 * float(x[41]))+ (0.24415155 * float(x[42]))+ (-0.73331743 * float(x[43]))+ (-2.9811563 * float(x[44]))+ (3.3390324 * float(x[45]))+ (-5.4258814 * float(x[46]))+ (-0.49885902 * float(x[47]))+ (2.1898174 * float(x[48]))+ (0.24546205 * float(x[49]))) + -6.1251726), 0)
    h_2 = max((((-2.7757287 * float(x[0]))+ (1.8128835 * float(x[1]))+ (5.2241693 * float(x[2]))+ (2.558055 * float(x[3]))+ (3.6412015 * float(x[4]))+ (2.7187822 * float(x[5]))+ (-1.3952596 * float(x[6]))+ (-0.3564134 * float(x[7]))+ (-0.07202902 * float(x[8]))+ (0.42808697 * float(x[9]))+ (1.1815269 * float(x[10]))+ (-3.7102463 * float(x[11]))+ (0.51942533 * float(x[12]))+ (-0.5379457 * float(x[13]))+ (-2.4938855 * float(x[14]))+ (-1.0560555 * float(x[15]))+ (-1.5742855 * float(x[16]))+ (2.965323 * float(x[17]))+ (0.3087871 * float(x[18]))+ (1.6808012 * float(x[19]))+ (-0.07016428 * float(x[20]))+ (2.3621092 * float(x[21]))+ (2.1448693 * float(x[22]))+ (-0.8832662 * float(x[23]))+ (3.4302342 * float(x[24]))+ (0.5711817 * float(x[25]))+ (-1.3860494 * float(x[26]))+ (-0.36797386 * float(x[27]))+ (0.3464506 * float(x[28]))+ (2.2855275 * float(x[29]))+ (4.539479 * float(x[30]))+ (1.674523 * float(x[31]))+ (-0.23266008 * float(x[32]))+ (0.11143141 * float(x[33]))+ (2.8267365 * float(x[34]))+ (-3.184866 * float(x[35]))+ (0.31655443 * float(x[36]))+ (2.3005676 * float(x[37]))+ (0.065167055 * float(x[38]))+ (2.1966028 * float(x[39]))+ (1.4783736 * float(x[40]))+ (1.5554422 * float(x[41]))+ (-0.7880006 * float(x[42]))+ (2.0017707 * float(x[43]))+ (2.254852 * float(x[44]))+ (-1.5031312 * float(x[45]))+ (1.7121683 * float(x[46]))+ (0.53374827 * float(x[47]))+ (-2.9400094 * float(x[48]))+ (2.6926022 * float(x[49]))) + 0.9761923), 0)
    h_3 = max((((-4.9150963 * float(x[0]))+ (0.18148895 * float(x[1]))+ (1.110432 * float(x[2]))+ (-1.9907815 * float(x[3]))+ (-0.021307312 * float(x[4]))+ (-0.36010647 * float(x[5]))+ (1.0060532 * float(x[6]))+ (0.8453117 * float(x[7]))+ (-0.76621294 * float(x[8]))+ (-1.9981102 * float(x[9]))+ (2.7854211 * float(x[10]))+ (0.5911908 * float(x[11]))+ (1.9882283 * float(x[12]))+ (-1.4727883 * float(x[13]))+ (-2.4739733 * float(x[14]))+ (-3.9598753 * float(x[15]))+ (-2.3859353 * float(x[16]))+ (0.6836788 * float(x[17]))+ (-0.671088 * float(x[18]))+ (2.251076 * float(x[19]))+ (-0.55937517 * float(x[20]))+ (-0.81126094 * float(x[21]))+ (1.2637697 * float(x[22]))+ (-0.80751646 * float(x[23]))+ (2.2180536 * float(x[24]))+ (0.036282096 * float(x[25]))+ (-4.2919507 * float(x[26]))+ (-2.4156287 * float(x[27]))+ (1.3867396 * float(x[28]))+ (0.36627305 * float(x[29]))+ (-1.9555681 * float(x[30]))+ (-1.4650315 * float(x[31]))+ (0.21105817 * float(x[32]))+ (-2.3112266 * float(x[33]))+ (4.9916267 * float(x[34]))+ (-0.70509034 * float(x[35]))+ (-0.091428205 * float(x[36]))+ (0.30816296 * float(x[37]))+ (0.5909529 * float(x[38]))+ (4.866355 * float(x[39]))+ (-2.3597562 * float(x[40]))+ (0.08632329 * float(x[41]))+ (0.92956567 * float(x[42]))+ (0.3419781 * float(x[43]))+ (2.7107098 * float(x[44]))+ (-1.4724903 * float(x[45]))+ (0.2933969 * float(x[46]))+ (-1.762782 * float(x[47]))+ (-4.0013475 * float(x[48]))+ (0.70270526 * float(x[49]))) + 1.0230534), 0)
    h_4 = max((((-0.7555764 * float(x[0]))+ (3.0591002 * float(x[1]))+ (1.5871551 * float(x[2]))+ (-0.025898922 * float(x[3]))+ (1.552265 * float(x[4]))+ (2.442969 * float(x[5]))+ (-1.6981577 * float(x[6]))+ (-1.8009365 * float(x[7]))+ (2.4738584 * float(x[8]))+ (1.7181516 * float(x[9]))+ (0.24143137 * float(x[10]))+ (-1.3662816 * float(x[11]))+ (-0.9751241 * float(x[12]))+ (-1.4074539 * float(x[13]))+ (-1.8345842 * float(x[14]))+ (1.5852404 * float(x[15]))+ (-0.68026763 * float(x[16]))+ (-0.34780562 * float(x[17]))+ (1.4350561 * float(x[18]))+ (-0.39950994 * float(x[19]))+ (-0.37892455 * float(x[20]))+ (2.2464404 * float(x[21]))+ (1.3299215 * float(x[22]))+ (0.7786262 * float(x[23]))+ (1.4529482 * float(x[24]))+ (2.322513 * float(x[25]))+ (-1.24853 * float(x[26]))+ (-0.21365754 * float(x[27]))+ (-1.8845934 * float(x[28]))+ (-1.1000768 * float(x[29]))+ (3.116922 * float(x[30]))+ (1.2412367 * float(x[31]))+ (-1.6275187 * float(x[32]))+ (2.04237 * float(x[33]))+ (-1.0982722 * float(x[34]))+ (0.34770176 * float(x[35]))+ (-0.571639 * float(x[36]))+ (0.53083944 * float(x[37]))+ (-0.061879426 * float(x[38]))+ (-1.6179554 * float(x[39]))+ (1.568511 * float(x[40]))+ (0.7482442 * float(x[41]))+ (-2.029663 * float(x[42]))+ (-0.91177577 * float(x[43]))+ (-1.3775896 * float(x[44]))+ (-0.7500508 * float(x[45]))+ (0.7391668 * float(x[46]))+ (-0.20839314 * float(x[47]))+ (0.06331116 * float(x[48]))+ (0.31504238 * float(x[49]))) + 0.009010044), 0)
    h_5 = max((((1.0921938 * float(x[0]))+ (0.7439978 * float(x[1]))+ (1.4409341 * float(x[2]))+ (1.0087687 * float(x[3]))+ (-1.9315312 * float(x[4]))+ (2.0469513 * float(x[5]))+ (1.5874817 * float(x[6]))+ (-1.5447596 * float(x[7]))+ (-2.4554098 * float(x[8]))+ (-1.0559686 * float(x[9]))+ (-2.8787897 * float(x[10]))+ (-1.0932925 * float(x[11]))+ (1.0934788 * float(x[12]))+ (-0.21672648 * float(x[13]))+ (0.47094527 * float(x[14]))+ (0.42741582 * float(x[15]))+ (-0.13400955 * float(x[16]))+ (2.8633008 * float(x[17]))+ (-2.3697731 * float(x[18]))+ (1.7735418 * float(x[19]))+ (0.6335402 * float(x[20]))+ (2.5242436 * float(x[21]))+ (-1.6620343 * float(x[22]))+ (-0.61983913 * float(x[23]))+ (-0.052255943 * float(x[24]))+ (0.02670862 * float(x[25]))+ (1.3947818 * float(x[26]))+ (2.3116379 * float(x[27]))+ (-0.028328095 * float(x[28]))+ (1.787717 * float(x[29]))+ (0.31425887 * float(x[30]))+ (0.92037576 * float(x[31]))+ (0.776449 * float(x[32]))+ (-1.6227018 * float(x[33]))+ (0.43477288 * float(x[34]))+ (-1.6351528 * float(x[35]))+ (-1.9314864 * float(x[36]))+ (-1.2109113 * float(x[37]))+ (-2.1058304 * float(x[38]))+ (0.5846048 * float(x[39]))+ (1.8441056 * float(x[40]))+ (0.46979773 * float(x[41]))+ (-0.74980956 * float(x[42]))+ (0.95466727 * float(x[43]))+ (-1.6538596 * float(x[44]))+ (3.3233688 * float(x[45]))+ (-1.8949581 * float(x[46]))+ (3.046687 * float(x[47]))+ (0.5083433 * float(x[48]))+ (2.0144424 * float(x[49]))) + -0.85732245), 0)
    o[0] = (1.7747458 * h_0)+ (2.2599578 * h_1)+ (3.389686 * h_2)+ (-2.8500743 * h_3)+ (-4.8376474 * h_4)+ (-4.283121 * h_5) + -1.8355514

    if num_output_logits == 1:
        return o[0] >= 0
    else:
        return argmax(o)


def Validate(cleanvalfile):
    #Binary
    if n_classes == 2:
        with open(cleanvalfile, 'r') as valcsvfile:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
            valcsvreader = csv.reader(valcsvfile)
            for valrow in valcsvreader:
                if len(valrow) == 0:
                    continue
                if int(classify(valrow[:-1])) == int(float(valrow[-1])):
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
                pred = int(classify(valrow[:-1]))
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
            print(str(','.join(str(j) for j in ([i for i in dirtyrow]))) + ',' + str(get_key(int(classify(cleanrow)), classmapping)))



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
        cleanfile = tempdir + os.sep + "clean.csv"
        preprocessedfile = tempdir + os.sep + "prep.csv"
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x,y: x
        classmapping = {}


    #Predict
    if not args.validate:
        Predict(cleanfile, preprocessedfile, args.headerless, get_key, classmapping)


    #Validate
    else: 
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap=313
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





            def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
                #check for numpy/scipy is imported
                try:
                    from scipy.sparse import coo_matrix #required for multiclass metrics
                    try:
                        np.array
                    except:
                        import numpy as np
                except:
                    raise ValueError("Scipy and Numpy Required for Multiclass Metrics")
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

