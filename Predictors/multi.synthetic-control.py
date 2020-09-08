#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class synthetic-control.csv -o synthetic-control_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:13:27.14. Finished on: Sep-04-2020 12:17:48.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         6-way classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 16.67%
Training accuracy:                   100.00% (420/420 correct)
Validation accuracy:                 100.00% (180/180 correct)
Overall Model accuracy:              100.00% (600/600 correct)
Overall Improvement over best guess: 83.33% (of possible 83.33%)
Model capacity (MEC):                346 bits
Generalization ratio:                1.73 bits/bit
Model efficiency:                    0.24%/parameter
Confusion Matrix:
 [16.67% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 16.67% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 16.67% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 16.67% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 16.67% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 16.67%]
Overfitting:                         No
Note: Labels have been remapped to 'Normal'=0, 'Cyclic'=1, 'Increasing_trend'=2, 'Decreasing_trend'=3, 'Upward_shift'=4, 'Downward_shift'=5.
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
TRAINFILE = "synthetic-control.csv"


#Number of output logits
num_output_logits = 6

#Number of attributes
num_attr = 61
n_classes = 6


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
    h_0 = max((((-0.080157004 * float(x[0]))+ (-0.010061183 * float(x[1]))+ (-0.3068043 * float(x[2]))+ (0.24472998 * float(x[3]))+ (0.2484288 * float(x[4]))+ (0.15830255 * float(x[5]))+ (-0.07162977 * float(x[6]))+ (0.17264968 * float(x[7]))+ (-0.18371245 * float(x[8]))+ (0.22688824 * float(x[9]))+ (-0.24938487 * float(x[10]))+ (-0.27465078 * float(x[11]))+ (-0.07400208 * float(x[12]))+ (0.2575888 * float(x[13]))+ (-0.1949171 * float(x[14]))+ (-0.2045876 * float(x[15]))+ (0.21537288 * float(x[16]))+ (-0.057701837 * float(x[17]))+ (-0.1066285 * float(x[18]))+ (0.006837817 * float(x[19]))+ (0.17873418 * float(x[20]))+ (-0.19535786 * float(x[21]))+ (0.14215985 * float(x[22]))+ (-0.062089473 * float(x[23]))+ (-0.19259799 * float(x[24]))+ (0.07802512 * float(x[25]))+ (-0.22119908 * float(x[26]))+ (-0.06468577 * float(x[27]))+ (0.2772614 * float(x[28]))+ (0.09805469 * float(x[29]))+ (-0.17331766 * float(x[30]))+ (0.10144217 * float(x[31]))+ (0.12982532 * float(x[32]))+ (0.23893575 * float(x[33]))+ (-0.29186037 * float(x[34]))+ (-0.27553383 * float(x[35]))+ (0.11514716 * float(x[36]))+ (0.251805 * float(x[37]))+ (-0.10002716 * float(x[38]))+ (-0.18708207 * float(x[39]))+ (0.14851348 * float(x[40]))+ (0.063645124 * float(x[41]))+ (-0.19661361 * float(x[42]))+ (0.10244857 * float(x[43]))+ (0.25722635 * float(x[44]))+ (-0.28613248 * float(x[45]))+ (-0.11913131 * float(x[46]))+ (-0.28383276 * float(x[47]))+ (0.103188634 * float(x[48]))+ (0.02018949 * float(x[49])))+ ((-0.3138786 * float(x[50]))+ (0.19596288 * float(x[51]))+ (-0.007465444 * float(x[52]))+ (-0.017882712 * float(x[53]))+ (-0.25211275 * float(x[54]))+ (-0.15623148 * float(x[55]))+ (7.3467534e-05 * float(x[56]))+ (0.19536091 * float(x[57]))+ (0.18869138 * float(x[58]))+ (-0.21016453 * float(x[59]))+ (0.062344663 * float(x[60]))) + 0.6463508), 0)
    h_1 = max((((0.027737929 * float(x[0]))+ (0.2737655 * float(x[1]))+ (0.5052219 * float(x[2]))+ (0.745619 * float(x[3]))+ (0.42568251 * float(x[4]))+ (0.7555689 * float(x[5]))+ (0.29395533 * float(x[6]))+ (0.119503334 * float(x[7]))+ (0.024668818 * float(x[8]))+ (-0.17367923 * float(x[9]))+ (-0.35123527 * float(x[10]))+ (-0.3411414 * float(x[11]))+ (0.043217365 * float(x[12]))+ (0.16615441 * float(x[13]))+ (0.1352407 * float(x[14]))+ (0.113837935 * float(x[15]))+ (0.31691977 * float(x[16]))+ (0.23715547 * float(x[17]))+ (0.27343833 * float(x[18]))+ (0.18617024 * float(x[19]))+ (0.0072742044 * float(x[20]))+ (-0.347196 * float(x[21]))+ (-0.084137745 * float(x[22]))+ (-0.009326522 * float(x[23]))+ (-0.076710016 * float(x[24]))+ (0.003416884 * float(x[25]))+ (0.17031486 * float(x[26]))+ (-0.011771956 * float(x[27]))+ (-0.22940344 * float(x[28]))+ (-0.102525145 * float(x[29]))+ (-0.044068906 * float(x[30]))+ (0.12670629 * float(x[31]))+ (0.10084787 * float(x[32]))+ (0.010927294 * float(x[33]))+ (-0.07139301 * float(x[34]))+ (-0.001443114 * float(x[35]))+ (0.027943492 * float(x[36]))+ (-0.07749268 * float(x[37]))+ (0.13508825 * float(x[38]))+ (-0.13682586 * float(x[39]))+ (0.014927186 * float(x[40]))+ (-0.058067262 * float(x[41]))+ (-0.16963774 * float(x[42]))+ (-0.34888834 * float(x[43]))+ (0.27155387 * float(x[44]))+ (0.118704505 * float(x[45]))+ (0.044652835 * float(x[46]))+ (0.09737967 * float(x[47]))+ (0.04056753 * float(x[48]))+ (-0.055325177 * float(x[49])))+ ((0.035936233 * float(x[50]))+ (-0.2873673 * float(x[51]))+ (-0.31748313 * float(x[52]))+ (-0.37950456 * float(x[53]))+ (-0.37509802 * float(x[54]))+ (-0.12902537 * float(x[55]))+ (-0.1512498 * float(x[56]))+ (-0.18014286 * float(x[57]))+ (-0.33100206 * float(x[58]))+ (-0.24052884 * float(x[59]))+ (-0.3654196 * float(x[60]))) + 1.1265694), 0)
    h_2 = max((((-0.26345536 * float(x[0]))+ (0.047123894 * float(x[1]))+ (0.060514867 * float(x[2]))+ (0.16431627 * float(x[3]))+ (-0.09161745 * float(x[4]))+ (0.23101936 * float(x[5]))+ (-0.034655184 * float(x[6]))+ (0.16639523 * float(x[7]))+ (-0.07640202 * float(x[8]))+ (-0.28931734 * float(x[9]))+ (-0.12402501 * float(x[10]))+ (-0.104067974 * float(x[11]))+ (-0.2781665 * float(x[12]))+ (0.014187092 * float(x[13]))+ (0.1689364 * float(x[14]))+ (-0.0330723 * float(x[15]))+ (0.09328607 * float(x[16]))+ (0.19178693 * float(x[17]))+ (0.21396112 * float(x[18]))+ (-0.19029185 * float(x[19]))+ (0.18925913 * float(x[20]))+ (0.02001328 * float(x[21]))+ (-0.18157434 * float(x[22]))+ (0.10897914 * float(x[23]))+ (0.26109505 * float(x[24]))+ (-0.3067313 * float(x[25]))+ (0.21083191 * float(x[26]))+ (0.24710655 * float(x[27]))+ (0.007144615 * float(x[28]))+ (0.23348954 * float(x[29]))+ (-0.33298218 * float(x[30]))+ (-0.2517261 * float(x[31]))+ (-0.10634181 * float(x[32]))+ (-0.26896003 * float(x[33]))+ (-0.0022978573 * float(x[34]))+ (-0.199763 * float(x[35]))+ (0.09099002 * float(x[36]))+ (0.022292793 * float(x[37]))+ (0.20332272 * float(x[38]))+ (-0.24584755 * float(x[39]))+ (0.027338466 * float(x[40]))+ (-0.063713744 * float(x[41]))+ (0.11441005 * float(x[42]))+ (0.19894587 * float(x[43]))+ (-0.16752473 * float(x[44]))+ (-0.002245934 * float(x[45]))+ (0.044109423 * float(x[46]))+ (-0.1426126 * float(x[47]))+ (0.23497732 * float(x[48]))+ (0.22204182 * float(x[49])))+ ((-0.32310766 * float(x[50]))+ (-0.09721822 * float(x[51]))+ (-0.076399736 * float(x[52]))+ (-0.22583169 * float(x[53]))+ (-0.052127443 * float(x[54]))+ (0.16611987 * float(x[55]))+ (0.17946735 * float(x[56]))+ (0.23467149 * float(x[57]))+ (-0.11052366 * float(x[58]))+ (-0.2865848 * float(x[59]))+ (-0.017141685 * float(x[60]))) + 0.29384053), 0)
    h_3 = max((((0.12151797 * float(x[0]))+ (-0.30708197 * float(x[1]))+ (-0.17191324 * float(x[2]))+ (-0.23234822 * float(x[3]))+ (-0.01855438 * float(x[4]))+ (0.025006743 * float(x[5]))+ (0.15435481 * float(x[6]))+ (-0.0020694037 * float(x[7]))+ (0.13842127 * float(x[8]))+ (0.15000512 * float(x[9]))+ (-0.15799269 * float(x[10]))+ (0.0040403227 * float(x[11]))+ (0.12750132 * float(x[12]))+ (0.136496 * float(x[13]))+ (-0.15156554 * float(x[14]))+ (-0.24668737 * float(x[15]))+ (-0.15149327 * float(x[16]))+ (0.10975259 * float(x[17]))+ (-0.25811693 * float(x[18]))+ (-0.26195398 * float(x[19]))+ (0.14111009 * float(x[20]))+ (-0.36394945 * float(x[21]))+ (-0.34587258 * float(x[22]))+ (0.036768116 * float(x[23]))+ (-0.06378447 * float(x[24]))+ (0.09546559 * float(x[25]))+ (-0.27456793 * float(x[26]))+ (-0.1461714 * float(x[27]))+ (0.15405913 * float(x[28]))+ (-0.29182833 * float(x[29]))+ (-0.3380109 * float(x[30]))+ (-0.028259588 * float(x[31]))+ (-0.13053986 * float(x[32]))+ (-0.34360975 * float(x[33]))+ (0.003342334 * float(x[34]))+ (0.18003114 * float(x[35]))+ (0.13482703 * float(x[36]))+ (-0.031865787 * float(x[37]))+ (0.16199163 * float(x[38]))+ (-0.35924643 * float(x[39]))+ (-0.057341237 * float(x[40]))+ (-0.1338748 * float(x[41]))+ (0.18371676 * float(x[42]))+ (-0.3279299 * float(x[43]))+ (0.17913853 * float(x[44]))+ (-0.08074048 * float(x[45]))+ (-0.24633837 * float(x[46]))+ (-0.1017069 * float(x[47]))+ (-0.2508646 * float(x[48]))+ (0.15791431 * float(x[49])))+ ((-0.31572872 * float(x[50]))+ (0.09864715 * float(x[51]))+ (0.08061876 * float(x[52]))+ (-0.15825349 * float(x[53]))+ (0.00042499154 * float(x[54]))+ (-0.034491736 * float(x[55]))+ (-0.114561416 * float(x[56]))+ (-0.34298643 * float(x[57]))+ (-0.010836296 * float(x[58]))+ (0.09861122 * float(x[59]))+ (0.22728465 * float(x[60]))) + -0.10230655), 0)
    h_4 = max((((0.44538394 * float(x[0]))+ (-0.41555095 * float(x[1]))+ (0.08853802 * float(x[2]))+ (-0.08330985 * float(x[3]))+ (0.120972484 * float(x[4]))+ (0.40046474 * float(x[5]))+ (0.4144469 * float(x[6]))+ (-0.16301915 * float(x[7]))+ (-0.28391343 * float(x[8]))+ (-0.3459257 * float(x[9]))+ (-0.5222288 * float(x[10]))+ (0.0065407394 * float(x[11]))+ (-0.23070337 * float(x[12]))+ (0.0047727246 * float(x[13]))+ (0.3671687 * float(x[14]))+ (0.114531934 * float(x[15]))+ (0.20769274 * float(x[16]))+ (0.1556453 * float(x[17]))+ (0.1181014 * float(x[18]))+ (-0.21560954 * float(x[19]))+ (0.1956203 * float(x[20]))+ (-0.13163947 * float(x[21]))+ (0.06352926 * float(x[22]))+ (-0.28515434 * float(x[23]))+ (0.012753063 * float(x[24]))+ (-0.045493122 * float(x[25]))+ (0.06223284 * float(x[26]))+ (-0.036585025 * float(x[27]))+ (0.16048767 * float(x[28]))+ (-0.25678235 * float(x[29]))+ (0.0007213045 * float(x[30]))+ (-0.15194033 * float(x[31]))+ (0.18775061 * float(x[32]))+ (-0.20595533 * float(x[33]))+ (0.098609254 * float(x[34]))+ (-0.018346157 * float(x[35]))+ (-0.1994783 * float(x[36]))+ (-0.23351792 * float(x[37]))+ (-0.0010622981 * float(x[38]))+ (-0.20280349 * float(x[39]))+ (0.2117078 * float(x[40]))+ (-0.14182472 * float(x[41]))+ (0.08160901 * float(x[42]))+ (-0.24064273 * float(x[43]))+ (0.048396282 * float(x[44]))+ (0.11392511 * float(x[45]))+ (-0.26194 * float(x[46]))+ (-0.15360983 * float(x[47]))+ (-0.13490224 * float(x[48]))+ (-0.3246832 * float(x[49])))+ ((-0.21759105 * float(x[50]))+ (-0.5222975 * float(x[51]))+ (0.09456417 * float(x[52]))+ (-0.19025768 * float(x[53]))+ (0.17231394 * float(x[54]))+ (-0.009457762 * float(x[55]))+ (-0.28716546 * float(x[56]))+ (0.2926939 * float(x[57]))+ (-0.08626923 * float(x[58]))+ (-0.0005042799 * float(x[59]))+ (-0.15354039 * float(x[60]))) + -0.22840321), 0)
    o[0] = (0.71111834 * h_0)+ (1.0206599 * h_1)+ (-0.012981228 * h_2)+ (-0.225484 * h_3)+ (-3.486385 * h_4) + 2.0331283
    o[1] = (-0.16370353 * h_0)+ (1.3135283 * h_1)+ (0.23498467 * h_2)+ (-0.5425425 * h_3)+ (-0.98056686 * h_4) + -9.661571
    o[2] = (-0.4276376 * h_0)+ (-1.8225205 * h_1)+ (0.1674361 * h_2)+ (-0.044976603 * h_3)+ (-0.2626553 * h_4) + 8.279044
    o[3] = (0.5839762 * h_0)+ (0.7088243 * h_1)+ (-0.62528956 * h_2)+ (0.3835761 * h_3)+ (-0.44680625 * h_4) + -3.3348472
    o[4] = (0.43278393 * h_0)+ (-1.1273282 * h_1)+ (0.59359384 * h_2)+ (0.4316907 * h_3)+ (0.13297367 * h_4) + -1.7787971
    o[5] = (0.03807363 * h_0)+ (-0.20108132 * h_1)+ (0.26485035 * h_2)+ (-0.20265236 * h_3)+ (-0.027589748 * h_4) + -3.4856453

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
        model_cap=346
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

