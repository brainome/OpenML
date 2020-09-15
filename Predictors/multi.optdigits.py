#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class optdigits.csv -o optdigits_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:27:20.73. Finished on: Sep-04-2020 12:02:01.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         10-way classifier
Best-guess accuracy:                 10.18%
Overall Model accuracy:              100.00% (5620/5620 correct)
Overall Improvement over best guess: 89.82% (of possible 89.82%)
Model capacity (MEC):                685 bits
Generalization ratio:                8.20 bits/bit
Model efficiency:                    0.13%/parameter
Confusion Matrix:
 [9.86% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 10.16% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 9.91% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 10.18% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 10.11% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 9.93% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 9.93% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 10.07% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 9.86% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 10.00%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
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
TRAINFILE = "optdigits.csv"


#Number of output logits
num_output_logits = 10

#Number of attributes
num_attr = 64
n_classes = 10


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
    h_0 = max((((-0.05788495 * float(x[0]))+ (0.43966097 * float(x[1]))+ (-0.18978727 * float(x[2]))+ (0.119735524 * float(x[3]))+ (0.4610541 * float(x[4]))+ (0.21506135 * float(x[5]))+ (-0.014016901 * float(x[6]))+ (-0.02300198 * float(x[7]))+ (-1.5815963 * float(x[8]))+ (0.4538578 * float(x[9]))+ (0.08645676 * float(x[10]))+ (0.09245887 * float(x[11]))+ (0.6662256 * float(x[12]))+ (0.28276598 * float(x[13]))+ (-0.1694762 * float(x[14]))+ (0.27921095 * float(x[15]))+ (-1.1670731 * float(x[16]))+ (0.075144924 * float(x[17]))+ (-0.91919166 * float(x[18]))+ (-0.5387028 * float(x[19]))+ (0.26698115 * float(x[20]))+ (0.4703196 * float(x[21]))+ (0.94897175 * float(x[22]))+ (-0.06824306 * float(x[23]))+ (-0.1899944 * float(x[24]))+ (-0.6049681 * float(x[25]))+ (-0.29969764 * float(x[26]))+ (-0.31991786 * float(x[27]))+ (0.25806537 * float(x[28]))+ (0.1522694 * float(x[29]))+ (-0.4444924 * float(x[30]))+ (-0.7083374 * float(x[31]))+ (0.15116628 * float(x[32]))+ (-0.34406558 * float(x[33]))+ (-0.12604798 * float(x[34]))+ (0.09544468 * float(x[35]))+ (0.25245458 * float(x[36]))+ (0.15964358 * float(x[37]))+ (0.11611851 * float(x[38]))+ (-0.15969272 * float(x[39]))+ (-0.34816396 * float(x[40]))+ (-0.88769305 * float(x[41]))+ (0.1831776 * float(x[42]))+ (0.051109053 * float(x[43]))+ (0.44347855 * float(x[44]))+ (-0.12019694 * float(x[45]))+ (0.43474656 * float(x[46]))+ (-1.0848819 * float(x[47]))+ (-1.0326306 * float(x[48]))+ (-0.30996883 * float(x[49])))+ ((-0.20319256 * float(x[50]))+ (-0.2581277 * float(x[51]))+ (0.026796117 * float(x[52]))+ (0.19813718 * float(x[53]))+ (0.4270553 * float(x[54]))+ (0.59564775 * float(x[55]))+ (0.37247446 * float(x[56]))+ (0.031822607 * float(x[57]))+ (0.0063046613 * float(x[58]))+ (-0.33770952 * float(x[59]))+ (-0.42973775 * float(x[60]))+ (-0.2867865 * float(x[61]))+ (-0.16059396 * float(x[62]))+ (0.609192 * float(x[63]))) + 0.59489816), 0)
    h_1 = max((((0.17756505 * float(x[0]))+ (0.71856564 * float(x[1]))+ (0.0057048486 * float(x[2]))+ (0.4435954 * float(x[3]))+ (0.3316572 * float(x[4]))+ (0.54941386 * float(x[5]))+ (0.31441522 * float(x[6]))+ (0.026585465 * float(x[7]))+ (1.7432479 * float(x[8]))+ (-0.0030622382 * float(x[9]))+ (0.17533843 * float(x[10]))+ (0.6280317 * float(x[11]))+ (0.36802945 * float(x[12]))+ (0.25803187 * float(x[13]))+ (-0.39600897 * float(x[14]))+ (0.1931125 * float(x[15]))+ (-1.1516591 * float(x[16]))+ (-0.08138419 * float(x[17]))+ (-0.11413482 * float(x[18]))+ (0.34156507 * float(x[19]))+ (-0.014359609 * float(x[20]))+ (0.10464383 * float(x[21]))+ (-0.69932705 * float(x[22]))+ (0.4431705 * float(x[23]))+ (-2.6506765 * float(x[24]))+ (-0.19427171 * float(x[25]))+ (0.29097834 * float(x[26]))+ (-0.14031346 * float(x[27]))+ (0.053358205 * float(x[28]))+ (0.22552907 * float(x[29]))+ (-0.13071814 * float(x[30]))+ (-0.9895516 * float(x[31]))+ (0.21517245 * float(x[32]))+ (0.73003244 * float(x[33]))+ (0.19656745 * float(x[34]))+ (0.33261645 * float(x[35]))+ (-0.15675524 * float(x[36]))+ (0.028763102 * float(x[37]))+ (0.5702753 * float(x[38]))+ (-0.06795021 * float(x[39]))+ (-0.24900983 * float(x[40]))+ (-0.8784126 * float(x[41]))+ (0.07280853 * float(x[42]))+ (-0.5039021 * float(x[43]))+ (-0.05399233 * float(x[44]))+ (0.3993701 * float(x[45]))+ (0.5069846 * float(x[46]))+ (1.2925516 * float(x[47]))+ (-0.31310615 * float(x[48]))+ (-0.69386667 * float(x[49])))+ ((0.31513935 * float(x[50]))+ (-0.14447387 * float(x[51]))+ (0.08331204 * float(x[52]))+ (-0.1387436 * float(x[53]))+ (-0.049907617 * float(x[54]))+ (0.15258767 * float(x[55]))+ (-0.5689134 * float(x[56]))+ (-0.15113057 * float(x[57]))+ (-0.08750878 * float(x[58]))+ (0.37357885 * float(x[59]))+ (0.23700617 * float(x[60]))+ (-0.18629016 * float(x[61]))+ (-0.5343721 * float(x[62]))+ (-0.7191979 * float(x[63]))) + 1.6834917), 0)
    h_2 = max((((0.0036874115 * float(x[0]))+ (0.5293112 * float(x[1]))+ (0.03909645 * float(x[2]))+ (-0.086141504 * float(x[3]))+ (0.00030757324 * float(x[4]))+ (0.20996512 * float(x[5]))+ (-0.28182313 * float(x[6]))+ (-0.92879105 * float(x[7]))+ (-0.1515256 * float(x[8]))+ (0.10519723 * float(x[9]))+ (0.005839189 * float(x[10]))+ (-0.020685839 * float(x[11]))+ (0.38032284 * float(x[12]))+ (0.1472402 * float(x[13]))+ (-0.3047177 * float(x[14]))+ (-1.2998061 * float(x[15]))+ (-0.03012117 * float(x[16]))+ (0.10337969 * float(x[17]))+ (-0.18572113 * float(x[18]))+ (-0.05826 * float(x[19]))+ (0.26787895 * float(x[20]))+ (0.12161126 * float(x[21]))+ (0.5233559 * float(x[22]))+ (-1.9801241 * float(x[23]))+ (-2.4588692 * float(x[24]))+ (0.08487844 * float(x[25]))+ (-0.5362697 * float(x[26]))+ (-0.47019657 * float(x[27]))+ (-0.2536116 * float(x[28]))+ (0.1907163 * float(x[29]))+ (-0.43044356 * float(x[30]))+ (-0.14258057 * float(x[31]))+ (0.22191821 * float(x[32]))+ (-0.29545334 * float(x[33]))+ (-0.1367134 * float(x[34]))+ (-0.50654763 * float(x[35]))+ (-0.014420254 * float(x[36]))+ (0.04704617 * float(x[37]))+ (-0.83405906 * float(x[38]))+ (0.034205325 * float(x[39]))+ (-0.3676669 * float(x[40]))+ (-0.5188995 * float(x[41]))+ (0.70971686 * float(x[42]))+ (0.40671945 * float(x[43]))+ (-0.58309335 * float(x[44]))+ (0.052170847 * float(x[45]))+ (0.027214075 * float(x[46]))+ (0.08665535 * float(x[47]))+ (-1.5169169 * float(x[48]))+ (0.2075998 * float(x[49])))+ ((0.39125445 * float(x[50]))+ (0.33924183 * float(x[51]))+ (0.5754691 * float(x[52]))+ (0.80817604 * float(x[53]))+ (-0.07389656 * float(x[54]))+ (0.7485733 * float(x[55]))+ (0.30123603 * float(x[56]))+ (0.19184443 * float(x[57]))+ (0.081784 * float(x[58]))+ (0.69662684 * float(x[59]))+ (0.61272675 * float(x[60]))+ (0.69455653 * float(x[61]))+ (-0.036001537 * float(x[62]))+ (0.18694025 * float(x[63]))) + 0.5660584), 0)
    h_3 = max((((0.24346119 * float(x[0]))+ (-0.6695494 * float(x[1]))+ (0.04816689 * float(x[2]))+ (0.16154923 * float(x[3]))+ (-0.13243163 * float(x[4]))+ (0.3929696 * float(x[5]))+ (0.19376816 * float(x[6]))+ (1.2014138 * float(x[7]))+ (-1.2854667 * float(x[8]))+ (-0.14972453 * float(x[9]))+ (-0.32500368 * float(x[10]))+ (-0.0032963252 * float(x[11]))+ (-0.21789688 * float(x[12]))+ (-0.13198367 * float(x[13]))+ (-0.7220486 * float(x[14]))+ (0.69679344 * float(x[15]))+ (2.0092704 * float(x[16]))+ (-0.5333859 * float(x[17]))+ (-0.1279495 * float(x[18]))+ (0.4409394 * float(x[19]))+ (-0.16147272 * float(x[20]))+ (-0.93865764 * float(x[21]))+ (-0.8387007 * float(x[22]))+ (-0.7531346 * float(x[23]))+ (-1.3515433 * float(x[24]))+ (0.30309042 * float(x[25]))+ (0.1729273 * float(x[26]))+ (0.27666122 * float(x[27]))+ (0.20684882 * float(x[28]))+ (-0.06911327 * float(x[29]))+ (-0.70271736 * float(x[30]))+ (0.01892348 * float(x[31]))+ (-0.08365011 * float(x[32]))+ (0.7889153 * float(x[33]))+ (0.039125536 * float(x[34]))+ (0.21587096 * float(x[35]))+ (0.18247749 * float(x[36]))+ (0.07039006 * float(x[37]))+ (0.2112706 * float(x[38]))+ (-0.17075193 * float(x[39]))+ (-0.1316057 * float(x[40]))+ (-0.5669164 * float(x[41]))+ (0.566455 * float(x[42]))+ (0.48719266 * float(x[43]))+ (-0.05255088 * float(x[44]))+ (-0.1600211 * float(x[45]))+ (0.20615104 * float(x[46]))+ (1.0162749 * float(x[47]))+ (-2.1751459 * float(x[48]))+ (-0.4899978 * float(x[49])))+ ((0.18518353 * float(x[50]))+ (0.1394159 * float(x[51]))+ (0.31613037 * float(x[52]))+ (-0.14645685 * float(x[53]))+ (0.022072252 * float(x[54]))+ (0.122697376 * float(x[55]))+ (-0.36383566 * float(x[56]))+ (0.5891329 * float(x[57]))+ (-0.37879983 * float(x[58]))+ (0.15540722 * float(x[59]))+ (0.08625349 * float(x[60]))+ (0.008324034 * float(x[61]))+ (-0.057212573 * float(x[62]))+ (0.40352917 * float(x[63]))) + -0.6357012), 0)
    h_4 = max((((-0.23739678 * float(x[0]))+ (-0.2576605 * float(x[1]))+ (0.10236244 * float(x[2]))+ (-0.1513352 * float(x[3]))+ (-0.2925658 * float(x[4]))+ (0.065015204 * float(x[5]))+ (0.0716687 * float(x[6]))+ (0.7163472 * float(x[7]))+ (-1.1351646 * float(x[8]))+ (0.22740196 * float(x[9]))+ (-0.3828706 * float(x[10]))+ (0.14670438 * float(x[11]))+ (0.22025523 * float(x[12]))+ (-0.088476606 * float(x[13]))+ (-0.030150717 * float(x[14]))+ (-0.407902 * float(x[15]))+ (1.4799434 * float(x[16]))+ (0.057546068 * float(x[17]))+ (0.22618271 * float(x[18]))+ (0.23668995 * float(x[19]))+ (0.28487167 * float(x[20]))+ (0.14037783 * float(x[21]))+ (-0.40381932 * float(x[22]))+ (-0.55111045 * float(x[23]))+ (3.0817685 * float(x[24]))+ (-0.10241336 * float(x[25]))+ (0.29611737 * float(x[26]))+ (-0.13056222 * float(x[27]))+ (0.046786223 * float(x[28]))+ (-0.022113008 * float(x[29]))+ (0.8479753 * float(x[30]))+ (1.0754938 * float(x[31]))+ (0.18294735 * float(x[32]))+ (0.34266987 * float(x[33]))+ (0.3835257 * float(x[34]))+ (-0.47822636 * float(x[35]))+ (0.10921115 * float(x[36]))+ (0.33471832 * float(x[37]))+ (0.803854 * float(x[38]))+ (0.038292635 * float(x[39]))+ (0.68864083 * float(x[40]))+ (0.65387857 * float(x[41]))+ (-0.037821602 * float(x[42]))+ (0.5666191 * float(x[43]))+ (0.2752045 * float(x[44]))+ (-0.23402913 * float(x[45]))+ (-0.14821981 * float(x[46]))+ (-1.0197215 * float(x[47]))+ (1.9224132 * float(x[48]))+ (0.5993261 * float(x[49])))+ ((-0.31659746 * float(x[50]))+ (0.1328481 * float(x[51]))+ (0.46286005 * float(x[52]))+ (-0.14772367 * float(x[53]))+ (0.069667116 * float(x[54]))+ (-0.43445653 * float(x[55]))+ (-0.13122496 * float(x[56]))+ (1.3539027 * float(x[57]))+ (-0.29649955 * float(x[58]))+ (-0.35818666 * float(x[59]))+ (-0.37341562 * float(x[60]))+ (0.1562021 * float(x[61]))+ (-0.0011908751 * float(x[62]))+ (0.53368044 * float(x[63]))) + 0.33888122), 0)
    h_5 = max((((-0.0787193 * float(x[0]))+ (-0.23188905 * float(x[1]))+ (-0.004309778 * float(x[2]))+ (0.42402118 * float(x[3]))+ (-0.14235781 * float(x[4]))+ (-0.03184747 * float(x[5]))+ (-0.2969173 * float(x[6]))+ (0.08608961 * float(x[7]))+ (0.62706065 * float(x[8]))+ (0.44434747 * float(x[9]))+ (0.011687914 * float(x[10]))+ (0.085139416 * float(x[11]))+ (0.22889842 * float(x[12]))+ (0.08339538 * float(x[13]))+ (-0.23507892 * float(x[14]))+ (-0.73511475 * float(x[15]))+ (-1.4804102 * float(x[16]))+ (-0.4812294 * float(x[17]))+ (-0.023114886 * float(x[18]))+ (0.030384433 * float(x[19]))+ (0.09491526 * float(x[20]))+ (0.33122596 * float(x[21]))+ (0.3835404 * float(x[22]))+ (0.6328864 * float(x[23]))+ (1.3629109 * float(x[24]))+ (0.147546 * float(x[25]))+ (0.18582301 * float(x[26]))+ (-0.18904415 * float(x[27]))+ (-0.08869284 * float(x[28]))+ (0.560224 * float(x[29]))+ (0.8580463 * float(x[30]))+ (-0.75140625 * float(x[31]))+ (0.18401632 * float(x[32]))+ (0.31409147 * float(x[33]))+ (0.07340416 * float(x[34]))+ (0.20488532 * float(x[35]))+ (0.12085111 * float(x[36]))+ (0.28146213 * float(x[37]))+ (-0.08423588 * float(x[38]))+ (0.07148241 * float(x[39]))+ (-0.034192935 * float(x[40]))+ (0.3871249 * float(x[41]))+ (0.70039314 * float(x[42]))+ (0.369861 * float(x[43]))+ (-0.092665 * float(x[44]))+ (-0.29735583 * float(x[45]))+ (-0.12484775 * float(x[46]))+ (0.2867116 * float(x[47]))+ (-0.39564362 * float(x[48]))+ (-0.56059474 * float(x[49])))+ ((0.035321575 * float(x[50]))+ (0.096925445 * float(x[51]))+ (-0.27522385 * float(x[52]))+ (-0.23638187 * float(x[53]))+ (-0.0649992 * float(x[54]))+ (0.29288107 * float(x[55]))+ (0.25966564 * float(x[56]))+ (-0.80405384 * float(x[57]))+ (-0.29420742 * float(x[58]))+ (-0.24472706 * float(x[59]))+ (-0.21568626 * float(x[60]))+ (-0.06117085 * float(x[61]))+ (-0.04354546 * float(x[62]))+ (0.56838554 * float(x[63]))) + 0.77458245), 0)
    h_6 = max((((-0.22382328 * float(x[0]))+ (-0.24525675 * float(x[1]))+ (-0.3030559 * float(x[2]))+ (-0.025705207 * float(x[3]))+ (-0.19914508 * float(x[4]))+ (-0.26004428 * float(x[5]))+ (-1.0853698 * float(x[6]))+ (-0.5328859 * float(x[7]))+ (1.2099276 * float(x[8]))+ (-0.29185957 * float(x[9]))+ (-0.37371457 * float(x[10]))+ (-0.5602312 * float(x[11]))+ (-0.34783325 * float(x[12]))+ (0.028887324 * float(x[13]))+ (0.37531164 * float(x[14]))+ (0.6745555 * float(x[15]))+ (-1.0359317 * float(x[16]))+ (0.49802786 * float(x[17]))+ (0.24843709 * float(x[18]))+ (0.4825578 * float(x[19]))+ (0.2698295 * float(x[20]))+ (0.33391517 * float(x[21]))+ (0.21110328 * float(x[22]))+ (0.24635169 * float(x[23]))+ (1.8393595 * float(x[24]))+ (0.3079205 * float(x[25]))+ (-0.10727662 * float(x[26]))+ (0.66234076 * float(x[27]))+ (-0.07582084 * float(x[28]))+ (0.16282897 * float(x[29]))+ (0.08284181 * float(x[30]))+ (0.9957658 * float(x[31]))+ (0.012062981 * float(x[32]))+ (-0.30593878 * float(x[33]))+ (0.35369977 * float(x[34]))+ (0.11129013 * float(x[35]))+ (0.3840568 * float(x[36]))+ (0.099095 * float(x[37]))+ (-0.20689371 * float(x[38]))+ (-0.19287291 * float(x[39]))+ (0.12081926 * float(x[40]))+ (0.7235062 * float(x[41]))+ (0.7357959 * float(x[42]))+ (0.5549396 * float(x[43]))+ (0.33262232 * float(x[44]))+ (0.12413938 * float(x[45]))+ (-0.16729783 * float(x[46]))+ (-2.3633397 * float(x[47]))+ (1.5502731 * float(x[48]))+ (0.5556102 * float(x[49])))+ ((0.09268025 * float(x[50]))+ (-0.23409483 * float(x[51]))+ (0.10059859 * float(x[52]))+ (0.11972157 * float(x[53]))+ (0.36226073 * float(x[54]))+ (-0.063870154 * float(x[55]))+ (-0.68571854 * float(x[56]))+ (-0.51972455 * float(x[57]))+ (-0.28029007 * float(x[58]))+ (-0.063790396 * float(x[59]))+ (0.19023535 * float(x[60]))+ (0.019174738 * float(x[61]))+ (-0.08788079 * float(x[62]))+ (1.3980392 * float(x[63]))) + 0.7658377), 0)
    h_7 = max((((-0.26073182 * float(x[0]))+ (-0.75446427 * float(x[1]))+ (-0.1532407 * float(x[2]))+ (0.35526946 * float(x[3]))+ (-0.08611779 * float(x[4]))+ (-0.021233454 * float(x[5]))+ (-0.0062835384 * float(x[6]))+ (-0.9339486 * float(x[7]))+ (-1.2799685 * float(x[8]))+ (0.6486173 * float(x[9]))+ (-0.12785093 * float(x[10]))+ (-0.24861744 * float(x[11]))+ (0.1307319 * float(x[12]))+ (0.20948672 * float(x[13]))+ (0.06832506 * float(x[14]))+ (0.17063631 * float(x[15]))+ (-1.5367287 * float(x[16]))+ (0.32810333 * float(x[17]))+ (-0.17856593 * float(x[18]))+ (0.28688332 * float(x[19]))+ (0.631669 * float(x[20]))+ (0.6028293 * float(x[21]))+ (0.6681011 * float(x[22]))+ (0.93747675 * float(x[23]))+ (-0.9537451 * float(x[24]))+ (-0.34975305 * float(x[25]))+ (-0.17675343 * float(x[26]))+ (-0.06932374 * float(x[27]))+ (0.3408838 * float(x[28]))+ (0.29031897 * float(x[29]))+ (-0.032526273 * float(x[30]))+ (-1.4218242 * float(x[31]))+ (0.046831653 * float(x[32]))+ (-0.53484875 * float(x[33]))+ (0.10479832 * float(x[34]))+ (-0.24342237 * float(x[35]))+ (0.15252809 * float(x[36]))+ (0.60355353 * float(x[37]))+ (-0.11710677 * float(x[38]))+ (0.2389811 * float(x[39]))+ (0.06487225 * float(x[40]))+ (-0.103177905 * float(x[41]))+ (-0.6961424 * float(x[42]))+ (-0.52237284 * float(x[43]))+ (-0.035792734 * float(x[44]))+ (-0.12383828 * float(x[45]))+ (-0.44699723 * float(x[46]))+ (-1.2808093 * float(x[47]))+ (-0.291173 * float(x[48]))+ (0.57511955 * float(x[49])))+ ((0.071585596 * float(x[50]))+ (-0.26870862 * float(x[51]))+ (0.039028317 * float(x[52]))+ (0.012396716 * float(x[53]))+ (-0.009970949 * float(x[54]))+ (0.21130139 * float(x[55]))+ (-0.3365187 * float(x[56]))+ (0.36287934 * float(x[57]))+ (0.007481934 * float(x[58]))+ (0.11566151 * float(x[59]))+ (0.032482345 * float(x[60]))+ (0.2273405 * float(x[61]))+ (-0.12373569 * float(x[62]))+ (-0.25562355 * float(x[63]))) + -0.9808635), 0)
    h_8 = max((((0.02289675 * float(x[0]))+ (-0.11498413 * float(x[1]))+ (-0.11021209 * float(x[2]))+ (-0.45591602 * float(x[3]))+ (-0.066657595 * float(x[4]))+ (-0.28212258 * float(x[5]))+ (-1.2997181 * float(x[6]))+ (0.6518352 * float(x[7]))+ (1.0671499 * float(x[8]))+ (-0.4369566 * float(x[9]))+ (-0.27619427 * float(x[10]))+ (-0.09102347 * float(x[11]))+ (-0.058786776 * float(x[12]))+ (-0.4552574 * float(x[13]))+ (0.60933995 * float(x[14]))+ (-0.30797333 * float(x[15]))+ (0.0571284 * float(x[16]))+ (-0.7656716 * float(x[17]))+ (-0.39524266 * float(x[18]))+ (0.0027754882 * float(x[19]))+ (-0.1901032 * float(x[20]))+ (-0.3905252 * float(x[21]))+ (0.1993077 * float(x[22]))+ (1.120878 * float(x[23]))+ (-0.21706848 * float(x[24]))+ (0.66373837 * float(x[25]))+ (0.12945971 * float(x[26]))+ (-0.27423516 * float(x[27]))+ (-0.14846401 * float(x[28]))+ (-0.46583286 * float(x[29]))+ (-0.18858404 * float(x[30]))+ (0.7008963 * float(x[31]))+ (0.16024822 * float(x[32]))+ (0.4772018 * float(x[33]))+ (-0.08238218 * float(x[34]))+ (0.03479341 * float(x[35]))+ (-0.19325821 * float(x[36]))+ (0.46790433 * float(x[37]))+ (0.49439496 * float(x[38]))+ (-0.0145847285 * float(x[39]))+ (-0.12693802 * float(x[40]))+ (0.5276601 * float(x[41]))+ (0.69415826 * float(x[42]))+ (-0.22429027 * float(x[43]))+ (0.18389761 * float(x[44]))+ (0.61274844 * float(x[45]))+ (0.46266514 * float(x[46]))+ (1.6147349 * float(x[47]))+ (-1.0288802 * float(x[48]))+ (-0.581096 * float(x[49])))+ ((0.3024385 * float(x[50]))+ (0.48391196 * float(x[51]))+ (0.46849248 * float(x[52]))+ (0.23245324 * float(x[53]))+ (-0.31077504 * float(x[54]))+ (-0.30476108 * float(x[55]))+ (0.050788004 * float(x[56]))+ (0.8530584 * float(x[57]))+ (-0.09193158 * float(x[58]))+ (0.09503667 * float(x[59]))+ (0.18251772 * float(x[60]))+ (0.42240644 * float(x[61]))+ (0.035072904 * float(x[62]))+ (-0.69963855 * float(x[63]))) + 0.56109154), 0)
    o[0] = (-0.55077124 * h_0)+ (0.11564779 * h_1)+ (0.3296978 * h_2)+ (-1.3720556 * h_3)+ (0.09436417 * h_4)+ (0.3325775 * h_5)+ (-0.056786634 * h_6)+ (-1.3468438 * h_7)+ (0.054460205 * h_8) + -2.017136
    o[1] = (0.3218587 * h_0)+ (-0.5019281 * h_1)+ (-0.041801494 * h_2)+ (0.69359857 * h_3)+ (0.28823912 * h_4)+ (-0.23508056 * h_5)+ (0.51771677 * h_6)+ (0.4562602 * h_7)+ (-1.1363637 * h_8) + -1.1898141
    o[2] = (0.1611223 * h_0)+ (-0.81942046 * h_1)+ (1.0890225 * h_2)+ (-0.17098986 * h_3)+ (0.48411644 * h_4)+ (-0.049803104 * h_5)+ (-0.25535476 * h_6)+ (-0.6007243 * h_7)+ (-1.0558705 * h_8) + 1.4185649
    o[3] = (1.0662854 * h_0)+ (0.16505532 * h_1)+ (-0.09909981 * h_2)+ (-0.50648713 * h_3)+ (-1.3472806 * h_4)+ (-1.7794603 * h_5)+ (0.04613147 * h_6)+ (-0.014071425 * h_7)+ (0.6066205 * h_8) + 1.5548269
    o[4] = (-1.0042309 * h_0)+ (-0.65122545 * h_1)+ (-1.1118389 * h_2)+ (-0.33793238 * h_3)+ (1.0893338 * h_4)+ (0.031093394 * h_5)+ (0.8385203 * h_6)+ (0.14048736 * h_7)+ (0.48896173 * h_8) + 1.7775307
    o[5] = (-1.1240798 * h_0)+ (0.7342903 * h_1)+ (-0.0503076 * h_2)+ (-0.18604264 * h_3)+ (0.4178892 * h_4)+ (-0.7277428 * h_5)+ (-0.9008357 * h_6)+ (-0.63759255 * h_7)+ (-0.9009362 * h_8) + 1.3598765
    o[6] = (0.61966056 * h_0)+ (-0.57408917 * h_1)+ (-0.0660806 * h_2)+ (0.33110029 * h_3)+ (-0.36748642 * h_4)+ (0.44820464 * h_5)+ (0.16017008 * h_6)+ (-2.641902 * h_7)+ (0.55879295 * h_8) + 0.98085785
    o[7] = (1.1729008 * h_0)+ (-0.14954795 * h_1)+ (-0.6438452 * h_2)+ (-0.20948215 * h_3)+ (0.40216416 * h_4)+ (0.7653108 * h_5)+ (-0.71849895 * h_6)+ (-0.5941991 * h_7)+ (-0.019884292 * h_8) + 1.2796029
    o[8] = (0.16094047 * h_0)+ (-0.03361831 * h_1)+ (-0.015725408 * h_2)+ (-0.4206301 * h_3)+ (-0.5216192 * h_4)+ (-0.10549004 * h_5)+ (0.8652081 * h_6)+ (-0.5022139 * h_7)+ (-0.9764712 * h_8) + 2.4933708
    o[9] = (-0.21147233 * h_0)+ (0.07924062 * h_1)+ (-0.44786853 * h_2)+ (-0.26028088 * h_3)+ (-0.46261883 * h_4)+ (0.3078085 * h_5)+ (0.0040729474 * h_6)+ (0.8182437 * h_7)+ (-0.8283481 * h_8) + 1.1076397

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
        model_cap=685
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


