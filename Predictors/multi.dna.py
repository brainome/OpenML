#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target class dna.csv -o dna_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:14:29.41. Finished on: Sep-04-2020 10:45:30.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         3-way classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 51.91%
Overall Model accuracy:              97.99% (3122/3186 correct)
Overall Improvement over best guess: 46.08% (of possible 48.09%)
Model capacity (MEC):                1659 bits
Generalization ratio:                1.88 bits/bit
Model efficiency:                    0.02%/parameter
Confusion Matrix:
 [51.00% 0.35% 0.56%]
 [0.35% 23.45% 0.28%]
 [0.28% 0.19% 23.54%]
Overfitting:                         No
Note: Labels have been remapped to '3'=0, '1'=1, '2'=2.
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
TRAINFILE = "dna.csv"


#Number of output logits
num_output_logits = 3

#Number of attributes
num_attr = 180
n_classes = 3


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
    clean.mapping={'3': 0, '1': 1, '2': 2}

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
    h_0 = max((((0.079912096 * float(x[0]))+ (0.04719547 * float(x[1]))+ (-0.10658926 * float(x[2]))+ (0.06666761 * float(x[3]))+ (0.121887304 * float(x[4]))+ (0.124535404 * float(x[5]))+ (0.05579227 * float(x[6]))+ (0.16969249 * float(x[7]))+ (0.043025676 * float(x[8]))+ (0.21404858 * float(x[9]))+ (-0.17960949 * float(x[10]))+ (-0.16736881 * float(x[11]))+ (0.019017361 * float(x[12]))+ (0.060501546 * float(x[13]))+ (-0.013025131 * float(x[14]))+ (-0.25750738 * float(x[15]))+ (0.13050139 * float(x[16]))+ (-0.024656931 * float(x[17]))+ (0.0526538 * float(x[18]))+ (0.07940339 * float(x[19]))+ (0.120639265 * float(x[20]))+ (0.037560374 * float(x[21]))+ (0.05927018 * float(x[22]))+ (-0.121827476 * float(x[23]))+ (0.008155044 * float(x[24]))+ (0.15161648 * float(x[25]))+ (-0.12971476 * float(x[26]))+ (-0.09026833 * float(x[27]))+ (0.29687092 * float(x[28]))+ (0.106913835 * float(x[29]))+ (-0.093001224 * float(x[30]))+ (0.027059773 * float(x[31]))+ (0.12876911 * float(x[32]))+ (0.1188448 * float(x[33]))+ (-0.09622709 * float(x[34]))+ (-0.18047303 * float(x[35]))+ (0.1741735 * float(x[36]))+ (0.18554588 * float(x[37]))+ (-0.060460195 * float(x[38]))+ (-0.05176976 * float(x[39]))+ (0.14860368 * float(x[40]))+ (0.18268937 * float(x[41]))+ (0.026074262 * float(x[42]))+ (0.13918224 * float(x[43]))+ (0.24295741 * float(x[44]))+ (-0.19293417 * float(x[45]))+ (0.05289107 * float(x[46]))+ (-0.025817174 * float(x[47]))+ (0.05804466 * float(x[48]))+ (0.1011742 * float(x[49])))+ ((-0.14184529 * float(x[50]))+ (0.16495214 * float(x[51]))+ (0.030562192 * float(x[52]))+ (0.14522447 * float(x[53]))+ (-0.089080624 * float(x[54]))+ (-0.11462578 * float(x[55]))+ (0.08495803 * float(x[56]))+ (0.24905254 * float(x[57]))+ (0.16234581 * float(x[58]))+ (-0.031215934 * float(x[59]))+ (0.24597658 * float(x[60]))+ (0.18370584 * float(x[61]))+ (-0.07989688 * float(x[62]))+ (-0.15087043 * float(x[63]))+ (0.0015489999 * float(x[64]))+ (0.085554354 * float(x[65]))+ (-0.0750849 * float(x[66]))+ (0.1540395 * float(x[67]))+ (0.09123571 * float(x[68]))+ (-0.10541116 * float(x[69]))+ (0.055113822 * float(x[70]))+ (-0.076490924 * float(x[71]))+ (0.07826993 * float(x[72]))+ (0.22056746 * float(x[73]))+ (0.0323256 * float(x[74]))+ (-0.10401266 * float(x[75]))+ (0.10662183 * float(x[76]))+ (0.052566674 * float(x[77]))+ (0.05859468 * float(x[78]))+ (0.13222992 * float(x[79]))+ (0.17633621 * float(x[80]))+ (-0.025984827 * float(x[81]))+ (-0.24892677 * float(x[82]))+ (0.2862138 * float(x[83]))+ (-0.2936075 * float(x[84]))+ (0.0972217 * float(x[85]))+ (0.25502962 * float(x[86]))+ (0.18872955 * float(x[87]))+ (0.1994889 * float(x[88]))+ (-0.4049175 * float(x[89]))+ (0.45241132 * float(x[90]))+ (0.1958024 * float(x[91]))+ (-0.24730538 * float(x[92]))+ (0.7214157 * float(x[93]))+ (0.52348024 * float(x[94]))+ (0.60405487 * float(x[95]))+ (-0.17624183 * float(x[96]))+ (0.20966676 * float(x[97]))+ (-0.025659226 * float(x[98]))+ (-0.15658912 * float(x[99])))+ ((-0.0750363 * float(x[100]))+ (0.38648316 * float(x[101]))+ (0.30141088 * float(x[102]))+ (0.17612928 * float(x[103]))+ (-0.5130019 * float(x[104]))+ (0.26165983 * float(x[105]))+ (0.33653843 * float(x[106]))+ (0.31108296 * float(x[107]))+ (-0.038891688 * float(x[108]))+ (0.039840013 * float(x[109]))+ (-0.15836632 * float(x[110]))+ (0.2112746 * float(x[111]))+ (-0.091384396 * float(x[112]))+ (-0.011203899 * float(x[113]))+ (0.13306569 * float(x[114]))+ (-0.09137627 * float(x[115]))+ (-0.1254848 * float(x[116]))+ (0.08240769 * float(x[117]))+ (-0.043621678 * float(x[118]))+ (-0.017468117 * float(x[119]))+ (-0.0864945 * float(x[120]))+ (0.071260616 * float(x[121]))+ (-0.17685294 * float(x[122]))+ (0.20999454 * float(x[123]))+ (-0.019135952 * float(x[124]))+ (-0.035516076 * float(x[125]))+ (0.102527246 * float(x[126]))+ (0.16072148 * float(x[127]))+ (0.0015023912 * float(x[128]))+ (0.1600607 * float(x[129]))+ (-0.013486995 * float(x[130]))+ (-0.15256967 * float(x[131]))+ (0.114213824 * float(x[132]))+ (0.027870031 * float(x[133]))+ (0.015824374 * float(x[134]))+ (0.11888066 * float(x[135]))+ (0.15940207 * float(x[136]))+ (0.06492599 * float(x[137]))+ (0.11555024 * float(x[138]))+ (0.121914886 * float(x[139]))+ (0.05540769 * float(x[140]))+ (0.025418457 * float(x[141]))+ (0.22343723 * float(x[142]))+ (-0.008945764 * float(x[143]))+ (-0.002306179 * float(x[144]))+ (-0.02816742 * float(x[145]))+ (0.19021851 * float(x[146]))+ (0.043448824 * float(x[147]))+ (0.0554376 * float(x[148]))+ (0.102754116 * float(x[149])))+ ((0.115678646 * float(x[150]))+ (0.22116767 * float(x[151]))+ (-0.08993103 * float(x[152]))+ (-0.03202047 * float(x[153]))+ (0.05637125 * float(x[154]))+ (-0.09692515 * float(x[155]))+ (0.12862858 * float(x[156]))+ (-0.009349857 * float(x[157]))+ (0.10219688 * float(x[158]))+ (0.19003132 * float(x[159]))+ (0.2112205 * float(x[160]))+ (-0.091385424 * float(x[161]))+ (0.18609856 * float(x[162]))+ (0.029669717 * float(x[163]))+ (0.16747667 * float(x[164]))+ (0.12455799 * float(x[165]))+ (-0.03177075 * float(x[166]))+ (-0.11206898 * float(x[167]))+ (0.18367463 * float(x[168]))+ (-0.100865334 * float(x[169]))+ (0.13685282 * float(x[170]))+ (0.20843601 * float(x[171]))+ (-0.14834468 * float(x[172]))+ (-0.120155305 * float(x[173]))+ (0.06877416 * float(x[174]))+ (-0.063007765 * float(x[175]))+ (-0.023476161 * float(x[176]))+ (0.23642667 * float(x[177]))+ (0.22062853 * float(x[178]))+ (0.0481746 * float(x[179]))) + 1.6730355), 0)
    h_1 = max((((0.054143842 * float(x[0]))+ (-0.11010358 * float(x[1]))+ (0.093405925 * float(x[2]))+ (0.1330539 * float(x[3]))+ (-0.09630688 * float(x[4]))+ (-0.064549096 * float(x[5]))+ (-0.0020903638 * float(x[6]))+ (0.05696651 * float(x[7]))+ (0.22920366 * float(x[8]))+ (0.21881123 * float(x[9]))+ (0.06972929 * float(x[10]))+ (0.17410533 * float(x[11]))+ (0.16395546 * float(x[12]))+ (-0.13288493 * float(x[13]))+ (0.09590839 * float(x[14]))+ (0.052810684 * float(x[15]))+ (0.17376538 * float(x[16]))+ (0.024481952 * float(x[17]))+ (0.007411305 * float(x[18]))+ (0.0816924 * float(x[19]))+ (0.18367669 * float(x[20]))+ (-0.015253325 * float(x[21]))+ (-0.07284387 * float(x[22]))+ (0.057616528 * float(x[23]))+ (-0.0051799533 * float(x[24]))+ (-0.055841032 * float(x[25]))+ (0.17308187 * float(x[26]))+ (0.07453923 * float(x[27]))+ (0.15353519 * float(x[28]))+ (-0.010148573 * float(x[29]))+ (0.011951473 * float(x[30]))+ (0.108245805 * float(x[31]))+ (-0.06505477 * float(x[32]))+ (-0.07077549 * float(x[33]))+ (0.04144012 * float(x[34]))+ (0.03833761 * float(x[35]))+ (-0.018067924 * float(x[36]))+ (0.035434693 * float(x[37]))+ (0.19236857 * float(x[38]))+ (0.22524081 * float(x[39]))+ (0.0759402 * float(x[40]))+ (0.26693007 * float(x[41]))+ (-0.0028272788 * float(x[42]))+ (0.018630775 * float(x[43]))+ (0.10020188 * float(x[44]))+ (0.2191912 * float(x[45]))+ (-0.03491697 * float(x[46]))+ (0.29610628 * float(x[47]))+ (0.048745338 * float(x[48]))+ (-0.065527104 * float(x[49])))+ ((0.069530234 * float(x[50]))+ (0.0058632535 * float(x[51]))+ (0.14739549 * float(x[52]))+ (0.013354191 * float(x[53]))+ (0.28147793 * float(x[54]))+ (0.054853417 * float(x[55]))+ (0.10229167 * float(x[56]))+ (0.21502689 * float(x[57]))+ (0.07674705 * float(x[58]))+ (0.0792022 * float(x[59]))+ (0.13569754 * float(x[60]))+ (0.11333314 * float(x[61]))+ (0.16951884 * float(x[62]))+ (0.21026328 * float(x[63]))+ (0.08159284 * float(x[64]))+ (-0.052550714 * float(x[65]))+ (0.22126828 * float(x[66]))+ (0.028658615 * float(x[67]))+ (0.13769168 * float(x[68]))+ (0.030789599 * float(x[69]))+ (-0.0037094154 * float(x[70]))+ (-0.09070969 * float(x[71]))+ (0.093103364 * float(x[72]))+ (0.03557382 * float(x[73]))+ (-0.049571496 * float(x[74]))+ (0.19785218 * float(x[75]))+ (-0.086662754 * float(x[76]))+ (0.09515902 * float(x[77]))+ (0.0285326 * float(x[78]))+ (-0.019126344 * float(x[79]))+ (-0.05390199 * float(x[80]))+ (0.16935094 * float(x[81]))+ (-0.039749406 * float(x[82]))+ (0.16045317 * float(x[83]))+ (-0.2903659 * float(x[84]))+ (0.18651319 * float(x[85]))+ (0.24807744 * float(x[86]))+ (0.1038863 * float(x[87]))+ (0.3172513 * float(x[88]))+ (-0.45627496 * float(x[89]))+ (0.20955561 * float(x[90]))+ (0.08151459 * float(x[91]))+ (-0.13696113 * float(x[92]))+ (0.484852 * float(x[93]))+ (0.53474486 * float(x[94]))+ (0.38923952 * float(x[95]))+ (-0.06530854 * float(x[96]))+ (0.18178773 * float(x[97]))+ (0.008971374 * float(x[98]))+ (-0.21328495 * float(x[99])))+ ((0.13472283 * float(x[100]))+ (0.03670875 * float(x[101]))+ (0.1320704 * float(x[102]))+ (0.057378285 * float(x[103]))+ (-0.2163067 * float(x[104]))+ (0.13625722 * float(x[105]))+ (0.21850039 * float(x[106]))+ (0.042038076 * float(x[107]))+ (-0.078508146 * float(x[108]))+ (0.10798604 * float(x[109]))+ (0.014967954 * float(x[110]))+ (-0.024901573 * float(x[111]))+ (0.039688658 * float(x[112]))+ (-0.045601465 * float(x[113]))+ (-0.0147854 * float(x[114]))+ (-0.010873281 * float(x[115]))+ (0.009054205 * float(x[116]))+ (-0.0043644295 * float(x[117]))+ (0.22400035 * float(x[118]))+ (0.16763644 * float(x[119]))+ (-0.16191874 * float(x[120]))+ (0.021986114 * float(x[121]))+ (-0.17898825 * float(x[122]))+ (0.17386572 * float(x[123]))+ (-0.05956451 * float(x[124]))+ (-0.1355931 * float(x[125]))+ (0.17008704 * float(x[126]))+ (-0.10084739 * float(x[127]))+ (0.101147234 * float(x[128]))+ (-0.027255055 * float(x[129]))+ (-0.048708055 * float(x[130]))+ (0.08219116 * float(x[131]))+ (-0.06936517 * float(x[132]))+ (-0.1289902 * float(x[133]))+ (0.100027785 * float(x[134]))+ (0.008541349 * float(x[135]))+ (-0.14135817 * float(x[136]))+ (-0.0944809 * float(x[137]))+ (-0.05209531 * float(x[138]))+ (0.067265145 * float(x[139]))+ (-0.11311898 * float(x[140]))+ (0.064781256 * float(x[141]))+ (0.0502946 * float(x[142]))+ (0.030820308 * float(x[143]))+ (-0.075160936 * float(x[144]))+ (-0.08024164 * float(x[145]))+ (-0.012180565 * float(x[146]))+ (-0.053382315 * float(x[147]))+ (0.0011183809 * float(x[148]))+ (0.036105577 * float(x[149])))+ ((0.07422688 * float(x[150]))+ (-0.016783144 * float(x[151]))+ (0.09329832 * float(x[152]))+ (0.09445417 * float(x[153]))+ (0.01202679 * float(x[154]))+ (0.109746404 * float(x[155]))+ (0.021393958 * float(x[156]))+ (-0.10783606 * float(x[157]))+ (0.185543 * float(x[158]))+ (0.17427632 * float(x[159]))+ (0.0973773 * float(x[160]))+ (0.10993012 * float(x[161]))+ (-0.03831851 * float(x[162]))+ (0.021995338 * float(x[163]))+ (0.12216036 * float(x[164]))+ (-0.018150777 * float(x[165]))+ (0.06529538 * float(x[166]))+ (0.023209149 * float(x[167]))+ (-0.04454658 * float(x[168]))+ (0.08156573 * float(x[169]))+ (0.099383794 * float(x[170]))+ (-0.036828443 * float(x[171]))+ (0.06886344 * float(x[172]))+ (0.036325034 * float(x[173]))+ (0.03488654 * float(x[174]))+ (0.18924153 * float(x[175]))+ (-0.10476854 * float(x[176]))+ (0.16680618 * float(x[177]))+ (0.07508872 * float(x[178]))+ (-0.025061997 * float(x[179]))) + 1.1128464), 0)
    h_2 = max((((0.15443705 * float(x[0]))+ (0.15366308 * float(x[1]))+ (0.06790407 * float(x[2]))+ (0.2149446 * float(x[3]))+ (0.091325894 * float(x[4]))+ (-0.15791321 * float(x[5]))+ (0.035498574 * float(x[6]))+ (-0.0023508389 * float(x[7]))+ (-0.015065908 * float(x[8]))+ (-0.039028794 * float(x[9]))+ (-0.0111221885 * float(x[10]))+ (0.008546926 * float(x[11]))+ (-0.0008478128 * float(x[12]))+ (0.30326185 * float(x[13]))+ (0.075697094 * float(x[14]))+ (0.0015728112 * float(x[15]))+ (-0.061692897 * float(x[16]))+ (-0.026664326 * float(x[17]))+ (-0.07167673 * float(x[18]))+ (0.017386096 * float(x[19]))+ (-0.16111492 * float(x[20]))+ (-0.020922689 * float(x[21]))+ (0.053517472 * float(x[22]))+ (0.13190772 * float(x[23]))+ (-0.22397266 * float(x[24]))+ (-0.21242185 * float(x[25]))+ (-0.23552383 * float(x[26]))+ (0.0782731 * float(x[27]))+ (-0.093315125 * float(x[28]))+ (-0.10375294 * float(x[29]))+ (0.03832303 * float(x[30]))+ (0.06693255 * float(x[31]))+ (0.03376849 * float(x[32]))+ (-0.044785243 * float(x[33]))+ (0.041056413 * float(x[34]))+ (-0.001890824 * float(x[35]))+ (-0.036024462 * float(x[36]))+ (0.0380037 * float(x[37]))+ (-0.18442507 * float(x[38]))+ (0.020493867 * float(x[39]))+ (-0.042074986 * float(x[40]))+ (0.13167435 * float(x[41]))+ (-0.16238366 * float(x[42]))+ (0.09872266 * float(x[43]))+ (-0.023788625 * float(x[44]))+ (-0.07842458 * float(x[45]))+ (0.02992591 * float(x[46]))+ (-0.032393865 * float(x[47]))+ (0.1890473 * float(x[48]))+ (0.07572996 * float(x[49])))+ ((0.026724977 * float(x[50]))+ (-0.26194292 * float(x[51]))+ (0.10481375 * float(x[52]))+ (-0.19406946 * float(x[53]))+ (-0.14811313 * float(x[54]))+ (-0.038897187 * float(x[55]))+ (-0.09114678 * float(x[56]))+ (-0.08039801 * float(x[57]))+ (0.077088945 * float(x[58]))+ (-0.014421911 * float(x[59]))+ (-0.17946824 * float(x[60]))+ (-0.056573723 * float(x[61]))+ (-0.06508418 * float(x[62]))+ (-0.12834717 * float(x[63]))+ (-0.048349332 * float(x[64]))+ (-0.3433283 * float(x[65]))+ (-0.08493929 * float(x[66]))+ (0.028929554 * float(x[67]))+ (-0.23769537 * float(x[68]))+ (-0.02173654 * float(x[69]))+ (0.11642971 * float(x[70]))+ (-0.17289415 * float(x[71]))+ (-0.14196332 * float(x[72]))+ (-0.17652723 * float(x[73]))+ (-0.11095466 * float(x[74]))+ (-0.39087585 * float(x[75]))+ (0.026788348 * float(x[76]))+ (-0.17370127 * float(x[77]))+ (0.2143119 * float(x[78]))+ (0.037839033 * float(x[79]))+ (-0.038184796 * float(x[80]))+ (-0.49922824 * float(x[81]))+ (0.10390189 * float(x[82]))+ (-0.42545572 * float(x[83]))+ (0.76144695 * float(x[84]))+ (-0.22834224 * float(x[85]))+ (-0.23527561 * float(x[86]))+ (-0.26761383 * float(x[87]))+ (-0.47306556 * float(x[88]))+ (0.6072918 * float(x[89]))+ (0.04585356 * float(x[90]))+ (-0.21146862 * float(x[91]))+ (0.4350447 * float(x[92]))+ (-0.19780467 * float(x[93]))+ (-0.33487913 * float(x[94]))+ (-0.38284862 * float(x[95]))+ (0.21044128 * float(x[96]))+ (-0.073981285 * float(x[97]))+ (-0.049895592 * float(x[98]))+ (-0.015051271 * float(x[99])))+ ((0.06892467 * float(x[100]))+ (-0.10101237 * float(x[101]))+ (0.0104386285 * float(x[102]))+ (-0.050735693 * float(x[103]))+ (0.21508752 * float(x[104]))+ (-0.079540916 * float(x[105]))+ (0.108531915 * float(x[106]))+ (-0.24318406 * float(x[107]))+ (0.17775546 * float(x[108]))+ (0.21319719 * float(x[109]))+ (0.16701636 * float(x[110]))+ (-0.09791249 * float(x[111]))+ (0.031956688 * float(x[112]))+ (-0.050521135 * float(x[113]))+ (0.032516338 * float(x[114]))+ (0.06275443 * float(x[115]))+ (0.01000965 * float(x[116]))+ (-0.07581079 * float(x[117]))+ (-0.06199918 * float(x[118]))+ (-0.058506634 * float(x[119]))+ (0.20842597 * float(x[120]))+ (0.14254174 * float(x[121]))+ (-0.10259057 * float(x[122]))+ (-0.042094402 * float(x[123]))+ (0.08200633 * float(x[124]))+ (0.03926927 * float(x[125]))+ (0.01951823 * float(x[126]))+ (0.22653478 * float(x[127]))+ (0.07733536 * float(x[128]))+ (0.043334305 * float(x[129]))+ (-0.12897982 * float(x[130]))+ (0.07869463 * float(x[131]))+ (-0.1296507 * float(x[132]))+ (0.045262817 * float(x[133]))+ (0.0631215 * float(x[134]))+ (-0.13815303 * float(x[135]))+ (0.112090856 * float(x[136]))+ (0.22560148 * float(x[137]))+ (-0.18262038 * float(x[138]))+ (-0.10453227 * float(x[139]))+ (0.05203183 * float(x[140]))+ (0.05090063 * float(x[141]))+ (0.05145297 * float(x[142]))+ (0.21083632 * float(x[143]))+ (0.13668722 * float(x[144]))+ (0.20928489 * float(x[145]))+ (-0.09315635 * float(x[146]))+ (-0.083331935 * float(x[147]))+ (0.0013589908 * float(x[148]))+ (0.22533754 * float(x[149])))+ ((-0.091573164 * float(x[150]))+ (0.072403476 * float(x[151]))+ (0.036435854 * float(x[152]))+ (-0.0397236 * float(x[153]))+ (-0.13826258 * float(x[154]))+ (0.015696047 * float(x[155]))+ (-0.33174065 * float(x[156]))+ (-0.056836434 * float(x[157]))+ (-0.11667425 * float(x[158]))+ (-0.14387509 * float(x[159]))+ (-0.10628564 * float(x[160]))+ (-0.15701868 * float(x[161]))+ (-0.040357444 * float(x[162]))+ (0.017666431 * float(x[163]))+ (0.01843802 * float(x[164]))+ (-0.013439377 * float(x[165]))+ (0.11672317 * float(x[166]))+ (0.2137736 * float(x[167]))+ (-0.05033686 * float(x[168]))+ (0.006600113 * float(x[169]))+ (0.030114349 * float(x[170]))+ (0.32646245 * float(x[171]))+ (0.25524914 * float(x[172]))+ (0.23737365 * float(x[173]))+ (-0.015637536 * float(x[174]))+ (-0.14246099 * float(x[175]))+ (-0.07586651 * float(x[176]))+ (0.03655184 * float(x[177]))+ (-0.06821135 * float(x[178]))+ (0.10141197 * float(x[179]))) + 0.37627333), 0)
    h_3 = max((((0.04787253 * float(x[0]))+ (0.010677034 * float(x[1]))+ (-0.082090154 * float(x[2]))+ (-0.055258613 * float(x[3]))+ (0.036934163 * float(x[4]))+ (-0.1004173 * float(x[5]))+ (-0.09983233 * float(x[6]))+ (0.024134383 * float(x[7]))+ (0.05197505 * float(x[8]))+ (0.063489184 * float(x[9]))+ (-0.00244674 * float(x[10]))+ (0.009031729 * float(x[11]))+ (0.09882205 * float(x[12]))+ (0.21190336 * float(x[13]))+ (0.18357646 * float(x[14]))+ (0.035054084 * float(x[15]))+ (0.014583209 * float(x[16]))+ (0.024944305 * float(x[17]))+ (0.029471034 * float(x[18]))+ (-0.08549208 * float(x[19]))+ (0.10151127 * float(x[20]))+ (0.009635764 * float(x[21]))+ (0.08205818 * float(x[22]))+ (0.2264353 * float(x[23]))+ (0.0831179 * float(x[24]))+ (-0.010440558 * float(x[25]))+ (-0.081777595 * float(x[26]))+ (-0.091802925 * float(x[27]))+ (0.034329947 * float(x[28]))+ (0.06669852 * float(x[29]))+ (-0.14486438 * float(x[30]))+ (-0.014854373 * float(x[31]))+ (-0.1962592 * float(x[32]))+ (0.085970655 * float(x[33]))+ (0.07436411 * float(x[34]))+ (-0.2322722 * float(x[35]))+ (-0.13353051 * float(x[36]))+ (-0.05654655 * float(x[37]))+ (0.036868613 * float(x[38]))+ (-0.12635633 * float(x[39]))+ (0.0045717526 * float(x[40]))+ (0.037082072 * float(x[41]))+ (-0.0054702167 * float(x[42]))+ (-0.11194378 * float(x[43]))+ (-0.31976467 * float(x[44]))+ (-0.078252204 * float(x[45]))+ (0.020922223 * float(x[46]))+ (-0.029895857 * float(x[47]))+ (-0.09157843 * float(x[48]))+ (-0.11060833 * float(x[49])))+ ((-0.0038805902 * float(x[50]))+ (0.01761827 * float(x[51]))+ (0.023398442 * float(x[52]))+ (-0.051378667 * float(x[53]))+ (-0.11968568 * float(x[54]))+ (0.17064711 * float(x[55]))+ (-0.10121061 * float(x[56]))+ (-0.007672736 * float(x[57]))+ (0.008072936 * float(x[58]))+ (-0.17366433 * float(x[59]))+ (-0.36861366 * float(x[60]))+ (-0.19460456 * float(x[61]))+ (-0.23306228 * float(x[62]))+ (-0.08958605 * float(x[63]))+ (0.02884283 * float(x[64]))+ (-0.066751026 * float(x[65]))+ (-0.036438093 * float(x[66]))+ (-0.24495682 * float(x[67]))+ (-0.009398924 * float(x[68]))+ (-0.022905309 * float(x[69]))+ (-0.012078234 * float(x[70]))+ (0.12242673 * float(x[71]))+ (-0.063678 * float(x[72]))+ (-5.7119272e-05 * float(x[73]))+ (-0.27563113 * float(x[74]))+ (-0.076353535 * float(x[75]))+ (0.070042334 * float(x[76]))+ (0.026454424 * float(x[77]))+ (0.113279685 * float(x[78]))+ (0.16891551 * float(x[79]))+ (0.09356398 * float(x[80]))+ (-0.40244862 * float(x[81]))+ (0.08837949 * float(x[82]))+ (-0.3351688 * float(x[83]))+ (0.5065279 * float(x[84]))+ (-0.060124446 * float(x[85]))+ (-0.42941478 * float(x[86]))+ (-0.113125764 * float(x[87]))+ (-0.18443175 * float(x[88]))+ (0.5569084 * float(x[89]))+ (-0.07330672 * float(x[90]))+ (-0.25454122 * float(x[91]))+ (0.18214922 * float(x[92]))+ (-0.28893238 * float(x[93]))+ (-0.12248431 * float(x[94]))+ (-0.22280589 * float(x[95]))+ (0.04018698 * float(x[96]))+ (-0.41526783 * float(x[97]))+ (-0.0023132723 * float(x[98]))+ (0.12597354 * float(x[99])))+ ((-0.11876596 * float(x[100]))+ (-0.030875169 * float(x[101]))+ (0.0046126684 * float(x[102]))+ (-0.072009526 * float(x[103]))+ (0.15897271 * float(x[104]))+ (0.04980028 * float(x[105]))+ (-0.21776713 * float(x[106]))+ (0.016648384 * float(x[107]))+ (-0.041013647 * float(x[108]))+ (0.1757171 * float(x[109]))+ (0.025441106 * float(x[110]))+ (0.05136485 * float(x[111]))+ (-0.18259479 * float(x[112]))+ (0.06948043 * float(x[113]))+ (-0.051462978 * float(x[114]))+ (-0.036245193 * float(x[115]))+ (0.098897755 * float(x[116]))+ (-0.09786486 * float(x[117]))+ (-0.16069205 * float(x[118]))+ (0.15331852 * float(x[119]))+ (-0.111345224 * float(x[120]))+ (0.18495767 * float(x[121]))+ (-0.0041939667 * float(x[122]))+ (0.011120626 * float(x[123]))+ (0.06502249 * float(x[124]))+ (0.017246041 * float(x[125]))+ (-0.18111584 * float(x[126]))+ (-0.030615645 * float(x[127]))+ (-0.098207094 * float(x[128]))+ (0.06857976 * float(x[129]))+ (-0.12546033 * float(x[130]))+ (-0.09718956 * float(x[131]))+ (0.007583184 * float(x[132]))+ (-0.16719182 * float(x[133]))+ (0.06932083 * float(x[134]))+ (-0.19229925 * float(x[135]))+ (-0.1683591 * float(x[136]))+ (0.13842613 * float(x[137]))+ (0.07444322 * float(x[138]))+ (0.07920057 * float(x[139]))+ (0.14744505 * float(x[140]))+ (-0.19419023 * float(x[141]))+ (0.05708399 * float(x[142]))+ (0.14767332 * float(x[143]))+ (-0.012356255 * float(x[144]))+ (0.18685153 * float(x[145]))+ (-0.04083514 * float(x[146]))+ (-0.06659722 * float(x[147]))+ (0.07194487 * float(x[148]))+ (0.2621321 * float(x[149])))+ ((-0.12124776 * float(x[150]))+ (-0.109565124 * float(x[151]))+ (0.042775977 * float(x[152]))+ (0.13570671 * float(x[153]))+ (-0.28435758 * float(x[154]))+ (-0.14885493 * float(x[155]))+ (-0.04932666 * float(x[156]))+ (-0.14342348 * float(x[157]))+ (-0.093697794 * float(x[158]))+ (-0.17298725 * float(x[159]))+ (-0.08211859 * float(x[160]))+ (-0.08483763 * float(x[161]))+ (-0.043317493 * float(x[162]))+ (0.00397515 * float(x[163]))+ (0.24350147 * float(x[164]))+ (-0.13791338 * float(x[165]))+ (0.16964824 * float(x[166]))+ (0.023099128 * float(x[167]))+ (-0.043601222 * float(x[168]))+ (0.04243561 * float(x[169]))+ (0.13574915 * float(x[170]))+ (0.046587523 * float(x[171]))+ (-0.12694488 * float(x[172]))+ (0.1806328 * float(x[173]))+ (0.121312514 * float(x[174]))+ (0.0031504775 * float(x[175]))+ (0.09836473 * float(x[176]))+ (-0.07315789 * float(x[177]))+ (0.031011894 * float(x[178]))+ (0.21788149 * float(x[179]))) + 1.0638208), 0)
    h_4 = max((((0.10942245 * float(x[0]))+ (0.12511958 * float(x[1]))+ (0.0017307157 * float(x[2]))+ (-0.07886069 * float(x[3]))+ (0.017476847 * float(x[4]))+ (-0.050937764 * float(x[5]))+ (-0.073745206 * float(x[6]))+ (0.19905499 * float(x[7]))+ (-0.03958804 * float(x[8]))+ (-0.08893051 * float(x[9]))+ (0.13741848 * float(x[10]))+ (-0.032333642 * float(x[11]))+ (0.031890098 * float(x[12]))+ (0.20861606 * float(x[13]))+ (0.050548337 * float(x[14]))+ (-0.17126653 * float(x[15]))+ (0.0074347095 * float(x[16]))+ (-0.025283443 * float(x[17]))+ (-0.10095611 * float(x[18]))+ (0.11049401 * float(x[19]))+ (0.07742032 * float(x[20]))+ (-0.12066929 * float(x[21]))+ (0.014516962 * float(x[22]))+ (-0.12238472 * float(x[23]))+ (0.11120667 * float(x[24]))+ (0.07307157 * float(x[25]))+ (-0.00033815738 * float(x[26]))+ (-0.28763816 * float(x[27]))+ (-0.10808196 * float(x[28]))+ (-0.33453354 * float(x[29]))+ (-0.09614433 * float(x[30]))+ (0.1410508 * float(x[31]))+ (-0.10655303 * float(x[32]))+ (-0.18036236 * float(x[33]))+ (-0.06253924 * float(x[34]))+ (-0.13904572 * float(x[35]))+ (-0.034754485 * float(x[36]))+ (0.031878047 * float(x[37]))+ (-0.2908135 * float(x[38]))+ (-0.24665335 * float(x[39]))+ (0.043540128 * float(x[40]))+ (-0.05912363 * float(x[41]))+ (-0.23390447 * float(x[42]))+ (-0.009315262 * float(x[43]))+ (-0.113107026 * float(x[44]))+ (-0.3069487 * float(x[45]))+ (-0.098300956 * float(x[46]))+ (0.05038888 * float(x[47]))+ (-0.21818337 * float(x[48]))+ (0.0768399 * float(x[49])))+ ((-0.043883726 * float(x[50]))+ (-0.16205984 * float(x[51]))+ (-0.0052292454 * float(x[52]))+ (0.054979563 * float(x[53]))+ (-0.2158227 * float(x[54]))+ (-0.14881323 * float(x[55]))+ (-0.0045705573 * float(x[56]))+ (-0.1327781 * float(x[57]))+ (0.080990344 * float(x[58]))+ (-0.16959669 * float(x[59]))+ (-0.22305423 * float(x[60]))+ (0.17330876 * float(x[61]))+ (-0.15826906 * float(x[62]))+ (-0.31175327 * float(x[63]))+ (-0.07464457 * float(x[64]))+ (-0.32985154 * float(x[65]))+ (-0.3622492 * float(x[66]))+ (-0.0869223 * float(x[67]))+ (-0.24166083 * float(x[68]))+ (0.06306205 * float(x[69]))+ (-0.16030413 * float(x[70]))+ (-0.20597471 * float(x[71]))+ (-0.45226863 * float(x[72]))+ (0.11179101 * float(x[73]))+ (-0.16356498 * float(x[74]))+ (-0.27192053 * float(x[75]))+ (-0.06061456 * float(x[76]))+ (-0.010961215 * float(x[77]))+ (-0.056749355 * float(x[78]))+ (0.031990595 * float(x[79]))+ (-0.024008151 * float(x[80]))+ (-0.51310766 * float(x[81]))+ (0.05458471 * float(x[82]))+ (-0.23480088 * float(x[83]))+ (0.35146812 * float(x[84]))+ (-0.3884966 * float(x[85]))+ (-0.3108649 * float(x[86]))+ (-0.24735002 * float(x[87]))+ (-0.24295017 * float(x[88]))+ (0.39156222 * float(x[89]))+ (0.30235425 * float(x[90]))+ (-0.14279924 * float(x[91]))+ (0.110806674 * float(x[92]))+ (-0.084049605 * float(x[93]))+ (0.015242946 * float(x[94]))+ (-0.122193485 * float(x[95]))+ (-0.23541422 * float(x[96]))+ (0.109619476 * float(x[97]))+ (-0.18525824 * float(x[98]))+ (0.041178815 * float(x[99])))+ ((0.14614896 * float(x[100]))+ (0.075414084 * float(x[101]))+ (0.15278211 * float(x[102]))+ (0.107581936 * float(x[103]))+ (-0.11019092 * float(x[104]))+ (0.04754181 * float(x[105]))+ (0.0055290475 * float(x[106]))+ (0.08980642 * float(x[107]))+ (0.28792825 * float(x[108]))+ (0.09149414 * float(x[109]))+ (-0.05472113 * float(x[110]))+ (-0.011462789 * float(x[111]))+ (0.111420944 * float(x[112]))+ (-0.122540385 * float(x[113]))+ (0.16523144 * float(x[114]))+ (0.1301378 * float(x[115]))+ (0.07190478 * float(x[116]))+ (-0.0015488159 * float(x[117]))+ (0.08465944 * float(x[118]))+ (-0.33837909 * float(x[119]))+ (0.11059327 * float(x[120]))+ (0.013733466 * float(x[121]))+ (0.053599175 * float(x[122]))+ (-0.033986546 * float(x[123]))+ (-0.049446344 * float(x[124]))+ (0.10055152 * float(x[125]))+ (0.1830167 * float(x[126]))+ (-0.04710355 * float(x[127]))+ (-0.14575434 * float(x[128]))+ (-0.035407905 * float(x[129]))+ (0.1804399 * float(x[130]))+ (-0.023355655 * float(x[131]))+ (0.10625559 * float(x[132]))+ (-0.053888477 * float(x[133]))+ (-0.061399836 * float(x[134]))+ (-0.004345323 * float(x[135]))+ (0.1684694 * float(x[136]))+ (-0.027394742 * float(x[137]))+ (-0.032711487 * float(x[138]))+ (-0.106930666 * float(x[139]))+ (0.03392255 * float(x[140]))+ (0.11316349 * float(x[141]))+ (0.15950672 * float(x[142]))+ (0.17582497 * float(x[143]))+ (-0.092667304 * float(x[144]))+ (0.056369856 * float(x[145]))+ (0.15158372 * float(x[146]))+ (-0.003721848 * float(x[147]))+ (-0.097564384 * float(x[148]))+ (0.028687835 * float(x[149])))+ ((-0.005858499 * float(x[150]))+ (0.011860998 * float(x[151]))+ (-0.1018822 * float(x[152]))+ (-0.108368285 * float(x[153]))+ (0.01813592 * float(x[154]))+ (-0.032962102 * float(x[155]))+ (-0.33546698 * float(x[156]))+ (-0.10277805 * float(x[157]))+ (-0.06208784 * float(x[158]))+ (0.16834544 * float(x[159]))+ (0.008842679 * float(x[160]))+ (-0.037341397 * float(x[161]))+ (0.070635006 * float(x[162]))+ (-0.106667385 * float(x[163]))+ (-0.008643035 * float(x[164]))+ (0.06936094 * float(x[165]))+ (0.15381856 * float(x[166]))+ (0.16226757 * float(x[167]))+ (-0.061648443 * float(x[168]))+ (-0.036211863 * float(x[169]))+ (-0.072095476 * float(x[170]))+ (0.2579537 * float(x[171]))+ (0.20262977 * float(x[172]))+ (0.014816076 * float(x[173]))+ (-0.068582855 * float(x[174]))+ (0.064573765 * float(x[175]))+ (-0.18783733 * float(x[176]))+ (-0.055937264 * float(x[177]))+ (0.119074695 * float(x[178]))+ (0.03982316 * float(x[179]))) + 0.9047376), 0)
    h_5 = max((((-0.08290115 * float(x[0]))+ (0.09253307 * float(x[1]))+ (0.046144236 * float(x[2]))+ (-0.08574138 * float(x[3]))+ (-0.15585043 * float(x[4]))+ (0.12611334 * float(x[5]))+ (0.0021016302 * float(x[6]))+ (-3.5938414e-05 * float(x[7]))+ (-0.0042526815 * float(x[8]))+ (-0.010566898 * float(x[9]))+ (-0.12317894 * float(x[10]))+ (0.15540297 * float(x[11]))+ (0.116340235 * float(x[12]))+ (-0.0626231 * float(x[13]))+ (0.11310673 * float(x[14]))+ (-0.13785376 * float(x[15]))+ (-0.014483525 * float(x[16]))+ (-0.13517873 * float(x[17]))+ (-0.10282558 * float(x[18]))+ (0.012969455 * float(x[19]))+ (-0.1352525 * float(x[20]))+ (0.0028195959 * float(x[21]))+ (-0.022329545 * float(x[22]))+ (0.21215774 * float(x[23]))+ (-0.14916784 * float(x[24]))+ (-0.107246466 * float(x[25]))+ (0.06991375 * float(x[26]))+ (-0.14135109 * float(x[27]))+ (0.024265127 * float(x[28]))+ (0.07481639 * float(x[29]))+ (-0.021820888 * float(x[30]))+ (-0.057868402 * float(x[31]))+ (0.10309603 * float(x[32]))+ (0.04101567 * float(x[33]))+ (0.08184481 * float(x[34]))+ (-0.2913156 * float(x[35]))+ (-0.054285098 * float(x[36]))+ (0.24449208 * float(x[37]))+ (0.121371046 * float(x[38]))+ (-0.08710874 * float(x[39]))+ (-0.042212546 * float(x[40]))+ (0.16452736 * float(x[41]))+ (-0.17522484 * float(x[42]))+ (0.075758636 * float(x[43]))+ (0.01652953 * float(x[44]))+ (-0.107974164 * float(x[45]))+ (-0.034311228 * float(x[46]))+ (0.023864726 * float(x[47]))+ (-0.020989742 * float(x[48]))+ (0.112965465 * float(x[49])))+ ((-0.0033415335 * float(x[50]))+ (0.014346411 * float(x[51]))+ (0.11661069 * float(x[52]))+ (0.054885034 * float(x[53]))+ (-0.25257322 * float(x[54]))+ (0.21226107 * float(x[55]))+ (-0.1728312 * float(x[56]))+ (-0.09964029 * float(x[57]))+ (0.01676757 * float(x[58]))+ (-0.016392944 * float(x[59]))+ (-0.13733065 * float(x[60]))+ (-0.0054292292 * float(x[61]))+ (-0.20816903 * float(x[62]))+ (-0.1815201 * float(x[63]))+ (0.030922417 * float(x[64]))+ (0.0038121254 * float(x[65]))+ (-0.33947638 * float(x[66]))+ (-0.07515706 * float(x[67]))+ (-0.15376312 * float(x[68]))+ (-0.03622251 * float(x[69]))+ (0.13674413 * float(x[70]))+ (-0.16011034 * float(x[71]))+ (-0.40992627 * float(x[72]))+ (0.0010024891 * float(x[73]))+ (-0.23615994 * float(x[74]))+ (-0.115035966 * float(x[75]))+ (-0.039166402 * float(x[76]))+ (-0.12339947 * float(x[77]))+ (0.15069468 * float(x[78]))+ (-0.10596403 * float(x[79]))+ (0.06672603 * float(x[80]))+ (-0.5449823 * float(x[81]))+ (-0.11103477 * float(x[82]))+ (-0.6792846 * float(x[83]))+ (0.63911706 * float(x[84]))+ (-0.06010796 * float(x[85]))+ (-0.52715015 * float(x[86]))+ (-0.27585232 * float(x[87]))+ (0.09188882 * float(x[88]))+ (0.38657883 * float(x[89]))+ (0.29919532 * float(x[90]))+ (0.34740254 * float(x[91]))+ (-0.25057772 * float(x[92]))+ (0.5752404 * float(x[93]))+ (0.6143749 * float(x[94]))+ (0.6081147 * float(x[95]))+ (-0.1591261 * float(x[96]))+ (0.19168186 * float(x[97]))+ (-0.05351437 * float(x[98]))+ (-0.22898594 * float(x[99])))+ ((-0.11595872 * float(x[100]))+ (0.0090403445 * float(x[101]))+ (-0.017133394 * float(x[102]))+ (0.3175434 * float(x[103]))+ (-0.47624803 * float(x[104]))+ (0.16032748 * float(x[105]))+ (0.18347012 * float(x[106]))+ (0.20162295 * float(x[107]))+ (0.0814634 * float(x[108]))+ (0.08390395 * float(x[109]))+ (0.115086965 * float(x[110]))+ (0.2229456 * float(x[111]))+ (-0.15145196 * float(x[112]))+ (0.10218989 * float(x[113]))+ (0.1909151 * float(x[114]))+ (-0.1097287 * float(x[115]))+ (-0.028684732 * float(x[116]))+ (0.034262475 * float(x[117]))+ (0.11598937 * float(x[118]))+ (0.0743145 * float(x[119]))+ (0.12870198 * float(x[120]))+ (0.123860024 * float(x[121]))+ (0.14847209 * float(x[122]))+ (0.39317796 * float(x[123]))+ (-0.092537746 * float(x[124]))+ (0.025215417 * float(x[125]))+ (0.041478854 * float(x[126]))+ (0.08968117 * float(x[127]))+ (0.09387278 * float(x[128]))+ (-0.028283998 * float(x[129]))+ (0.09732393 * float(x[130]))+ (0.084782794 * float(x[131]))+ (0.077554345 * float(x[132]))+ (0.056619808 * float(x[133]))+ (0.21252088 * float(x[134]))+ (0.05799327 * float(x[135]))+ (0.24811086 * float(x[136]))+ (0.24505164 * float(x[137]))+ (0.0886728 * float(x[138]))+ (-0.04521915 * float(x[139]))+ (-0.008804774 * float(x[140]))+ (0.012260301 * float(x[141]))+ (0.12681635 * float(x[142]))+ (0.06384659 * float(x[143]))+ (-0.03479014 * float(x[144]))+ (0.14512429 * float(x[145]))+ (0.11538376 * float(x[146]))+ (0.21783264 * float(x[147]))+ (0.023480138 * float(x[148]))+ (0.11681484 * float(x[149])))+ ((0.24338973 * float(x[150]))+ (-0.0042499504 * float(x[151]))+ (0.12626342 * float(x[152]))+ (-0.024001747 * float(x[153]))+ (0.10962051 * float(x[154]))+ (0.12391644 * float(x[155]))+ (0.07175095 * float(x[156]))+ (0.061346892 * float(x[157]))+ (0.004594824 * float(x[158]))+ (0.04530612 * float(x[159]))+ (-0.04532513 * float(x[160]))+ (-0.06880726 * float(x[161]))+ (0.0429406 * float(x[162]))+ (0.1827126 * float(x[163]))+ (0.08370582 * float(x[164]))+ (-0.07294508 * float(x[165]))+ (0.06302772 * float(x[166]))+ (-0.05644685 * float(x[167]))+ (0.093165666 * float(x[168]))+ (0.0032618858 * float(x[169]))+ (0.2723692 * float(x[170]))+ (0.18204153 * float(x[171]))+ (0.26484084 * float(x[172]))+ (0.19947514 * float(x[173]))+ (0.047192406 * float(x[174]))+ (0.09971996 * float(x[175]))+ (0.19975716 * float(x[176]))+ (0.11664729 * float(x[177]))+ (0.037124958 * float(x[178]))+ (0.0992145 * float(x[179]))) + 0.48905334), 0)
    h_6 = max((((0.066243075 * float(x[0]))+ (0.11504682 * float(x[1]))+ (-0.016600795 * float(x[2]))+ (0.046031117 * float(x[3]))+ (0.02302254 * float(x[4]))+ (0.057020184 * float(x[5]))+ (-0.02987652 * float(x[6]))+ (0.039954778 * float(x[7]))+ (-0.029946115 * float(x[8]))+ (-0.008001549 * float(x[9]))+ (0.12681529 * float(x[10]))+ (0.15268855 * float(x[11]))+ (-0.18540126 * float(x[12]))+ (0.13536854 * float(x[13]))+ (7.352782e-05 * float(x[14]))+ (0.22159393 * float(x[15]))+ (0.23931539 * float(x[16]))+ (0.1990611 * float(x[17]))+ (0.18957043 * float(x[18]))+ (0.13473213 * float(x[19]))+ (0.024175884 * float(x[20]))+ (-0.22188899 * float(x[21]))+ (0.33424205 * float(x[22]))+ (0.12780516 * float(x[23]))+ (0.03329991 * float(x[24]))+ (0.18788041 * float(x[25]))+ (0.21079388 * float(x[26]))+ (0.0003937817 * float(x[27]))+ (0.07712068 * float(x[28]))+ (0.01595331 * float(x[29]))+ (0.22179662 * float(x[30]))+ (0.16071653 * float(x[31]))+ (0.012177189 * float(x[32]))+ (0.13246882 * float(x[33]))+ (-0.066658884 * float(x[34]))+ (0.18225627 * float(x[35]))+ (0.07756561 * float(x[36]))+ (0.18157747 * float(x[37]))+ (0.16680232 * float(x[38]))+ (0.10481652 * float(x[39]))+ (0.08344304 * float(x[40]))+ (0.13517836 * float(x[41]))+ (0.054544277 * float(x[42]))+ (-0.014409556 * float(x[43]))+ (0.20366818 * float(x[44]))+ (0.037252616 * float(x[45]))+ (0.19506083 * float(x[46]))+ (0.0668787 * float(x[47]))+ (0.2523314 * float(x[48]))+ (0.11886295 * float(x[49])))+ ((0.18953463 * float(x[50]))+ (0.12413078 * float(x[51]))+ (-0.015037264 * float(x[52]))+ (0.121525355 * float(x[53]))+ (0.028220568 * float(x[54]))+ (-0.03524034 * float(x[55]))+ (0.32982758 * float(x[56]))+ (0.17179912 * float(x[57]))+ (0.27832174 * float(x[58]))+ (-0.08193532 * float(x[59]))+ (0.24411629 * float(x[60]))+ (-0.030822251 * float(x[61]))+ (0.025689753 * float(x[62]))+ (0.42026025 * float(x[63]))+ (0.17264861 * float(x[64]))+ (0.25663328 * float(x[65]))+ (0.32406288 * float(x[66]))+ (0.14987306 * float(x[67]))+ (0.17025043 * float(x[68]))+ (0.13222861 * float(x[69]))+ (-0.09641187 * float(x[70]))+ (0.15056488 * float(x[71]))+ (0.4050537 * float(x[72]))+ (-0.016065925 * float(x[73]))+ (0.47286126 * float(x[74]))+ (0.15073459 * float(x[75]))+ (-0.059948307 * float(x[76]))+ (0.15747796 * float(x[77]))+ (-0.17731827 * float(x[78]))+ (0.0029129644 * float(x[79]))+ (0.0056347526 * float(x[80]))+ (0.41126108 * float(x[81]))+ (-0.093987815 * float(x[82]))+ (0.41432592 * float(x[83]))+ (-0.35988867 * float(x[84]))+ (0.36752066 * float(x[85]))+ (0.5663999 * float(x[86]))+ (0.357946 * float(x[87]))+ (0.049462575 * float(x[88]))+ (-0.39402193 * float(x[89]))+ (-0.5597547 * float(x[90]))+ (-0.06777419 * float(x[91]))+ (0.27459386 * float(x[92]))+ (-0.50702375 * float(x[93]))+ (-0.52369434 * float(x[94]))+ (-0.6059407 * float(x[95]))+ (0.51163965 * float(x[96]))+ (-0.22814663 * float(x[97]))+ (0.17936018 * float(x[98]))+ (0.15352933 * float(x[99])))+ ((0.05773578 * float(x[100]))+ (-0.13785085 * float(x[101]))+ (-0.07090786 * float(x[102]))+ (-0.17268436 * float(x[103]))+ (0.53426254 * float(x[104]))+ (-0.16101775 * float(x[105]))+ (-0.19535153 * float(x[106]))+ (-0.15914544 * float(x[107]))+ (0.07952367 * float(x[108]))+ (-0.036664475 * float(x[109]))+ (0.15375805 * float(x[110]))+ (0.11235152 * float(x[111]))+ (-0.04704425 * float(x[112]))+ (0.050604895 * float(x[113]))+ (-0.14643402 * float(x[114]))+ (0.06336118 * float(x[115]))+ (0.21367948 * float(x[116]))+ (-0.0119887665 * float(x[117]))+ (-0.022250403 * float(x[118]))+ (0.058439717 * float(x[119]))+ (-0.1855461 * float(x[120]))+ (-0.0056023994 * float(x[121]))+ (0.051223304 * float(x[122]))+ (0.0009983453 * float(x[123]))+ (0.038754188 * float(x[124]))+ (0.1498266 * float(x[125]))+ (-0.041860893 * float(x[126]))+ (0.16018398 * float(x[127]))+ (-0.03449901 * float(x[128]))+ (0.1329829 * float(x[129]))+ (0.06273637 * float(x[130]))+ (-0.19932382 * float(x[131]))+ (-0.26890197 * float(x[132]))+ (-0.08434665 * float(x[133]))+ (-0.034534574 * float(x[134]))+ (0.03621405 * float(x[135]))+ (-0.024433829 * float(x[136]))+ (-0.06192606 * float(x[137]))+ (0.050819114 * float(x[138]))+ (0.044347916 * float(x[139]))+ (-0.08276656 * float(x[140]))+ (0.035953417 * float(x[141]))+ (-0.10673011 * float(x[142]))+ (-0.14217834 * float(x[143]))+ (-0.09780541 * float(x[144]))+ (0.02271856 * float(x[145]))+ (-0.052354075 * float(x[146]))+ (-0.14999849 * float(x[147]))+ (-0.068893194 * float(x[148]))+ (-0.029504208 * float(x[149])))+ ((-0.26964632 * float(x[150]))+ (0.20856802 * float(x[151]))+ (0.034916066 * float(x[152]))+ (-0.11859066 * float(x[153]))+ (0.06582584 * float(x[154]))+ (0.26627213 * float(x[155]))+ (0.06644707 * float(x[156]))+ (-0.054879002 * float(x[157]))+ (0.14418148 * float(x[158]))+ (0.07356879 * float(x[159]))+ (0.15658784 * float(x[160]))+ (0.14066064 * float(x[161]))+ (-0.21908936 * float(x[162]))+ (-0.05056927 * float(x[163]))+ (-0.073079675 * float(x[164]))+ (0.086640365 * float(x[165]))+ (-0.053774793 * float(x[166]))+ (0.13185151 * float(x[167]))+ (-0.09608469 * float(x[168]))+ (-0.019710606 * float(x[169]))+ (-0.06442975 * float(x[170]))+ (-0.13549408 * float(x[171]))+ (-0.09883597 * float(x[172]))+ (0.06385905 * float(x[173]))+ (0.042112987 * float(x[174]))+ (0.026667127 * float(x[175]))+ (-0.14516476 * float(x[176]))+ (0.018799894 * float(x[177]))+ (0.02158336 * float(x[178]))+ (0.14466767 * float(x[179]))) + 0.40566996), 0)
    h_7 = max((((0.04426203 * float(x[0]))+ (0.06364184 * float(x[1]))+ (-0.07386172 * float(x[2]))+ (0.05240482 * float(x[3]))+ (0.22631937 * float(x[4]))+ (0.09116243 * float(x[5]))+ (0.019866867 * float(x[6]))+ (0.090465695 * float(x[7]))+ (0.039644998 * float(x[8]))+ (0.05867298 * float(x[9]))+ (0.14379382 * float(x[10]))+ (-0.050595634 * float(x[11]))+ (-0.03105334 * float(x[12]))+ (-0.052606795 * float(x[13]))+ (-0.011837247 * float(x[14]))+ (0.19681872 * float(x[15]))+ (0.1245652 * float(x[16]))+ (0.18474637 * float(x[17]))+ (0.11593466 * float(x[18]))+ (0.15757017 * float(x[19]))+ (-0.005433248 * float(x[20]))+ (0.025084138 * float(x[21]))+ (0.24271436 * float(x[22]))+ (-0.1560949 * float(x[23]))+ (0.048375037 * float(x[24]))+ (-0.01032871 * float(x[25]))+ (0.2137765 * float(x[26]))+ (0.14174797 * float(x[27]))+ (-0.026273781 * float(x[28]))+ (0.14881909 * float(x[29]))+ (0.010144048 * float(x[30]))+ (0.07643767 * float(x[31]))+ (0.09882717 * float(x[32]))+ (0.1532376 * float(x[33]))+ (0.12805258 * float(x[34]))+ (0.17633922 * float(x[35]))+ (-0.16076733 * float(x[36]))+ (0.078627385 * float(x[37]))+ (-0.05133584 * float(x[38]))+ (0.09577901 * float(x[39]))+ (-0.085390694 * float(x[40]))+ (-0.023378104 * float(x[41]))+ (0.124721296 * float(x[42]))+ (-0.0039327433 * float(x[43]))+ (-0.044608623 * float(x[44]))+ (0.19808306 * float(x[45]))+ (0.099577956 * float(x[46]))+ (-0.0007033828 * float(x[47]))+ (0.116042696 * float(x[48]))+ (-0.062446453 * float(x[49])))+ ((0.13516547 * float(x[50]))+ (0.19739981 * float(x[51]))+ (0.14072837 * float(x[52]))+ (0.14671607 * float(x[53]))+ (0.078397006 * float(x[54]))+ (0.106656775 * float(x[55]))+ (0.13748287 * float(x[56]))+ (0.116936624 * float(x[57]))+ (-0.08687157 * float(x[58]))+ (-0.120663166 * float(x[59]))+ (0.013690729 * float(x[60]))+ (0.1247065 * float(x[61]))+ (0.2407673 * float(x[62]))+ (0.24245754 * float(x[63]))+ (0.15012616 * float(x[64]))+ (0.20360531 * float(x[65]))+ (0.13227327 * float(x[66]))+ (-0.019713309 * float(x[67]))+ (-0.042506468 * float(x[68]))+ (0.083787456 * float(x[69]))+ (0.011474134 * float(x[70]))+ (-0.053371686 * float(x[71]))+ (0.18938786 * float(x[72]))+ (-0.109013066 * float(x[73]))+ (0.080749355 * float(x[74]))+ (0.36528584 * float(x[75]))+ (0.07304883 * float(x[76]))+ (-0.009695832 * float(x[77]))+ (-0.14754622 * float(x[78]))+ (-0.06957367 * float(x[79]))+ (0.22573517 * float(x[80]))+ (0.36458892 * float(x[81]))+ (-0.116973944 * float(x[82]))+ (0.28311846 * float(x[83]))+ (-0.26316082 * float(x[84]))+ (-0.13048632 * float(x[85]))+ (0.25098518 * float(x[86]))+ (0.20572013 * float(x[87]))+ (-0.15290679 * float(x[88]))+ (-0.009455862 * float(x[89]))+ (-0.46073678 * float(x[90]))+ (-0.47766215 * float(x[91]))+ (0.45039785 * float(x[92]))+ (-0.8664885 * float(x[93]))+ (-0.63670444 * float(x[94]))+ (-0.5295427 * float(x[95]))+ (0.41974685 * float(x[96]))+ (-0.1694418 * float(x[97]))+ (0.1427749 * float(x[98]))+ (0.33145097 * float(x[99])))+ ((0.13178742 * float(x[100]))+ (-0.20597218 * float(x[101]))+ (0.032495216 * float(x[102]))+ (-0.19111499 * float(x[103]))+ (0.2832117 * float(x[104]))+ (-0.18855345 * float(x[105]))+ (-0.11772082 * float(x[106]))+ (-0.15237106 * float(x[107]))+ (-0.10613749 * float(x[108]))+ (-0.010860335 * float(x[109]))+ (-0.026685461 * float(x[110]))+ (-0.0120571405 * float(x[111]))+ (-0.067881405 * float(x[112]))+ (0.07257635 * float(x[113]))+ (-0.033564363 * float(x[114]))+ (0.08814359 * float(x[115]))+ (0.014820218 * float(x[116]))+ (0.099091984 * float(x[117]))+ (0.07607116 * float(x[118]))+ (0.14598341 * float(x[119]))+ (-0.044202853 * float(x[120]))+ (-0.2013184 * float(x[121]))+ (0.14018805 * float(x[122]))+ (-0.1379652 * float(x[123]))+ (0.053727604 * float(x[124]))+ (0.019414859 * float(x[125]))+ (-0.18448332 * float(x[126]))+ (-0.14243 * float(x[127]))+ (0.07457782 * float(x[128]))+ (0.09450479 * float(x[129]))+ (0.18580568 * float(x[130]))+ (0.060826115 * float(x[131]))+ (-0.32481676 * float(x[132]))+ (0.035448954 * float(x[133]))+ (-0.2455864 * float(x[134]))+ (-0.06658824 * float(x[135]))+ (-0.05702411 * float(x[136]))+ (0.07399666 * float(x[137]))+ (0.16785367 * float(x[138]))+ (-0.0849794 * float(x[139]))+ (0.06308496 * float(x[140]))+ (-0.17407511 * float(x[141]))+ (0.19097905 * float(x[142]))+ (0.20961529 * float(x[143]))+ (0.02074871 * float(x[144]))+ (0.064215556 * float(x[145]))+ (0.09779169 * float(x[146]))+ (-0.14573385 * float(x[147]))+ (0.40771228 * float(x[148]))+ (-0.101749755 * float(x[149])))+ ((-0.16312739 * float(x[150]))+ (-0.11565716 * float(x[151]))+ (-0.15809698 * float(x[152]))+ (-0.029309371 * float(x[153]))+ (0.009537938 * float(x[154]))+ (-0.021271266 * float(x[155]))+ (-0.121843494 * float(x[156]))+ (-0.16454038 * float(x[157]))+ (0.082877114 * float(x[158]))+ (-0.061834525 * float(x[159]))+ (0.25326413 * float(x[160]))+ (-0.0798426 * float(x[161]))+ (-0.24838978 * float(x[162]))+ (-0.029085774 * float(x[163]))+ (0.070835404 * float(x[164]))+ (0.13291268 * float(x[165]))+ (-0.0841563 * float(x[166]))+ (0.31467834 * float(x[167]))+ (-0.20273896 * float(x[168]))+ (0.07053534 * float(x[169]))+ (-0.17498107 * float(x[170]))+ (-0.060400873 * float(x[171]))+ (-0.16692868 * float(x[172]))+ (-0.056676675 * float(x[173]))+ (0.03519724 * float(x[174]))+ (-0.11192416 * float(x[175]))+ (0.111293614 * float(x[176]))+ (-0.07927316 * float(x[177]))+ (-0.0979815 * float(x[178]))+ (0.23476985 * float(x[179]))) + 0.30116493), 0)
    h_8 = max((((-0.18677406 * float(x[0]))+ (0.21580581 * float(x[1]))+ (0.12095247 * float(x[2]))+ (0.1087435 * float(x[3]))+ (0.19570404 * float(x[4]))+ (0.19391221 * float(x[5]))+ (-0.011798013 * float(x[6]))+ (-0.20580347 * float(x[7]))+ (-0.07332052 * float(x[8]))+ (0.11368211 * float(x[9]))+ (0.083146125 * float(x[10]))+ (0.14710622 * float(x[11]))+ (0.14849786 * float(x[12]))+ (0.10992609 * float(x[13]))+ (-0.1371288 * float(x[14]))+ (0.14307414 * float(x[15]))+ (0.18107742 * float(x[16]))+ (-0.0028135185 * float(x[17]))+ (0.009047718 * float(x[18]))+ (0.08422853 * float(x[19]))+ (-0.0101754265 * float(x[20]))+ (-0.22380933 * float(x[21]))+ (0.1263545 * float(x[22]))+ (0.16486719 * float(x[23]))+ (-0.21485801 * float(x[24]))+ (-0.20795552 * float(x[25]))+ (-0.07618936 * float(x[26]))+ (0.067697 * float(x[27]))+ (-0.2744048 * float(x[28]))+ (0.11270069 * float(x[29]))+ (0.16148098 * float(x[30]))+ (0.28477633 * float(x[31]))+ (-0.08749266 * float(x[32]))+ (0.033302072 * float(x[33]))+ (-0.028721526 * float(x[34]))+ (0.15545316 * float(x[35]))+ (-0.06837831 * float(x[36]))+ (-0.037530243 * float(x[37]))+ (0.18675072 * float(x[38]))+ (0.17073464 * float(x[39]))+ (-0.0663174 * float(x[40]))+ (-0.14899135 * float(x[41]))+ (-0.09733013 * float(x[42]))+ (-0.010491504 * float(x[43]))+ (-0.058702923 * float(x[44]))+ (0.073025085 * float(x[45]))+ (-0.020637723 * float(x[46]))+ (-0.23370682 * float(x[47]))+ (0.08636642 * float(x[48]))+ (-0.021002801 * float(x[49])))+ ((0.04727933 * float(x[50]))+ (0.15671502 * float(x[51]))+ (0.10989065 * float(x[52]))+ (-0.10894418 * float(x[53]))+ (0.068728946 * float(x[54]))+ (0.28842118 * float(x[55]))+ (0.1165308 * float(x[56]))+ (-0.06375649 * float(x[57]))+ (-0.043642644 * float(x[58]))+ (0.04114102 * float(x[59]))+ (-0.087073855 * float(x[60]))+ (-0.017018016 * float(x[61]))+ (0.013108842 * float(x[62]))+ (0.19021545 * float(x[63]))+ (0.056181595 * float(x[64]))+ (0.116881385 * float(x[65]))+ (0.2690102 * float(x[66]))+ (-0.22871074 * float(x[67]))+ (-0.24743934 * float(x[68]))+ (0.23528582 * float(x[69]))+ (0.14034273 * float(x[70]))+ (0.14751498 * float(x[71]))+ (0.10537767 * float(x[72]))+ (0.048209406 * float(x[73]))+ (-0.010423018 * float(x[74]))+ (0.10586355 * float(x[75]))+ (0.005846024 * float(x[76]))+ (0.1387832 * float(x[77]))+ (-0.06505009 * float(x[78]))+ (0.17141476 * float(x[79]))+ (0.033650417 * float(x[80]))+ (0.061127465 * float(x[81]))+ (0.12865038 * float(x[82]))+ (-0.14760238 * float(x[83]))+ (0.54160774 * float(x[84]))+ (-0.21619791 * float(x[85]))+ (-0.003947259 * float(x[86]))+ (-0.098859526 * float(x[87]))+ (-0.53496766 * float(x[88]))+ (0.6959115 * float(x[89]))+ (-0.704353 * float(x[90]))+ (-0.61685234 * float(x[91]))+ (0.8525504 * float(x[92]))+ (-0.81545913 * float(x[93]))+ (-0.9907545 * float(x[94]))+ (-0.7483279 * float(x[95]))+ (0.45500648 * float(x[96]))+ (-0.2998734 * float(x[97]))+ (0.21058214 * float(x[98]))+ (0.4567503 * float(x[99])))+ ((0.16432698 * float(x[100]))+ (-0.43145958 * float(x[101]))+ (-0.19595404 * float(x[102]))+ (-0.37779003 * float(x[103]))+ (0.6223049 * float(x[104]))+ (-0.43825227 * float(x[105]))+ (-0.16473022 * float(x[106]))+ (-0.21753958 * float(x[107]))+ (0.29089502 * float(x[108]))+ (0.018605731 * float(x[109]))+ (-0.0523855 * float(x[110]))+ (-0.023111496 * float(x[111]))+ (-0.087295264 * float(x[112]))+ (-0.07902113 * float(x[113]))+ (0.049624797 * float(x[114]))+ (0.049366962 * float(x[115]))+ (0.043350317 * float(x[116]))+ (-0.30785096 * float(x[117]))+ (-0.17942348 * float(x[118]))+ (-0.004390089 * float(x[119]))+ (-0.27383795 * float(x[120]))+ (0.15529089 * float(x[121]))+ (0.104424775 * float(x[122]))+ (-0.20986852 * float(x[123]))+ (0.04890754 * float(x[124]))+ (0.21900837 * float(x[125]))+ (-0.098281965 * float(x[126]))+ (0.029014807 * float(x[127]))+ (0.09699566 * float(x[128]))+ (0.08174599 * float(x[129]))+ (0.062668994 * float(x[130]))+ (-0.019140087 * float(x[131]))+ (0.05641527 * float(x[132]))+ (-0.038338933 * float(x[133]))+ (-0.12590568 * float(x[134]))+ (-0.17839159 * float(x[135]))+ (0.093557164 * float(x[136]))+ (-0.19464323 * float(x[137]))+ (0.05584781 * float(x[138]))+ (0.031262357 * float(x[139]))+ (0.17815824 * float(x[140]))+ (-0.09080454 * float(x[141]))+ (-0.01428129 * float(x[142]))+ (0.06777269 * float(x[143]))+ (0.17208932 * float(x[144]))+ (0.049997013 * float(x[145]))+ (0.18785468 * float(x[146]))+ (-0.13154414 * float(x[147]))+ (0.1211986 * float(x[148]))+ (0.1068871 * float(x[149])))+ ((-0.10692254 * float(x[150]))+ (0.027575811 * float(x[151]))+ (-0.04150314 * float(x[152]))+ (-0.15862814 * float(x[153]))+ (-0.037689712 * float(x[154]))+ (-0.20115195 * float(x[155]))+ (-0.13958108 * float(x[156]))+ (-0.094743714 * float(x[157]))+ (0.12538697 * float(x[158]))+ (-0.018527327 * float(x[159]))+ (-0.075780064 * float(x[160]))+ (-0.14413291 * float(x[161]))+ (-0.044872466 * float(x[162]))+ (0.10327228 * float(x[163]))+ (-0.007865003 * float(x[164]))+ (0.036487907 * float(x[165]))+ (0.12514551 * float(x[166]))+ (0.18838179 * float(x[167]))+ (-0.05935291 * float(x[168]))+ (-0.07481246 * float(x[169]))+ (0.1624781 * float(x[170]))+ (0.051153515 * float(x[171]))+ (0.060950503 * float(x[172]))+ (0.09552273 * float(x[173]))+ (0.028853977 * float(x[174]))+ (0.07668953 * float(x[175]))+ (0.017766112 * float(x[176]))+ (-0.17933074 * float(x[177]))+ (-0.22611581 * float(x[178]))+ (0.07071656 * float(x[179]))) + -0.021459227), 0)
    o[0] = (1.0345132 * h_0)+ (0.6794335 * h_1)+ (-0.63719916 * h_2)+ (-0.549475 * h_3)+ (-0.56922823 * h_4)+ (-0.11759058 * h_5)+ (0.42581317 * h_6)+ (0.12426109 * h_7)+ (-1.2362633 * h_8) + 1.7052668
    o[1] = (-0.57774895 * h_0)+ (-0.6362367 * h_1)+ (0.34965366 * h_2)+ (0.279549 * h_3)+ (-0.07644529 * h_4)+ (-0.80176747 * h_5)+ (0.96512765 * h_6)+ (1.0414151 * h_7)+ (0.1942949 * h_8) + 1.1421406
    o[2] = (0.24217854 * h_0)+ (-0.69468397 * h_1)+ (0.7755445 * h_2)+ (0.65151197 * h_3)+ (0.9123719 * h_4)+ (0.98899573 * h_5)+ (-0.85740316 * h_6)+ (-0.40746945 * h_7)+ (-0.48709467 * h_8) + 0.9386313

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
        model_cap=1659
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


