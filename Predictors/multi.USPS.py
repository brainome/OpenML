#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target int0 USPS.csv -o USPS_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:44:14.35. Finished on: Sep-04-2020 13:02:05.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         10-way classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 16.70%
Overall Model accuracy:              97.54% (9070/9298 correct)
Overall Improvement over best guess: 80.84% (of possible 83.3%)
Model capacity (MEC):                3214 bits
Generalization ratio:                2.82 bits/bit
Model efficiency:                    0.02%/parameter
Confusion Matrix:
 [8.82% 0.03% 0.04% 0.00% 0.00% 0.00% 0.03% 0.01% 0.03% 0.00%]
 [0.03% 7.24% 0.09% 0.01% 0.15% 0.01% 0.08% 0.05% 0.01% 0.03%]
 [0.05% 0.06% 8.88% 0.02% 0.00% 0.04% 0.01% 0.01% 0.05% 0.02%]
 [0.00% 0.00% 0.08% 8.30% 0.00% 0.00% 0.00% 0.01% 0.01% 0.12%]
 [0.00% 0.08% 0.01% 0.01% 8.60% 0.01% 0.02% 0.01% 0.11% 0.01%]
 [0.01% 0.00% 0.06% 0.01% 0.00% 13.55% 0.00% 0.01% 0.00% 0.00%]
 [0.02% 0.04% 0.03% 0.01% 0.04% 0.01% 16.51% 0.01% 0.01% 0.01%]
 [0.00% 0.08% 0.02% 0.01% 0.11% 0.01% 0.03% 7.32% 0.02% 0.01%]
 [0.01% 0.04% 0.03% 0.04% 0.06% 0.00% 0.01% 0.08% 9.71% 0.00%]
 [0.00% 0.00% 0.08% 0.09% 0.03% 0.02% 0.00% 0.00% 0.01% 8.60%]
Overfitting:                         No
Note: Labels have been remapped to '7'=0, '6'=1, '5'=2, '8'=3, '4'=4, '2'=5, '1'=6, '9'=7, '3'=8, '10'=9.
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
TRAINFILE = "USPS.csv"


#Number of output logits
num_output_logits = 10

#Number of attributes
num_attr = 256
n_classes = 10


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="int0"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="int0"
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
    clean.mapping={'7': 0, '6': 1, '5': 2, '8': 3, '4': 4, '2': 5, '1': 6, '9': 7, '3': 8, '10': 9}

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
    h_0 = max((((-0.13979213 * float(x[0]))+ (-0.21559022 * float(x[1]))+ (-0.5277819 * float(x[2]))+ (-0.35145432 * float(x[3]))+ (-0.3750054 * float(x[4]))+ (-0.53667796 * float(x[5]))+ (-0.88010746 * float(x[6]))+ (-0.30662695 * float(x[7]))+ (0.26170525 * float(x[8]))+ (0.5297886 * float(x[9]))+ (0.19080828 * float(x[10]))+ (-0.16618961 * float(x[11]))+ (-0.117376596 * float(x[12]))+ (0.07947198 * float(x[13]))+ (-0.29112288 * float(x[14]))+ (-0.30459872 * float(x[15]))+ (0.11067876 * float(x[16]))+ (-0.13098705 * float(x[17]))+ (-0.38802022 * float(x[18]))+ (-0.4536627 * float(x[19]))+ (-0.29449198 * float(x[20]))+ (-0.40975443 * float(x[21]))+ (-0.47927183 * float(x[22]))+ (-0.50210077 * float(x[23]))+ (-0.3517965 * float(x[24]))+ (-0.08809386 * float(x[25]))+ (0.051759876 * float(x[26]))+ (-0.26155886 * float(x[27]))+ (-0.22348371 * float(x[28]))+ (-0.043403026 * float(x[29]))+ (-0.26209563 * float(x[30]))+ (-0.17980555 * float(x[31]))+ (0.24293222 * float(x[32]))+ (0.31157866 * float(x[33]))+ (-0.10892539 * float(x[34]))+ (-0.15037297 * float(x[35]))+ (0.35481647 * float(x[36]))+ (0.3075206 * float(x[37]))+ (-0.5739617 * float(x[38]))+ (0.022166442 * float(x[39]))+ (0.1358903 * float(x[40]))+ (-0.1412929 * float(x[41]))+ (0.3502598 * float(x[42]))+ (0.16898602 * float(x[43]))+ (-0.124577634 * float(x[44]))+ (-0.28260168 * float(x[45]))+ (-0.21512666 * float(x[46]))+ (-0.34172022 * float(x[47]))+ (0.27380168 * float(x[48]))+ (0.30644554 * float(x[49])))+ ((-0.03515042 * float(x[50]))+ (0.06098704 * float(x[51]))+ (0.31576848 * float(x[52]))+ (0.09120344 * float(x[53]))+ (-0.63303936 * float(x[54]))+ (0.2716591 * float(x[55]))+ (-0.15267676 * float(x[56]))+ (-0.51785547 * float(x[57]))+ (0.14354266 * float(x[58]))+ (0.06720089 * float(x[59]))+ (-0.023410242 * float(x[60]))+ (-0.09253016 * float(x[61]))+ (-0.11964952 * float(x[62]))+ (-0.11688474 * float(x[63]))+ (0.10601388 * float(x[64]))+ (-0.033094935 * float(x[65]))+ (-0.22987512 * float(x[66]))+ (-0.41281593 * float(x[67]))+ (0.08450494 * float(x[68]))+ (0.362756 * float(x[69]))+ (-0.07889462 * float(x[70]))+ (0.3877196 * float(x[71]))+ (-0.38803512 * float(x[72]))+ (-0.54432106 * float(x[73]))+ (-0.32965204 * float(x[74]))+ (-0.404747 * float(x[75]))+ (-0.696434 * float(x[76]))+ (-0.71344 * float(x[77]))+ (-0.24001202 * float(x[78]))+ (-0.013080519 * float(x[79]))+ (-0.016044723 * float(x[80]))+ (-0.1945305 * float(x[81]))+ (-0.2788753 * float(x[82]))+ (0.05361585 * float(x[83]))+ (0.7379304 * float(x[84]))+ (0.37673935 * float(x[85]))+ (-0.20036787 * float(x[86]))+ (0.038408168 * float(x[87]))+ (-0.38373753 * float(x[88]))+ (-0.06244721 * float(x[89]))+ (0.032159142 * float(x[90]))+ (-0.39988795 * float(x[91]))+ (-0.94612694 * float(x[92]))+ (-1.0973088 * float(x[93]))+ (-0.57012606 * float(x[94]))+ (-0.08840909 * float(x[95]))+ (0.14924647 * float(x[96]))+ (0.14657155 * float(x[97]))+ (0.29218337 * float(x[98]))+ (0.054630328 * float(x[99])))+ ((0.56939054 * float(x[100]))+ (0.18773898 * float(x[101]))+ (-0.6569899 * float(x[102]))+ (-0.32700878 * float(x[103]))+ (-0.28535292 * float(x[104]))+ (-0.25294685 * float(x[105]))+ (-0.2212504 * float(x[106]))+ (-0.11303262 * float(x[107]))+ (-0.48678014 * float(x[108]))+ (-0.50227505 * float(x[109]))+ (-0.328121 * float(x[110]))+ (0.08760521 * float(x[111]))+ (0.09761571 * float(x[112]))+ (0.4443281 * float(x[113]))+ (0.25357264 * float(x[114]))+ (-0.3010714 * float(x[115]))+ (0.081148475 * float(x[116]))+ (0.12406397 * float(x[117]))+ (-0.5685515 * float(x[118]))+ (-0.37668502 * float(x[119]))+ (-0.46117187 * float(x[120]))+ (-0.95100963 * float(x[121]))+ (-1.1098254 * float(x[122]))+ (-0.49802828 * float(x[123]))+ (-0.2999718 * float(x[124]))+ (0.25763428 * float(x[125]))+ (0.31606197 * float(x[126]))+ (0.3214439 * float(x[127]))+ (0.20128952 * float(x[128]))+ (0.5931458 * float(x[129]))+ (0.35722184 * float(x[130]))+ (-0.22593202 * float(x[131]))+ (0.024583489 * float(x[132]))+ (0.28345725 * float(x[133]))+ (-0.33618724 * float(x[134]))+ (-0.51167697 * float(x[135]))+ (-0.47089309 * float(x[136]))+ (-0.5814443 * float(x[137]))+ (-0.014798365 * float(x[138]))+ (-0.008192876 * float(x[139]))+ (0.23089832 * float(x[140]))+ (0.35152715 * float(x[141]))+ (0.61952865 * float(x[142]))+ (0.29060015 * float(x[143]))+ (-0.016177919 * float(x[144]))+ (0.36601764 * float(x[145]))+ (0.49526387 * float(x[146]))+ (-0.3163787 * float(x[147]))+ (-0.11290441 * float(x[148]))+ (0.026101053 * float(x[149])))+ ((-0.06402993 * float(x[150]))+ (0.16086182 * float(x[151]))+ (-0.21172671 * float(x[152]))+ (0.07807809 * float(x[153]))+ (0.21410584 * float(x[154]))+ (-0.2134223 * float(x[155]))+ (0.37585434 * float(x[156]))+ (0.33477667 * float(x[157]))+ (0.33006355 * float(x[158]))+ (0.14297383 * float(x[159]))+ (-0.015227102 * float(x[160]))+ (-0.09492937 * float(x[161]))+ (0.19730362 * float(x[162]))+ (0.05921921 * float(x[163]))+ (0.21213445 * float(x[164]))+ (-0.035720494 * float(x[165]))+ (-0.18463929 * float(x[166]))+ (-0.26470646 * float(x[167]))+ (-0.5222401 * float(x[168]))+ (0.11183063 * float(x[169]))+ (0.36267596 * float(x[170]))+ (-0.009308231 * float(x[171]))+ (0.13230509 * float(x[172]))+ (0.3452336 * float(x[173]))+ (-0.05734812 * float(x[174]))+ (-0.18190917 * float(x[175]))+ (-0.3177767 * float(x[176]))+ (0.14979038 * float(x[177]))+ (0.47957215 * float(x[178]))+ (0.4856375 * float(x[179]))+ (0.33290464 * float(x[180]))+ (0.07458058 * float(x[181]))+ (-0.07100532 * float(x[182]))+ (-0.69632185 * float(x[183]))+ (-0.8976531 * float(x[184]))+ (0.14211212 * float(x[185]))+ (0.18147029 * float(x[186]))+ (-0.032455955 * float(x[187]))+ (0.2129857 * float(x[188]))+ (0.51314443 * float(x[189]))+ (-0.054459337 * float(x[190]))+ (0.0038775187 * float(x[191]))+ (-0.050770484 * float(x[192]))+ (0.16066836 * float(x[193]))+ (0.5259334 * float(x[194]))+ (0.3093548 * float(x[195]))+ (0.04743602 * float(x[196]))+ (0.12055787 * float(x[197]))+ (-0.20255928 * float(x[198]))+ (-0.39991766 * float(x[199])))+ ((0.5297452 * float(x[200]))+ (0.4043141 * float(x[201]))+ (0.3098513 * float(x[202]))+ (0.32739815 * float(x[203]))+ (0.14817673 * float(x[204]))+ (0.09131353 * float(x[205]))+ (0.08820879 * float(x[206]))+ (0.04597801 * float(x[207]))+ (-0.023460565 * float(x[208]))+ (-0.09148136 * float(x[209]))+ (-0.04175005 * float(x[210]))+ (0.22746633 * float(x[211]))+ (-0.056203496 * float(x[212]))+ (0.25273424 * float(x[213]))+ (0.8793218 * float(x[214]))+ (1.043106 * float(x[215]))+ (1.1885895 * float(x[216]))+ (0.5647888 * float(x[217]))+ (0.56655616 * float(x[218]))+ (0.64947385 * float(x[219]))+ (0.41635394 * float(x[220]))+ (0.18853544 * float(x[221]))+ (-0.045051027 * float(x[222]))+ (0.060511738 * float(x[223]))+ (-0.21228832 * float(x[224]))+ (-0.089489326 * float(x[225]))+ (-0.48505762 * float(x[226]))+ (0.048266187 * float(x[227]))+ (-0.022723358 * float(x[228]))+ (-0.038888197 * float(x[229]))+ (0.84382117 * float(x[230]))+ (0.5953518 * float(x[231]))+ (0.5969239 * float(x[232]))+ (0.3265879 * float(x[233]))+ (0.6738381 * float(x[234]))+ (0.81462604 * float(x[235]))+ (0.534831 * float(x[236]))+ (0.2753026 * float(x[237]))+ (0.062506266 * float(x[238]))+ (-0.13769484 * float(x[239]))+ (-0.29158476 * float(x[240]))+ (-0.17004481 * float(x[241]))+ (-0.17519855 * float(x[242]))+ (0.01436293 * float(x[243]))+ (0.08938072 * float(x[244]))+ (-0.2183506 * float(x[245]))+ (0.93480855 * float(x[246]))+ (1.0393375 * float(x[247]))+ (0.30406302 * float(x[248]))+ (0.1616747 * float(x[249])))+ ((0.36057624 * float(x[250]))+ (0.552009 * float(x[251]))+ (0.43712664 * float(x[252]))+ (0.19269265 * float(x[253]))+ (-0.15874644 * float(x[254]))+ (-0.09706971 * float(x[255]))) + 0.9593867), 0)
    h_1 = max((((-0.39054048 * float(x[0]))+ (-0.3115254 * float(x[1]))+ (-0.1486937 * float(x[2]))+ (-0.38924006 * float(x[3]))+ (-0.9808412 * float(x[4]))+ (-0.88772154 * float(x[5]))+ (-0.1824013 * float(x[6]))+ (-0.4206184 * float(x[7]))+ (-0.35528994 * float(x[8]))+ (0.2723017 * float(x[9]))+ (0.14498946 * float(x[10]))+ (-0.28805622 * float(x[11]))+ (-0.27400392 * float(x[12]))+ (-0.454423 * float(x[13]))+ (-0.28968927 * float(x[14]))+ (-0.2741368 * float(x[15]))+ (-0.047157638 * float(x[16]))+ (-0.1212129 * float(x[17]))+ (0.10538788 * float(x[18]))+ (-0.14941366 * float(x[19]))+ (-0.62888676 * float(x[20]))+ (-0.8140536 * float(x[21]))+ (-0.22783488 * float(x[22]))+ (-0.04256133 * float(x[23]))+ (0.1927795 * float(x[24]))+ (0.32750586 * float(x[25]))+ (0.4483516 * float(x[26]))+ (-0.101826854 * float(x[27]))+ (-0.19775657 * float(x[28]))+ (-0.35064664 * float(x[29]))+ (-0.19098404 * float(x[30]))+ (-0.2402091 * float(x[31]))+ (0.1085553 * float(x[32]))+ (0.06913621 * float(x[33]))+ (-0.039696377 * float(x[34]))+ (0.042357825 * float(x[35]))+ (-0.14246197 * float(x[36]))+ (-0.3431509 * float(x[37]))+ (0.013139023 * float(x[38]))+ (0.2092007 * float(x[39]))+ (-0.030586414 * float(x[40]))+ (-0.3447582 * float(x[41]))+ (0.25246713 * float(x[42]))+ (0.4187211 * float(x[43]))+ (-0.06962704 * float(x[44]))+ (-0.04096671 * float(x[45]))+ (-0.108595125 * float(x[46]))+ (0.11523933 * float(x[47]))+ (-0.034320284 * float(x[48]))+ (-0.17199063 * float(x[49])))+ ((-0.17756216 * float(x[50]))+ (-0.0013854502 * float(x[51]))+ (0.0054912935 * float(x[52]))+ (-0.45852298 * float(x[53]))+ (-0.054615363 * float(x[54]))+ (0.1616817 * float(x[55]))+ (0.047410537 * float(x[56]))+ (-0.21147202 * float(x[57]))+ (0.0062326207 * float(x[58]))+ (0.20564936 * float(x[59]))+ (0.26838604 * float(x[60]))+ (0.16467181 * float(x[61]))+ (0.36656308 * float(x[62]))+ (0.38939932 * float(x[63]))+ (-0.017785072 * float(x[64]))+ (-0.12552314 * float(x[65]))+ (-0.2163932 * float(x[66]))+ (0.1353962 * float(x[67]))+ (-0.31193125 * float(x[68]))+ (-0.24454768 * float(x[69]))+ (-0.012666378 * float(x[70]))+ (-0.5159572 * float(x[71]))+ (-0.17392905 * float(x[72]))+ (-0.27848122 * float(x[73]))+ (-0.27380884 * float(x[74]))+ (-0.20243047 * float(x[75]))+ (0.29313886 * float(x[76]))+ (0.52878714 * float(x[77]))+ (0.44947556 * float(x[78]))+ (0.368725 * float(x[79]))+ (-0.076345965 * float(x[80]))+ (-0.17651334 * float(x[81]))+ (-0.058682296 * float(x[82]))+ (-0.25239304 * float(x[83]))+ (-0.43250686 * float(x[84]))+ (0.17625014 * float(x[85]))+ (0.33559382 * float(x[86]))+ (-0.24246337 * float(x[87]))+ (-0.28071508 * float(x[88]))+ (-0.23209451 * float(x[89]))+ (-0.014266089 * float(x[90]))+ (-0.3108983 * float(x[91]))+ (-0.1874542 * float(x[92]))+ (0.638798 * float(x[93]))+ (0.51721203 * float(x[94]))+ (0.028334888 * float(x[95]))+ (-0.18588504 * float(x[96]))+ (-0.17048663 * float(x[97]))+ (-0.22600828 * float(x[98]))+ (-0.27428523 * float(x[99])))+ ((-0.31390515 * float(x[100]))+ (0.2906482 * float(x[101]))+ (0.43747336 * float(x[102]))+ (0.22096407 * float(x[103]))+ (0.19302882 * float(x[104]))+ (0.30545008 * float(x[105]))+ (0.21384747 * float(x[106]))+ (-0.48608893 * float(x[107]))+ (-0.5647729 * float(x[108]))+ (-0.24570173 * float(x[109]))+ (-0.07059771 * float(x[110]))+ (-0.39053407 * float(x[111]))+ (-0.43929628 * float(x[112]))+ (-0.539115 * float(x[113]))+ (-0.20976143 * float(x[114]))+ (-0.06307409 * float(x[115]))+ (-0.02064331 * float(x[116]))+ (0.071428135 * float(x[117]))+ (-0.066066176 * float(x[118]))+ (0.024043621 * float(x[119]))+ (-0.1518809 * float(x[120]))+ (-0.05629497 * float(x[121]))+ (-0.28704083 * float(x[122]))+ (-0.28239122 * float(x[123]))+ (-0.893004 * float(x[124]))+ (-0.5565111 * float(x[125]))+ (-0.32547584 * float(x[126]))+ (-0.1797993 * float(x[127]))+ (-0.6084555 * float(x[128]))+ (-0.7383782 * float(x[129]))+ (-0.6451821 * float(x[130]))+ (-0.25642997 * float(x[131]))+ (-0.14012983 * float(x[132]))+ (-0.17643727 * float(x[133]))+ (-0.16517925 * float(x[134]))+ (0.35063174 * float(x[135]))+ (0.25834355 * float(x[136]))+ (0.20531854 * float(x[137]))+ (-0.22415839 * float(x[138]))+ (-0.104767 * float(x[139]))+ (-0.4054274 * float(x[140]))+ (-0.094517946 * float(x[141]))+ (0.20328861 * float(x[142]))+ (0.2040818 * float(x[143]))+ (-0.1486534 * float(x[144]))+ (-0.22608447 * float(x[145]))+ (-0.8077064 * float(x[146]))+ (-0.5699845 * float(x[147]))+ (-0.21797028 * float(x[148]))+ (-0.56208235 * float(x[149])))+ ((-0.30424663 * float(x[150]))+ (0.6377431 * float(x[151]))+ (0.3060603 * float(x[152]))+ (-0.215333 * float(x[153]))+ (-0.41038582 * float(x[154]))+ (-0.39046115 * float(x[155]))+ (-0.3968659 * float(x[156]))+ (0.1640313 * float(x[157]))+ (0.84623945 * float(x[158]))+ (0.4082225 * float(x[159]))+ (0.27791613 * float(x[160]))+ (0.15338281 * float(x[161]))+ (-0.27991354 * float(x[162]))+ (-0.21247984 * float(x[163]))+ (0.1923801 * float(x[164]))+ (-0.27900785 * float(x[165]))+ (-0.32347286 * float(x[166]))+ (0.74610656 * float(x[167]))+ (0.14412297 * float(x[168]))+ (-0.5492468 * float(x[169]))+ (-0.39506665 * float(x[170]))+ (0.12931715 * float(x[171]))+ (-0.1250295 * float(x[172]))+ (0.5289481 * float(x[173]))+ (1.0731082 * float(x[174]))+ (0.3146806 * float(x[175]))+ (0.38327572 * float(x[176]))+ (0.14662677 * float(x[177]))+ (-0.25148377 * float(x[178]))+ (-0.016101988 * float(x[179]))+ (0.86040527 * float(x[180]))+ (0.2041313 * float(x[181]))+ (-0.503197 * float(x[182]))+ (0.11750287 * float(x[183]))+ (-0.1522 * float(x[184]))+ (0.010982688 * float(x[185]))+ (0.07944289 * float(x[186]))+ (0.18364298 * float(x[187]))+ (0.5188867 * float(x[188]))+ (0.7621731 * float(x[189]))+ (0.7093369 * float(x[190]))+ (0.29832086 * float(x[191]))+ (-0.1750039 * float(x[192]))+ (-0.014380012 * float(x[193]))+ (-0.34615323 * float(x[194]))+ (-0.17665064 * float(x[195]))+ (0.7713493 * float(x[196]))+ (0.7482213 * float(x[197]))+ (-0.33033022 * float(x[198]))+ (-0.5297711 * float(x[199])))+ ((-0.27194935 * float(x[200]))+ (0.4791783 * float(x[201]))+ (0.094044626 * float(x[202]))+ (-0.11107943 * float(x[203]))+ (0.35812813 * float(x[204]))+ (0.45242897 * float(x[205]))+ (0.27357146 * float(x[206]))+ (-0.10659802 * float(x[207]))+ (-0.33902898 * float(x[208]))+ (-0.30635294 * float(x[209]))+ (-0.07641957 * float(x[210]))+ (-0.3111797 * float(x[211]))+ (-0.027737027 * float(x[212]))+ (0.7761815 * float(x[213]))+ (1.2909195 * float(x[214]))+ (1.0204092 * float(x[215]))+ (0.84917116 * float(x[216]))+ (1.1335461 * float(x[217]))+ (0.3182422 * float(x[218]))+ (0.01840894 * float(x[219]))+ (0.5728301 * float(x[220]))+ (0.549331 * float(x[221]))+ (0.13350664 * float(x[222]))+ (-0.073565565 * float(x[223]))+ (-0.16919953 * float(x[224]))+ (-0.19886081 * float(x[225]))+ (0.022802087 * float(x[226]))+ (0.19940886 * float(x[227]))+ (0.45332304 * float(x[228]))+ (0.75494426 * float(x[229]))+ (1.3636672 * float(x[230]))+ (1.2409008 * float(x[231]))+ (0.5687898 * float(x[232]))+ (0.7429536 * float(x[233]))+ (0.06697539 * float(x[234]))+ (0.1707083 * float(x[235]))+ (0.4078466 * float(x[236]))+ (0.36894155 * float(x[237]))+ (0.19646841 * float(x[238]))+ (-0.14365706 * float(x[239]))+ (-0.086921714 * float(x[240]))+ (-0.03552601 * float(x[241]))+ (-0.046785522 * float(x[242]))+ (0.28672114 * float(x[243]))+ (0.5201164 * float(x[244]))+ (0.7616702 * float(x[245]))+ (0.41247368 * float(x[246]))+ (-0.47704756 * float(x[247]))+ (-0.16284467 * float(x[248]))+ (0.53943485 * float(x[249])))+ ((0.031588648 * float(x[250]))+ (-0.044545908 * float(x[251]))+ (0.21469499 * float(x[252]))+ (0.14340748 * float(x[253]))+ (-0.13130276 * float(x[254]))+ (-0.077130586 * float(x[255]))) + 0.855665), 0)
    h_2 = max((((-0.24214472 * float(x[0]))+ (-0.26876524 * float(x[1]))+ (-0.20310351 * float(x[2]))+ (-0.24779509 * float(x[3]))+ (-0.57592446 * float(x[4]))+ (-0.5609239 * float(x[5]))+ (-0.11189817 * float(x[6]))+ (0.95987296 * float(x[7]))+ (0.28075936 * float(x[8]))+ (-0.388918 * float(x[9]))+ (0.21351328 * float(x[10]))+ (0.5403939 * float(x[11]))+ (0.2133117 * float(x[12]))+ (-0.039657056 * float(x[13]))+ (-0.06347935 * float(x[14]))+ (-0.09394551 * float(x[15]))+ (-0.13617255 * float(x[16]))+ (-0.011250596 * float(x[17]))+ (0.17376493 * float(x[18]))+ (-0.05437823 * float(x[19]))+ (-0.35166585 * float(x[20]))+ (-0.06395412 * float(x[21]))+ (0.46252394 * float(x[22]))+ (0.85732585 * float(x[23]))+ (0.13108806 * float(x[24]))+ (-0.10104812 * float(x[25]))+ (-0.07948692 * float(x[26]))+ (0.70493263 * float(x[27]))+ (0.679735 * float(x[28]))+ (0.25268662 * float(x[29]))+ (-0.006348806 * float(x[30]))+ (-0.26919875 * float(x[31]))+ (0.13867491 * float(x[32]))+ (0.24523324 * float(x[33]))+ (0.31078985 * float(x[34]))+ (0.09043466 * float(x[35]))+ (0.22246133 * float(x[36]))+ (0.095393986 * float(x[37]))+ (0.26709452 * float(x[38]))+ (0.41425994 * float(x[39]))+ (0.3807582 * float(x[40]))+ (0.5050959 * float(x[41]))+ (-0.06302396 * float(x[42]))+ (-0.011939777 * float(x[43]))+ (0.2808297 * float(x[44]))+ (0.14261435 * float(x[45]))+ (-0.0772495 * float(x[46]))+ (-0.33143145 * float(x[47]))+ (0.22797301 * float(x[48]))+ (0.4685321 * float(x[49])))+ ((0.36008233 * float(x[50]))+ (0.46539983 * float(x[51]))+ (0.5685782 * float(x[52]))+ (-0.07038567 * float(x[53]))+ (0.30917305 * float(x[54]))+ (0.6019658 * float(x[55]))+ (0.7894449 * float(x[56]))+ (0.79622704 * float(x[57]))+ (-0.17348222 * float(x[58]))+ (0.13926439 * float(x[59]))+ (0.046798915 * float(x[60]))+ (0.047466233 * float(x[61]))+ (-0.23378162 * float(x[62]))+ (-0.47830257 * float(x[63]))+ (0.049739815 * float(x[64]))+ (0.16440427 * float(x[65]))+ (0.1276617 * float(x[66]))+ (0.3207582 * float(x[67]))+ (0.53521097 * float(x[68]))+ (0.104730554 * float(x[69]))+ (0.2592843 * float(x[70]))+ (-0.11648396 * float(x[71]))+ (-0.2969685 * float(x[72]))+ (-0.23581074 * float(x[73]))+ (-0.24980037 * float(x[74]))+ (0.5340236 * float(x[75]))+ (0.3094839 * float(x[76]))+ (-0.046945557 * float(x[77]))+ (0.056560885 * float(x[78]))+ (-0.03407777 * float(x[79]))+ (-0.070016816 * float(x[80]))+ (0.18812358 * float(x[81]))+ (0.22109778 * float(x[82]))+ (0.14394891 * float(x[83]))+ (-0.13541509 * float(x[84]))+ (-0.015810717 * float(x[85]))+ (-0.21705136 * float(x[86]))+ (-0.71489125 * float(x[87]))+ (-0.36638618 * float(x[88]))+ (-0.11849044 * float(x[89]))+ (-0.42493242 * float(x[90]))+ (0.020667499 * float(x[91]))+ (0.245503 * float(x[92]))+ (0.18354087 * float(x[93]))+ (0.17913724 * float(x[94]))+ (-0.10492929 * float(x[95]))+ (-0.047795612 * float(x[96]))+ (-0.116444595 * float(x[97]))+ (0.11757372 * float(x[98]))+ (-0.23171717 * float(x[99])))+ ((-0.55008817 * float(x[100]))+ (-0.4122012 * float(x[101]))+ (-0.6487515 * float(x[102]))+ (-0.5143935 * float(x[103]))+ (0.31699488 * float(x[104]))+ (0.615435 * float(x[105]))+ (-0.32499644 * float(x[106]))+ (-0.2382695 * float(x[107]))+ (0.05876309 * float(x[108]))+ (-0.07419194 * float(x[109]))+ (0.006371944 * float(x[110]))+ (-0.15661488 * float(x[111]))+ (-0.18919854 * float(x[112]))+ (-0.14204803 * float(x[113]))+ (-0.26918325 * float(x[114]))+ (-0.15267219 * float(x[115]))+ (-0.10085513 * float(x[116]))+ (0.2189223 * float(x[117]))+ (0.21684796 * float(x[118]))+ (-0.15480563 * float(x[119]))+ (-0.28560826 * float(x[120]))+ (-0.2338788 * float(x[121]))+ (-0.67010707 * float(x[122]))+ (-0.4784257 * float(x[123]))+ (-0.13177878 * float(x[124]))+ (-0.29805785 * float(x[125]))+ (-0.3148968 * float(x[126]))+ (-0.2873146 * float(x[127]))+ (-0.34476286 * float(x[128]))+ (-0.12966795 * float(x[129]))+ (0.08634156 * float(x[130]))+ (-0.11266771 * float(x[131]))+ (-0.11872482 * float(x[132]))+ (0.5728204 * float(x[133]))+ (0.7022671 * float(x[134]))+ (0.5980065 * float(x[135]))+ (-0.7828023 * float(x[136]))+ (-0.9184546 * float(x[137]))+ (-0.8643404 * float(x[138]))+ (-0.033742845 * float(x[139]))+ (-0.17966671 * float(x[140]))+ (-0.33714032 * float(x[141]))+ (-0.34530723 * float(x[142]))+ (-0.09116586 * float(x[143]))+ (-0.16472958 * float(x[144]))+ (0.012655595 * float(x[145]))+ (0.13920791 * float(x[146]))+ (-0.122459374 * float(x[147]))+ (-0.4099998 * float(x[148]))+ (0.16894835 * float(x[149])))+ ((-0.22834638 * float(x[150]))+ (0.12991916 * float(x[151]))+ (0.034031782 * float(x[152]))+ (-0.40427303 * float(x[153]))+ (-0.43287653 * float(x[154]))+ (0.25268784 * float(x[155]))+ (-0.15107782 * float(x[156]))+ (-0.2900292 * float(x[157]))+ (-0.32931632 * float(x[158]))+ (-0.12904888 * float(x[159]))+ (-0.03617702 * float(x[160]))+ (0.02469661 * float(x[161]))+ (0.21853608 * float(x[162]))+ (-0.3145321 * float(x[163]))+ (0.056155086 * float(x[164]))+ (0.2149616 * float(x[165]))+ (-0.37391207 * float(x[166]))+ (0.3800183 * float(x[167]))+ (0.6518668 * float(x[168]))+ (0.80811006 * float(x[169]))+ (0.45057732 * float(x[170]))+ (0.018364077 * float(x[171]))+ (-0.073814526 * float(x[172]))+ (-0.065630145 * float(x[173]))+ (-0.25581235 * float(x[174]))+ (-0.16077052 * float(x[175]))+ (0.071368925 * float(x[176]))+ (0.28181687 * float(x[177]))+ (-0.07874961 * float(x[178]))+ (0.0036878258 * float(x[179]))+ (0.85436946 * float(x[180]))+ (0.26339707 * float(x[181]))+ (-0.25579184 * float(x[182]))+ (0.4399763 * float(x[183]))+ (0.018837668 * float(x[184]))+ (0.0637271 * float(x[185]))+ (-0.07175151 * float(x[186]))+ (-0.7397341 * float(x[187]))+ (-0.106200285 * float(x[188]))+ (0.040097654 * float(x[189]))+ (-0.1516103 * float(x[190]))+ (-0.13255753 * float(x[191]))+ (0.1272838 * float(x[192]))+ (-0.19579312 * float(x[193]))+ (-0.23255363 * float(x[194]))+ (-0.06848069 * float(x[195]))+ (0.69579893 * float(x[196]))+ (-0.4072859 * float(x[197]))+ (-0.5505535 * float(x[198]))+ (-0.034503356 * float(x[199])))+ ((-0.7274532 * float(x[200]))+ (-0.5487462 * float(x[201]))+ (-0.10840019 * float(x[202]))+ (-0.16344278 * float(x[203]))+ (0.14342357 * float(x[204]))+ (0.15657568 * float(x[205]))+ (-0.097577706 * float(x[206]))+ (-0.014536375 * float(x[207]))+ (0.008151637 * float(x[208]))+ (-0.10324865 * float(x[209]))+ (-0.49928126 * float(x[210]))+ (-0.3503035 * float(x[211]))+ (0.7555112 * float(x[212]))+ (0.19457626 * float(x[213]))+ (-0.095366 * float(x[214]))+ (0.53935826 * float(x[215]))+ (0.15801533 * float(x[216]))+ (-0.20183006 * float(x[217]))+ (0.23067807 * float(x[218]))+ (0.8562467 * float(x[219]))+ (0.5042808 * float(x[220]))+ (-0.008760418 * float(x[221]))+ (-0.15400885 * float(x[222]))+ (-0.24065465 * float(x[223]))+ (-0.15683015 * float(x[224]))+ (-0.16195641 * float(x[225]))+ (-0.3536254 * float(x[226]))+ (-0.003402947 * float(x[227]))+ (0.6573261 * float(x[228]))+ (0.69397086 * float(x[229]))+ (0.8524669 * float(x[230]))+ (1.0167837 * float(x[231]))+ (0.92294365 * float(x[232]))+ (0.7022212 * float(x[233]))+ (0.51792866 * float(x[234]))+ (0.608811 * float(x[235]))+ (0.3482908 * float(x[236]))+ (-0.20971228 * float(x[237]))+ (-0.2039741 * float(x[238]))+ (-0.09628345 * float(x[239]))+ (-0.2442735 * float(x[240]))+ (-0.14777534 * float(x[241]))+ (-0.31025124 * float(x[242]))+ (-0.101737835 * float(x[243]))+ (0.12791982 * float(x[244]))+ (0.14592434 * float(x[245]))+ (0.36571267 * float(x[246]))+ (0.33251113 * float(x[247]))+ (0.24221516 * float(x[248]))+ (0.1028477 * float(x[249])))+ ((-0.38383973 * float(x[250]))+ (-0.01900065 * float(x[251]))+ (0.2040402 * float(x[252]))+ (-0.16083051 * float(x[253]))+ (-0.21468899 * float(x[254]))+ (-0.07690726 * float(x[255]))) + 0.98608285), 0)
    h_3 = max((((-0.0935231 * float(x[0]))+ (-0.02681428 * float(x[1]))+ (-0.24110381 * float(x[2]))+ (0.111209676 * float(x[3]))+ (0.37465948 * float(x[4]))+ (0.08803393 * float(x[5]))+ (-0.4850135 * float(x[6]))+ (-0.17787084 * float(x[7]))+ (-0.20857413 * float(x[8]))+ (-0.45662776 * float(x[9]))+ (-0.32828125 * float(x[10]))+ (-0.38997784 * float(x[11]))+ (-0.0028547044 * float(x[12]))+ (0.2515792 * float(x[13]))+ (-0.048253577 * float(x[14]))+ (-0.21796991 * float(x[15]))+ (-0.101561345 * float(x[16]))+ (-0.02764816 * float(x[17]))+ (-0.225769 * float(x[18]))+ (0.13200688 * float(x[19]))+ (0.53323895 * float(x[20]))+ (0.030781046 * float(x[21]))+ (-0.57664186 * float(x[22]))+ (0.43760923 * float(x[23]))+ (0.17298102 * float(x[24]))+ (-0.2706328 * float(x[25]))+ (-0.57120734 * float(x[26]))+ (-0.6066816 * float(x[27]))+ (-0.12313301 * float(x[28]))+ (0.13208646 * float(x[29]))+ (-0.24313064 * float(x[30]))+ (-0.24530138 * float(x[31]))+ (-0.06670963 * float(x[32]))+ (-0.05605688 * float(x[33]))+ (-0.24869636 * float(x[34]))+ (-0.3419738 * float(x[35]))+ (0.14433265 * float(x[36]))+ (0.32605964 * float(x[37]))+ (0.041853108 * float(x[38]))+ (0.7134112 * float(x[39]))+ (0.10843775 * float(x[40]))+ (-0.19519833 * float(x[41]))+ (-0.15747489 * float(x[42]))+ (-0.43745935 * float(x[43]))+ (-0.046259806 * float(x[44]))+ (0.09237853 * float(x[45]))+ (-0.034560826 * float(x[46]))+ (-0.31676248 * float(x[47]))+ (-0.16105235 * float(x[48]))+ (0.23926489 * float(x[49])))+ ((-0.30696315 * float(x[50]))+ (-0.731514 * float(x[51]))+ (-0.16154495 * float(x[52]))+ (0.31834108 * float(x[53]))+ (-0.10224458 * float(x[54]))+ (0.50785965 * float(x[55]))+ (-0.13970557 * float(x[56]))+ (-0.15307532 * float(x[57]))+ (-0.06502076 * float(x[58]))+ (0.10248005 * float(x[59]))+ (0.42920056 * float(x[60]))+ (0.6415776 * float(x[61]))+ (0.34669396 * float(x[62]))+ (0.16895321 * float(x[63]))+ (0.18317531 * float(x[64]))+ (0.22893304 * float(x[65]))+ (0.09937397 * float(x[66]))+ (-0.2937891 * float(x[67]))+ (-0.0680229 * float(x[68]))+ (0.046509884 * float(x[69]))+ (0.10297205 * float(x[70]))+ (0.46495208 * float(x[71]))+ (0.44522023 * float(x[72]))+ (0.3628224 * float(x[73]))+ (0.38758627 * float(x[74]))+ (-0.06421502 * float(x[75]))+ (-0.0062942356 * float(x[76]))+ (0.0932168 * float(x[77]))+ (-0.017783798 * float(x[78]))+ (-0.15560232 * float(x[79]))+ (0.13773607 * float(x[80]))+ (0.1702326 * float(x[81]))+ (-0.0077732108 * float(x[82]))+ (-0.26339173 * float(x[83]))+ (0.22909495 * float(x[84]))+ (-0.28326595 * float(x[85]))+ (-0.39344394 * float(x[86]))+ (0.18314542 * float(x[87]))+ (0.47003537 * float(x[88]))+ (0.05983512 * float(x[89]))+ (0.15306962 * float(x[90]))+ (-0.11106243 * float(x[91]))+ (-0.433836 * float(x[92]))+ (-0.804952 * float(x[93]))+ (-0.8076235 * float(x[94]))+ (-0.40107158 * float(x[95]))+ (0.0003859824 * float(x[96]))+ (-0.33523276 * float(x[97]))+ (-0.5286761 * float(x[98]))+ (-0.5139555 * float(x[99])))+ ((0.215261 * float(x[100]))+ (-0.3216999 * float(x[101]))+ (-0.72070915 * float(x[102]))+ (-0.19863674 * float(x[103]))+ (0.082228094 * float(x[104]))+ (-0.5994612 * float(x[105]))+ (-0.3851041 * float(x[106]))+ (0.01373103 * float(x[107]))+ (-0.08610183 * float(x[108]))+ (-0.43813792 * float(x[109]))+ (-0.7280833 * float(x[110]))+ (-0.39919114 * float(x[111]))+ (0.025579542 * float(x[112]))+ (-0.3355116 * float(x[113]))+ (-0.5317297 * float(x[114]))+ (-0.48547167 * float(x[115]))+ (0.19806704 * float(x[116]))+ (0.37941784 * float(x[117]))+ (0.22340183 * float(x[118]))+ (0.6357267 * float(x[119]))+ (0.56284314 * float(x[120]))+ (-0.6150045 * float(x[121]))+ (-0.4407442 * float(x[122]))+ (0.5720603 * float(x[123]))+ (0.3823029 * float(x[124]))+ (0.24954103 * float(x[125]))+ (0.12282388 * float(x[126]))+ (-0.10278531 * float(x[127]))+ (-0.2401806 * float(x[128]))+ (-0.19596823 * float(x[129]))+ (-0.21963653 * float(x[130]))+ (-0.20640455 * float(x[131]))+ (-0.32986456 * float(x[132]))+ (0.0553351 * float(x[133]))+ (-0.3396915 * float(x[134]))+ (0.10878226 * float(x[135]))+ (0.26746845 * float(x[136]))+ (0.04213293 * float(x[137]))+ (-0.029167496 * float(x[138]))+ (0.20627238 * float(x[139]))+ (-0.051463436 * float(x[140]))+ (0.0024821418 * float(x[141]))+ (0.26208842 * float(x[142]))+ (0.22718647 * float(x[143]))+ (-0.08737929 * float(x[144]))+ (0.002747605 * float(x[145]))+ (0.35227197 * float(x[146]))+ (-0.0037565634 * float(x[147]))+ (-0.5759831 * float(x[148]))+ (-0.76086414 * float(x[149])))+ ((-0.854889 * float(x[150]))+ (0.018255504 * float(x[151]))+ (0.1428116 * float(x[152]))+ (0.32035503 * float(x[153]))+ (0.3692545 * float(x[154]))+ (0.12661798 * float(x[155]))+ (0.04380362 * float(x[156]))+ (0.2590253 * float(x[157]))+ (0.49885118 * float(x[158]))+ (0.1483685 * float(x[159]))+ (0.19493027 * float(x[160]))+ (0.47047243 * float(x[161]))+ (0.60621005 * float(x[162]))+ (0.39786944 * float(x[163]))+ (0.38796195 * float(x[164]))+ (0.28141654 * float(x[165]))+ (0.037580673 * float(x[166]))+ (0.11460912 * float(x[167]))+ (0.10803977 * float(x[168]))+ (0.441656 * float(x[169]))+ (-0.12363698 * float(x[170]))+ (-0.4941321 * float(x[171]))+ (0.1377838 * float(x[172]))+ (0.5786099 * float(x[173]))+ (0.025216695 * float(x[174]))+ (-0.14126733 * float(x[175]))+ (0.28829682 * float(x[176]))+ (0.48602813 * float(x[177]))+ (0.3477619 * float(x[178]))+ (0.41235086 * float(x[179]))+ (0.4013815 * float(x[180]))+ (-0.053365476 * float(x[181]))+ (0.045997985 * float(x[182]))+ (0.11227825 * float(x[183]))+ (-0.12573466 * float(x[184]))+ (-0.08084784 * float(x[185]))+ (-0.43750212 * float(x[186]))+ (0.078381486 * float(x[187]))+ (0.11471333 * float(x[188]))+ (0.1389157 * float(x[189]))+ (0.117808886 * float(x[190]))+ (-0.01250126 * float(x[191]))+ (0.2082596 * float(x[192]))+ (0.1653616 * float(x[193]))+ (0.30812111 * float(x[194]))+ (0.39166442 * float(x[195]))+ (-0.032058775 * float(x[196]))+ (-0.22249243 * float(x[197]))+ (-0.19314101 * float(x[198]))+ (0.10934231 * float(x[199])))+ ((-0.1617326 * float(x[200]))+ (-0.33998755 * float(x[201]))+ (0.069682255 * float(x[202]))+ (0.271903 * float(x[203]))+ (-0.027367363 * float(x[204]))+ (-0.12560825 * float(x[205]))+ (0.2175283 * float(x[206]))+ (0.19757393 * float(x[207]))+ (-0.03768005 * float(x[208]))+ (-0.124062024 * float(x[209]))+ (0.22042094 * float(x[210]))+ (0.2696515 * float(x[211]))+ (0.18901388 * float(x[212]))+ (-0.060957532 * float(x[213]))+ (-0.039799392 * float(x[214]))+ (0.31930217 * float(x[215]))+ (0.4284395 * float(x[216]))+ (0.2595469 * float(x[217]))+ (0.12925433 * float(x[218]))+ (0.2917712 * float(x[219]))+ (0.014266202 * float(x[220]))+ (-0.26187426 * float(x[221]))+ (0.110934086 * float(x[222]))+ (0.25805977 * float(x[223]))+ (-0.0936135 * float(x[224]))+ (-0.028596994 * float(x[225]))+ (0.1619526 * float(x[226]))+ (0.09715267 * float(x[227]))+ (0.07433015 * float(x[228]))+ (-0.13693991 * float(x[229]))+ (-0.5413494 * float(x[230]))+ (-0.49281478 * float(x[231]))+ (-0.19287592 * float(x[232]))+ (-0.45180115 * float(x[233]))+ (-0.3660423 * float(x[234]))+ (0.26700482 * float(x[235]))+ (0.23722935 * float(x[236]))+ (0.24325038 * float(x[237]))+ (0.2517764 * float(x[238]))+ (0.1466778 * float(x[239]))+ (-0.045082405 * float(x[240]))+ (0.011999039 * float(x[241]))+ (0.057076193 * float(x[242]))+ (-0.08852908 * float(x[243]))+ (-0.50333875 * float(x[244]))+ (-0.62431216 * float(x[245]))+ (-0.9047032 * float(x[246]))+ (-0.82996494 * float(x[247]))+ (-0.7947748 * float(x[248]))+ (-0.7408836 * float(x[249])))+ ((-0.23159385 * float(x[250]))+ (0.40434837 * float(x[251]))+ (0.6120974 * float(x[252]))+ (0.22462955 * float(x[253]))+ (0.09339184 * float(x[254]))+ (0.20571175 * float(x[255]))) + 0.6803548), 0)
    h_4 = max((((0.1323699 * float(x[0]))+ (0.28329328 * float(x[1]))+ (0.16632245 * float(x[2]))+ (0.112388864 * float(x[3]))+ (0.20242982 * float(x[4]))+ (0.42873344 * float(x[5]))+ (0.095647715 * float(x[6]))+ (-0.41319126 * float(x[7]))+ (-0.27276295 * float(x[8]))+ (-0.102243096 * float(x[9]))+ (0.30234548 * float(x[10]))+ (0.5774866 * float(x[11]))+ (0.40399328 * float(x[12]))+ (0.15029623 * float(x[13]))+ (-0.014587532 * float(x[14]))+ (-0.08709398 * float(x[15]))+ (0.2450403 * float(x[16]))+ (0.41596296 * float(x[17]))+ (0.50740016 * float(x[18]))+ (0.17924426 * float(x[19]))+ (0.43012232 * float(x[20]))+ (0.54903746 * float(x[21]))+ (-0.15842542 * float(x[22]))+ (-0.68211055 * float(x[23]))+ (-0.0425742 * float(x[24]))+ (0.2203845 * float(x[25]))+ (-0.12547094 * float(x[26]))+ (-0.1790978 * float(x[27]))+ (-0.124290444 * float(x[28]))+ (-0.30920988 * float(x[29]))+ (0.04743774 * float(x[30]))+ (0.14485557 * float(x[31]))+ (0.4459213 * float(x[32]))+ (0.4737099 * float(x[33]))+ (0.49627912 * float(x[34]))+ (0.31545454 * float(x[35]))+ (0.785672 * float(x[36]))+ (0.3893379 * float(x[37]))+ (0.10079913 * float(x[38]))+ (0.14526442 * float(x[39]))+ (0.13130906 * float(x[40]))+ (0.5636589 * float(x[41]))+ (-0.014084022 * float(x[42]))+ (-0.33923137 * float(x[43]))+ (-0.5337752 * float(x[44]))+ (-0.40207347 * float(x[45]))+ (0.06678332 * float(x[46]))+ (0.06392102 * float(x[47]))+ (0.47166854 * float(x[48]))+ (0.60950077 * float(x[49])))+ ((0.33686998 * float(x[50]))+ (0.38878787 * float(x[51]))+ (0.31225443 * float(x[52]))+ (0.12819389 * float(x[53]))+ (0.2694559 * float(x[54]))+ (-0.19810496 * float(x[55]))+ (-0.81447875 * float(x[56]))+ (0.035459734 * float(x[57]))+ (0.210273 * float(x[58]))+ (0.15074399 * float(x[59]))+ (-0.027235016 * float(x[60]))+ (-0.054819874 * float(x[61]))+ (-0.09283293 * float(x[62]))+ (-0.002708206 * float(x[63]))+ (0.24085012 * float(x[64]))+ (0.10229795 * float(x[65]))+ (-0.34525853 * float(x[66]))+ (0.07882853 * float(x[67]))+ (0.28292716 * float(x[68]))+ (0.26944202 * float(x[69]))+ (0.56369376 * float(x[70]))+ (-0.19134215 * float(x[71]))+ (-0.97787994 * float(x[72]))+ (-0.4728985 * float(x[73]))+ (-0.052136682 * float(x[74]))+ (0.26344696 * float(x[75]))+ (0.2854422 * float(x[76]))+ (0.03825879 * float(x[77]))+ (0.19220738 * float(x[78]))+ (0.16259457 * float(x[79]))+ (0.09244766 * float(x[80]))+ (0.07983916 * float(x[81]))+ (-0.20527868 * float(x[82]))+ (0.3818925 * float(x[83]))+ (0.9838613 * float(x[84]))+ (0.57317746 * float(x[85]))+ (0.9326137 * float(x[86]))+ (0.34173098 * float(x[87]))+ (-0.17123125 * float(x[88]))+ (-0.18597373 * float(x[89]))+ (-0.49451825 * float(x[90]))+ (-0.29865387 * float(x[91]))+ (-0.17602517 * float(x[92]))+ (0.034174234 * float(x[93]))+ (0.0581019 * float(x[94]))+ (0.13334456 * float(x[95]))+ (0.10668935 * float(x[96]))+ (0.10917303 * float(x[97]))+ (0.0012148513 * float(x[98]))+ (0.35580766 * float(x[99])))+ ((0.17139013 * float(x[100]))+ (-0.15287788 * float(x[101]))+ (-0.0052213278 * float(x[102]))+ (-0.25890326 * float(x[103]))+ (0.21366268 * float(x[104]))+ (0.52123463 * float(x[105]))+ (-0.063683145 * float(x[106]))+ (-0.51217216 * float(x[107]))+ (-0.24224865 * float(x[108]))+ (-0.060608067 * float(x[109]))+ (-0.19758312 * float(x[110]))+ (-0.048650987 * float(x[111]))+ (-0.05385309 * float(x[112]))+ (-0.24091655 * float(x[113]))+ (-0.020782646 * float(x[114]))+ (-0.35341036 * float(x[115]))+ (-0.5282704 * float(x[116]))+ (-0.059769586 * float(x[117]))+ (-0.33318356 * float(x[118]))+ (-0.7127551 * float(x[119]))+ (-0.14491494 * float(x[120]))+ (0.5675182 * float(x[121]))+ (0.18617307 * float(x[122]))+ (-0.2156707 * float(x[123]))+ (-0.11017911 * float(x[124]))+ (-0.20397042 * float(x[125]))+ (-0.3553237 * float(x[126]))+ (-0.16102564 * float(x[127]))+ (-0.26978162 * float(x[128]))+ (-0.3777134 * float(x[129]))+ (-0.14970955 * float(x[130]))+ (-0.45784932 * float(x[131]))+ (-0.7748466 * float(x[132]))+ (0.10720573 * float(x[133]))+ (0.0691528 * float(x[134]))+ (-0.28072202 * float(x[135]))+ (-0.38026023 * float(x[136]))+ (-0.234186 * float(x[137]))+ (-0.0847209 * float(x[138]))+ (-0.24805473 * float(x[139]))+ (-0.3236983 * float(x[140]))+ (-0.4417173 * float(x[141]))+ (-0.3684676 * float(x[142]))+ (-0.34721616 * float(x[143]))+ (-0.321296 * float(x[144]))+ (-0.33353153 * float(x[145]))+ (-0.17714965 * float(x[146]))+ (0.027659832 * float(x[147]))+ (-0.77683944 * float(x[148]))+ (-0.20886022 * float(x[149])))+ ((0.17061605 * float(x[150]))+ (0.20348705 * float(x[151]))+ (0.44598177 * float(x[152]))+ (-0.09175559 * float(x[153]))+ (0.23931712 * float(x[154]))+ (-0.05408199 * float(x[155]))+ (-0.42628047 * float(x[156]))+ (-0.5197674 * float(x[157]))+ (-0.40948442 * float(x[158]))+ (-0.2261682 * float(x[159]))+ (0.04150873 * float(x[160]))+ (-0.2717174 * float(x[161]))+ (0.08711313 * float(x[162]))+ (0.27157503 * float(x[163]))+ (-0.17477621 * float(x[164]))+ (-0.28118 * float(x[165]))+ (0.11991944 * float(x[166]))+ (0.15768498 * float(x[167]))+ (0.07071043 * float(x[168]))+ (0.13179521 * float(x[169]))+ (0.3633809 * float(x[170]))+ (0.14209267 * float(x[171]))+ (0.12626775 * float(x[172]))+ (-0.21912214 * float(x[173]))+ (-0.29825255 * float(x[174]))+ (-0.32536894 * float(x[175]))+ (0.015789935 * float(x[176]))+ (0.12980708 * float(x[177]))+ (0.17876494 * float(x[178]))+ (0.43022177 * float(x[179]))+ (-0.07079243 * float(x[180]))+ (0.002947173 * float(x[181]))+ (0.30893302 * float(x[182]))+ (0.06675745 * float(x[183]))+ (-0.4351741 * float(x[184]))+ (-0.19658695 * float(x[185]))+ (0.12125734 * float(x[186]))+ (0.03431327 * float(x[187]))+ (0.06613851 * float(x[188]))+ (-0.085735366 * float(x[189]))+ (-0.20861255 * float(x[190]))+ (-0.2619653 * float(x[191]))+ (0.20248476 * float(x[192]))+ (0.18077281 * float(x[193]))+ (0.10104491 * float(x[194]))+ (-0.048292153 * float(x[195]))+ (-0.084866956 * float(x[196]))+ (-0.20249014 * float(x[197]))+ (0.5581251 * float(x[198]))+ (0.4892849 * float(x[199])))+ ((0.22694656 * float(x[200]))+ (0.30505177 * float(x[201]))+ (0.40253675 * float(x[202]))+ (0.34977272 * float(x[203]))+ (-0.08829243 * float(x[204]))+ (-0.31769547 * float(x[205]))+ (-0.26791325 * float(x[206]))+ (0.07641243 * float(x[207]))+ (0.0925196 * float(x[208]))+ (0.06720526 * float(x[209]))+ (-0.16503654 * float(x[210]))+ (-0.040733714 * float(x[211]))+ (0.047994547 * float(x[212]))+ (-0.21310377 * float(x[213]))+ (0.15022995 * float(x[214]))+ (0.07475793 * float(x[215]))+ (0.021368247 * float(x[216]))+ (0.15960939 * float(x[217]))+ (0.12278893 * float(x[218]))+ (0.22314763 * float(x[219]))+ (-0.23848137 * float(x[220]))+ (-0.62176543 * float(x[221]))+ (-0.33313638 * float(x[222]))+ (0.10785819 * float(x[223]))+ (-0.099482305 * float(x[224]))+ (-0.036067884 * float(x[225]))+ (-0.26984242 * float(x[226]))+ (-0.20072864 * float(x[227]))+ (-0.1655963 * float(x[228]))+ (-0.010842593 * float(x[229]))+ (-0.28709128 * float(x[230]))+ (-0.6556987 * float(x[231]))+ (-0.2753951 * float(x[232]))+ (-0.09431216 * float(x[233]))+ (-0.01702465 * float(x[234]))+ (0.05930416 * float(x[235]))+ (-0.043377385 * float(x[236]))+ (-0.16646054 * float(x[237]))+ (-0.17067428 * float(x[238]))+ (-0.091924615 * float(x[239]))+ (0.11835745 * float(x[240]))+ (0.095979646 * float(x[241]))+ (-0.055686727 * float(x[242]))+ (-0.18795179 * float(x[243]))+ (-0.32807264 * float(x[244]))+ (-0.15576622 * float(x[245]))+ (0.13143301 * float(x[246]))+ (0.17219545 * float(x[247]))+ (0.10214533 * float(x[248]))+ (-0.2404885 * float(x[249])))+ ((-0.049341224 * float(x[250]))+ (0.11835274 * float(x[251]))+ (0.20619641 * float(x[252]))+ (0.16041857 * float(x[253]))+ (0.17747489 * float(x[254]))+ (0.07847797 * float(x[255]))) + 0.66515785), 0)
    h_5 = max((((-0.08688536 * float(x[0]))+ (-0.110441856 * float(x[1]))+ (-0.10532739 * float(x[2]))+ (-0.20025119 * float(x[3]))+ (-0.06978638 * float(x[4]))+ (-0.2790055 * float(x[5]))+ (0.050641876 * float(x[6]))+ (0.54399437 * float(x[7]))+ (0.8694399 * float(x[8]))+ (1.0680279 * float(x[9]))+ (0.4106554 * float(x[10]))+ (-0.099291906 * float(x[11]))+ (-0.31170574 * float(x[12]))+ (-0.39334857 * float(x[13]))+ (-0.021181468 * float(x[14]))+ (0.05303753 * float(x[15]))+ (-0.27310163 * float(x[16]))+ (-0.19734754 * float(x[17]))+ (-0.3477736 * float(x[18]))+ (-0.033801835 * float(x[19]))+ (-0.17844686 * float(x[20]))+ (-0.2893729 * float(x[21]))+ (0.14050323 * float(x[22]))+ (0.6609724 * float(x[23]))+ (0.37788475 * float(x[24]))+ (0.18669817 * float(x[25]))+ (-0.06364498 * float(x[26]))+ (-0.36730614 * float(x[27]))+ (-0.16188706 * float(x[28]))+ (-0.4870958 * float(x[29]))+ (-0.2575129 * float(x[30]))+ (0.057500005 * float(x[31]))+ (-0.15167373 * float(x[32]))+ (-0.21206143 * float(x[33]))+ (-0.25769547 * float(x[34]))+ (0.016937578 * float(x[35]))+ (0.041067086 * float(x[36]))+ (0.1732169 * float(x[37]))+ (-0.22996458 * float(x[38]))+ (-0.7233038 * float(x[39]))+ (-0.35573328 * float(x[40]))+ (0.34359732 * float(x[41]))+ (0.05361869 * float(x[42]))+ (-0.2949147 * float(x[43]))+ (0.16739091 * float(x[44]))+ (-0.25413075 * float(x[45]))+ (-0.53925407 * float(x[46]))+ (-0.09113937 * float(x[47]))+ (-0.2649982 * float(x[48]))+ (-0.36829293 * float(x[49])))+ ((0.13276808 * float(x[50]))+ (0.112160474 * float(x[51]))+ (-0.058458727 * float(x[52]))+ (0.4329403 * float(x[53]))+ (-0.028454913 * float(x[54]))+ (-0.93856716 * float(x[55]))+ (-0.21513106 * float(x[56]))+ (0.78557485 * float(x[57]))+ (0.42023996 * float(x[58]))+ (0.053566944 * float(x[59]))+ (0.34041733 * float(x[60]))+ (-0.35703698 * float(x[61]))+ (-0.654943 * float(x[62]))+ (-0.24986333 * float(x[63]))+ (-0.3046245 * float(x[64]))+ (-0.2787727 * float(x[65]))+ (0.25331897 * float(x[66]))+ (0.3490087 * float(x[67]))+ (-0.12999238 * float(x[68]))+ (0.12318094 * float(x[69]))+ (0.44063777 * float(x[70]))+ (-0.18766013 * float(x[71]))+ (-0.03189756 * float(x[72]))+ (0.3353253 * float(x[73]))+ (0.69281256 * float(x[74]))+ (0.9535614 * float(x[75]))+ (0.6373168 * float(x[76]))+ (-0.2091914 * float(x[77]))+ (-0.5283058 * float(x[78]))+ (-0.1849146 * float(x[79]))+ (-0.1575486 * float(x[80]))+ (-0.32041082 * float(x[81]))+ (0.16627729 * float(x[82]))+ (0.24664968 * float(x[83]))+ (-0.32364675 * float(x[84]))+ (-0.1438279 * float(x[85]))+ (0.08078419 * float(x[86]))+ (-0.09095242 * float(x[87]))+ (-0.45447177 * float(x[88]))+ (-0.37005785 * float(x[89]))+ (-0.018880112 * float(x[90]))+ (0.47342002 * float(x[91]))+ (0.42431355 * float(x[92]))+ (-0.11751491 * float(x[93]))+ (-0.25935152 * float(x[94]))+ (-0.16500409 * float(x[95]))+ (-0.3015484 * float(x[96]))+ (-0.4183645 * float(x[97]))+ (-0.31030425 * float(x[98]))+ (0.15731782 * float(x[99])))+ ((-0.077029325 * float(x[100]))+ (-0.15667851 * float(x[101]))+ (0.05756592 * float(x[102]))+ (0.061776165 * float(x[103]))+ (-0.38559428 * float(x[104]))+ (-0.4004651 * float(x[105]))+ (-0.2222414 * float(x[106]))+ (-0.17856494 * float(x[107]))+ (0.14779967 * float(x[108]))+ (-0.18264757 * float(x[109]))+ (-0.19142541 * float(x[110]))+ (-0.10600822 * float(x[111]))+ (-0.2354176 * float(x[112]))+ (-0.2781233 * float(x[113]))+ (-0.32518357 * float(x[114]))+ (0.14314954 * float(x[115]))+ (0.037171762 * float(x[116]))+ (0.1032653 * float(x[117]))+ (0.02251671 * float(x[118]))+ (-0.21838215 * float(x[119]))+ (-0.3256836 * float(x[120]))+ (0.06834547 * float(x[121]))+ (0.21202368 * float(x[122]))+ (-0.037146274 * float(x[123]))+ (0.03716347 * float(x[124]))+ (-0.11924313 * float(x[125]))+ (-0.16267759 * float(x[126]))+ (-0.06918093 * float(x[127]))+ (0.15522134 * float(x[128]))+ (-0.18467668 * float(x[129]))+ (0.16115038 * float(x[130]))+ (0.43133232 * float(x[131]))+ (0.070722 * float(x[132]))+ (-0.0864698 * float(x[133]))+ (-0.22510676 * float(x[134]))+ (-0.2904677 * float(x[135]))+ (-0.06391168 * float(x[136]))+ (0.089063846 * float(x[137]))+ (-0.21650234 * float(x[138]))+ (-0.12028833 * float(x[139]))+ (-0.05737202 * float(x[140]))+ (-0.1298124 * float(x[141]))+ (-0.1813061 * float(x[142]))+ (-0.018868024 * float(x[143]))+ (0.20205535 * float(x[144]))+ (0.18356271 * float(x[145]))+ (0.51211375 * float(x[146]))+ (0.914363 * float(x[147]))+ (0.38100007 * float(x[148]))+ (-0.0764381 * float(x[149])))+ ((-0.51836157 * float(x[150]))+ (-0.030676922 * float(x[151]))+ (0.12418447 * float(x[152]))+ (-0.15825446 * float(x[153]))+ (-0.3680701 * float(x[154]))+ (-0.12119804 * float(x[155]))+ (-0.10694645 * float(x[156]))+ (-0.41643927 * float(x[157]))+ (-0.19629265 * float(x[158]))+ (0.09662338 * float(x[159]))+ (0.056351468 * float(x[160]))+ (0.407799 * float(x[161]))+ (0.55311275 * float(x[162]))+ (0.4646045 * float(x[163]))+ (0.53660554 * float(x[164]))+ (0.35243717 * float(x[165]))+ (0.22871415 * float(x[166]))+ (0.20462967 * float(x[167]))+ (0.23958047 * float(x[168]))+ (-0.106396176 * float(x[169]))+ (-0.012766117 * float(x[170]))+ (-0.059103146 * float(x[171]))+ (-0.3188599 * float(x[172]))+ (-0.7385658 * float(x[173]))+ (-0.38098463 * float(x[174]))+ (-0.093252994 * float(x[175]))+ (0.32427403 * float(x[176]))+ (0.25728342 * float(x[177]))+ (0.34550947 * float(x[178]))+ (0.478496 * float(x[179]))+ (0.30263814 * float(x[180]))+ (0.058490343 * float(x[181]))+ (0.35748976 * float(x[182]))+ (0.08888938 * float(x[183]))+ (-0.037482083 * float(x[184]))+ (-0.21858494 * float(x[185]))+ (0.034718733 * float(x[186]))+ (0.051745012 * float(x[187]))+ (-0.5471123 * float(x[188]))+ (-0.42548963 * float(x[189]))+ (-0.06921938 * float(x[190]))+ (0.14346737 * float(x[191]))+ (0.19497441 * float(x[192]))+ (0.3042904 * float(x[193]))+ (0.28906047 * float(x[194]))+ (0.5043963 * float(x[195]))+ (0.6100021 * float(x[196]))+ (0.019061195 * float(x[197]))+ (-0.07780471 * float(x[198]))+ (-0.23714739 * float(x[199])))+ ((-0.2029401 * float(x[200]))+ (-0.39625296 * float(x[201]))+ (-0.056406043 * float(x[202]))+ (0.18269381 * float(x[203]))+ (-0.112957634 * float(x[204]))+ (-0.2936874 * float(x[205]))+ (-0.062789515 * float(x[206]))+ (-0.06863805 * float(x[207]))+ (0.3384081 * float(x[208]))+ (0.56896836 * float(x[209]))+ (0.39538547 * float(x[210]))+ (0.4302972 * float(x[211]))+ (0.5061394 * float(x[212]))+ (0.19951433 * float(x[213]))+ (-0.06725015 * float(x[214]))+ (-0.15294164 * float(x[215]))+ (0.03752797 * float(x[216]))+ (-0.33819968 * float(x[217]))+ (0.13233232 * float(x[218]))+ (0.25501093 * float(x[219]))+ (-0.19577937 * float(x[220]))+ (-0.1797887 * float(x[221]))+ (-0.29575044 * float(x[222]))+ (-0.12443072 * float(x[223]))+ (0.17574628 * float(x[224]))+ (0.5267046 * float(x[225]))+ (0.5892421 * float(x[226]))+ (0.3033393 * float(x[227]))+ (0.14864449 * float(x[228]))+ (0.026580546 * float(x[229]))+ (0.3974623 * float(x[230]))+ (0.41173607 * float(x[231]))+ (0.056602143 * float(x[232]))+ (-0.14537688 * float(x[233]))+ (0.14568025 * float(x[234]))+ (0.03075519 * float(x[235]))+ (-0.4300712 * float(x[236]))+ (-0.19484414 * float(x[237]))+ (-0.13621977 * float(x[238]))+ (0.13250734 * float(x[239]))+ (0.13423772 * float(x[240]))+ (0.24234468 * float(x[241]))+ (0.3227775 * float(x[242]))+ (0.28619885 * float(x[243]))+ (0.06203667 * float(x[244]))+ (-0.11494822 * float(x[245]))+ (0.13566776 * float(x[246]))+ (-0.18225123 * float(x[247]))+ (-0.77200234 * float(x[248]))+ (-0.63924575 * float(x[249])))+ ((-0.44654146 * float(x[250]))+ (-0.3137929 * float(x[251]))+ (-0.15896846 * float(x[252]))+ (-0.062389676 * float(x[253]))+ (-0.05236992 * float(x[254]))+ (0.18596682 * float(x[255]))) + 0.6833981), 0)
    h_6 = max((((-0.14697699 * float(x[0]))+ (0.0028499137 * float(x[1]))+ (-0.23459874 * float(x[2]))+ (-0.49435076 * float(x[3]))+ (-0.31786695 * float(x[4]))+ (0.0057003014 * float(x[5]))+ (0.34483877 * float(x[6]))+ (-0.021892142 * float(x[7]))+ (-0.45977524 * float(x[8]))+ (-1.0318943 * float(x[9]))+ (-0.8113641 * float(x[10]))+ (-0.11735391 * float(x[11]))+ (-0.3338977 * float(x[12]))+ (-0.5065701 * float(x[13]))+ (-0.41637635 * float(x[14]))+ (-0.06951197 * float(x[15]))+ (-0.17635328 * float(x[16]))+ (-0.2483366 * float(x[17]))+ (-0.28174204 * float(x[18]))+ (-0.72433406 * float(x[19]))+ (-0.36926118 * float(x[20]))+ (0.17722379 * float(x[21]))+ (-0.12268507 * float(x[22]))+ (-0.3277784 * float(x[23]))+ (-0.5511822 * float(x[24]))+ (-0.17477916 * float(x[25]))+ (-0.56504256 * float(x[26]))+ (0.1306588 * float(x[27]))+ (-0.170522 * float(x[28]))+ (-0.5055771 * float(x[29]))+ (-0.17579632 * float(x[30]))+ (0.04209409 * float(x[31]))+ (-0.1029562 * float(x[32]))+ (0.12510283 * float(x[33]))+ (0.04873237 * float(x[34]))+ (-0.35633883 * float(x[35]))+ (-0.033564426 * float(x[36]))+ (0.08935682 * float(x[37]))+ (0.096995376 * float(x[38]))+ (0.06610741 * float(x[39]))+ (0.30433953 * float(x[40]))+ (0.52791995 * float(x[41]))+ (0.33828133 * float(x[42]))+ (0.14607255 * float(x[43]))+ (0.092274226 * float(x[44]))+ (-0.24612126 * float(x[45]))+ (-0.12428063 * float(x[46]))+ (0.00642712 * float(x[47]))+ (-0.008086629 * float(x[48]))+ (-0.021738209 * float(x[49])))+ ((0.45162913 * float(x[50]))+ (0.20150307 * float(x[51]))+ (-0.1617158 * float(x[52]))+ (0.41624406 * float(x[53]))+ (0.5517586 * float(x[54]))+ (0.62768555 * float(x[55]))+ (0.3512306 * float(x[56]))+ (0.26569834 * float(x[57]))+ (0.35192713 * float(x[58]))+ (0.1391587 * float(x[59]))+ (-0.0267496 * float(x[60]))+ (-0.31400502 * float(x[61]))+ (-0.22527148 * float(x[62]))+ (-0.076903485 * float(x[63]))+ (-0.38755062 * float(x[64]))+ (-0.42469192 * float(x[65]))+ (-0.09441238 * float(x[66]))+ (-0.0769846 * float(x[67]))+ (-0.25851518 * float(x[68]))+ (0.13815792 * float(x[69]))+ (0.7132654 * float(x[70]))+ (-0.06989294 * float(x[71]))+ (-0.6580114 * float(x[72]))+ (-0.42036852 * float(x[73]))+ (0.28209496 * float(x[74]))+ (0.17948489 * float(x[75]))+ (0.05125999 * float(x[76]))+ (-0.28834355 * float(x[77]))+ (-0.0032617785 * float(x[78]))+ (-0.11673178 * float(x[79]))+ (-0.48627633 * float(x[80]))+ (-0.4471597 * float(x[81]))+ (-0.22556882 * float(x[82]))+ (0.15253325 * float(x[83]))+ (0.4221878 * float(x[84]))+ (0.7570761 * float(x[85]))+ (0.68352324 * float(x[86]))+ (-0.05325571 * float(x[87]))+ (-0.5357038 * float(x[88]))+ (0.28969303 * float(x[89]))+ (0.52818465 * float(x[90]))+ (0.13299438 * float(x[91]))+ (-0.046156477 * float(x[92]))+ (-0.0049493457 * float(x[93]))+ (0.17703567 * float(x[94]))+ (0.2212428 * float(x[95]))+ (-0.4067191 * float(x[96]))+ (-0.14019011 * float(x[97]))+ (0.029361598 * float(x[98]))+ (0.104220256 * float(x[99])))+ ((0.8213681 * float(x[100]))+ (0.6794726 * float(x[101]))+ (0.21394433 * float(x[102]))+ (-0.51205105 * float(x[103]))+ (-0.63551897 * float(x[104]))+ (0.25546885 * float(x[105]))+ (0.45444492 * float(x[106]))+ (0.3238076 * float(x[107]))+ (0.32162446 * float(x[108]))+ (0.32720247 * float(x[109]))+ (0.33060944 * float(x[110]))+ (0.47465765 * float(x[111]))+ (-0.094051294 * float(x[112]))+ (0.073028296 * float(x[113]))+ (-0.07345682 * float(x[114]))+ (-0.028855575 * float(x[115]))+ (0.02468583 * float(x[116]))+ (0.2627308 * float(x[117]))+ (-0.31088707 * float(x[118]))+ (-0.64916813 * float(x[119]))+ (-0.80014205 * float(x[120]))+ (-0.3583532 * float(x[121]))+ (-0.42775416 * float(x[122]))+ (-0.33869916 * float(x[123]))+ (-0.12913093 * float(x[124]))+ (0.21646616 * float(x[125]))+ (0.1622336 * float(x[126]))+ (0.36584005 * float(x[127]))+ (0.48539728 * float(x[128]))+ (0.4515654 * float(x[129]))+ (0.3106032 * float(x[130]))+ (0.0656702 * float(x[131]))+ (0.36811796 * float(x[132]))+ (0.36118475 * float(x[133]))+ (-0.2576732 * float(x[134]))+ (-0.89197546 * float(x[135]))+ (-0.59699404 * float(x[136]))+ (-0.65795314 * float(x[137]))+ (-0.710181 * float(x[138]))+ (-0.18263367 * float(x[139]))+ (-0.08860707 * float(x[140]))+ (0.14367889 * float(x[141]))+ (0.06281234 * float(x[142]))+ (0.26489174 * float(x[143]))+ (0.20135821 * float(x[144]))+ (0.5904709 * float(x[145]))+ (0.49587348 * float(x[146]))+ (0.12915516 * float(x[147]))+ (0.24779539 * float(x[148]))+ (0.54309726 * float(x[149])))+ ((0.3525353 * float(x[150]))+ (-0.0489775 * float(x[151]))+ (-0.30418697 * float(x[152]))+ (-0.45360988 * float(x[153]))+ (-0.043517496 * float(x[154]))+ (0.6203535 * float(x[155]))+ (0.31085163 * float(x[156]))+ (0.34716013 * float(x[157]))+ (0.007665446 * float(x[158]))+ (-0.073246986 * float(x[159]))+ (-0.043687075 * float(x[160]))+ (0.15804483 * float(x[161]))+ (0.29145986 * float(x[162]))+ (-0.05723595 * float(x[163]))+ (0.15228434 * float(x[164]))+ (0.5630118 * float(x[165]))+ (0.43954542 * float(x[166]))+ (0.093273915 * float(x[167]))+ (-0.2120952 * float(x[168]))+ (0.2211855 * float(x[169]))+ (0.47278285 * float(x[170]))+ (0.38209558 * float(x[171]))+ (-0.23012494 * float(x[172]))+ (0.048356973 * float(x[173]))+ (-0.114242 * float(x[174]))+ (-0.2530764 * float(x[175]))+ (-0.19633655 * float(x[176]))+ (-0.021905866 * float(x[177]))+ (0.15207244 * float(x[178]))+ (0.008392206 * float(x[179]))+ (0.25063586 * float(x[180]))+ (0.6246592 * float(x[181]))+ (0.7030987 * float(x[182]))+ (-0.049394727 * float(x[183]))+ (-0.27763996 * float(x[184]))+ (0.26397955 * float(x[185]))+ (0.4792325 * float(x[186]))+ (-0.25167137 * float(x[187]))+ (-0.6043836 * float(x[188]))+ (-0.23423024 * float(x[189]))+ (-0.20351316 * float(x[190]))+ (-0.16564852 * float(x[191]))+ (-0.00809213 * float(x[192]))+ (0.08173825 * float(x[193]))+ (0.4143558 * float(x[194]))+ (0.30646998 * float(x[195]))+ (0.21312965 * float(x[196]))+ (0.48478392 * float(x[197]))+ (0.6625207 * float(x[198]))+ (0.35660693 * float(x[199])))+ ((0.44911602 * float(x[200]))+ (0.635767 * float(x[201]))+ (0.064727284 * float(x[202]))+ (-0.23730831 * float(x[203]))+ (-0.5037324 * float(x[204]))+ (-0.37662026 * float(x[205]))+ (-0.163893 * float(x[206]))+ (-0.026891338 * float(x[207]))+ (0.09735825 * float(x[208]))+ (0.24836093 * float(x[209]))+ (0.05653606 * float(x[210]))+ (0.2534844 * float(x[211]))+ (-0.14627522 * float(x[212]))+ (0.25638172 * float(x[213]))+ (0.48755553 * float(x[214]))+ (0.49513084 * float(x[215]))+ (0.29841948 * float(x[216]))+ (-0.029157741 * float(x[217]))+ (-0.26630887 * float(x[218]))+ (-0.20428982 * float(x[219]))+ (-0.10082022 * float(x[220]))+ (-0.37015697 * float(x[221]))+ (-0.21358374 * float(x[222]))+ (-0.00074815593 * float(x[223]))+ (0.14497085 * float(x[224]))+ (0.14128537 * float(x[225]))+ (0.20053157 * float(x[226]))+ (0.1191679 * float(x[227]))+ (0.17123294 * float(x[228]))+ (-0.016286153 * float(x[229]))+ (-0.2170366 * float(x[230]))+ (-0.46192253 * float(x[231]))+ (0.051177274 * float(x[232]))+ (0.25062045 * float(x[233]))+ (-0.102547795 * float(x[234]))+ (-0.26216376 * float(x[235]))+ (-0.15318944 * float(x[236]))+ (-0.49704763 * float(x[237]))+ (-0.27255803 * float(x[238]))+ (-0.20386012 * float(x[239]))+ (-0.10349304 * float(x[240]))+ (-0.12649603 * float(x[241]))+ (-0.066784404 * float(x[242]))+ (0.1335287 * float(x[243]))+ (0.17252669 * float(x[244]))+ (0.071512476 * float(x[245]))+ (-0.4329888 * float(x[246]))+ (-0.42875597 * float(x[247]))+ (0.401768 * float(x[248]))+ (0.2616923 * float(x[249])))+ ((-0.3160432 * float(x[250]))+ (-0.39443424 * float(x[251]))+ (-0.41622847 * float(x[252]))+ (-0.35829931 * float(x[253]))+ (-0.08020838 * float(x[254]))+ (0.042309448 * float(x[255]))) + 0.73221123), 0)
    h_7 = max((((0.011681974 * float(x[0]))+ (0.046113618 * float(x[1]))+ (0.011635943 * float(x[2]))+ (-0.17022529 * float(x[3]))+ (0.04544728 * float(x[4]))+ (0.016887119 * float(x[5]))+ (-0.3514544 * float(x[6]))+ (-0.4662044 * float(x[7]))+ (-0.90173405 * float(x[8]))+ (-0.69314873 * float(x[9]))+ (0.011901032 * float(x[10]))+ (0.17690617 * float(x[11]))+ (0.14557882 * float(x[12]))+ (0.23111966 * float(x[13]))+ (0.12986927 * float(x[14]))+ (0.09567546 * float(x[15]))+ (0.010258613 * float(x[16]))+ (0.11310068 * float(x[17]))+ (-0.047517423 * float(x[18]))+ (-0.0027776225 * float(x[19]))+ (0.14068067 * float(x[20]))+ (0.26622745 * float(x[21]))+ (-0.051138334 * float(x[22]))+ (-0.41405132 * float(x[23]))+ (-0.35103884 * float(x[24]))+ (0.05995642 * float(x[25]))+ (0.30005398 * float(x[26]))+ (0.39841506 * float(x[27]))+ (0.28063506 * float(x[28]))+ (0.42675507 * float(x[29]))+ (0.14823268 * float(x[30]))+ (0.06905749 * float(x[31]))+ (-0.12817572 * float(x[32]))+ (-0.06970528 * float(x[33]))+ (-0.34472284 * float(x[34]))+ (-0.3821926 * float(x[35]))+ (0.19731574 * float(x[36]))+ (0.54339576 * float(x[37]))+ (-0.17538299 * float(x[38]))+ (-0.13553053 * float(x[39]))+ (0.30355617 * float(x[40]))+ (0.34166056 * float(x[41]))+ (0.22948444 * float(x[42]))+ (0.36716452 * float(x[43]))+ (0.090777084 * float(x[44]))+ (0.22801627 * float(x[45]))+ (-0.038350277 * float(x[46]))+ (-0.1791828 * float(x[47]))+ (-0.06070661 * float(x[48]))+ (-0.14342538 * float(x[49])))+ ((-0.41820377 * float(x[50]))+ (-0.510647 * float(x[51]))+ (-0.2989638 * float(x[52]))+ (-0.069463626 * float(x[53]))+ (-0.70273566 * float(x[54]))+ (-0.20605026 * float(x[55]))+ (-0.32016683 * float(x[56]))+ (-0.73084563 * float(x[57]))+ (-0.35042652 * float(x[58]))+ (-0.31620833 * float(x[59]))+ (-0.5434309 * float(x[60]))+ (-0.50283486 * float(x[61]))+ (-0.6427023 * float(x[62]))+ (-0.36972812 * float(x[63]))+ (-0.015780011 * float(x[64]))+ (-0.043266654 * float(x[65]))+ (-0.27563754 * float(x[66]))+ (-0.30513176 * float(x[67]))+ (-0.1033802 * float(x[68]))+ (-0.19116873 * float(x[69]))+ (-0.25795656 * float(x[70]))+ (-0.06730618 * float(x[71]))+ (-0.40911245 * float(x[72]))+ (-1.1267159 * float(x[73]))+ (-1.0277072 * float(x[74]))+ (-0.5106357 * float(x[75]))+ (-0.22667691 * float(x[76]))+ (-0.20203136 * float(x[77]))+ (-0.6951665 * float(x[78]))+ (-0.59448737 * float(x[79]))+ (-0.028806902 * float(x[80]))+ (0.27040073 * float(x[81]))+ (0.14486852 * float(x[82]))+ (-0.06645843 * float(x[83]))+ (-0.38194802 * float(x[84]))+ (-0.13308458 * float(x[85]))+ (-0.35437244 * float(x[86]))+ (0.0011836146 * float(x[87]))+ (-0.28440008 * float(x[88]))+ (-0.34167996 * float(x[89]))+ (-0.44879264 * float(x[90]))+ (-0.29852226 * float(x[91]))+ (0.11001953 * float(x[92]))+ (0.1410756 * float(x[93]))+ (-0.37480035 * float(x[94]))+ (-0.62158775 * float(x[95]))+ (-0.14892973 * float(x[96]))+ (0.16988629 * float(x[97]))+ (0.14209339 * float(x[98]))+ (-0.28428093 * float(x[99])))+ ((-0.25274795 * float(x[100]))+ (0.049036514 * float(x[101]))+ (-0.14855382 * float(x[102]))+ (-0.14401913 * float(x[103]))+ (0.15366971 * float(x[104]))+ (0.1993224 * float(x[105]))+ (0.48527464 * float(x[106]))+ (0.1427592 * float(x[107]))+ (0.15265833 * float(x[108]))+ (0.08345076 * float(x[109]))+ (-0.13360542 * float(x[110]))+ (-0.41126278 * float(x[111]))+ (-0.23364043 * float(x[112]))+ (0.010851462 * float(x[113]))+ (-0.01243849 * float(x[114]))+ (-0.31298116 * float(x[115]))+ (-0.14677468 * float(x[116]))+ (-0.45705965 * float(x[117]))+ (-0.28084943 * float(x[118]))+ (-0.34029967 * float(x[119]))+ (-0.07012273 * float(x[120]))+ (-0.27284765 * float(x[121]))+ (0.2321879 * float(x[122]))+ (0.31323808 * float(x[123]))+ (0.33096784 * float(x[124]))+ (0.15801698 * float(x[125]))+ (-0.086543694 * float(x[126]))+ (-0.08762499 * float(x[127]))+ (-0.21711424 * float(x[128]))+ (0.06638731 * float(x[129]))+ (0.2668387 * float(x[130]))+ (-0.0018841525 * float(x[131]))+ (0.008823412 * float(x[132]))+ (-0.24895589 * float(x[133]))+ (-0.15061852 * float(x[134]))+ (0.28699663 * float(x[135]))+ (0.6028168 * float(x[136]))+ (-0.51322675 * float(x[137]))+ (-0.14797719 * float(x[138]))+ (0.41914505 * float(x[139]))+ (0.540489 * float(x[140]))+ (0.27158627 * float(x[141]))+ (0.038163632 * float(x[142]))+ (-0.3427051 * float(x[143]))+ (-0.26778442 * float(x[144]))+ (-0.0814611 * float(x[145]))+ (0.4549467 * float(x[146]))+ (0.70814526 * float(x[147]))+ (0.34069386 * float(x[148]))+ (-0.050615318 * float(x[149])))+ ((0.4349252 * float(x[150]))+ (0.9736467 * float(x[151]))+ (1.1958814 * float(x[152]))+ (-0.15646465 * float(x[153]))+ (-0.44880936 * float(x[154]))+ (0.032594163 * float(x[155]))+ (-0.10496277 * float(x[156]))+ (-0.40958473 * float(x[157]))+ (-0.46465516 * float(x[158]))+ (-0.18587343 * float(x[159]))+ (-0.23998065 * float(x[160]))+ (0.06939152 * float(x[161]))+ (0.44965473 * float(x[162]))+ (0.4302904 * float(x[163]))+ (-0.19263175 * float(x[164]))+ (-0.36167756 * float(x[165]))+ (0.36293682 * float(x[166]))+ (0.6952555 * float(x[167]))+ (0.88124895 * float(x[168]))+ (0.3396931 * float(x[169]))+ (0.29252186 * float(x[170]))+ (0.031989045 * float(x[171]))+ (-0.16520488 * float(x[172]))+ (-0.4998737 * float(x[173]))+ (-0.26483282 * float(x[174]))+ (-0.21144141 * float(x[175]))+ (-0.09052052 * float(x[176]))+ (0.098643005 * float(x[177]))+ (0.29362673 * float(x[178]))+ (0.41604277 * float(x[179]))+ (-0.031256035 * float(x[180]))+ (0.15832669 * float(x[181]))+ (0.80695814 * float(x[182]))+ (0.5760722 * float(x[183]))+ (0.19767444 * float(x[184]))+ (-0.14872307 * float(x[185]))+ (0.08926673 * float(x[186]))+ (-0.22854112 * float(x[187]))+ (-0.05733971 * float(x[188]))+ (-0.07598026 * float(x[189]))+ (-0.13700399 * float(x[190]))+ (-0.34789371 * float(x[191]))+ (0.16348405 * float(x[192]))+ (0.3314122 * float(x[193]))+ (0.3343089 * float(x[194]))+ (0.26822063 * float(x[195]))+ (0.4123989 * float(x[196]))+ (0.4450298 * float(x[197]))+ (0.20700198 * float(x[198]))+ (0.5053113 * float(x[199])))+ ((0.4538991 * float(x[200]))+ (-0.18422219 * float(x[201]))+ (0.018303305 * float(x[202]))+ (-0.20525293 * float(x[203]))+ (0.2619537 * float(x[204]))+ (0.04542526 * float(x[205]))+ (-0.32366842 * float(x[206]))+ (-0.34077254 * float(x[207]))+ (0.2073809 * float(x[208]))+ (0.50034106 * float(x[209]))+ (0.20669493 * float(x[210]))+ (-0.16515638 * float(x[211]))+ (-0.2391259 * float(x[212]))+ (0.095630266 * float(x[213]))+ (-0.15625212 * float(x[214]))+ (0.33331954 * float(x[215]))+ (0.7798286 * float(x[216]))+ (0.5733764 * float(x[217]))+ (0.19831434 * float(x[218]))+ (0.03654064 * float(x[219]))+ (0.09122174 * float(x[220]))+ (-0.11960039 * float(x[221]))+ (0.05929763 * float(x[222]))+ (-0.23756729 * float(x[223]))+ (0.22548868 * float(x[224]))+ (0.2622956 * float(x[225]))+ (0.21729895 * float(x[226]))+ (-0.07583329 * float(x[227]))+ (-0.30970967 * float(x[228]))+ (-0.25168127 * float(x[229]))+ (-0.043107487 * float(x[230]))+ (0.27766192 * float(x[231]))+ (0.65112627 * float(x[232]))+ (0.54603696 * float(x[233]))+ (0.37412822 * float(x[234]))+ (0.13034457 * float(x[235]))+ (0.19501466 * float(x[236]))+ (0.2305205 * float(x[237]))+ (0.20143159 * float(x[238]))+ (-0.1715213 * float(x[239]))+ (0.045155283 * float(x[240]))+ (0.019284055 * float(x[241]))+ (0.31320027 * float(x[242]))+ (0.32588872 * float(x[243]))+ (0.11472407 * float(x[244]))+ (0.043133706 * float(x[245]))+ (-0.33502644 * float(x[246]))+ (-0.5355631 * float(x[247]))+ (-0.69153947 * float(x[248]))+ (-0.4388646 * float(x[249])))+ ((-0.3984076 * float(x[250]))+ (-0.10488199 * float(x[251]))+ (0.06189162 * float(x[252]))+ (0.037597153 * float(x[253]))+ (0.11932916 * float(x[254]))+ (-0.1306941 * float(x[255]))) + 0.65813583), 0)
    h_8 = max((((-0.005069298 * float(x[0]))+ (0.04456749 * float(x[1]))+ (0.3923781 * float(x[2]))+ (0.15450317 * float(x[3]))+ (0.108089864 * float(x[4]))+ (0.5237805 * float(x[5]))+ (0.40286222 * float(x[6]))+ (-0.3748428 * float(x[7]))+ (-0.30584067 * float(x[8]))+ (0.83349425 * float(x[9]))+ (0.4408065 * float(x[10]))+ (0.2768141 * float(x[11]))+ (0.2270074 * float(x[12]))+ (0.32910874 * float(x[13]))+ (0.17433172 * float(x[14]))+ (0.17461374 * float(x[15]))+ (-0.126851 * float(x[16]))+ (0.1674246 * float(x[17]))+ (0.42187744 * float(x[18]))+ (0.1982409 * float(x[19]))+ (0.036349654 * float(x[20]))+ (0.12514476 * float(x[21]))+ (0.7725715 * float(x[22]))+ (0.41291076 * float(x[23]))+ (0.38954163 * float(x[24]))+ (0.4215986 * float(x[25]))+ (0.5862666 * float(x[26]))+ (0.12447844 * float(x[27]))+ (0.11584735 * float(x[28]))+ (0.24747719 * float(x[29]))+ (0.23772088 * float(x[30]))+ (-0.013502281 * float(x[31]))+ (-0.06322849 * float(x[32]))+ (0.12907769 * float(x[33]))+ (0.3824438 * float(x[34]))+ (0.0063295728 * float(x[35]))+ (-0.32457864 * float(x[36]))+ (-0.091154434 * float(x[37]))+ (0.5991544 * float(x[38]))+ (0.84393615 * float(x[39]))+ (0.43820414 * float(x[40]))+ (-0.08019068 * float(x[41]))+ (0.06667273 * float(x[42]))+ (0.07216661 * float(x[43]))+ (0.44132212 * float(x[44]))+ (0.17769277 * float(x[45]))+ (-0.11161087 * float(x[46]))+ (0.118711725 * float(x[47]))+ (0.33873302 * float(x[48]))+ (0.41887397 * float(x[49])))+ ((0.5131398 * float(x[50]))+ (-0.2875046 * float(x[51]))+ (-0.27932417 * float(x[52]))+ (0.3049671 * float(x[53]))+ (0.5583002 * float(x[54]))+ (0.002349366 * float(x[55]))+ (-0.1708791 * float(x[56]))+ (-0.25091892 * float(x[57]))+ (-0.5414433 * float(x[58]))+ (-0.5433139 * float(x[59]))+ (-0.095513 * float(x[60]))+ (-0.121412836 * float(x[61]))+ (-0.107553855 * float(x[62]))+ (0.15772215 * float(x[63]))+ (0.4793163 * float(x[64]))+ (0.7209959 * float(x[65]))+ (0.5765421 * float(x[66]))+ (0.059817158 * float(x[67]))+ (-0.11770091 * float(x[68]))+ (0.59896713 * float(x[69]))+ (0.045632657 * float(x[70]))+ (-0.68897724 * float(x[71]))+ (-0.014763872 * float(x[72]))+ (0.2201885 * float(x[73]))+ (-0.21048988 * float(x[74]))+ (-0.87633604 * float(x[75]))+ (-0.68243694 * float(x[76]))+ (-0.46992138 * float(x[77]))+ (-0.074714944 * float(x[78]))+ (0.04211849 * float(x[79]))+ (0.46692497 * float(x[80]))+ (0.18839847 * float(x[81]))+ (0.18203978 * float(x[82]))+ (0.31675383 * float(x[83]))+ (0.19935118 * float(x[84]))+ (0.23939979 * float(x[85]))+ (0.036430474 * float(x[86]))+ (-0.2042778 * float(x[87]))+ (0.018144118 * float(x[88]))+ (0.12791488 * float(x[89]))+ (0.36198947 * float(x[90]))+ (0.25739056 * float(x[91]))+ (-0.33604234 * float(x[92]))+ (-0.3101905 * float(x[93]))+ (-0.23491853 * float(x[94]))+ (0.21465167 * float(x[95]))+ (0.35565397 * float(x[96]))+ (-0.15320252 * float(x[97]))+ (-0.11093045 * float(x[98]))+ (0.11553396 * float(x[99])))+ ((0.0025837896 * float(x[100]))+ (-0.27176252 * float(x[101]))+ (-0.1322554 * float(x[102]))+ (-0.08316495 * float(x[103]))+ (0.15679793 * float(x[104]))+ (-0.26661733 * float(x[105]))+ (0.26719156 * float(x[106]))+ (0.64502 * float(x[107]))+ (0.2306081 * float(x[108]))+ (-0.3192865 * float(x[109]))+ (-0.1542175 * float(x[110]))+ (0.02301603 * float(x[111]))+ (-0.046170216 * float(x[112]))+ (-0.28474334 * float(x[113]))+ (-0.22547205 * float(x[114]))+ (-0.21378107 * float(x[115]))+ (-0.10551538 * float(x[116]))+ (-0.19408187 * float(x[117]))+ (-0.011214647 * float(x[118]))+ (0.072440326 * float(x[119]))+ (-0.057029653 * float(x[120]))+ (-0.43269566 * float(x[121]))+ (-0.10917115 * float(x[122]))+ (0.3739138 * float(x[123]))+ (0.046615742 * float(x[124]))+ (-0.07132786 * float(x[125]))+ (-0.3986826 * float(x[126]))+ (-0.078123055 * float(x[127]))+ (-0.08452254 * float(x[128]))+ (-0.24869582 * float(x[129]))+ (-0.2614113 * float(x[130]))+ (0.24327369 * float(x[131]))+ (0.57262325 * float(x[132]))+ (0.47569814 * float(x[133]))+ (0.19153045 * float(x[134]))+ (0.3604572 * float(x[135]))+ (-0.14469458 * float(x[136]))+ (-0.37787423 * float(x[137]))+ (-0.36003146 * float(x[138]))+ (-0.4844399 * float(x[139]))+ (-0.097783715 * float(x[140]))+ (-0.046074044 * float(x[141]))+ (-0.3480314 * float(x[142]))+ (-0.06181901 * float(x[143]))+ (-0.22956449 * float(x[144]))+ (-0.7132314 * float(x[145]))+ (-0.37916556 * float(x[146]))+ (0.2570922 * float(x[147]))+ (0.47865105 * float(x[148]))+ (0.67069936 * float(x[149])))+ ((0.4018387 * float(x[150]))+ (0.76314217 * float(x[151]))+ (-0.011818617 * float(x[152]))+ (-0.17983851 * float(x[153]))+ (0.27225265 * float(x[154]))+ (-0.20523584 * float(x[155]))+ (-0.096915364 * float(x[156]))+ (-0.015929794 * float(x[157]))+ (-0.287369 * float(x[158]))+ (-0.16954465 * float(x[159]))+ (-0.29067105 * float(x[160]))+ (-0.7462837 * float(x[161]))+ (-0.81187004 * float(x[162]))+ (-0.4644582 * float(x[163]))+ (-0.25387025 * float(x[164]))+ (-0.059936494 * float(x[165]))+ (-0.09907885 * float(x[166]))+ (0.14305902 * float(x[167]))+ (-0.21577951 * float(x[168]))+ (0.30706686 * float(x[169]))+ (0.2163104 * float(x[170]))+ (0.13275103 * float(x[171]))+ (0.09081992 * float(x[172]))+ (-0.08440695 * float(x[173]))+ (-0.17116146 * float(x[174]))+ (0.016273875 * float(x[175]))+ (-0.1889053 * float(x[176]))+ (-0.49675784 * float(x[177]))+ (-0.5337789 * float(x[178]))+ (-0.60087425 * float(x[179]))+ (-0.85224384 * float(x[180]))+ (-0.93300396 * float(x[181]))+ (-1.0425893 * float(x[182]))+ (-0.67984277 * float(x[183]))+ (-0.13235371 * float(x[184]))+ (0.7685106 * float(x[185]))+ (0.5328698 * float(x[186]))+ (0.39786455 * float(x[187]))+ (0.37565306 * float(x[188]))+ (0.1473218 * float(x[189]))+ (0.036190204 * float(x[190]))+ (0.16227624 * float(x[191]))+ (0.18663558 * float(x[192]))+ (-0.11289678 * float(x[193]))+ (-0.0007681645 * float(x[194]))+ (0.12497716 * float(x[195]))+ (-0.4104614 * float(x[196]))+ (-0.68796694 * float(x[197]))+ (-0.8467753 * float(x[198]))+ (-0.7993954 * float(x[199])))+ ((-0.038421996 * float(x[200]))+ (0.5239612 * float(x[201]))+ (0.19064607 * float(x[202]))+ (0.07821451 * float(x[203]))+ (0.2265895 * float(x[204]))+ (0.08455243 * float(x[205]))+ (0.065481015 * float(x[206]))+ (0.14763065 * float(x[207]))+ (-0.031252407 * float(x[208]))+ (0.05088519 * float(x[209]))+ (0.09955073 * float(x[210]))+ (0.07954148 * float(x[211]))+ (-0.0059897895 * float(x[212]))+ (-0.091435 * float(x[213]))+ (-0.23123054 * float(x[214]))+ (-0.52421194 * float(x[215]))+ (0.22316667 * float(x[216]))+ (0.039261658 * float(x[217]))+ (-0.32792726 * float(x[218]))+ (0.22317778 * float(x[219]))+ (0.45393524 * float(x[220]))+ (0.4118285 * float(x[221]))+ (0.037169147 * float(x[222]))+ (0.23584019 * float(x[223]))+ (-0.07023873 * float(x[224]))+ (-0.16227941 * float(x[225]))+ (-0.29003322 * float(x[226]))+ (-0.14811596 * float(x[227]))+ (0.29242417 * float(x[228]))+ (0.4469172 * float(x[229]))+ (-0.06300526 * float(x[230]))+ (-0.37034336 * float(x[231]))+ (0.33879393 * float(x[232]))+ (-0.22994292 * float(x[233]))+ (-0.34227654 * float(x[234]))+ (0.28363302 * float(x[235]))+ (0.407566 * float(x[236]))+ (0.22881195 * float(x[237]))+ (0.15675971 * float(x[238]))+ (0.0064327097 * float(x[239]))+ (-0.042050242 * float(x[240]))+ (-0.16518152 * float(x[241]))+ (-0.06887794 * float(x[242]))+ (-0.24008052 * float(x[243]))+ (0.48590693 * float(x[244]))+ (0.8062789 * float(x[245]))+ (0.7822842 * float(x[246]))+ (-0.16377166 * float(x[247]))+ (-0.38103497 * float(x[248]))+ (0.02088075 * float(x[249])))+ ((0.096236475 * float(x[250]))+ (0.050164632 * float(x[251]))+ (0.18065262 * float(x[252]))+ (0.16995369 * float(x[253]))+ (0.23367923 * float(x[254]))+ (0.03869929 * float(x[255]))) + 0.5125481), 0)
    h_9 = max((((-0.37414292 * float(x[0]))+ (-0.44599423 * float(x[1]))+ (-0.28514996 * float(x[2]))+ (-0.17408423 * float(x[3]))+ (-0.5220658 * float(x[4]))+ (-0.11951234 * float(x[5]))+ (-0.14149024 * float(x[6]))+ (-0.69461673 * float(x[7]))+ (-0.34538618 * float(x[8]))+ (-0.41582975 * float(x[9]))+ (-0.5864203 * float(x[10]))+ (-0.807154 * float(x[11]))+ (-0.6436701 * float(x[12]))+ (-0.403935 * float(x[13]))+ (-0.3858718 * float(x[14]))+ (-0.40448332 * float(x[15]))+ (-0.31617767 * float(x[16]))+ (-0.1277683 * float(x[17]))+ (0.16400006 * float(x[18]))+ (-0.055378646 * float(x[19]))+ (0.2136555 * float(x[20]))+ (0.4742953 * float(x[21]))+ (0.43024117 * float(x[22]))+ (-0.5634189 * float(x[23]))+ (-0.23046398 * float(x[24]))+ (0.25852802 * float(x[25]))+ (0.21117885 * float(x[26]))+ (0.1546055 * float(x[27]))+ (-0.1267895 * float(x[28]))+ (-0.19644274 * float(x[29]))+ (-0.42984533 * float(x[30]))+ (-0.5048862 * float(x[31]))+ (-0.069957085 * float(x[32]))+ (0.14365105 * float(x[33]))+ (0.16664769 * float(x[34]))+ (-0.048909973 * float(x[35]))+ (0.13248165 * float(x[36]))+ (0.33775064 * float(x[37]))+ (0.44988254 * float(x[38]))+ (-0.4847692 * float(x[39]))+ (-0.5146289 * float(x[40]))+ (0.102897584 * float(x[41]))+ (0.37037766 * float(x[42]))+ (0.9311416 * float(x[43]))+ (0.10336264 * float(x[44]))+ (-0.17665765 * float(x[45]))+ (-0.41329172 * float(x[46]))+ (-0.52677655 * float(x[47]))+ (-0.16310805 * float(x[48]))+ (-0.3122606 * float(x[49])))+ ((-0.46515146 * float(x[50]))+ (-0.15530668 * float(x[51]))+ (-0.0056305486 * float(x[52]))+ (-0.3003895 * float(x[53]))+ (-0.06668005 * float(x[54]))+ (-0.21864885 * float(x[55]))+ (-0.18209927 * float(x[56]))+ (0.046418905 * float(x[57]))+ (0.15763514 * float(x[58]))+ (0.6700755 * float(x[59]))+ (0.07575395 * float(x[60]))+ (-0.2000306 * float(x[61]))+ (-0.1265067 * float(x[62]))+ (-0.20242251 * float(x[63]))+ (0.025356537 * float(x[64]))+ (0.26553732 * float(x[65]))+ (0.034569733 * float(x[66]))+ (0.26312378 * float(x[67]))+ (0.33389458 * float(x[68]))+ (-0.38094437 * float(x[69]))+ (-0.1334038 * float(x[70]))+ (-0.37262014 * float(x[71]))+ (0.016616927 * float(x[72]))+ (-0.1509333 * float(x[73]))+ (-0.015420101 * float(x[74]))+ (-0.006026694 * float(x[75]))+ (-0.31711438 * float(x[76]))+ (0.0351884 * float(x[77]))+ (0.17463769 * float(x[78]))+ (-0.23171835 * float(x[79]))+ (-0.0943643 * float(x[80]))+ (0.3051406 * float(x[81]))+ (0.20621094 * float(x[82]))+ (-0.20813482 * float(x[83]))+ (-0.31821647 * float(x[84]))+ (-0.50391966 * float(x[85]))+ (0.04858476 * float(x[86]))+ (0.15111469 * float(x[87]))+ (0.28288934 * float(x[88]))+ (0.27963075 * float(x[89]))+ (0.17322007 * float(x[90]))+ (-0.04037172 * float(x[91]))+ (-0.25607193 * float(x[92]))+ (0.13141143 * float(x[93]))+ (0.1036801 * float(x[94]))+ (-0.21828881 * float(x[95]))+ (-0.0027569006 * float(x[96]))+ (0.16608587 * float(x[97]))+ (0.44978714 * float(x[98]))+ (0.1912529 * float(x[99])))+ ((-0.20323947 * float(x[100]))+ (0.0040223184 * float(x[101]))+ (0.26087672 * float(x[102]))+ (0.19322917 * float(x[103]))+ (0.19754875 * float(x[104]))+ (0.18198141 * float(x[105]))+ (-0.17722677 * float(x[106]))+ (-0.0037140239 * float(x[107]))+ (-0.22294281 * float(x[108]))+ (-0.18553343 * float(x[109]))+ (0.07129362 * float(x[110]))+ (-0.2196896 * float(x[111]))+ (-0.0896594 * float(x[112]))+ (0.0113195 * float(x[113]))+ (0.22455761 * float(x[114]))+ (0.40384865 * float(x[115]))+ (0.19994408 * float(x[116]))+ (0.044694785 * float(x[117]))+ (0.20945539 * float(x[118]))+ (-0.36522627 * float(x[119]))+ (-0.77188843 * float(x[120]))+ (-0.82015055 * float(x[121]))+ (-0.3148125 * float(x[122]))+ (0.07986348 * float(x[123]))+ (-0.19010004 * float(x[124]))+ (-0.2687315 * float(x[125]))+ (-0.050976094 * float(x[126]))+ (-0.019731011 * float(x[127]))+ (-0.04752153 * float(x[128]))+ (0.040993974 * float(x[129]))+ (-0.1933275 * float(x[130]))+ (0.192425 * float(x[131]))+ (-0.054553416 * float(x[132]))+ (-0.19624165 * float(x[133]))+ (0.13397723 * float(x[134]))+ (-0.08649667 * float(x[135]))+ (-0.26887062 * float(x[136]))+ (0.10767419 * float(x[137]))+ (0.043384638 * float(x[138]))+ (-0.1201765 * float(x[139]))+ (-0.008125257 * float(x[140]))+ (0.09576168 * float(x[141]))+ (0.18457896 * float(x[142]))+ (0.20586151 * float(x[143]))+ (-0.2530857 * float(x[144]))+ (0.046365567 * float(x[145]))+ (-0.18193214 * float(x[146]))+ (0.17212033 * float(x[147]))+ (-0.11627617 * float(x[148]))+ (-0.19294715 * float(x[149])))+ ((0.108938314 * float(x[150]))+ (0.13255972 * float(x[151]))+ (0.06349803 * float(x[152]))+ (0.23021899 * float(x[153]))+ (-0.05862692 * float(x[154]))+ (-0.32738137 * float(x[155]))+ (-0.3666798 * float(x[156]))+ (0.0891567 * float(x[157]))+ (0.47056228 * float(x[158]))+ (0.046158735 * float(x[159]))+ (-0.06106682 * float(x[160]))+ (0.22915825 * float(x[161]))+ (0.49325576 * float(x[162]))+ (0.59119415 * float(x[163]))+ (-0.03667911 * float(x[164]))+ (-0.30049208 * float(x[165]))+ (-0.11887409 * float(x[166]))+ (-0.40738544 * float(x[167]))+ (-0.32700288 * float(x[168]))+ (-0.02634882 * float(x[169]))+ (-0.22696446 * float(x[170]))+ (-0.3092672 * float(x[171]))+ (-0.21578625 * float(x[172]))+ (0.16397314 * float(x[173]))+ (0.36832252 * float(x[174]))+ (0.18118747 * float(x[175]))+ (-0.09578125 * float(x[176]))+ (0.17923552 * float(x[177]))+ (0.6463426 * float(x[178]))+ (0.9126201 * float(x[179]))+ (0.26530874 * float(x[180]))+ (-0.07103449 * float(x[181]))+ (0.037691936 * float(x[182]))+ (-0.51544034 * float(x[183]))+ (-0.5971164 * float(x[184]))+ (0.23780426 * float(x[185]))+ (0.5587508 * float(x[186]))+ (-0.029842213 * float(x[187]))+ (-0.0957316 * float(x[188]))+ (0.07972341 * float(x[189]))+ (0.2684819 * float(x[190]))+ (0.08878885 * float(x[191]))+ (-0.2001972 * float(x[192]))+ (0.089093104 * float(x[193]))+ (0.57900923 * float(x[194]))+ (0.5360711 * float(x[195]))+ (0.22804588 * float(x[196]))+ (0.193103 * float(x[197]))+ (0.47644126 * float(x[198]))+ (-0.70742273 * float(x[199])))+ ((-1.0433965 * float(x[200]))+ (0.65764564 * float(x[201]))+ (0.6227641 * float(x[202]))+ (0.14822835 * float(x[203]))+ (0.4456059 * float(x[204]))+ (0.39809957 * float(x[205]))+ (0.23157935 * float(x[206]))+ (-0.11329977 * float(x[207]))+ (-0.30562854 * float(x[208]))+ (-0.061247136 * float(x[209]))+ (0.06834388 * float(x[210]))+ (0.11217246 * float(x[211]))+ (-0.13717727 * float(x[212]))+ (0.46881813 * float(x[213]))+ (0.8045686 * float(x[214]))+ (0.044805985 * float(x[215]))+ (-0.17040555 * float(x[216]))+ (0.5061218 * float(x[217]))+ (0.050055996 * float(x[218]))+ (0.32053605 * float(x[219]))+ (0.74260426 * float(x[220]))+ (0.233851 * float(x[221]))+ (-0.16248107 * float(x[222]))+ (-0.36091515 * float(x[223]))+ (-0.25764143 * float(x[224]))+ (-0.20067239 * float(x[225]))+ (-0.4419351 * float(x[226]))+ (-0.3755169 * float(x[227]))+ (-0.15835904 * float(x[228]))+ (0.7447842 * float(x[229]))+ (0.8331115 * float(x[230]))+ (0.7298065 * float(x[231]))+ (0.6845692 * float(x[232]))+ (0.25130725 * float(x[233]))+ (-0.14690226 * float(x[234]))+ (0.19720294 * float(x[235]))+ (0.5385303 * float(x[236]))+ (0.080782264 * float(x[237]))+ (-0.21629176 * float(x[238]))+ (-0.41488022 * float(x[239]))+ (-0.41880354 * float(x[240]))+ (-0.5176497 * float(x[241]))+ (-0.43686166 * float(x[242]))+ (-0.41052768 * float(x[243]))+ (-0.07533455 * float(x[244]))+ (0.2611853 * float(x[245]))+ (0.11683651 * float(x[246]))+ (-0.2542902 * float(x[247]))+ (0.108687885 * float(x[248]))+ (0.14471212 * float(x[249])))+ ((-0.25092974 * float(x[250]))+ (0.038378697 * float(x[251]))+ (0.17361112 * float(x[252]))+ (-0.009407543 * float(x[253]))+ (-0.3363786 * float(x[254]))+ (-0.22819583 * float(x[255]))) + 0.76547736), 0)
    h_10 = max((((0.012815213 * float(x[0]))+ (0.36439013 * float(x[1]))+ (0.21896824 * float(x[2]))+ (0.07164836 * float(x[3]))+ (0.33815402 * float(x[4]))+ (0.63732946 * float(x[5]))+ (-0.010295243 * float(x[6]))+ (-0.3314965 * float(x[7]))+ (0.7080595 * float(x[8]))+ (0.7217857 * float(x[9]))+ (-0.44520032 * float(x[10]))+ (0.3791529 * float(x[11]))+ (0.36082292 * float(x[12]))+ (-0.055707894 * float(x[13]))+ (-0.07081363 * float(x[14]))+ (0.2138366 * float(x[15]))+ (0.11347973 * float(x[16]))+ (0.009881791 * float(x[17]))+ (0.090840116 * float(x[18]))+ (0.11067269 * float(x[19]))+ (0.6722033 * float(x[20]))+ (0.499893 * float(x[21]))+ (0.06874541 * float(x[22]))+ (0.29565945 * float(x[23]))+ (1.2711134 * float(x[24]))+ (0.9172937 * float(x[25]))+ (-0.22948319 * float(x[26]))+ (0.21097182 * float(x[27]))+ (-0.028032446 * float(x[28]))+ (-0.20071803 * float(x[29]))+ (0.03190235 * float(x[30]))+ (0.36175787 * float(x[31]))+ (-0.22137655 * float(x[32]))+ (-0.12730938 * float(x[33]))+ (-0.29986957 * float(x[34]))+ (-0.36350852 * float(x[35]))+ (0.07209157 * float(x[36]))+ (0.37152845 * float(x[37]))+ (0.20121145 * float(x[38]))+ (0.5510889 * float(x[39]))+ (0.7193195 * float(x[40]))+ (0.4197947 * float(x[41]))+ (0.0031232461 * float(x[42]))+ (0.4501526 * float(x[43]))+ (0.20292342 * float(x[44]))+ (-0.1066453 * float(x[45]))+ (0.4233642 * float(x[46]))+ (0.63282675 * float(x[47]))+ (-0.065450765 * float(x[48]))+ (-0.2905709 * float(x[49])))+ ((-0.520374 * float(x[50]))+ (-0.20120661 * float(x[51]))+ (-0.058903508 * float(x[52]))+ (0.46959123 * float(x[53]))+ (0.111953475 * float(x[54]))+ (0.00836957 * float(x[55]))+ (0.21532603 * float(x[56]))+ (0.12353478 * float(x[57]))+ (0.66856927 * float(x[58]))+ (0.7705771 * float(x[59]))+ (0.76644766 * float(x[60]))+ (0.5676433 * float(x[61]))+ (0.6550143 * float(x[62]))+ (0.4869074 * float(x[63]))+ (0.08002602 * float(x[64]))+ (-0.28172398 * float(x[65]))+ (-0.5387022 * float(x[66]))+ (0.039656356 * float(x[67]))+ (-0.24168195 * float(x[68]))+ (-0.4638101 * float(x[69]))+ (-1.3477226 * float(x[70]))+ (-1.320433 * float(x[71]))+ (-0.50019014 * float(x[72]))+ (0.30104053 * float(x[73]))+ (0.60146993 * float(x[74]))+ (0.05031431 * float(x[75]))+ (0.3082187 * float(x[76]))+ (0.7016053 * float(x[77]))+ (0.4039651 * float(x[78]))+ (0.3688355 * float(x[79]))+ (0.25607675 * float(x[80]))+ (0.004977483 * float(x[81]))+ (-0.21359533 * float(x[82]))+ (0.06378251 * float(x[83]))+ (0.17058031 * float(x[84]))+ (-0.4071631 * float(x[85]))+ (-1.2029363 * float(x[86]))+ (-1.0000749 * float(x[87]))+ (-0.5110905 * float(x[88]))+ (-0.042820975 * float(x[89]))+ (-0.23043974 * float(x[90]))+ (-0.89870876 * float(x[91]))+ (-0.17355521 * float(x[92]))+ (0.36734056 * float(x[93]))+ (0.15046014 * float(x[94]))+ (-0.13046627 * float(x[95]))+ (0.06974369 * float(x[96]))+ (0.028688952 * float(x[97]))+ (-0.27417737 * float(x[98]))+ (0.030068964 * float(x[99])))+ ((-0.17926514 * float(x[100]))+ (-0.62917376 * float(x[101]))+ (-0.92010105 * float(x[102]))+ (-0.37760773 * float(x[103]))+ (-0.54928225 * float(x[104]))+ (-0.24001998 * float(x[105]))+ (0.35469058 * float(x[106]))+ (-0.20919502 * float(x[107]))+ (0.06124936 * float(x[108]))+ (0.088136815 * float(x[109]))+ (-0.31581813 * float(x[110]))+ (-0.20970379 * float(x[111]))+ (0.113448665 * float(x[112]))+ (-0.12877874 * float(x[113]))+ (-0.24534306 * float(x[114]))+ (-0.39980066 * float(x[115]))+ (-0.7500786 * float(x[116]))+ (-0.6132644 * float(x[117]))+ (-0.75136554 * float(x[118]))+ (-0.22514217 * float(x[119]))+ (-0.28929868 * float(x[120]))+ (-0.37480724 * float(x[121]))+ (0.3890006 * float(x[122]))+ (0.02608498 * float(x[123]))+ (-0.18163006 * float(x[124]))+ (-0.011704367 * float(x[125]))+ (-0.28524375 * float(x[126]))+ (-0.090336405 * float(x[127]))+ (-0.104172915 * float(x[128]))+ (-0.30456746 * float(x[129]))+ (-0.42456737 * float(x[130]))+ (-0.31962243 * float(x[131]))+ (-0.26098776 * float(x[132]))+ (-0.36250165 * float(x[133]))+ (-0.42823702 * float(x[134]))+ (-0.3039104 * float(x[135]))+ (-0.3980557 * float(x[136]))+ (-0.7302758 * float(x[137]))+ (-0.46464825 * float(x[138]))+ (-0.32224953 * float(x[139]))+ (-0.17168988 * float(x[140]))+ (0.08531321 * float(x[141]))+ (-0.16842538 * float(x[142]))+ (0.0974269 * float(x[143]))+ (-0.18987462 * float(x[144]))+ (-0.32217827 * float(x[145]))+ (-0.3638707 * float(x[146]))+ (-0.094661154 * float(x[147]))+ (-0.2301555 * float(x[148]))+ (-0.73167497 * float(x[149])))+ ((-0.3524909 * float(x[150]))+ (-0.15521553 * float(x[151]))+ (0.07177409 * float(x[152]))+ (-0.59894985 * float(x[153]))+ (-0.17284189 * float(x[154]))+ (-0.11849335 * float(x[155]))+ (-0.48826525 * float(x[156]))+ (-0.06761856 * float(x[157]))+ (-0.008969965 * float(x[158]))+ (0.039114844 * float(x[159]))+ (-0.26533002 * float(x[160]))+ (-0.40095332 * float(x[161]))+ (-0.17956865 * float(x[162]))+ (-0.041230798 * float(x[163]))+ (-0.32888445 * float(x[164]))+ (-0.4851197 * float(x[165]))+ (-0.2697395 * float(x[166]))+ (-0.2779091 * float(x[167]))+ (-0.032172997 * float(x[168]))+ (-0.3807372 * float(x[169]))+ (-0.049750414 * float(x[170]))+ (0.037922602 * float(x[171]))+ (-0.24980938 * float(x[172]))+ (-0.23950309 * float(x[173]))+ (-0.027467474 * float(x[174]))+ (-0.21445444 * float(x[175]))+ (-0.29182163 * float(x[176]))+ (-0.479384 * float(x[177]))+ (-0.23828958 * float(x[178]))+ (0.10078036 * float(x[179]))+ (-0.3369637 * float(x[180]))+ (-0.60969937 * float(x[181]))+ (-0.61943847 * float(x[182]))+ (-0.7581849 * float(x[183]))+ (0.011153901 * float(x[184]))+ (-0.4790432 * float(x[185]))+ (0.18821812 * float(x[186]))+ (0.4919472 * float(x[187]))+ (-0.18417245 * float(x[188]))+ (0.026408838 * float(x[189]))+ (-0.016774189 * float(x[190]))+ (-0.17659493 * float(x[191]))+ (-0.14128444 * float(x[192]))+ (-0.26481673 * float(x[193]))+ (-0.235974 * float(x[194]))+ (0.0025249962 * float(x[195]))+ (0.096505076 * float(x[196]))+ (-0.055876993 * float(x[197]))+ (-0.094818585 * float(x[198]))+ (-0.1642704 * float(x[199])))+ ((0.75673676 * float(x[200]))+ (0.18531683 * float(x[201]))+ (0.15030268 * float(x[202]))+ (0.48315272 * float(x[203]))+ (0.040021766 * float(x[204]))+ (0.28199637 * float(x[205]))+ (0.055862408 * float(x[206]))+ (-0.17262137 * float(x[207]))+ (-0.15231742 * float(x[208]))+ (-0.25441998 * float(x[209]))+ (0.009245901 * float(x[210]))+ (0.18551402 * float(x[211]))+ (0.32968935 * float(x[212]))+ (0.3903295 * float(x[213]))+ (0.31135243 * float(x[214]))+ (0.77304864 * float(x[215]))+ (1.3380799 * float(x[216]))+ (0.38607615 * float(x[217]))+ (-0.0657373 * float(x[218]))+ (0.24394235 * float(x[219]))+ (0.4871268 * float(x[220]))+ (0.59059423 * float(x[221]))+ (0.2817883 * float(x[222]))+ (0.106354125 * float(x[223]))+ (0.21514156 * float(x[224]))+ (0.09571816 * float(x[225]))+ (-0.1252348 * float(x[226]))+ (0.13592333 * float(x[227]))+ (0.43575966 * float(x[228]))+ (0.4327383 * float(x[229]))+ (0.1871258 * float(x[230]))+ (-0.025351888 * float(x[231]))+ (0.5690699 * float(x[232]))+ (0.141318 * float(x[233]))+ (0.039176926 * float(x[234]))+ (0.09973829 * float(x[235]))+ (0.44995525 * float(x[236]))+ (0.27420688 * float(x[237]))+ (0.063185036 * float(x[238]))+ (0.011699242 * float(x[239]))+ (0.13048732 * float(x[240]))+ (0.0064401957 * float(x[241]))+ (0.0025334626 * float(x[242]))+ (0.0831842 * float(x[243]))+ (0.4400294 * float(x[244]))+ (0.23279019 * float(x[245]))+ (-0.18237449 * float(x[246]))+ (-0.29040393 * float(x[247]))+ (0.17432487 * float(x[248]))+ (0.43755972 * float(x[249])))+ ((-0.093341865 * float(x[250]))+ (0.08651118 * float(x[251]))+ (0.19002014 * float(x[252]))+ (0.004959957 * float(x[253]))+ (0.113133356 * float(x[254]))+ (0.17253323 * float(x[255]))) + 0.34405926), 0)
    h_11 = max((((-0.37892818 * float(x[0]))+ (-0.5359066 * float(x[1]))+ (-0.12322863 * float(x[2]))+ (0.0640763 * float(x[3]))+ (0.17276594 * float(x[4]))+ (0.10564559 * float(x[5]))+ (-0.13578843 * float(x[6]))+ (0.4162366 * float(x[7]))+ (0.5868077 * float(x[8]))+ (0.7279111 * float(x[9]))+ (0.48943254 * float(x[10]))+ (0.17125578 * float(x[11]))+ (-0.06858468 * float(x[12]))+ (-0.049675066 * float(x[13]))+ (-0.3412794 * float(x[14]))+ (-0.3346598 * float(x[15]))+ (-0.45137507 * float(x[16]))+ (-0.6309593 * float(x[17]))+ (0.017736321 * float(x[18]))+ (0.7197913 * float(x[19]))+ (0.61809003 * float(x[20]))+ (0.0855581 * float(x[21]))+ (0.5168944 * float(x[22]))+ (1.0783832 * float(x[23]))+ (0.7589005 * float(x[24]))+ (0.41230735 * float(x[25]))+ (0.9855174 * float(x[26]))+ (0.34921974 * float(x[27]))+ (0.061550967 * float(x[28]))+ (-0.054443456 * float(x[29]))+ (-0.24966235 * float(x[30]))+ (-0.34312612 * float(x[31]))+ (-0.43469825 * float(x[32]))+ (-0.36028507 * float(x[33]))+ (0.06736696 * float(x[34]))+ (0.83816314 * float(x[35]))+ (0.3964415 * float(x[36]))+ (0.27800825 * float(x[37]))+ (0.71631426 * float(x[38]))+ (0.76524144 * float(x[39]))+ (0.5710595 * float(x[40]))+ (-0.40111378 * float(x[41]))+ (0.3512261 * float(x[42]))+ (0.38832545 * float(x[43]))+ (0.2520257 * float(x[44]))+ (-0.19445556 * float(x[45]))+ (-0.29543388 * float(x[46]))+ (-0.2755433 * float(x[47]))+ (-0.26712736 * float(x[48]))+ (-0.11086167 * float(x[49])))+ ((0.54101986 * float(x[50]))+ (0.6495918 * float(x[51]))+ (0.5720777 * float(x[52]))+ (0.41118166 * float(x[53]))+ (0.5169247 * float(x[54]))+ (0.15150489 * float(x[55]))+ (0.18269685 * float(x[56]))+ (-0.16421866 * float(x[57]))+ (0.17282422 * float(x[58]))+ (0.4248105 * float(x[59]))+ (-0.05402557 * float(x[60]))+ (-0.11385987 * float(x[61]))+ (-0.3068901 * float(x[62]))+ (-0.259869 * float(x[63]))+ (-0.26584685 * float(x[64]))+ (0.1088379 * float(x[65]))+ (0.8657154 * float(x[66]))+ (0.63526225 * float(x[67]))+ (0.1975015 * float(x[68]))+ (0.4835704 * float(x[69]))+ (0.33017245 * float(x[70]))+ (0.46923044 * float(x[71]))+ (0.46703905 * float(x[72]))+ (0.49549788 * float(x[73]))+ (0.5644177 * float(x[74]))+ (0.6226602 * float(x[75]))+ (0.01332802 * float(x[76]))+ (0.15346673 * float(x[77]))+ (-0.08239603 * float(x[78]))+ (-0.2589084 * float(x[79]))+ (-0.5958193 * float(x[80]))+ (0.04119054 * float(x[81]))+ (0.3689989 * float(x[82]))+ (0.51963615 * float(x[83]))+ (0.07371254 * float(x[84]))+ (0.2359446 * float(x[85]))+ (0.26466817 * float(x[86]))+ (0.17562145 * float(x[87]))+ (0.11747399 * float(x[88]))+ (0.15429768 * float(x[89]))+ (0.34298927 * float(x[90]))+ (0.75758725 * float(x[91]))+ (0.37193537 * float(x[92]))+ (0.24583717 * float(x[93]))+ (0.062094335 * float(x[94]))+ (-0.28984502 * float(x[95]))+ (-0.3330322 * float(x[96]))+ (0.018674687 * float(x[97]))+ (0.4605677 * float(x[98]))+ (0.7624791 * float(x[99])))+ ((0.45882022 * float(x[100]))+ (0.33670986 * float(x[101]))+ (0.17080429 * float(x[102]))+ (-0.055688046 * float(x[103]))+ (-0.29105353 * float(x[104]))+ (-0.32337195 * float(x[105]))+ (-0.09619312 * float(x[106]))+ (0.030332258 * float(x[107]))+ (0.026747802 * float(x[108]))+ (0.08380754 * float(x[109]))+ (-0.1140243 * float(x[110]))+ (-0.45423838 * float(x[111]))+ (-0.33018798 * float(x[112]))+ (-0.052125707 * float(x[113]))+ (0.018591644 * float(x[114]))+ (0.4554977 * float(x[115]))+ (0.26921448 * float(x[116]))+ (-0.115634315 * float(x[117]))+ (-0.23423143 * float(x[118]))+ (-0.3383948 * float(x[119]))+ (-0.37301934 * float(x[120]))+ (-0.2662087 * float(x[121]))+ (-0.1696153 * float(x[122]))+ (-0.13254586 * float(x[123]))+ (-0.047509275 * float(x[124]))+ (-0.09881218 * float(x[125]))+ (-0.07615242 * float(x[126]))+ (-0.45294032 * float(x[127]))+ (-0.51613253 * float(x[128]))+ (-0.13755263 * float(x[129]))+ (-0.09755587 * float(x[130]))+ (-0.07706361 * float(x[131]))+ (-0.18861702 * float(x[132]))+ (-0.7109175 * float(x[133]))+ (-0.029894946 * float(x[134]))+ (0.081267305 * float(x[135]))+ (-0.17397188 * float(x[136]))+ (0.36866897 * float(x[137]))+ (0.15990745 * float(x[138]))+ (0.021081902 * float(x[139]))+ (0.51849735 * float(x[140]))+ (0.13809517 * float(x[141]))+ (-0.03193973 * float(x[142]))+ (-0.60535306 * float(x[143]))+ (-0.46664414 * float(x[144]))+ (-0.4301859 * float(x[145]))+ (-0.49482134 * float(x[146]))+ (0.06050349 * float(x[147]))+ (0.12366678 * float(x[148]))+ (0.1957558 * float(x[149])))+ ((-0.07502714 * float(x[150]))+ (-0.37207648 * float(x[151]))+ (-0.3253419 * float(x[152]))+ (0.59327024 * float(x[153]))+ (-0.18561336 * float(x[154]))+ (-0.435766 * float(x[155]))+ (0.27819404 * float(x[156]))+ (0.30975506 * float(x[157]))+ (-0.17765218 * float(x[158]))+ (-0.29162455 * float(x[159]))+ (-0.59044087 * float(x[160]))+ (-0.46059343 * float(x[161]))+ (-0.48454538 * float(x[162]))+ (0.25524825 * float(x[163]))+ (0.30013952 * float(x[164]))+ (0.13788025 * float(x[165]))+ (-0.18654642 * float(x[166]))+ (-0.42740616 * float(x[167]))+ (0.12453088 * float(x[168]))+ (0.08844263 * float(x[169]))+ (-0.75816494 * float(x[170]))+ (-0.40314648 * float(x[171]))+ (0.42194507 * float(x[172]))+ (-0.016859695 * float(x[173]))+ (0.08632338 * float(x[174]))+ (-0.241905 * float(x[175]))+ (-0.3753776 * float(x[176]))+ (-0.366705 * float(x[177]))+ (-0.21963285 * float(x[178]))+ (-0.1480844 * float(x[179]))+ (-0.6016806 * float(x[180]))+ (0.10968701 * float(x[181]))+ (0.24076878 * float(x[182]))+ (0.84376705 * float(x[183]))+ (1.1913102 * float(x[184]))+ (0.05072971 * float(x[185]))+ (-0.619912 * float(x[186]))+ (0.11145524 * float(x[187]))+ (0.39351076 * float(x[188]))+ (-0.3456148 * float(x[189]))+ (-0.0894348 * float(x[190]))+ (-0.07351494 * float(x[191]))+ (-0.36577505 * float(x[192]))+ (-0.22669946 * float(x[193]))+ (-0.21140097 * float(x[194]))+ (-0.28032228 * float(x[195]))+ (-0.4445489 * float(x[196]))+ (-0.0946132 * float(x[197]))+ (0.45284557 * float(x[198]))+ (0.6975935 * float(x[199])))+ ((0.47366402 * float(x[200]))+ (-0.13696907 * float(x[201]))+ (-0.019443879 * float(x[202]))+ (0.54943234 * float(x[203]))+ (0.30139753 * float(x[204]))+ (-0.5247429 * float(x[205]))+ (-0.7162459 * float(x[206]))+ (-0.5663138 * float(x[207]))+ (-0.37076092 * float(x[208]))+ (-0.3103513 * float(x[209]))+ (0.105548084 * float(x[210]))+ (-0.22706999 * float(x[211]))+ (-0.58926547 * float(x[212]))+ (-0.44670728 * float(x[213]))+ (-0.22859097 * float(x[214]))+ (-0.31159377 * float(x[215]))+ (-0.6339948 * float(x[216]))+ (-0.0020493725 * float(x[217]))+ (0.55119854 * float(x[218]))+ (0.24044244 * float(x[219]))+ (0.042679377 * float(x[220]))+ (-0.33071828 * float(x[221]))+ (-0.62094986 * float(x[222]))+ (-0.7738415 * float(x[223]))+ (-0.5667116 * float(x[224]))+ (-0.4691918 * float(x[225]))+ (-0.2995434 * float(x[226]))+ (-0.39574286 * float(x[227]))+ (-0.86223954 * float(x[228]))+ (-0.63649994 * float(x[229]))+ (-0.123836115 * float(x[230]))+ (0.23559035 * float(x[231]))+ (-0.0019608121 * float(x[232]))+ (0.16025187 * float(x[233]))+ (0.48417535 * float(x[234]))+ (-0.010625754 * float(x[235]))+ (-0.44161424 * float(x[236]))+ (-0.18845789 * float(x[237]))+ (-0.55467385 * float(x[238]))+ (-0.5906486 * float(x[239]))+ (-0.33729824 * float(x[240]))+ (-0.49626458 * float(x[241]))+ (-0.47091225 * float(x[242]))+ (-0.8729575 * float(x[243]))+ (-0.79260427 * float(x[244]))+ (-0.6537528 * float(x[245]))+ (0.08800642 * float(x[246]))+ (0.5578311 * float(x[247]))+ (0.19996265 * float(x[248]))+ (0.022595603 * float(x[249])))+ ((0.47201806 * float(x[250]))+ (-0.28601187 * float(x[251]))+ (-0.5050899 * float(x[252]))+ (-0.15868405 * float(x[253]))+ (-0.4798746 * float(x[254]))+ (-0.5441003 * float(x[255]))) + 0.65479386), 0)
    o[0] = (1.128773 * h_0)+ (0.5088746 * h_1)+ (-0.5727249 * h_2)+ (-0.95853215 * h_3)+ (-1.2593365 * h_4)+ (0.4135864 * h_5)+ (0.36779907 * h_6)+ (1.2002417 * h_7)+ (-0.99349695 * h_8)+ (-0.4862918 * h_9)+ (-1.2104443 * h_10)+ (-0.49596873 * h_11) + -1.217348
    o[1] = (1.052441 * h_0)+ (-0.16598397 * h_1)+ (-0.013625728 * h_2)+ (1.8637662 * h_3)+ (1.209786 * h_4)+ (-2.6065352 * h_5)+ (-0.37232247 * h_6)+ (-0.8887105 * h_7)+ (1.7330259 * h_8)+ (0.529268 * h_9)+ (-0.52913296 * h_10)+ (-0.13447064 * h_11) + 1.1186478
    o[2] = (0.40682608 * h_0)+ (-1.8651996 * h_1)+ (-3.054344 * h_2)+ (-0.20928669 * h_3)+ (-0.3174634 * h_4)+ (-0.9454243 * h_5)+ (-0.51077515 * h_6)+ (0.6653381 * h_7)+ (-1.9219515 * h_8)+ (0.6990551 * h_9)+ (-3.5790963 * h_10)+ (1.0454899 * h_11) + 0.558157
    o[3] = (-1.1395502 * h_0)+ (-1.244496 * h_1)+ (-0.9497547 * h_2)+ (-0.6263594 * h_3)+ (1.1636789 * h_4)+ (-1.0206695 * h_5)+ (0.8222916 * h_6)+ (-0.35257193 * h_7)+ (-2.2579632 * h_8)+ (-1.56904 * h_9)+ (1.5853536 * h_10)+ (0.3525592 * h_11) + 1.2300606
    o[4] = (-1.4241292 * h_0)+ (1.0727566 * h_1)+ (-0.847516 * h_2)+ (0.5855983 * h_3)+ (-1.1968703 * h_4)+ (-0.40264606 * h_5)+ (-0.38148907 * h_6)+ (-1.1497664 * h_7)+ (1.1360996 * h_8)+ (0.82231015 * h_9)+ (0.5367145 * h_10)+ (0.09475952 * h_11) + 0.7897781
    o[5] = (-0.08012699 * h_0)+ (0.7205105 * h_1)+ (0.92103505 * h_2)+ (0.9278761 * h_3)+ (-3.159018 * h_4)+ (-1.1674412 * h_5)+ (0.40226507 * h_6)+ (-0.14063062 * h_7)+ (-0.95856166 * h_8)+ (-1.8048235 * h_9)+ (0.10520244 * h_10)+ (0.10744192 * h_11) + 1.122782
    o[6] = (0.48042068 * h_0)+ (-0.8063311 * h_1)+ (0.34667668 * h_2)+ (-1.0216495 * h_3)+ (-1.7031068 * h_4)+ (-0.41863874 * h_5)+ (1.7971386 * h_6)+ (-1.9974526 * h_7)+ (-3.7706387 * h_8)+ (0.6418041 * h_9)+ (0.6749793 * h_10)+ (-0.41294694 * h_11) + 0.66106945
    o[7] = (-1.9918638 * h_0)+ (0.03783981 * h_1)+ (1.1387246 * h_2)+ (-2.2165778 * h_3)+ (0.8310607 * h_4)+ (0.1984298 * h_5)+ (-0.65542614 * h_6)+ (0.40783042 * h_7)+ (-0.6028108 * h_8)+ (1.056327 * h_9)+ (-1.6640737 * h_10)+ (-0.091745324 * h_11) + 1.2264391
    o[8] = (-0.8572704 * h_0)+ (0.17474554 * h_1)+ (0.68496424 * h_2)+ (0.9008446 * h_3)+ (0.4620001 * h_4)+ (0.93584454 * h_5)+ (0.21915688 * h_6)+ (0.17349872 * h_7)+ (-0.37800637 * h_8)+ (0.15906832 * h_9)+ (0.42279246 * h_10)+ (-0.82796067 * h_11) + 1.1419586
    o[9] = (-1.8186576 * h_0)+ (-0.5845302 * h_1)+ (-1.029293 * h_2)+ (-0.86418533 * h_3)+ (0.009087085 * h_4)+ (1.0824142 * h_5)+ (-1.0610803 * h_6)+ (-0.85019904 * h_7)+ (0.7943663 * h_8)+ (-1.1834588 * h_9)+ (-0.635137 * h_10)+ (0.93421555 * h_11) + 1.304411

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
        model_cap=3214
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


