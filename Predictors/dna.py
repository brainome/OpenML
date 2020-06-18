#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/4965245/dna.arff -o Predictors/dna_NN.py -target class -stopat 96.5 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 0:00:58.65. Finished on: Jun-07-2020 05:02:00.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        3-way classifier
Best-guess accuracy:                51.93%
Model accuracy:                     96.79% (3084/3186 correct)
Improvement over best guess:        44.86% (of possible 48.07%)
Model capacity (MEC):               1475 bits
Generalization ratio:               2.09 bits/bit
Confusion Matrix:
 [50.41% 0.63% 0.88%]
 [0.25% 23.23% 0.60%]
 [0.60% 0.25% 23.16%]

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
    h_0 = max((((0.3919979 * float(x[0]))+ (0.11629091 * float(x[1]))+ (0.15546931 * float(x[2]))+ (0.21697903 * float(x[3]))+ (0.7093236 * float(x[4]))+ (0.09791653 * float(x[5]))+ (0.26774955 * float(x[6]))+ (0.27406546 * float(x[7]))+ (0.29458782 * float(x[8]))+ (0.17283162 * float(x[9]))+ (0.39288193 * float(x[10]))+ (0.4202322 * float(x[11]))+ (0.055763807 * float(x[12]))+ (0.2788813 * float(x[13]))+ (0.15796703 * float(x[14]))+ (0.4129511 * float(x[15]))+ (0.017347904 * float(x[16]))+ (0.3498514 * float(x[17]))+ (0.45659918 * float(x[18]))+ (0.07047257 * float(x[19]))+ (0.04252744 * float(x[20]))+ (0.1581184 * float(x[21]))+ (0.3185499 * float(x[22]))+ (0.27228814 * float(x[23]))+ (0.17239307 * float(x[24]))+ (0.04277604 * float(x[25]))+ (0.28789893 * float(x[26]))+ (0.24322851 * float(x[27]))+ (0.15368606 * float(x[28]))+ (0.02183324 * float(x[29]))+ (0.3656466 * float(x[30]))+ (0.21652961 * float(x[31]))+ (0.48350352 * float(x[32]))+ (0.06637977 * float(x[33]))+ (0.794981 * float(x[34]))+ (-0.08395943 * float(x[35]))+ (0.004896153 * float(x[36]))+ (0.94840026 * float(x[37]))+ (-0.08437427 * float(x[38]))+ (0.37152606 * float(x[39]))+ (0.27101278 * float(x[40]))+ (0.018964311 * float(x[41]))+ (0.15486592 * float(x[42]))+ (0.07385642 * float(x[43]))+ (0.055240545 * float(x[44]))+ (-0.1041616 * float(x[45]))+ (0.35284635 * float(x[46]))+ (0.31996468 * float(x[47]))+ (0.22463064 * float(x[48]))+ (0.19300935 * float(x[49])))+ ((0.0681699 * float(x[50]))+ (0.030217597 * float(x[51]))+ (0.3968268 * float(x[52]))+ (0.033249572 * float(x[53]))+ (0.13484637 * float(x[54]))+ (0.4457788 * float(x[55]))+ (0.09707857 * float(x[56]))+ (-0.062961355 * float(x[57]))+ (0.5860226 * float(x[58]))+ (0.19316505 * float(x[59]))+ (-0.14570409 * float(x[60]))+ (0.19095352 * float(x[61]))+ (-0.031507865 * float(x[62]))+ (-0.1933099 * float(x[63]))+ (0.14227521 * float(x[64]))+ (0.5780266 * float(x[65]))+ (0.034100343 * float(x[66]))+ (0.35904238 * float(x[67]))+ (-0.07884412 * float(x[68]))+ (0.018044632 * float(x[69]))+ (0.45399192 * float(x[70]))+ (0.21173452 * float(x[71]))+ (0.061798412 * float(x[72]))+ (0.4989775 * float(x[73]))+ (-0.121861406 * float(x[74]))+ (-0.12618004 * float(x[75]))+ (0.42851382 * float(x[76]))+ (0.12429778 * float(x[77]))+ (0.27178916 * float(x[78]))+ (0.15741122 * float(x[79]))+ (0.060168684 * float(x[80]))+ (-0.16879697 * float(x[81]))+ (0.5720383 * float(x[82]))+ (0.26025146 * float(x[83]))+ (0.9800508 * float(x[84]))+ (0.22621597 * float(x[85]))+ (-0.076126754 * float(x[86]))+ (0.24812675 * float(x[87]))+ (-0.15344243 * float(x[88]))+ (1.0579997 * float(x[89]))+ (0.055142507 * float(x[90]))+ (0.38774595 * float(x[91]))+ (0.21760198 * float(x[92]))+ (0.1989435 * float(x[93]))+ (0.19929637 * float(x[94]))+ (-0.16809286 * float(x[95]))+ (0.09697102 * float(x[96]))+ (0.30372584 * float(x[97]))+ (0.38604438 * float(x[98]))+ (0.1845595 * float(x[99])))+ ((0.09355457 * float(x[100]))+ (0.33079505 * float(x[101]))+ (0.43143818 * float(x[102]))+ (0.23272878 * float(x[103]))+ (0.18617599 * float(x[104]))+ (0.05184883 * float(x[105]))+ (0.17680356 * float(x[106]))+ (0.52395004 * float(x[107]))+ (0.51149064 * float(x[108]))+ (0.18972647 * float(x[109]))+ (0.03553414 * float(x[110]))+ (0.07202756 * float(x[111]))+ (0.13756129 * float(x[112]))+ (0.40967736 * float(x[113]))+ (0.03891416 * float(x[114]))+ (0.32513317 * float(x[115]))+ (0.07668371 * float(x[116]))+ (0.35296986 * float(x[117]))+ (0.32089525 * float(x[118]))+ (-0.0049199276 * float(x[119]))+ (-0.01950273 * float(x[120]))+ (0.3482694 * float(x[121]))+ (0.12739588 * float(x[122]))+ (0.4080048 * float(x[123]))+ (0.2890861 * float(x[124]))+ (0.12815799 * float(x[125]))+ (0.58231694 * float(x[126]))+ (0.315804 * float(x[127]))+ (0.15161364 * float(x[128]))+ (0.45689657 * float(x[129]))+ (0.41232497 * float(x[130]))+ (-0.07196666 * float(x[131]))+ (0.676783 * float(x[132]))+ (0.14443457 * float(x[133]))+ (0.273699 * float(x[134]))+ (0.05542238 * float(x[135]))+ (0.08116604 * float(x[136]))+ (0.6399786 * float(x[137]))+ (0.19162956 * float(x[138]))+ (0.12040508 * float(x[139]))+ (0.17381383 * float(x[140]))+ (0.12499675 * float(x[141]))+ (0.06093633 * float(x[142]))+ (0.6267415 * float(x[143]))+ (0.3773643 * float(x[144]))+ (0.5146978 * float(x[145]))+ (0.14369579 * float(x[146]))+ (0.07634251 * float(x[147]))+ (0.10215377 * float(x[148]))+ (0.67641914 * float(x[149])))+ ((0.11253899 * float(x[150]))+ (0.33469847 * float(x[151]))+ (0.45456782 * float(x[152]))+ (0.21507299 * float(x[153]))+ (0.37506256 * float(x[154]))+ (0.26888442 * float(x[155]))+ (0.029244471 * float(x[156]))+ (0.19217278 * float(x[157]))+ (0.32860073 * float(x[158]))+ (0.30136514 * float(x[159]))+ (0.44669056 * float(x[160]))+ (0.035397153 * float(x[161]))+ (0.1272355 * float(x[162]))+ (0.36018264 * float(x[163]))+ (0.087035015 * float(x[164]))+ (0.13972382 * float(x[165]))+ (0.43830487 * float(x[166]))+ (0.33082616 * float(x[167]))+ (0.22372861 * float(x[168]))+ (0.42848969 * float(x[169]))+ (0.6334945 * float(x[170]))+ (0.18609326 * float(x[171]))+ (0.17803617 * float(x[172]))+ (0.5051169 * float(x[173]))+ (0.32090214 * float(x[174]))+ (0.076709956 * float(x[175]))+ (0.20389943 * float(x[176]))+ (0.1137807 * float(x[177]))+ (0.61218864 * float(x[178]))+ (0.11731982 * float(x[179]))) + 1.2942344), 0)
    h_1 = max((((0.5251334 * float(x[0]))+ (0.34410042 * float(x[1]))+ (0.3879921 * float(x[2]))+ (0.2515425 * float(x[3]))+ (0.6944883 * float(x[4]))+ (0.45061448 * float(x[5]))+ (0.47571847 * float(x[6]))+ (0.33071077 * float(x[7]))+ (0.34299916 * float(x[8]))+ (0.12831369 * float(x[9]))+ (0.68032837 * float(x[10]))+ (0.4869655 * float(x[11]))+ (0.2459653 * float(x[12]))+ (0.30481553 * float(x[13]))+ (0.7396528 * float(x[14]))+ (0.30335784 * float(x[15]))+ (0.44309798 * float(x[16]))+ (0.22464086 * float(x[17]))+ (0.53413993 * float(x[18]))+ (0.5391067 * float(x[19]))+ (0.50691235 * float(x[20]))+ (0.46238324 * float(x[21]))+ (0.7661565 * float(x[22]))+ (0.18448213 * float(x[23]))+ (0.43324363 * float(x[24]))+ (0.91707104 * float(x[25]))+ (0.18912835 * float(x[26]))+ (0.3494672 * float(x[27]))+ (0.4079638 * float(x[28]))+ (0.5730925 * float(x[29]))+ (0.42701933 * float(x[30]))+ (0.12822121 * float(x[31]))+ (0.60271 * float(x[32]))+ (0.18721978 * float(x[33]))+ (1.0961975 * float(x[34]))+ (0.4830561 * float(x[35]))+ (0.18066858 * float(x[36]))+ (1.0471061 * float(x[37]))+ (0.28844586 * float(x[38]))+ (0.5769833 * float(x[39]))+ (0.4724927 * float(x[40]))+ (0.34290117 * float(x[41]))+ (0.2360196 * float(x[42]))+ (0.25769758 * float(x[43]))+ (0.61624014 * float(x[44]))+ (0.46137708 * float(x[45]))+ (0.7644138 * float(x[46]))+ (0.3954707 * float(x[47]))+ (0.3726883 * float(x[48]))+ (0.7364208 * float(x[49])))+ ((0.39150885 * float(x[50]))+ (0.312793 * float(x[51]))+ (0.5912534 * float(x[52]))+ (0.4949204 * float(x[53]))+ (0.3140605 * float(x[54]))+ (0.7493858 * float(x[55]))+ (0.37288108 * float(x[56]))+ (0.40291288 * float(x[57]))+ (0.5235295 * float(x[58]))+ (0.540137 * float(x[59]))+ (0.40325668 * float(x[60]))+ (0.31943443 * float(x[61]))+ (0.68027574 * float(x[62]))+ (0.38404855 * float(x[63]))+ (0.21786916 * float(x[64]))+ (0.9061081 * float(x[65]))+ (0.3436672 * float(x[66]))+ (0.46418357 * float(x[67]))+ (0.72716093 * float(x[68]))+ (0.4330817 * float(x[69]))+ (0.3805436 * float(x[70]))+ (0.6094331 * float(x[71]))+ (0.22801098 * float(x[72]))+ (0.2857172 * float(x[73]))+ (0.606325 * float(x[74]))+ (0.2784176 * float(x[75]))+ (0.4240771 * float(x[76]))+ (0.73541564 * float(x[77]))+ (0.39249817 * float(x[78]))+ (0.21441159 * float(x[79]))+ (0.46779636 * float(x[80]))+ (0.4950527 * float(x[81]))+ (0.8301702 * float(x[82]))+ (0.44201332 * float(x[83]))+ (0.26410234 * float(x[84]))+ (0.63576084 * float(x[85]))+ (0.31459114 * float(x[86]))+ (0.4312278 * float(x[87]))+ (0.5614245 * float(x[88]))+ (0.24734682 * float(x[89]))+ (0.25974062 * float(x[90]))+ (0.56372225 * float(x[91]))+ (0.48048174 * float(x[92]))+ (0.62646997 * float(x[93]))+ (0.4556078 * float(x[94]))+ (0.6328077 * float(x[95]))+ (0.26045495 * float(x[96]))+ (0.26308623 * float(x[97]))+ (0.9547197 * float(x[98]))+ (0.5736011 * float(x[99])))+ ((0.31190214 * float(x[100]))+ (0.6151791 * float(x[101]))+ (0.41025397 * float(x[102]))+ (0.28948036 * float(x[103]))+ (0.32510415 * float(x[104]))+ (0.3832757 * float(x[105]))+ (0.7069663 * float(x[106]))+ (0.8786789 * float(x[107]))+ (0.50965047 * float(x[108]))+ (0.43251345 * float(x[109]))+ (0.30495158 * float(x[110]))+ (0.380013 * float(x[111]))+ (0.6606948 * float(x[112]))+ (0.4358109 * float(x[113]))+ (0.23612934 * float(x[114]))+ (0.5521082 * float(x[115]))+ (0.29691702 * float(x[116]))+ (0.44240102 * float(x[117]))+ (0.350034 * float(x[118]))+ (0.60728604 * float(x[119]))+ (0.27594265 * float(x[120]))+ (0.60829633 * float(x[121]))+ (0.36080006 * float(x[122]))+ (0.39907113 * float(x[123]))+ (0.2821718 * float(x[124]))+ (0.5993151 * float(x[125]))+ (0.23803264 * float(x[126]))+ (0.41094664 * float(x[127]))+ (0.6188543 * float(x[128]))+ (0.31943044 * float(x[129]))+ (0.36394325 * float(x[130]))+ (0.5282837 * float(x[131]))+ (0.2465158 * float(x[132]))+ (0.081223175 * float(x[133]))+ (0.6940746 * float(x[134]))+ (0.29590198 * float(x[135]))+ (0.5771641 * float(x[136]))+ (0.38966188 * float(x[137]))+ (0.12188635 * float(x[138]))+ (0.5045541 * float(x[139]))+ (0.35717937 * float(x[140]))+ (0.40319127 * float(x[141]))+ (0.82359976 * float(x[142]))+ (0.2356885 * float(x[143]))+ (0.166691 * float(x[144]))+ (0.76431537 * float(x[145]))+ (0.20336068 * float(x[146]))+ (0.24834894 * float(x[147]))+ (0.6332122 * float(x[148]))+ (0.57712 * float(x[149])))+ ((0.37408993 * float(x[150]))+ (0.51398605 * float(x[151]))+ (0.54043514 * float(x[152]))+ (0.38165396 * float(x[153]))+ (0.611447 * float(x[154]))+ (0.55502087 * float(x[155]))+ (0.64788055 * float(x[156]))+ (0.29583305 * float(x[157]))+ (0.78519464 * float(x[158]))+ (0.19806452 * float(x[159]))+ (0.92949086 * float(x[160]))+ (0.44893327 * float(x[161]))+ (0.16801165 * float(x[162]))+ (0.25124282 * float(x[163]))+ (0.65073687 * float(x[164]))+ (0.4535771 * float(x[165]))+ (0.43047804 * float(x[166]))+ (0.5179849 * float(x[167]))+ (0.2775822 * float(x[168]))+ (0.20680405 * float(x[169]))+ (0.63770956 * float(x[170]))+ (0.24546857 * float(x[171]))+ (0.42327535 * float(x[172]))+ (0.5163051 * float(x[173]))+ (0.5187881 * float(x[174]))+ (0.1373507 * float(x[175]))+ (0.6446836 * float(x[176]))+ (0.17211878 * float(x[177]))+ (0.92782223 * float(x[178]))+ (0.25730383 * float(x[179]))) + 1.5073748), 0)
    h_2 = max((((0.10208567 * float(x[0]))+ (0.425116 * float(x[1]))+ (-0.022149848 * float(x[2]))+ (0.40709218 * float(x[3]))+ (0.11597975 * float(x[4]))+ (-0.0938958 * float(x[5]))+ (0.31831324 * float(x[6]))+ (0.1905591 * float(x[7]))+ (0.06572647 * float(x[8]))+ (0.41693392 * float(x[9]))+ (-0.053358767 * float(x[10]))+ (0.53335446 * float(x[11]))+ (0.072781995 * float(x[12]))+ (-0.101860106 * float(x[13]))+ (0.17052269 * float(x[14]))+ (0.123670444 * float(x[15]))+ (0.29466513 * float(x[16]))+ (0.42062047 * float(x[17]))+ (-0.007276264 * float(x[18]))+ (0.016119406 * float(x[19]))+ (0.58826 * float(x[20]))+ (0.00747744 * float(x[21]))+ (0.53101414 * float(x[22]))+ (0.09108944 * float(x[23]))+ (0.3044464 * float(x[24]))+ (-0.003325633 * float(x[25]))+ (-0.064493716 * float(x[26]))+ (0.061930973 * float(x[27]))+ (-0.08772382 * float(x[28]))+ (0.37678435 * float(x[29]))+ (0.10286445 * float(x[30]))+ (-0.017627804 * float(x[31]))+ (0.65455216 * float(x[32]))+ (0.39311975 * float(x[33]))+ (0.15977971 * float(x[34]))+ (0.014014048 * float(x[35]))+ (-0.011804784 * float(x[36]))+ (0.062819 * float(x[37]))+ (0.354823 * float(x[38]))+ (0.08706931 * float(x[39]))+ (0.14498802 * float(x[40]))+ (0.34626263 * float(x[41]))+ (0.31134844 * float(x[42]))+ (-0.012943924 * float(x[43]))+ (0.18818894 * float(x[44]))+ (0.41221857 * float(x[45]))+ (0.38082263 * float(x[46]))+ (-0.12449604 * float(x[47]))+ (0.71702206 * float(x[48]))+ (-0.0075726025 * float(x[49])))+ ((0.0946509 * float(x[50]))+ (0.14470227 * float(x[51]))+ (-0.07594649 * float(x[52]))+ (0.3791142 * float(x[53]))+ (0.34525526 * float(x[54]))+ (0.29968962 * float(x[55]))+ (0.0692666 * float(x[56]))+ (0.087410174 * float(x[57]))+ (0.24896729 * float(x[58]))+ (0.3470455 * float(x[59]))+ (0.352112 * float(x[60]))+ (-0.052988864 * float(x[61]))+ (0.4310699 * float(x[62]))+ (0.12239687 * float(x[63]))+ (0.39539245 * float(x[64]))+ (0.16427904 * float(x[65]))+ (0.04052586 * float(x[66]))+ (0.33210617 * float(x[67]))+ (0.2738435 * float(x[68]))+ (0.7814728 * float(x[69]))+ (0.011065762 * float(x[70]))+ (0.09391705 * float(x[71]))+ (-0.004056585 * float(x[72]))+ (-0.010627655 * float(x[73]))+ (0.71604276 * float(x[74]))+ (0.045478463 * float(x[75]))+ (0.39931378 * float(x[76]))+ (0.04734573 * float(x[77]))+ (0.41853175 * float(x[78]))+ (0.3702032 * float(x[79]))+ (-0.02908212 * float(x[80]))+ (0.019279117 * float(x[81]))+ (0.40780017 * float(x[82]))+ (-0.12888373 * float(x[83]))+ (0.6605519 * float(x[84]))+ (0.3223803 * float(x[85]))+ (-0.06524068 * float(x[86]))+ (-0.069027 * float(x[87]))+ (0.13884363 * float(x[88]))+ (0.7765358 * float(x[89]))+ (-0.29129136 * float(x[90]))+ (-0.29181972 * float(x[91]))+ (1.0662566 * float(x[92]))+ (-0.3678331 * float(x[93]))+ (-0.2600206 * float(x[94]))+ (-0.47794178 * float(x[95]))+ (0.46351233 * float(x[96]))+ (0.019423682 * float(x[97]))+ (0.4181044 * float(x[98]))+ (1.2563413 * float(x[99])))+ ((-0.12571198 * float(x[100]))+ (-0.16904397 * float(x[101]))+ (0.079964004 * float(x[102]))+ (-0.07580057 * float(x[103]))+ (0.99159884 * float(x[104]))+ (-0.14156058 * float(x[105]))+ (0.082353406 * float(x[106]))+ (0.15196103 * float(x[107]))+ (0.2060814 * float(x[108]))+ (0.36587557 * float(x[109]))+ (0.31055164 * float(x[110]))+ (0.18666665 * float(x[111]))+ (0.27154076 * float(x[112]))+ (0.23104459 * float(x[113]))+ (-0.00844374 * float(x[114]))+ (0.4097041 * float(x[115]))+ (0.30223176 * float(x[116]))+ (-0.0364135 * float(x[117]))+ (-0.16963144 * float(x[118]))+ (0.6611843 * float(x[119]))+ (0.19655292 * float(x[120]))+ (0.021271622 * float(x[121]))+ (0.1816175 * float(x[122]))+ (0.27777204 * float(x[123]))+ (-0.15118854 * float(x[124]))+ (0.37756917 * float(x[125]))+ (0.59621584 * float(x[126]))+ (-0.0016834064 * float(x[127]))+ (-0.009736616 * float(x[128]))+ (-0.00803839 * float(x[129]))+ (0.11136926 * float(x[130]))+ (0.7506483 * float(x[131]))+ (0.061293524 * float(x[132]))+ (-0.10458622 * float(x[133]))+ (0.6940538 * float(x[134]))+ (0.2883763 * float(x[135]))+ (0.28319207 * float(x[136]))+ (0.048842404 * float(x[137]))+ (-0.020987375 * float(x[138]))+ (0.24924944 * float(x[139]))+ (-0.08384072 * float(x[140]))+ (0.082567915 * float(x[141]))+ (-0.034288652 * float(x[142]))+ (0.28538823 * float(x[143]))+ (-0.0056917323 * float(x[144]))+ (0.05779948 * float(x[145]))+ (0.2997585 * float(x[146]))+ (0.050401445 * float(x[147]))+ (0.2280088 * float(x[148]))+ (0.027247248 * float(x[149])))+ ((-0.002120541 * float(x[150]))+ (0.16670151 * float(x[151]))+ (0.309116 * float(x[152]))+ (-0.040409517 * float(x[153]))+ (0.14858826 * float(x[154]))+ (0.41719598 * float(x[155]))+ (0.1261864 * float(x[156]))+ (0.24898453 * float(x[157]))+ (0.24080849 * float(x[158]))+ (0.098914355 * float(x[159]))+ (0.03972248 * float(x[160]))+ (0.2676967 * float(x[161]))+ (0.041148383 * float(x[162]))+ (-0.06306282 * float(x[163]))+ (0.42369336 * float(x[164]))+ (0.3709805 * float(x[165]))+ (-0.019331757 * float(x[166]))+ (-0.0054047666 * float(x[167]))+ (0.10193451 * float(x[168]))+ (0.25428185 * float(x[169]))+ (0.41514832 * float(x[170]))+ (0.06682168 * float(x[171]))+ (0.52770036 * float(x[172]))+ (-0.012943217 * float(x[173]))+ (0.47447562 * float(x[174]))+ (0.25794914 * float(x[175]))+ (-0.019965684 * float(x[176]))+ (-0.1319444 * float(x[177]))+ (0.2252565 * float(x[178]))+ (0.24668717 * float(x[179]))) + 0.96360433), 0)
    h_3 = max((((0.003073191 * float(x[0]))+ (0.15892453 * float(x[1]))+ (0.40047157 * float(x[2]))+ (0.0592575 * float(x[3]))+ (-0.0058702463 * float(x[4]))+ (0.40145427 * float(x[5]))+ (0.39801386 * float(x[6]))+ (0.110943496 * float(x[7]))+ (0.12435571 * float(x[8]))+ (-0.022225762 * float(x[9]))+ (0.24019274 * float(x[10]))+ (0.22575925 * float(x[11]))+ (-0.010468387 * float(x[12]))+ (0.11770137 * float(x[13]))+ (0.091759615 * float(x[14]))+ (0.13793102 * float(x[15]))+ (0.43548325 * float(x[16]))+ (-0.054522477 * float(x[17]))+ (0.09361713 * float(x[18]))+ (0.116139114 * float(x[19]))+ (0.02884795 * float(x[20]))+ (0.26918632 * float(x[21]))+ (0.48849142 * float(x[22]))+ (0.06653173 * float(x[23]))+ (0.19778872 * float(x[24]))+ (0.40083364 * float(x[25]))+ (-0.10354793 * float(x[26]))+ (0.08700532 * float(x[27]))+ (0.004316909 * float(x[28]))+ (0.1730904 * float(x[29]))+ (-0.08926571 * float(x[30]))+ (-0.03685528 * float(x[31]))+ (0.11250843 * float(x[32]))+ (0.24714091 * float(x[33]))+ (0.39491427 * float(x[34]))+ (-0.055856623 * float(x[35]))+ (-0.037363943 * float(x[36]))+ (0.0038650255 * float(x[37]))+ (0.035431802 * float(x[38]))+ (0.04463592 * float(x[39]))+ (0.19138308 * float(x[40]))+ (0.011409564 * float(x[41]))+ (-0.031576023 * float(x[42]))+ (0.14802374 * float(x[43]))+ (0.06702986 * float(x[44]))+ (0.13863946 * float(x[45]))+ (0.029988715 * float(x[46]))+ (-0.11797971 * float(x[47]))+ (-0.08548918 * float(x[48]))+ (0.38276252 * float(x[49])))+ ((0.0180778 * float(x[50]))+ (-0.030099038 * float(x[51]))+ (0.34289914 * float(x[52]))+ (-0.068124615 * float(x[53]))+ (-0.16599841 * float(x[54]))+ (0.1371017 * float(x[55]))+ (-0.031577088 * float(x[56]))+ (0.13914284 * float(x[57]))+ (-0.14144275 * float(x[58]))+ (0.0027358227 * float(x[59]))+ (0.10951958 * float(x[60]))+ (0.2339097 * float(x[61]))+ (0.19821501 * float(x[62]))+ (0.19099204 * float(x[63]))+ (0.03658532 * float(x[64]))+ (0.10978902 * float(x[65]))+ (-0.06334187 * float(x[66]))+ (0.24409024 * float(x[67]))+ (0.19767106 * float(x[68]))+ (0.089649655 * float(x[69]))+ (0.06606546 * float(x[70]))+ (0.097221926 * float(x[71]))+ (-0.021482807 * float(x[72]))+ (-0.03378144 * float(x[73]))+ (0.15666093 * float(x[74]))+ (0.009950952 * float(x[75]))+ (-0.0009525159 * float(x[76]))+ (0.01777095 * float(x[77]))+ (0.097317144 * float(x[78]))+ (0.10184365 * float(x[79]))+ (0.5632664 * float(x[80]))+ (-0.18294469 * float(x[81]))+ (0.451575 * float(x[82]))+ (-0.25765017 * float(x[83]))+ (0.9651735 * float(x[84]))+ (-0.08448929 * float(x[85]))+ (0.17813428 * float(x[86]))+ (-0.07252015 * float(x[87]))+ (-0.23185322 * float(x[88]))+ (0.9853979 * float(x[89]))+ (0.337318 * float(x[90]))+ (-0.30889875 * float(x[91]))+ (0.8024328 * float(x[92]))+ (-0.35467115 * float(x[93]))+ (-0.23346734 * float(x[94]))+ (0.13831846 * float(x[95]))+ (0.40917888 * float(x[96]))+ (-0.15484133 * float(x[97]))+ (0.51635486 * float(x[98]))+ (0.3759586 * float(x[99])))+ ((0.38179445 * float(x[100]))+ (0.028006805 * float(x[101]))+ (-0.063331395 * float(x[102]))+ (-0.043245867 * float(x[103]))+ (0.4551855 * float(x[104]))+ (-0.08043054 * float(x[105]))+ (0.14163972 * float(x[106]))+ (0.10017532 * float(x[107]))+ (0.14054981 * float(x[108]))+ (0.18001947 * float(x[109]))+ (0.2726702 * float(x[110]))+ (0.2738988 * float(x[111]))+ (0.3777711 * float(x[112]))+ (0.012280355 * float(x[113]))+ (0.43185955 * float(x[114]))+ (0.11067892 * float(x[115]))+ (0.019377861 * float(x[116]))+ (-0.11994528 * float(x[117]))+ (0.19684824 * float(x[118]))+ (0.30406663 * float(x[119]))+ (0.39863035 * float(x[120]))+ (0.07129612 * float(x[121]))+ (0.18563266 * float(x[122]))+ (0.055016607 * float(x[123]))+ (-0.00083508616 * float(x[124]))+ (0.17162015 * float(x[125]))+ (-0.113616474 * float(x[126]))+ (0.25370798 * float(x[127]))+ (0.4048048 * float(x[128]))+ (0.13373308 * float(x[129]))+ (-0.054977942 * float(x[130]))+ (0.53364414 * float(x[131]))+ (-0.09907727 * float(x[132]))+ (0.25783753 * float(x[133]))+ (0.24855763 * float(x[134]))+ (0.20982401 * float(x[135]))+ (0.17447826 * float(x[136]))+ (0.07069435 * float(x[137]))+ (-0.055063706 * float(x[138]))+ (0.22140194 * float(x[139]))+ (0.34827322 * float(x[140]))+ (0.062611096 * float(x[141]))+ (0.44646928 * float(x[142]))+ (-0.024166858 * float(x[143]))+ (-0.016205473 * float(x[144]))+ (0.51943916 * float(x[145]))+ (0.06813745 * float(x[146]))+ (0.11687711 * float(x[147]))+ (0.37595817 * float(x[148]))+ (0.18867505 * float(x[149])))+ ((0.27690846 * float(x[150]))+ (0.27112934 * float(x[151]))+ (-0.0025988072 * float(x[152]))+ (0.18048847 * float(x[153]))+ (0.23145548 * float(x[154]))+ (0.12272433 * float(x[155]))+ (0.17632313 * float(x[156]))+ (-0.06355931 * float(x[157]))+ (0.31659287 * float(x[158]))+ (-0.022038775 * float(x[159]))+ (0.24531077 * float(x[160]))+ (0.22358765 * float(x[161]))+ (0.25873393 * float(x[162]))+ (0.103616215 * float(x[163]))+ (0.06844001 * float(x[164]))+ (0.179002 * float(x[165]))+ (0.37183288 * float(x[166]))+ (0.1439688 * float(x[167]))+ (0.14310066 * float(x[168]))+ (-0.036592934 * float(x[169]))+ (0.2351105 * float(x[170]))+ (0.32266784 * float(x[171]))+ (0.106107414 * float(x[172]))+ (0.27960068 * float(x[173]))+ (0.10920458 * float(x[174]))+ (-0.009573479 * float(x[175]))+ (0.45563 * float(x[176]))+ (0.10736202 * float(x[177]))+ (0.035017055 * float(x[178]))+ (0.2921489 * float(x[179]))) + 0.98105276), 0)
    h_4 = max((((0.08111772 * float(x[0]))+ (0.24796717 * float(x[1]))+ (0.28285173 * float(x[2]))+ (0.087714024 * float(x[3]))+ (-0.15037818 * float(x[4]))+ (0.46449652 * float(x[5]))+ (0.42772362 * float(x[6]))+ (0.2688952 * float(x[7]))+ (0.055221714 * float(x[8]))+ (0.34440964 * float(x[9]))+ (0.06113079 * float(x[10]))+ (0.00081370096 * float(x[11]))+ (0.38151512 * float(x[12]))+ (0.29772598 * float(x[13]))+ (0.033977374 * float(x[14]))+ (0.38018152 * float(x[15]))+ (0.22319993 * float(x[16]))+ (-0.41416666 * float(x[17]))+ (0.30770832 * float(x[18]))+ (0.03357968 * float(x[19]))+ (-0.008214112 * float(x[20]))+ (0.3274214 * float(x[21]))+ (0.36141893 * float(x[22]))+ (-0.22146726 * float(x[23]))+ (0.5360317 * float(x[24]))+ (0.46913335 * float(x[25]))+ (-0.02138817 * float(x[26]))+ (-0.04049562 * float(x[27]))+ (0.3550485 * float(x[28]))+ (0.13228858 * float(x[29]))+ (0.13540506 * float(x[30]))+ (0.36962226 * float(x[31]))+ (0.14778082 * float(x[32]))+ (0.63920647 * float(x[33]))+ (0.5076101 * float(x[34]))+ (-0.24417183 * float(x[35]))+ (0.105313845 * float(x[36]))+ (0.47281438 * float(x[37]))+ (0.10737204 * float(x[38]))+ (0.19910009 * float(x[39]))+ (0.4592564 * float(x[40]))+ (0.2681296 * float(x[41]))+ (0.28053614 * float(x[42]))+ (0.108793736 * float(x[43]))+ (-0.06593946 * float(x[44]))+ (0.19478332 * float(x[45]))+ (0.021805186 * float(x[46]))+ (0.34403494 * float(x[47]))+ (-0.02567664 * float(x[48]))+ (0.07882852 * float(x[49])))+ ((-0.17581376 * float(x[50]))+ (0.022791445 * float(x[51]))+ (0.21030314 * float(x[52]))+ (0.15250055 * float(x[53]))+ (0.064822376 * float(x[54]))+ (0.40289345 * float(x[55]))+ (0.21053004 * float(x[56]))+ (0.26525384 * float(x[57]))+ (0.08371039 * float(x[58]))+ (0.16893436 * float(x[59]))+ (0.3119884 * float(x[60]))+ (0.32242188 * float(x[61]))+ (-0.059035603 * float(x[62]))+ (0.3483563 * float(x[63]))+ (0.09673589 * float(x[64]))+ (0.010706291 * float(x[65]))+ (-0.14642443 * float(x[66]))+ (0.46154746 * float(x[67]))+ (0.2953296 * float(x[68]))+ (-0.022881027 * float(x[69]))+ (0.15483607 * float(x[70]))+ (-0.26558444 * float(x[71]))+ (-0.008825445 * float(x[72]))+ (0.007857742 * float(x[73]))+ (0.1614955 * float(x[74]))+ (-0.046442255 * float(x[75]))+ (0.29076752 * float(x[76]))+ (-0.033216693 * float(x[77]))+ (-0.028459692 * float(x[78]))+ (0.10290564 * float(x[79]))+ (0.1563513 * float(x[80]))+ (-0.08531768 * float(x[81]))+ (0.34291336 * float(x[82]))+ (0.27860948 * float(x[83]))+ (0.7728773 * float(x[84]))+ (-0.19216865 * float(x[85]))+ (0.09909779 * float(x[86]))+ (0.020666461 * float(x[87]))+ (0.33207127 * float(x[88]))+ (0.21533081 * float(x[89]))+ (0.7713214 * float(x[90]))+ (0.71859723 * float(x[91]))+ (-0.4272476 * float(x[92]))+ (1.0658765 * float(x[93]))+ (1.0902582 * float(x[94]))+ (1.0999174 * float(x[95]))+ (-0.60801 * float(x[96]))+ (0.68928957 * float(x[97]))+ (0.14533171 * float(x[98]))+ (-0.33248243 * float(x[99])))+ ((0.4183193 * float(x[100]))+ (0.5934449 * float(x[101]))+ (0.6380869 * float(x[102]))+ (0.75658894 * float(x[103]))+ (-0.89403164 * float(x[104]))+ (0.6086743 * float(x[105]))+ (0.9150149 * float(x[106]))+ (0.81262946 * float(x[107]))+ (0.26308173 * float(x[108]))+ (0.13371424 * float(x[109]))+ (0.07604178 * float(x[110]))+ (0.5337336 * float(x[111]))+ (0.3882093 * float(x[112]))+ (-0.08255865 * float(x[113]))+ (0.90154403 * float(x[114]))+ (-0.07686635 * float(x[115]))+ (0.019461062 * float(x[116]))+ (0.539205 * float(x[117]))+ (0.45037133 * float(x[118]))+ (0.180956 * float(x[119]))+ (0.52296805 * float(x[120]))+ (0.027879236 * float(x[121]))+ (0.053870063 * float(x[122]))+ (0.60466975 * float(x[123]))+ (0.22762178 * float(x[124]))+ (0.10422428 * float(x[125]))+ (0.41498172 * float(x[126]))+ (0.17105259 * float(x[127]))+ (0.35404986 * float(x[128]))+ (0.23602217 * float(x[129]))+ (0.15489206 * float(x[130]))+ (0.31380185 * float(x[131]))+ (0.60955745 * float(x[132]))+ (0.43921968 * float(x[133]))+ (0.29244322 * float(x[134]))+ (0.6457019 * float(x[135]))+ (0.3599139 * float(x[136]))+ (0.12620884 * float(x[137]))+ (0.23464923 * float(x[138]))+ (0.28933826 * float(x[139]))+ (0.41979587 * float(x[140]))+ (0.32113957 * float(x[141]))+ (0.20651372 * float(x[142]))+ (0.36140177 * float(x[143]))+ (0.28876165 * float(x[144]))+ (0.38433027 * float(x[145]))+ (0.25370026 * float(x[146]))+ (0.19193947 * float(x[147]))+ (0.40732762 * float(x[148]))+ (-0.038625896 * float(x[149])))+ ((0.53805494 * float(x[150]))+ (0.3907461 * float(x[151]))+ (0.2743161 * float(x[152]))+ (0.48315653 * float(x[153]))+ (0.49143526 * float(x[154]))+ (0.14628938 * float(x[155]))+ (0.274151 * float(x[156]))+ (0.2903132 * float(x[157]))+ (0.34117606 * float(x[158]))+ (0.53516537 * float(x[159]))+ (0.16263144 * float(x[160]))+ (0.25662503 * float(x[161]))+ (0.5988529 * float(x[162]))+ (0.29283497 * float(x[163]))+ (-0.073328346 * float(x[164]))+ (0.53325987 * float(x[165]))+ (0.6102431 * float(x[166]))+ (-0.1706819 * float(x[167]))+ (0.5235256 * float(x[168]))+ (0.42927054 * float(x[169]))+ (0.08320915 * float(x[170]))+ (0.47592616 * float(x[171]))+ (0.07452332 * float(x[172]))+ (0.12421394 * float(x[173]))+ (0.116204254 * float(x[174]))+ (0.2534872 * float(x[175]))+ (0.289521 * float(x[176]))+ (0.52354455 * float(x[177]))+ (0.4245762 * float(x[178]))+ (0.24420224 * float(x[179]))) + 0.8915533), 0)
    h_5 = max((((-0.1179476 * float(x[0]))+ (0.115483515 * float(x[1]))+ (0.09387289 * float(x[2]))+ (0.06602503 * float(x[3]))+ (0.22417884 * float(x[4]))+ (0.041731138 * float(x[5]))+ (0.28341457 * float(x[6]))+ (0.4468879 * float(x[7]))+ (-0.10694761 * float(x[8]))+ (0.20366094 * float(x[9]))+ (0.23073062 * float(x[10]))+ (0.10936464 * float(x[11]))+ (0.3496345 * float(x[12]))+ (0.10393443 * float(x[13]))+ (-0.26699674 * float(x[14]))+ (0.21013023 * float(x[15]))+ (0.38053307 * float(x[16]))+ (0.61012137 * float(x[17]))+ (0.53182846 * float(x[18]))+ (0.36504993 * float(x[19]))+ (0.29509586 * float(x[20]))+ (-0.29542127 * float(x[21]))+ (0.23458886 * float(x[22]))+ (0.24098986 * float(x[23]))+ (0.3043634 * float(x[24]))+ (0.0047304644 * float(x[25]))+ (0.30657184 * float(x[26]))+ (0.82608765 * float(x[27]))+ (-0.193792 * float(x[28]))+ (-0.039150547 * float(x[29]))+ (0.17512739 * float(x[30]))+ (0.48978955 * float(x[31]))+ (-0.17710547 * float(x[32]))+ (0.056246392 * float(x[33]))+ (0.47714773 * float(x[34]))+ (0.60575324 * float(x[35]))+ (0.23533918 * float(x[36]))+ (-0.01719061 * float(x[37]))+ (0.4474584 * float(x[38]))+ (0.6046595 * float(x[39]))+ (-0.04363252 * float(x[40]))+ (0.03675907 * float(x[41]))+ (0.084994845 * float(x[42]))+ (0.3046117 * float(x[43]))+ (0.26502246 * float(x[44]))+ (0.38212472 * float(x[45]))+ (0.17622904 * float(x[46]))+ (0.53179127 * float(x[47]))+ (0.46900678 * float(x[48]))+ (0.1154186 * float(x[49])))+ ((0.5039692 * float(x[50]))+ (0.9333235 * float(x[51]))+ (0.07524504 * float(x[52]))+ (0.116931066 * float(x[53]))+ (0.6501591 * float(x[54]))+ (0.020259658 * float(x[55]))+ (0.50895107 * float(x[56]))+ (0.8725335 * float(x[57]))+ (0.3400066 * float(x[58]))+ (0.021907102 * float(x[59]))+ (0.7819407 * float(x[60]))+ (0.23091681 * float(x[61]))+ (0.41407678 * float(x[62]))+ (0.85293317 * float(x[63]))+ (0.26077822 * float(x[64]))+ (0.5514801 * float(x[65]))+ (0.91018224 * float(x[66]))+ (0.19513528 * float(x[67]))+ (0.096910045 * float(x[68]))+ (0.67512333 * float(x[69]))+ (-0.105885565 * float(x[70]))+ (0.9348237 * float(x[71]))+ (0.7070636 * float(x[72]))+ (0.14026101 * float(x[73]))+ (0.54629844 * float(x[74]))+ (0.52096343 * float(x[75]))+ (0.22259359 * float(x[76]))+ (0.43534425 * float(x[77]))+ (-0.08871227 * float(x[78]))+ (0.20975399 * float(x[79]))+ (-0.024433678 * float(x[80]))+ (1.1564453 * float(x[81]))+ (-0.11498274 * float(x[82]))+ (0.7017934 * float(x[83]))+ (-0.5475927 * float(x[84]))+ (0.6775212 * float(x[85]))+ (0.6158223 * float(x[86]))+ (0.71161777 * float(x[87]))+ (0.32411438 * float(x[88]))+ (-0.26509377 * float(x[89]))+ (-1.1954721 * float(x[90]))+ (-0.8148207 * float(x[91]))+ (1.2819186 * float(x[92]))+ (-1.5145909 * float(x[93]))+ (-1.2891587 * float(x[94]))+ (-1.4748285 * float(x[95]))+ (1.0276898 * float(x[96]))+ (-0.8930775 * float(x[97]))+ (0.45043987 * float(x[98]))+ (0.74844843 * float(x[99])))+ ((-0.093158975 * float(x[100]))+ (-0.24928086 * float(x[101]))+ (-0.4144542 * float(x[102]))+ (-0.336945 * float(x[103]))+ (0.8943813 * float(x[104]))+ (-0.24046375 * float(x[105]))+ (-0.71369094 * float(x[106]))+ (-0.14610144 * float(x[107]))+ (0.14184344 * float(x[108]))+ (0.39009476 * float(x[109]))+ (0.07810481 * float(x[110]))+ (-0.16456456 * float(x[111]))+ (-0.15618186 * float(x[112]))+ (0.19980626 * float(x[113]))+ (-0.12351139 * float(x[114]))+ (0.46605998 * float(x[115]))+ (0.09985291 * float(x[116]))+ (0.29704404 * float(x[117]))+ (0.29203418 * float(x[118]))+ (0.3208117 * float(x[119]))+ (0.009306723 * float(x[120]))+ (0.14388797 * float(x[121]))+ (0.08545916 * float(x[122]))+ (-0.15688305 * float(x[123]))+ (0.6292747 * float(x[124]))+ (0.012235737 * float(x[125]))+ (0.10188191 * float(x[126]))+ (0.14322366 * float(x[127]))+ (-0.21768418 * float(x[128]))+ (-0.005704901 * float(x[129]))+ (0.479709 * float(x[130]))+ (0.46921584 * float(x[131]))+ (0.28621686 * float(x[132]))+ (-0.17952594 * float(x[133]))+ (-0.3411078 * float(x[134]))+ (0.00753943 * float(x[135]))+ (-0.30852795 * float(x[136]))+ (-0.16008458 * float(x[137]))+ (-0.10826761 * float(x[138]))+ (-0.0014635004 * float(x[139]))+ (0.3267656 * float(x[140]))+ (0.09903093 * float(x[141]))+ (0.1796597 * float(x[142]))+ (0.05117433 * float(x[143]))+ (-0.019461915 * float(x[144]))+ (0.06490134 * float(x[145]))+ (0.053593624 * float(x[146]))+ (-0.023213653 * float(x[147]))+ (0.2593451 * float(x[148]))+ (0.446207 * float(x[149])))+ ((-0.36964113 * float(x[150]))+ (0.31127688 * float(x[151]))+ (0.0657027 * float(x[152]))+ (0.3016497 * float(x[153]))+ (0.24105404 * float(x[154]))+ (-0.15634954 * float(x[155]))+ (0.49567804 * float(x[156]))+ (0.16658884 * float(x[157]))+ (0.2896627 * float(x[158]))+ (0.53701746 * float(x[159]))+ (0.054972865 * float(x[160]))+ (0.033230774 * float(x[161]))+ (0.029138459 * float(x[162]))+ (0.24217446 * float(x[163]))+ (-0.030251935 * float(x[164]))+ (0.20054944 * float(x[165]))+ (0.07342768 * float(x[166]))+ (0.39398035 * float(x[167]))+ (-0.16798812 * float(x[168]))+ (-0.04676184 * float(x[169]))+ (-0.24750735 * float(x[170]))+ (-0.034849558 * float(x[171]))+ (0.073426485 * float(x[172]))+ (0.23463759 * float(x[173]))+ (0.33559728 * float(x[174]))+ (0.30182868 * float(x[175]))+ (0.2802978 * float(x[176]))+ (0.022256594 * float(x[177]))+ (0.10817806 * float(x[178]))+ (-0.16178668 * float(x[179]))) + 0.3847873), 0)
    h_6 = max((((-0.0130021125 * float(x[0]))+ (0.010620636 * float(x[1]))+ (0.11424478 * float(x[2]))+ (0.048273414 * float(x[3]))+ (0.07066713 * float(x[4]))+ (0.03253101 * float(x[5]))+ (0.2512646 * float(x[6]))+ (0.2900323 * float(x[7]))+ (-0.042014685 * float(x[8]))+ (0.11223045 * float(x[9]))+ (0.12649095 * float(x[10]))+ (0.042817485 * float(x[11]))+ (0.11917099 * float(x[12]))+ (0.15642107 * float(x[13]))+ (-0.10049072 * float(x[14]))+ (0.11496527 * float(x[15]))+ (0.07962459 * float(x[16]))+ (0.2198977 * float(x[17]))+ (0.25784174 * float(x[18]))+ (0.008992002 * float(x[19]))+ (0.09369707 * float(x[20]))+ (-0.0007039076 * float(x[21]))+ (0.09369646 * float(x[22]))+ (0.016258948 * float(x[23]))+ (0.18418883 * float(x[24]))+ (0.01943433 * float(x[25]))+ (0.18827777 * float(x[26]))+ (0.22574632 * float(x[27]))+ (0.03596286 * float(x[28]))+ (0.04514949 * float(x[29]))+ (0.12095528 * float(x[30]))+ (0.082711264 * float(x[31]))+ (0.002430061 * float(x[32]))+ (0.12760973 * float(x[33]))+ (0.23715864 * float(x[34]))+ (0.21776682 * float(x[35]))+ (0.14972183 * float(x[36]))+ (0.059298374 * float(x[37]))+ (0.22568591 * float(x[38]))+ (0.2709985 * float(x[39]))+ (0.1909258 * float(x[40]))+ (0.064284004 * float(x[41]))+ (0.11725543 * float(x[42]))+ (0.013460002 * float(x[43]))+ (0.097601086 * float(x[44]))+ (0.23364027 * float(x[45]))+ (0.10328078 * float(x[46]))+ (0.21979162 * float(x[47]))+ (0.21606034 * float(x[48]))+ (0.103047505 * float(x[49])))+ ((0.045166306 * float(x[50]))+ (0.19396849 * float(x[51]))+ (0.09678513 * float(x[52]))+ (0.11557123 * float(x[53]))+ (0.34411 * float(x[54]))+ (0.15261944 * float(x[55]))+ (0.120653264 * float(x[56]))+ (0.24765623 * float(x[57]))+ (0.19614865 * float(x[58]))+ (0.054422572 * float(x[59]))+ (0.403796 * float(x[60]))+ (0.046640348 * float(x[61]))+ (0.20371397 * float(x[62]))+ (0.38349777 * float(x[63]))+ (0.1085043 * float(x[64]))+ (0.19566774 * float(x[65]))+ (0.261704 * float(x[66]))+ (0.1188849 * float(x[67]))+ (0.16604333 * float(x[68]))+ (0.21030909 * float(x[69]))+ (-0.033163555 * float(x[70]))+ (0.25199586 * float(x[71]))+ (0.22451618 * float(x[72]))+ (0.053051695 * float(x[73]))+ (0.28881082 * float(x[74]))+ (0.22334501 * float(x[75]))+ (0.052823268 * float(x[76]))+ (0.18247367 * float(x[77]))+ (-0.059248094 * float(x[78]))+ (-0.026795305 * float(x[79]))+ (0.06341267 * float(x[80]))+ (0.43172902 * float(x[81]))+ (-0.09307417 * float(x[82]))+ (0.41788357 * float(x[83]))+ (-0.4891063 * float(x[84]))+ (0.21711352 * float(x[85]))+ (0.29657072 * float(x[86]))+ (0.28596878 * float(x[87]))+ (0.26500925 * float(x[88]))+ (-0.44607097 * float(x[89]))+ (0.037653435 * float(x[90]))+ (0.16242349 * float(x[91]))+ (-0.0145529155 * float(x[92]))+ (-0.012794364 * float(x[93]))+ (0.2828244 * float(x[94]))+ (0.29922992 * float(x[95]))+ (0.26426885 * float(x[96]))+ (0.08205601 * float(x[97]))+ (0.08442131 * float(x[98]))+ (0.103184864 * float(x[99])))+ ((-0.010629113 * float(x[100]))+ (0.121452615 * float(x[101]))+ (0.17181776 * float(x[102]))+ (-0.12334973 * float(x[103]))+ (0.037120968 * float(x[104]))+ (0.15116322 * float(x[105]))+ (0.10461576 * float(x[106]))+ (0.24210614 * float(x[107]))+ (0.021931846 * float(x[108]))+ (0.017604321 * float(x[109]))+ (0.074791946 * float(x[110]))+ (0.000575453 * float(x[111]))+ (0.046687286 * float(x[112]))+ (0.0453834 * float(x[113]))+ (0.15076582 * float(x[114]))+ (0.058206726 * float(x[115]))+ (0.07336111 * float(x[116]))+ (0.2087392 * float(x[117]))+ (0.29552135 * float(x[118]))+ (0.17417082 * float(x[119]))+ (0.20325418 * float(x[120]))+ (0.07592995 * float(x[121]))+ (-0.10268109 * float(x[122]))+ (0.14210606 * float(x[123]))+ (0.2546263 * float(x[124]))+ (0.05804185 * float(x[125]))+ (0.19227539 * float(x[126]))+ (-0.030117046 * float(x[127]))+ (0.023587056 * float(x[128]))+ (0.054601528 * float(x[129]))+ (0.1261854 * float(x[130]))+ (0.17487329 * float(x[131]))+ (0.19757123 * float(x[132]))+ (-0.06558495 * float(x[133]))+ (-0.07773648 * float(x[134]))+ (0.13333489 * float(x[135]))+ (-0.03214021 * float(x[136]))+ (-0.002696373 * float(x[137]))+ (0.04247904 * float(x[138]))+ (-0.026044166 * float(x[139]))+ (0.32191673 * float(x[140]))+ (0.0044373274 * float(x[141]))+ (0.12517528 * float(x[142]))+ (0.16697407 * float(x[143]))+ (0.10418002 * float(x[144]))+ (0.115615495 * float(x[145]))+ (-0.03005762 * float(x[146]))+ (0.057797477 * float(x[147]))+ (0.23505431 * float(x[148]))+ (0.06866699 * float(x[149])))+ ((0.024120007 * float(x[150]))+ (0.14829765 * float(x[151]))+ (0.11235498 * float(x[152]))+ (0.3042874 * float(x[153]))+ (0.15499422 * float(x[154]))+ (-0.008867196 * float(x[155]))+ (0.15374504 * float(x[156]))+ (0.019246036 * float(x[157]))+ (0.2744757 * float(x[158]))+ (0.16060778 * float(x[159]))+ (0.05351335 * float(x[160]))+ (0.11606816 * float(x[161]))+ (0.111444235 * float(x[162]))+ (0.054142877 * float(x[163]))+ (-0.012035546 * float(x[164]))+ (0.17241712 * float(x[165]))+ (0.0628907 * float(x[166]))+ (0.14949682 * float(x[167]))+ (-0.018065235 * float(x[168]))+ (-0.02419829 * float(x[169]))+ (-0.07005273 * float(x[170]))+ (0.008622448 * float(x[171]))+ (-0.1253556 * float(x[172]))+ (0.14366695 * float(x[173]))+ (0.13455996 * float(x[174]))+ (0.07925786 * float(x[175]))+ (0.14320385 * float(x[176]))+ (0.08765161 * float(x[177]))+ (0.18816905 * float(x[178]))+ (0.021673514 * float(x[179]))) + 0.14621131), 0)
    h_7 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))+ (0.0 * float(x[16]))+ (0.0 * float(x[17]))+ (0.0 * float(x[18]))+ (0.0 * float(x[19]))+ (0.0 * float(x[20]))+ (0.0 * float(x[21]))+ (0.0 * float(x[22]))+ (0.0 * float(x[23]))+ (0.0 * float(x[24]))+ (0.0 * float(x[25]))+ (0.0 * float(x[26]))+ (0.0 * float(x[27]))+ (0.0 * float(x[28]))+ (0.0 * float(x[29]))+ (0.0 * float(x[30]))+ (0.0 * float(x[31]))+ (0.0 * float(x[32]))+ (0.0 * float(x[33]))+ (0.0 * float(x[34]))+ (0.0 * float(x[35]))+ (0.0 * float(x[36]))+ (0.0 * float(x[37]))+ (0.0 * float(x[38]))+ (0.0 * float(x[39]))+ (0.0 * float(x[40]))+ (0.0 * float(x[41]))+ (0.0 * float(x[42]))+ (0.0 * float(x[43]))+ (0.0 * float(x[44]))+ (0.0 * float(x[45]))+ (0.0 * float(x[46]))+ (0.0 * float(x[47]))+ (0.0 * float(x[48]))+ (0.0 * float(x[49])))+ ((0.0 * float(x[50]))+ (0.0 * float(x[51]))+ (0.0 * float(x[52]))+ (0.0 * float(x[53]))+ (0.0 * float(x[54]))+ (0.0 * float(x[55]))+ (0.0 * float(x[56]))+ (0.0 * float(x[57]))+ (0.0 * float(x[58]))+ (0.0 * float(x[59]))+ (0.0 * float(x[60]))+ (0.0 * float(x[61]))+ (0.0 * float(x[62]))+ (0.0 * float(x[63]))+ (0.0 * float(x[64]))+ (0.0 * float(x[65]))+ (0.0 * float(x[66]))+ (0.0 * float(x[67]))+ (0.0 * float(x[68]))+ (0.0 * float(x[69]))+ (0.0 * float(x[70]))+ (0.0 * float(x[71]))+ (0.0 * float(x[72]))+ (0.0 * float(x[73]))+ (0.0 * float(x[74]))+ (0.0 * float(x[75]))+ (0.0 * float(x[76]))+ (0.0 * float(x[77]))+ (0.0 * float(x[78]))+ (0.0 * float(x[79]))+ (0.0 * float(x[80]))+ (0.0 * float(x[81]))+ (0.0 * float(x[82]))+ (0.0 * float(x[83]))+ (0.0 * float(x[84]))+ (0.0 * float(x[85]))+ (0.0 * float(x[86]))+ (0.0 * float(x[87]))+ (0.0 * float(x[88]))+ (0.0 * float(x[89]))+ (0.0 * float(x[90]))+ (0.0 * float(x[91]))+ (0.0 * float(x[92]))+ (0.0 * float(x[93]))+ (0.0 * float(x[94]))+ (0.0 * float(x[95]))+ (0.0 * float(x[96]))+ (0.0 * float(x[97]))+ (0.0 * float(x[98]))+ (0.0 * float(x[99])))+ ((0.0 * float(x[100]))+ (0.0 * float(x[101]))+ (0.0 * float(x[102]))+ (0.0 * float(x[103]))+ (0.0 * float(x[104]))+ (0.0 * float(x[105]))+ (0.0 * float(x[106]))+ (0.0 * float(x[107]))+ (0.0 * float(x[108]))+ (0.0 * float(x[109]))+ (0.0 * float(x[110]))+ (0.0 * float(x[111]))+ (0.0 * float(x[112]))+ (0.0 * float(x[113]))+ (0.0 * float(x[114]))+ (0.0 * float(x[115]))+ (0.0 * float(x[116]))+ (0.0 * float(x[117]))+ (0.0 * float(x[118]))+ (0.0 * float(x[119]))+ (0.0 * float(x[120]))+ (0.0 * float(x[121]))+ (0.0 * float(x[122]))+ (0.0 * float(x[123]))+ (0.0 * float(x[124]))+ (0.0 * float(x[125]))+ (0.0 * float(x[126]))+ (0.0 * float(x[127]))+ (0.0 * float(x[128]))+ (0.0 * float(x[129]))+ (0.0 * float(x[130]))+ (0.0 * float(x[131]))+ (0.0 * float(x[132]))+ (0.0 * float(x[133]))+ (0.0 * float(x[134]))+ (0.0 * float(x[135]))+ (0.0 * float(x[136]))+ (0.0 * float(x[137]))+ (0.0 * float(x[138]))+ (0.0 * float(x[139]))+ (0.0 * float(x[140]))+ (0.0 * float(x[141]))+ (0.0 * float(x[142]))+ (0.0 * float(x[143]))+ (0.0 * float(x[144]))+ (0.0 * float(x[145]))+ (0.0 * float(x[146]))+ (0.0 * float(x[147]))+ (0.0 * float(x[148]))+ (0.0 * float(x[149])))+ ((0.0 * float(x[150]))+ (0.0 * float(x[151]))+ (0.0 * float(x[152]))+ (0.0 * float(x[153]))+ (0.0 * float(x[154]))+ (0.0 * float(x[155]))+ (0.0 * float(x[156]))+ (0.0 * float(x[157]))+ (0.0 * float(x[158]))+ (0.0 * float(x[159]))+ (0.0 * float(x[160]))+ (0.0 * float(x[161]))+ (0.0 * float(x[162]))+ (0.0 * float(x[163]))+ (0.0 * float(x[164]))+ (0.0 * float(x[165]))+ (0.0 * float(x[166]))+ (0.0 * float(x[167]))+ (0.0 * float(x[168]))+ (0.0 * float(x[169]))+ (0.0 * float(x[170]))+ (0.0 * float(x[171]))+ (0.0 * float(x[172]))+ (0.0 * float(x[173]))+ (0.0 * float(x[174]))+ (0.0 * float(x[175]))+ (0.0 * float(x[176]))+ (0.0 * float(x[177]))+ (0.0 * float(x[178]))+ (0.0 * float(x[179]))) + 0.0), 0)
    o[0] = (0.13988172 * h_0)+ (1.4274255 * h_1)+ (0.7283724 * h_2)+ (-0.30148995 * h_3)+ (2.4784334 * h_4)+ (2.2583778 * h_5)+ (3.54412 * h_6)+ (0.0 * h_7) + 1.8013613
    o[1] = (1.148454 * h_0)+ (0.3652476 * h_1)+ (1.2372952 * h_2)+ (2.0067446 * h_3)+ (0.9960107 * h_4)+ (2.7380009 * h_5)+ (2.570123 * h_6)+ (0.0 * h_7) + 1.1230493
    o[2] = (1.8778068 * h_0)+ (0.07107647 * h_1)+ (1.0360659 * h_2)+ (1.9078939 * h_3)+ (1.9883645 * h_4)+ (1.7801372 * h_5)+ (1.7397224 * h_6)+ (0.0 * h_7) + 0.940415

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
        model_cap=1475
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

