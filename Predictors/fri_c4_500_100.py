#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 00:04:44
# Invocation: btc -server brain.brainome.ai Data/fri_c4_500_100.csv -o Models/fri_c4_500_100.py -v -v -v -stopat 89.00 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                56.60%
Model accuracy:                     88.00% (440/500 correct)
Improvement over best guess:        31.40% (of possible 43.4%)
Model capacity (MEC):               511 bits
Generalization ratio:               0.86 bits/bit
Model efficiency:                   0.06%/parameter
System behavior
True Negatives:                     36.40% (182/500)
True Positives:                     51.60% (258/500)
False Negatives:                    5.00% (25/500)
False Positives:                    7.00% (35/500)
True Pos. Rate/Sensitivity/Recall:  0.91
True Neg. Rate/Specificity:         0.84
Precision:                          0.88
F-1 Measure:                        0.90
False Negative Rate/Miss Rate:      0.09
Critical Success Index:             0.81

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
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="fri_c4_500_100.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 100
n_classes = 2


# Preprocessor for CSV files
def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist=[]
    clean.testfile=testfile
    clean.mapping={}
    

    def convert(cell):
        value=str(cell)
        try:
            result=int(value)
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                return result

    def convertclassid(cell):
        if (clean.testfile):
            return convert(cell)
        value=str(cell)
        if (value==''):
            raise ValueError("All cells in the target column must contain a class label.")

        if (not clean.mapping=={}):
            result=-1
            try:
                result=clean.mapping[cell]
            except:
                raise ValueError("Class label '"+value+"' encountered in input not defined in user-provided mapping.")
            if (not result==int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(int(result*100)/100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not result==int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result<0):
                raise ValueError("Integer class labels must be positive and contiguous.")

        return result

    rowcount=0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f=open(outfile,"w+")
        if (headerless==False):
            next(reader,None)
        outbuf=[]
        for row in reader:
            if (row==[]):  # Skip empty rows
                continue
            rowcount=rowcount+1
            rowlen=num_attr
            if (not testfile):
                rowlen=rowlen+1    
            if (not len(row)==rowlen):
                raise ValueError("Column count must match trained predictor. Row "+str(rowcount)+" differs.")
            i=0
            for elem in row:
                if(i+1<len(row)):
                    outbuf.append(str(convert(elem)))
                    outbuf.append(',')
                else:
                    classid=str(convertclassid(elem))
                    outbuf.append(classid)
                i=i+1
            if (len(outbuf)<IOBUF):
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)>=2):
            raise ValueError("Number of classes must be at least 2.")



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    o=[0]*num_output_logits
    h_0 = max((((-4.3790817 * float(x[0]))+ (-8.890103 * float(x[1]))+ (-2.6930432 * float(x[2]))+ (9.509924 * float(x[3]))+ (-4.0767097 * float(x[4]))+ (-1.2943401 * float(x[5]))+ (-0.9173002 * float(x[6]))+ (8.484183 * float(x[7]))+ (-5.244902 * float(x[8]))+ (-7.7444754 * float(x[9]))+ (1.0234814 * float(x[10]))+ (5.5860324 * float(x[11]))+ (-3.3172104 * float(x[12]))+ (-0.5291206 * float(x[13]))+ (-7.593316 * float(x[14]))+ (-4.8125877 * float(x[15]))+ (0.38806146 * float(x[16]))+ (-1.5442631 * float(x[17]))+ (2.8035765 * float(x[18]))+ (1.3301524 * float(x[19]))+ (-0.0035203204 * float(x[20]))+ (-2.4426022 * float(x[21]))+ (-0.49664593 * float(x[22]))+ (-2.717244 * float(x[23]))+ (5.8411064 * float(x[24]))+ (0.9407298 * float(x[25]))+ (5.3781586 * float(x[26]))+ (-1.028016 * float(x[27]))+ (0.642888 * float(x[28]))+ (-0.33295995 * float(x[29]))+ (4.9549646 * float(x[30]))+ (2.2974725 * float(x[31]))+ (0.084992744 * float(x[32]))+ (-1.582168 * float(x[33]))+ (-4.823058 * float(x[34]))+ (3.6501038 * float(x[35]))+ (-6.9299374 * float(x[36]))+ (-2.0842588 * float(x[37]))+ (2.9210398 * float(x[38]))+ (4.732485 * float(x[39]))+ (0.93300974 * float(x[40]))+ (-4.1828747 * float(x[41]))+ (-3.726367 * float(x[42]))+ (-4.831499 * float(x[43]))+ (-3.76911 * float(x[44]))+ (-3.8226748 * float(x[45]))+ (6.8277664 * float(x[46]))+ (-3.1551716 * float(x[47]))+ (-1.8674358 * float(x[48]))+ (2.187747 * float(x[49])))+ ((-5.3091903 * float(x[50]))+ (-0.43977973 * float(x[51]))+ (-3.3476627 * float(x[52]))+ (2.2605703 * float(x[53]))+ (0.23047268 * float(x[54]))+ (-2.364885 * float(x[55]))+ (0.24886237 * float(x[56]))+ (-2.3346372 * float(x[57]))+ (2.6020231 * float(x[58]))+ (2.9805024 * float(x[59]))+ (0.23218958 * float(x[60]))+ (-4.6871247 * float(x[61]))+ (4.41693 * float(x[62]))+ (-1.4395237 * float(x[63]))+ (-5.3729467 * float(x[64]))+ (5.375738 * float(x[65]))+ (1.3977236 * float(x[66]))+ (2.5332146 * float(x[67]))+ (0.54009885 * float(x[68]))+ (-0.9606403 * float(x[69]))+ (2.8119984 * float(x[70]))+ (4.77125 * float(x[71]))+ (-1.1572089 * float(x[72]))+ (-5.325202 * float(x[73]))+ (4.5168877 * float(x[74]))+ (-5.1202106 * float(x[75]))+ (0.017296197 * float(x[76]))+ (2.134806 * float(x[77]))+ (4.0240483 * float(x[78]))+ (0.82900745 * float(x[79]))+ (-4.8909364 * float(x[80]))+ (-1.0463645 * float(x[81]))+ (3.169758 * float(x[82]))+ (3.1655424 * float(x[83]))+ (0.9806236 * float(x[84]))+ (0.7799596 * float(x[85]))+ (-5.581534 * float(x[86]))+ (-1.2765183 * float(x[87]))+ (3.8178234 * float(x[88]))+ (-2.822624 * float(x[89]))+ (-1.7615427 * float(x[90]))+ (-0.64235544 * float(x[91]))+ (0.9661621 * float(x[92]))+ (-4.5082917 * float(x[93]))+ (4.5549173 * float(x[94]))+ (0.29151693 * float(x[95]))+ (1.2734151 * float(x[96]))+ (0.57792073 * float(x[97]))+ (-1.6069144 * float(x[98]))+ (3.0609045 * float(x[99]))) + 0.94651175), 0)
    h_1 = max((((-0.5166406 * float(x[0]))+ (0.011318958 * float(x[1]))+ (1.0638268 * float(x[2]))+ (2.3830683 * float(x[3]))+ (1.5430408 * float(x[4]))+ (0.15783848 * float(x[5]))+ (0.30157894 * float(x[6]))+ (0.0069912137 * float(x[7]))+ (2.0325084 * float(x[8]))+ (0.74044245 * float(x[9]))+ (-0.019619364 * float(x[10]))+ (-0.94738173 * float(x[11]))+ (-0.19685978 * float(x[12]))+ (0.3006455 * float(x[13]))+ (0.32514808 * float(x[14]))+ (-0.6784803 * float(x[15]))+ (-0.02750165 * float(x[16]))+ (-0.675524 * float(x[17]))+ (1.9653747 * float(x[18]))+ (-0.23647605 * float(x[19]))+ (-0.28361395 * float(x[20]))+ (-2.918739 * float(x[21]))+ (-0.008977726 * float(x[22]))+ (-1.4412946 * float(x[23]))+ (-2.0430503 * float(x[24]))+ (-3.5190754 * float(x[25]))+ (3.5140908 * float(x[26]))+ (-0.35902074 * float(x[27]))+ (-2.1183383 * float(x[28]))+ (1.9060038 * float(x[29]))+ (-1.9835283 * float(x[30]))+ (1.1294295 * float(x[31]))+ (0.6984786 * float(x[32]))+ (-0.68680316 * float(x[33]))+ (-2.101989 * float(x[34]))+ (-2.4857666 * float(x[35]))+ (1.1463867 * float(x[36]))+ (0.18265662 * float(x[37]))+ (0.9229982 * float(x[38]))+ (-0.1837126 * float(x[39]))+ (1.9444412 * float(x[40]))+ (-1.1794168 * float(x[41]))+ (-2.8640153 * float(x[42]))+ (-0.4019956 * float(x[43]))+ (-0.49583536 * float(x[44]))+ (-0.16110522 * float(x[45]))+ (-0.2014673 * float(x[46]))+ (1.0054948 * float(x[47]))+ (-0.53426063 * float(x[48]))+ (4.912813 * float(x[49])))+ ((3.6593037 * float(x[50]))+ (2.8500907 * float(x[51]))+ (3.0352023 * float(x[52]))+ (1.122927 * float(x[53]))+ (0.1870023 * float(x[54]))+ (2.0148315 * float(x[55]))+ (2.954379 * float(x[56]))+ (-0.14097652 * float(x[57]))+ (-1.9925752 * float(x[58]))+ (-1.710464 * float(x[59]))+ (1.8543453 * float(x[60]))+ (-2.0610013 * float(x[61]))+ (-0.42849427 * float(x[62]))+ (1.5128851 * float(x[63]))+ (-0.85976017 * float(x[64]))+ (-1.0244066 * float(x[65]))+ (2.3758495 * float(x[66]))+ (0.90006226 * float(x[67]))+ (2.5255241 * float(x[68]))+ (1.6216668 * float(x[69]))+ (-3.4545574 * float(x[70]))+ (-1.0375057 * float(x[71]))+ (-1.3913381 * float(x[72]))+ (0.8685317 * float(x[73]))+ (2.301325 * float(x[74]))+ (-2.6468892 * float(x[75]))+ (-0.96079224 * float(x[76]))+ (0.13130847 * float(x[77]))+ (-3.0068643 * float(x[78]))+ (1.7420689 * float(x[79]))+ (2.1434739 * float(x[80]))+ (2.4973586 * float(x[81]))+ (1.9932101 * float(x[82]))+ (1.6974128 * float(x[83]))+ (-0.3226492 * float(x[84]))+ (-0.48884714 * float(x[85]))+ (1.9291546 * float(x[86]))+ (2.4151704 * float(x[87]))+ (0.42596215 * float(x[88]))+ (1.6622946 * float(x[89]))+ (4.0516644 * float(x[90]))+ (3.5008168 * float(x[91]))+ (-1.8949883 * float(x[92]))+ (6.7876453 * float(x[93]))+ (2.683191 * float(x[94]))+ (-1.0650254 * float(x[95]))+ (-0.84534794 * float(x[96]))+ (-2.1208827 * float(x[97]))+ (3.0020938 * float(x[98]))+ (-1.4173195 * float(x[99]))) + -0.44410792), 0)
    h_2 = max((((-0.7423746 * float(x[0]))+ (-1.8835585 * float(x[1]))+ (-0.94089466 * float(x[2]))+ (-0.5207366 * float(x[3]))+ (-2.7166266 * float(x[4]))+ (0.67298937 * float(x[5]))+ (0.8763666 * float(x[6]))+ (-0.32912713 * float(x[7]))+ (-0.51716465 * float(x[8]))+ (-1.1691571 * float(x[9]))+ (2.6369555 * float(x[10]))+ (2.529255 * float(x[11]))+ (0.77867615 * float(x[12]))+ (1.3770249 * float(x[13]))+ (-1.8781048 * float(x[14]))+ (0.6152863 * float(x[15]))+ (0.5443032 * float(x[16]))+ (1.5409364 * float(x[17]))+ (2.62611 * float(x[18]))+ (1.5896019 * float(x[19]))+ (1.6932827 * float(x[20]))+ (-3.8382123 * float(x[21]))+ (-0.20807426 * float(x[22]))+ (-2.517244 * float(x[23]))+ (-1.0493437 * float(x[24]))+ (2.7155423 * float(x[25]))+ (-1.2581532 * float(x[26]))+ (-0.880437 * float(x[27]))+ (-1.6642615 * float(x[28]))+ (-1.0886111 * float(x[29]))+ (0.94313496 * float(x[30]))+ (1.6483958 * float(x[31]))+ (2.7653632 * float(x[32]))+ (1.6975695 * float(x[33]))+ (2.5300553 * float(x[34]))+ (2.9216812 * float(x[35]))+ (2.3884442 * float(x[36]))+ (-1.4702194 * float(x[37]))+ (1.7397078 * float(x[38]))+ (2.5410244 * float(x[39]))+ (-1.7540138 * float(x[40]))+ (-1.2391481 * float(x[41]))+ (0.7731731 * float(x[42]))+ (-0.8204786 * float(x[43]))+ (3.972285 * float(x[44]))+ (2.879413 * float(x[45]))+ (-3.2346165 * float(x[46]))+ (-2.4036512 * float(x[47]))+ (-1.2437485 * float(x[48]))+ (2.6460848 * float(x[49])))+ ((-1.1386584 * float(x[50]))+ (-0.21021768 * float(x[51]))+ (2.2374296 * float(x[52]))+ (0.5888352 * float(x[53]))+ (1.5403315 * float(x[54]))+ (2.8349307 * float(x[55]))+ (-1.5974598 * float(x[56]))+ (-0.9169107 * float(x[57]))+ (1.0775051 * float(x[58]))+ (0.12638202 * float(x[59]))+ (-1.5234584 * float(x[60]))+ (0.07654753 * float(x[61]))+ (4.305929 * float(x[62]))+ (-3.5594559 * float(x[63]))+ (-0.2635401 * float(x[64]))+ (1.0420936 * float(x[65]))+ (0.52945304 * float(x[66]))+ (-2.2863011 * float(x[67]))+ (-1.7572944 * float(x[68]))+ (-1.7681681 * float(x[69]))+ (3.540047 * float(x[70]))+ (-1.2354385 * float(x[71]))+ (0.9099983 * float(x[72]))+ (1.1643143 * float(x[73]))+ (2.6552913 * float(x[74]))+ (-2.2553117 * float(x[75]))+ (-0.56039464 * float(x[76]))+ (1.2450825 * float(x[77]))+ (2.9048405 * float(x[78]))+ (5.671111 * float(x[79]))+ (-2.645676 * float(x[80]))+ (-0.39274824 * float(x[81]))+ (-1.5101395 * float(x[82]))+ (2.0585253 * float(x[83]))+ (-1.1103303 * float(x[84]))+ (1.1547229 * float(x[85]))+ (-0.8661969 * float(x[86]))+ (0.27386698 * float(x[87]))+ (2.8189092 * float(x[88]))+ (2.172709 * float(x[89]))+ (-1.038976 * float(x[90]))+ (-2.1684675 * float(x[91]))+ (-2.679282 * float(x[92]))+ (-0.48661545 * float(x[93]))+ (-1.5640349 * float(x[94]))+ (-2.2974436 * float(x[95]))+ (3.4685647 * float(x[96]))+ (-2.8462138 * float(x[97]))+ (2.5960932 * float(x[98]))+ (-0.24241148 * float(x[99]))) + 2.9327657), 0)
    h_3 = max((((-1.5432996 * float(x[0]))+ (-0.8297493 * float(x[1]))+ (-0.57046676 * float(x[2]))+ (1.2077515 * float(x[3]))+ (0.12475707 * float(x[4]))+ (1.0084462 * float(x[5]))+ (1.3150548 * float(x[6]))+ (0.7403867 * float(x[7]))+ (-1.0292196 * float(x[8]))+ (2.401667 * float(x[9]))+ (-1.2018762 * float(x[10]))+ (2.533488 * float(x[11]))+ (0.7520614 * float(x[12]))+ (0.7781746 * float(x[13]))+ (0.45994848 * float(x[14]))+ (-0.1747604 * float(x[15]))+ (1.2589442 * float(x[16]))+ (1.2233323 * float(x[17]))+ (0.061848927 * float(x[18]))+ (0.015316819 * float(x[19]))+ (0.5181527 * float(x[20]))+ (1.1361508 * float(x[21]))+ (0.8780248 * float(x[22]))+ (2.1419172 * float(x[23]))+ (-0.7830571 * float(x[24]))+ (1.5199208 * float(x[25]))+ (-0.53368616 * float(x[26]))+ (-2.5284138 * float(x[27]))+ (-1.0489457 * float(x[28]))+ (-0.9648732 * float(x[29]))+ (-0.22024082 * float(x[30]))+ (0.028203247 * float(x[31]))+ (0.34953815 * float(x[32]))+ (-1.648904 * float(x[33]))+ (1.4267173 * float(x[34]))+ (0.76509196 * float(x[35]))+ (0.7611589 * float(x[36]))+ (1.4067248 * float(x[37]))+ (-1.0491899 * float(x[38]))+ (0.7633292 * float(x[39]))+ (-0.5369187 * float(x[40]))+ (0.46978956 * float(x[41]))+ (1.7378756 * float(x[42]))+ (0.9952099 * float(x[43]))+ (0.05682378 * float(x[44]))+ (0.9978055 * float(x[45]))+ (-1.863363 * float(x[46]))+ (1.5476667 * float(x[47]))+ (-0.7389481 * float(x[48]))+ (0.2572828 * float(x[49])))+ ((-0.7079126 * float(x[50]))+ (-0.5646025 * float(x[51]))+ (2.7453225 * float(x[52]))+ (1.0177716 * float(x[53]))+ (2.4408314 * float(x[54]))+ (-1.557997 * float(x[55]))+ (0.43781173 * float(x[56]))+ (-2.027049 * float(x[57]))+ (0.050169878 * float(x[58]))+ (0.55938256 * float(x[59]))+ (-1.6193482 * float(x[60]))+ (2.4053004 * float(x[61]))+ (-0.019632183 * float(x[62]))+ (0.15401322 * float(x[63]))+ (2.4188902 * float(x[64]))+ (1.0284874 * float(x[65]))+ (0.42877495 * float(x[66]))+ (0.056102213 * float(x[67]))+ (0.031339437 * float(x[68]))+ (-0.50832206 * float(x[69]))+ (0.46744633 * float(x[70]))+ (-1.3319637 * float(x[71]))+ (0.22055003 * float(x[72]))+ (1.908522 * float(x[73]))+ (2.104743 * float(x[74]))+ (-1.2897818 * float(x[75]))+ (0.26487806 * float(x[76]))+ (-1.0824089 * float(x[77]))+ (0.91278046 * float(x[78]))+ (1.0856367 * float(x[79]))+ (0.15923981 * float(x[80]))+ (-1.5357513 * float(x[81]))+ (-0.5851823 * float(x[82]))+ (1.08948 * float(x[83]))+ (-1.034221 * float(x[84]))+ (2.2188654 * float(x[85]))+ (-0.2548877 * float(x[86]))+ (-0.65145016 * float(x[87]))+ (1.4372343 * float(x[88]))+ (-0.017691908 * float(x[89]))+ (-0.7717939 * float(x[90]))+ (-1.464382 * float(x[91]))+ (-1.0557878 * float(x[92]))+ (-0.6553642 * float(x[93]))+ (0.27061847 * float(x[94]))+ (0.07991138 * float(x[95]))+ (0.6058721 * float(x[96]))+ (-0.5723913 * float(x[97]))+ (-0.80362695 * float(x[98]))+ (0.7683957 * float(x[99]))) + -1.0520521), 0)
    h_4 = max((((-0.58886766 * float(x[0]))+ (-0.17221063 * float(x[1]))+ (-0.9508673 * float(x[2]))+ (0.3408686 * float(x[3]))+ (-0.8409854 * float(x[4]))+ (-0.65373313 * float(x[5]))+ (-0.6180456 * float(x[6]))+ (0.45050123 * float(x[7]))+ (-0.40251526 * float(x[8]))+ (-0.0922269 * float(x[9]))+ (-0.16255751 * float(x[10]))+ (0.026208635 * float(x[11]))+ (0.9252448 * float(x[12]))+ (-1.0308193 * float(x[13]))+ (0.49337503 * float(x[14]))+ (-0.1061972 * float(x[15]))+ (-1.4324992 * float(x[16]))+ (-0.28259027 * float(x[17]))+ (-0.10501352 * float(x[18]))+ (2.5200856 * float(x[19]))+ (1.4733133 * float(x[20]))+ (0.5508722 * float(x[21]))+ (-0.95653564 * float(x[22]))+ (-1.29237 * float(x[23]))+ (-0.3602175 * float(x[24]))+ (0.69321036 * float(x[25]))+ (0.25387952 * float(x[26]))+ (0.19926265 * float(x[27]))+ (0.11513985 * float(x[28]))+ (1.2946386 * float(x[29]))+ (-1.8232427 * float(x[30]))+ (0.4129917 * float(x[31]))+ (1.3629478 * float(x[32]))+ (0.45936576 * float(x[33]))+ (1.6895175 * float(x[34]))+ (1.482883 * float(x[35]))+ (1.4431192 * float(x[36]))+ (-0.8013641 * float(x[37]))+ (2.0650706 * float(x[38]))+ (0.3078624 * float(x[39]))+ (0.2573658 * float(x[40]))+ (-0.58931005 * float(x[41]))+ (1.4992058 * float(x[42]))+ (0.3802346 * float(x[43]))+ (1.1423124 * float(x[44]))+ (0.33084968 * float(x[45]))+ (1.7493323 * float(x[46]))+ (0.16031809 * float(x[47]))+ (-0.1842502 * float(x[48]))+ (0.93871504 * float(x[49])))+ ((0.33952728 * float(x[50]))+ (-0.059512038 * float(x[51]))+ (-0.12584338 * float(x[52]))+ (0.025534766 * float(x[53]))+ (-1.9739621 * float(x[54]))+ (1.2667323 * float(x[55]))+ (-0.028309403 * float(x[56]))+ (-0.930106 * float(x[57]))+ (-0.1317413 * float(x[58]))+ (-1.3629308 * float(x[59]))+ (0.33328968 * float(x[60]))+ (1.1582438 * float(x[61]))+ (0.015209608 * float(x[62]))+ (-1.4276911 * float(x[63]))+ (1.1658971 * float(x[64]))+ (-0.063136674 * float(x[65]))+ (2.7006698 * float(x[66]))+ (1.951367 * float(x[67]))+ (-1.3559579 * float(x[68]))+ (0.18028188 * float(x[69]))+ (-1.0421156 * float(x[70]))+ (-1.1107514 * float(x[71]))+ (0.003299262 * float(x[72]))+ (-1.2353928 * float(x[73]))+ (0.2713406 * float(x[74]))+ (-0.14770894 * float(x[75]))+ (0.27438417 * float(x[76]))+ (0.7737508 * float(x[77]))+ (1.4619155 * float(x[78]))+ (0.37712488 * float(x[79]))+ (-0.33371487 * float(x[80]))+ (-1.6368339 * float(x[81]))+ (0.8388297 * float(x[82]))+ (2.128073 * float(x[83]))+ (-1.3689034 * float(x[84]))+ (1.9125862 * float(x[85]))+ (0.64596355 * float(x[86]))+ (-0.9485432 * float(x[87]))+ (1.9549669 * float(x[88]))+ (0.14246742 * float(x[89]))+ (1.2426866 * float(x[90]))+ (1.5748427 * float(x[91]))+ (-0.73410237 * float(x[92]))+ (0.6265429 * float(x[93]))+ (1.7426375 * float(x[94]))+ (-0.8086553 * float(x[95]))+ (0.6718231 * float(x[96]))+ (-0.9739242 * float(x[97]))+ (0.5719857 * float(x[98]))+ (-0.080402814 * float(x[99]))) + -1.0395457), 0)
    o[0] = (1.5206276 * h_0)+ (0.37059066 * h_1)+ (-1.9464973 * h_2)+ (1.450576 * h_3)+ (2.225311 * h_4) + -6.3989716

    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)

# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    
    if not args.validate: # Then predict
        if args.cleanfile:
            with open(args.csvfile,'r') as cleancsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                for cleanrow in cleancsvreader:
                    if len(cleanrow)==0:
                        continue
                print(str(','.join(str(j) for j in ([i for i in cleanrow])))+','+str(int(classify(cleanrow))))
        else:
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            with open(cleanfile,'r') as cleancsvfile, open(args.csvfile,'r') as dirtycsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                dirtycsvreader = csv.reader(dirtycsvfile)
                if (not args.headerless):
                        print(','.join(next(dirtycsvreader, None)+['Prediction']))
                for cleanrow,dirtyrow in zip(cleancsvreader,dirtycsvreader):
                    if len(cleanrow)==0:
                        continue
                    print(str(','.join(str(j) for j in ([i for i in dirtyrow])))+','+str(int(classify(cleanrow))))
            os.remove(cleanfile)
            
    else: # Then validate this predictor
        if n_classes==2:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanvalfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanvalfile, -1, args.headerless)
            with open(cleanvalfile,'r') as valcsvfile:
                count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0=0,0,0,0,0,0,0,0
                valcsvreader = csv.reader(valcsvfile)
                for valrow in valcsvreader:
                    if len(valrow)==0:
                        continue
                    if int(classify(valrow[:-1]))==int(float(valrow[-1])):
                        correct_count+=1
                        if int(float(valrow[-1]))==1:
                            num_class_1+=1
                            num_TP+=1
                        else:
                            num_class_0+=1
                            num_TN+=1
                    else:
                        if int(float(valrow[-1]))==1:
                            num_class_1+=1
                            num_FN+=1
                        else:
                            num_class_0+=1
                            num_FP+=1
                    count+=1
        else:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanvalfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanvalfile, -1, args.headerless)
            with open(cleanvalfile,'r') as valcsvfile:
                count,correct_count=0,0
                valcsvreader = csv.reader(valcsvfile)
                numeachclass={}
                for i,valrow in enumerate(valcsvreader):
                    if len(valrow)==0:
                        continue
                    if int(classify(valrow[:-1]))==int(float(valrow[-1])):
                        correct_count+=1
                    if int(float(valrow[-1])) in numeachclass.keys():
                        numeachclass[int(float(valrow[-1]))]+=1
                    else:
                        numeachclass[int(float(valrow[-1]))]=0
                    count+=1

        model_cap=511

        if n_classes==2:

            FN=float(num_FN)*100.0/float(count)
            FP=float(num_FP)*100.0/float(count)
            TN=float(num_TN)*100.0/float(count)
            TP=float(num_TP)*100.0/float(count)
            num_correct=correct_count

            if int(num_TP+num_FN)!=0:
                TPR=num_TP/(num_TP+num_FN) # Sensitivity, Recall
            if int(num_TN+num_FP)!=0:
                TNR=num_TN/(num_TN+num_FP) # Specificity, 
            if int(num_TP+num_FP)!=0:
                PPV=num_TP/(num_TP+num_FP) # Recall
            if int(num_FN+num_TP)!=0:
                FNR=num_FN/(num_FN+num_TP) # Miss rate
            if int(2*num_TP+num_FP+num_FN)!=0:
                FONE=2*num_TP/(2*num_TP+num_FP+num_FN) # F1 Score
            if int(num_TP+num_FN+num_FP)!=0:
                TS=num_TP/(num_TP+num_FN+num_FP) # Critical Success Index

            randguess=int(float(10000.0*max(num_class_1,num_class_0))/count)/100.0
            modelacc=int(float(num_correct*10000)/count)/100.0

            print("System Type:                        Binary classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")
            print("Model efficiency:                   {:.2f}%/parameter".format(int(100*(modelacc-randguess)/model_cap)/100.0))
            print("System behavior")
            print("True Negatives:                     {:.2f}%".format(TN)+" ("+str(int(num_TN))+"/"+str(count)+")")
            print("True Positives:                     {:.2f}%".format(TP)+" ("+str(int(num_TP))+"/"+str(count)+")")
            print("False Negatives:                    {:.2f}%".format(FN)+" ("+str(int(num_FN))+"/"+str(count)+")")
            print("False Positives:                    {:.2f}%".format(FP)+" ("+str(int(num_FP))+"/"+str(count)+")")
            if int(num_TP+num_FN)!=0:
                print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
            if int(num_TN+num_FP)!=0:
                print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
            if int(num_TP+num_FP)!=0:
                print("Precision:                          {:.2f}".format(PPV))
            if int(2*num_TP+num_FP+num_FN)!=0:
                print("F-1 Measure:                        {:.2f}".format(FONE))
            if int(num_TP+num_FN)!=0:
                print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
            if int(num_TP+num_FN+num_FP)!=0:    
                print("Critical Success Index:             {:.2f}".format(TS))
        else:
            num_correct=correct_count
            modelacc=int(float(num_correct*10000)/count)/100.0
            randguess=round(max(numeachclass.values())/sum(numeachclass.values())*100,2)
            print("System Type:                        "+str(n_classes)+"-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")






        os.remove(cleanvalfile)

