#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Feb-28-2020 19:07:52
# Invocation: btc Data/fri_c4_500_100.csv -o Models/fri_c4_500_100.py -v -v -v -stopat 89.00 -port 8090 -e 9
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                56.60%
Model accuracy:                     87.80% (439/500 correct)
Improvement over best guess:        31.20% (of possible 43.4%)
Model capacity (MEC):               613 bits
Generalization ratio:               0.71 bits/bit
Model efficiency:                   0.05%/parameter
System behavior
True Negatives:                     37.40% (187/500)
True Positives:                     50.40% (252/500)
False Negatives:                    6.20% (31/500)
False Positives:                    6.00% (30/500)
True Pos. Rate/Sensitivity/Recall:  0.89
True Neg. Rate/Specificity:         0.86
Precision:                          0.89
F-1 Measure:                        0.89
False Negative Rate/Miss Rate:      0.11
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
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mapped to 0 and 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(result)

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mappable to 0 and 1.")
        finally:
            if (result<0 or result>1):
                raise ValueError("Alpha version restriction: Integer class labels can only be 0 or 1.")
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
                outbuf.append("\n")
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)==2):
            raise ValueError("Number of classes must be 2.")


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    h_0 = max((((-2.5198913 * float(x[0]))+ (-5.731071 * float(x[1]))+ (0.07545418 * float(x[2]))+ (6.4905777 * float(x[3]))+ (1.4081197 * float(x[4]))+ (-1.0312269 * float(x[5]))+ (1.9601897 * float(x[6]))+ (0.94947016 * float(x[7]))+ (-0.7688579 * float(x[8]))+ (-4.0474176 * float(x[9]))+ (0.7321109 * float(x[10]))+ (7.5665736 * float(x[11]))+ (-3.2166786 * float(x[12]))+ (1.4010506 * float(x[13]))+ (-5.342237 * float(x[14]))+ (-0.54636425 * float(x[15]))+ (-5.1283975 * float(x[16]))+ (0.509654 * float(x[17]))+ (1.037588 * float(x[18]))+ (6.644039 * float(x[19]))+ (-1.6585683 * float(x[20]))+ (-2.0102513 * float(x[21]))+ (-4.149198 * float(x[22]))+ (-0.9628954 * float(x[23]))+ (-1.3460575 * float(x[24]))+ (0.84363633 * float(x[25]))+ (1.8979057 * float(x[26]))+ (2.2457373 * float(x[27]))+ (-2.9320407 * float(x[28]))+ (-5.1611853 * float(x[29]))+ (-4.885002 * float(x[30]))+ (4.166166 * float(x[31]))+ (-2.6524005 * float(x[32]))+ (-1.7350236 * float(x[33]))+ (-3.785319 * float(x[34]))+ (-0.0881782 * float(x[35]))+ (-5.954996 * float(x[36]))+ (0.47982642 * float(x[37]))+ (-1.8952608 * float(x[38]))+ (-3.3571467 * float(x[39]))+ (-1.8610979 * float(x[40]))+ (-0.13160346 * float(x[41]))+ (-3.4922159 * float(x[42]))+ (-6.525073 * float(x[43]))+ (2.3282537 * float(x[44]))+ (-5.5786004 * float(x[45]))+ (6.2294292 * float(x[46]))+ (-0.69653934 * float(x[47]))+ (-0.8978379 * float(x[48]))+ (-0.76328707 * float(x[49])))+ ((4.054951 * float(x[50]))+ (1.0557579 * float(x[51]))+ (-3.6586115 * float(x[52]))+ (3.2264738 * float(x[53]))+ (-1.9018395 * float(x[54]))+ (2.7565095 * float(x[55]))+ (7.3322678 * float(x[56]))+ (1.7930524 * float(x[57]))+ (4.6748 * float(x[58]))+ (0.45764294 * float(x[59]))+ (1.5944971 * float(x[60]))+ (1.2505381 * float(x[61]))+ (0.18868351 * float(x[62]))+ (-2.086594 * float(x[63]))+ (0.06676555 * float(x[64]))+ (-2.7656143 * float(x[65]))+ (-2.8365483 * float(x[66]))+ (3.5743139 * float(x[67]))+ (-0.73813665 * float(x[68]))+ (0.57009244 * float(x[69]))+ (1.9747229 * float(x[70]))+ (4.859552 * float(x[71]))+ (-3.5542321 * float(x[72]))+ (-6.5407863 * float(x[73]))+ (-1.9598666 * float(x[74]))+ (3.3480308 * float(x[75]))+ (-2.3026094 * float(x[76]))+ (-2.6383212 * float(x[77]))+ (-2.2868826 * float(x[78]))+ (1.3014797 * float(x[79]))+ (2.199601 * float(x[80]))+ (1.7704344 * float(x[81]))+ (0.37858975 * float(x[82]))+ (5.755072 * float(x[83]))+ (4.3227057 * float(x[84]))+ (-1.2376891 * float(x[85]))+ (-0.8221666 * float(x[86]))+ (-6.0957937 * float(x[87]))+ (4.5627537 * float(x[88]))+ (1.2429644 * float(x[89]))+ (6.9162216 * float(x[90]))+ (-1.646189 * float(x[91]))+ (-3.75528 * float(x[92]))+ (4.6740932 * float(x[93]))+ (-0.42812783 * float(x[94]))+ (-6.077069 * float(x[95]))+ (-7.754732 * float(x[96]))+ (0.10089382 * float(x[97]))+ (4.5169883 * float(x[98]))+ (-2.6986582 * float(x[99]))) + 4.445635), 0)
    h_1 = max((((-1.8156537 * float(x[0]))+ (-2.4986265 * float(x[1]))+ (-1.9085081 * float(x[2]))+ (0.9881215 * float(x[3]))+ (-1.8230106 * float(x[4]))+ (-0.011040924 * float(x[5]))+ (-1.7921008 * float(x[6]))+ (3.4899204 * float(x[7]))+ (-5.495736 * float(x[8]))+ (1.8614129 * float(x[9]))+ (-2.628661 * float(x[10]))+ (-1.635526 * float(x[11]))+ (1.3620236 * float(x[12]))+ (1.7807302 * float(x[13]))+ (4.4204426 * float(x[14]))+ (-1.1773434 * float(x[15]))+ (3.30192 * float(x[16]))+ (-2.5618958 * float(x[17]))+ (0.14288993 * float(x[18]))+ (0.4845079 * float(x[19]))+ (0.3431784 * float(x[20]))+ (-0.013629826 * float(x[21]))+ (-1.6204473 * float(x[22]))+ (2.0420332 * float(x[23]))+ (-5.0447073 * float(x[24]))+ (-2.1555781 * float(x[25]))+ (-3.0212462 * float(x[26]))+ (-2.0854468 * float(x[27]))+ (1.8080945 * float(x[28]))+ (1.6481861 * float(x[29]))+ (4.1253166 * float(x[30]))+ (-2.5556238 * float(x[31]))+ (-0.22087806 * float(x[32]))+ (1.8198243 * float(x[33]))+ (-0.29511166 * float(x[34]))+ (-2.7276485 * float(x[35]))+ (5.038899 * float(x[36]))+ (-1.1102055 * float(x[37]))+ (-1.4513798 * float(x[38]))+ (1.7391139 * float(x[39]))+ (3.1849697 * float(x[40]))+ (-1.743388 * float(x[41]))+ (4.502771 * float(x[42]))+ (1.1002638 * float(x[43]))+ (3.7505133 * float(x[44]))+ (-0.26549736 * float(x[45]))+ (-2.4709988 * float(x[46]))+ (-1.5160785 * float(x[47]))+ (1.1291883 * float(x[48]))+ (-2.033339 * float(x[49])))+ ((-1.499631 * float(x[50]))+ (0.20984381 * float(x[51]))+ (0.84149146 * float(x[52]))+ (3.919627 * float(x[53]))+ (1.0788093 * float(x[54]))+ (-0.8621963 * float(x[55]))+ (-0.48281834 * float(x[56]))+ (-1.3309491 * float(x[57]))+ (-2.423327 * float(x[58]))+ (-1.6849083 * float(x[59]))+ (0.88450795 * float(x[60]))+ (-0.54437953 * float(x[61]))+ (-0.7717983 * float(x[62]))+ (-1.1110448 * float(x[63]))+ (-2.4391487 * float(x[64]))+ (7.5182614 * float(x[65]))+ (4.3712044 * float(x[66]))+ (3.855235 * float(x[67]))+ (0.35910678 * float(x[68]))+ (-2.3948894 * float(x[69]))+ (-1.4519619 * float(x[70]))+ (-2.3505359 * float(x[71]))+ (-1.5638458 * float(x[72]))+ (0.68669045 * float(x[73]))+ (6.067931 * float(x[74]))+ (-4.471049 * float(x[75]))+ (0.18137951 * float(x[76]))+ (2.4721014 * float(x[77]))+ (-2.3321562 * float(x[78]))+ (6.521561 * float(x[79]))+ (-0.2872034 * float(x[80]))+ (0.21143648 * float(x[81]))+ (1.6661818 * float(x[82]))+ (1.5735087 * float(x[83]))+ (-0.27534753 * float(x[84]))+ (-2.0037353 * float(x[85]))+ (0.6152998 * float(x[86]))+ (-2.8206716 * float(x[87]))+ (-1.452339 * float(x[88]))+ (2.4361293 * float(x[89]))+ (-1.2943848 * float(x[90]))+ (1.7162344 * float(x[91]))+ (-1.6355529 * float(x[92]))+ (-4.8206005 * float(x[93]))+ (4.7757473 * float(x[94]))+ (0.5417273 * float(x[95]))+ (0.9538006 * float(x[96]))+ (1.6538587 * float(x[97]))+ (1.5629245 * float(x[98]))+ (-1.6328893 * float(x[99]))) + -1.098462), 0)
    h_2 = max((((-1.9441228 * float(x[0]))+ (-3.2736719 * float(x[1]))+ (-1.6027895 * float(x[2]))+ (-3.225448 * float(x[3]))+ (-3.727193 * float(x[4]))+ (-1.7874799 * float(x[5]))+ (0.5517678 * float(x[6]))+ (-3.0244668 * float(x[7]))+ (1.2109238 * float(x[8]))+ (-1.2654649 * float(x[9]))+ (0.14339685 * float(x[10]))+ (0.42130622 * float(x[11]))+ (-0.31689507 * float(x[12]))+ (-1.5412395 * float(x[13]))+ (0.2923538 * float(x[14]))+ (-3.1872237 * float(x[15]))+ (0.15216045 * float(x[16]))+ (-3.7366698 * float(x[17]))+ (-0.7747075 * float(x[18]))+ (2.0811307 * float(x[19]))+ (-2.795552 * float(x[20]))+ (-4.558029 * float(x[21]))+ (1.6785072 * float(x[22]))+ (3.7167995 * float(x[23]))+ (-3.4529173 * float(x[24]))+ (-0.53484106 * float(x[25]))+ (1.3631545 * float(x[26]))+ (0.68061364 * float(x[27]))+ (0.73576164 * float(x[28]))+ (0.17321451 * float(x[29]))+ (0.30325723 * float(x[30]))+ (2.1344798 * float(x[31]))+ (0.8354277 * float(x[32]))+ (-3.1401973 * float(x[33]))+ (0.8905476 * float(x[34]))+ (1.7633176 * float(x[35]))+ (-1.0117227 * float(x[36]))+ (1.1199293 * float(x[37]))+ (1.7135273 * float(x[38]))+ (-0.048464675 * float(x[39]))+ (-2.440389 * float(x[40]))+ (0.632013 * float(x[41]))+ (0.37381598 * float(x[42]))+ (-0.7236801 * float(x[43]))+ (3.316756 * float(x[44]))+ (0.5647525 * float(x[45]))+ (-0.9444609 * float(x[46]))+ (1.0659769 * float(x[47]))+ (1.7479955 * float(x[48]))+ (0.1263932 * float(x[49])))+ ((2.2907543 * float(x[50]))+ (1.5835857 * float(x[51]))+ (2.0224295 * float(x[52]))+ (-1.0991648 * float(x[53]))+ (0.8952407 * float(x[54]))+ (-1.7236267 * float(x[55]))+ (0.99374443 * float(x[56]))+ (1.4055493 * float(x[57]))+ (3.465802 * float(x[58]))+ (2.6808023 * float(x[59]))+ (-0.35207194 * float(x[60]))+ (5.3404865 * float(x[61]))+ (-0.9227122 * float(x[62]))+ (1.1666838 * float(x[63]))+ (2.3118982 * float(x[64]))+ (-1.6404421 * float(x[65]))+ (0.6432062 * float(x[66]))+ (-2.1447232 * float(x[67]))+ (0.447415 * float(x[68]))+ (-2.1786091 * float(x[69]))+ (0.37227848 * float(x[70]))+ (0.28103092 * float(x[71]))+ (-2.3467104 * float(x[72]))+ (-2.8712285 * float(x[73]))+ (-0.44648474 * float(x[74]))+ (0.46263444 * float(x[75]))+ (2.8366697 * float(x[76]))+ (0.5992206 * float(x[77]))+ (-1.9055035 * float(x[78]))+ (-1.5615264 * float(x[79]))+ (2.427969 * float(x[80]))+ (-0.2575926 * float(x[81]))+ (0.8527104 * float(x[82]))+ (0.40051398 * float(x[83]))+ (1.3900694 * float(x[84]))+ (-3.2522194 * float(x[85]))+ (-0.97176665 * float(x[86]))+ (-1.448337 * float(x[87]))+ (-1.6372727 * float(x[88]))+ (1.4254069 * float(x[89]))+ (1.0904953 * float(x[90]))+ (-0.31095234 * float(x[91]))+ (0.5798406 * float(x[92]))+ (-0.7720754 * float(x[93]))+ (-1.8568215 * float(x[94]))+ (-1.8282928 * float(x[95]))+ (-1.4479032 * float(x[96]))+ (-0.03829807 * float(x[97]))+ (-1.1182046 * float(x[98]))+ (-1.1617975 * float(x[99]))) + 2.945331), 0)
    h_3 = max((((2.297214 * float(x[0]))+ (3.1189237 * float(x[1]))+ (2.4421372 * float(x[2]))+ (-1.2428955 * float(x[3]))+ (0.7288078 * float(x[4]))+ (3.0378704 * float(x[5]))+ (1.8223789 * float(x[6]))+ (1.6387382 * float(x[7]))+ (-1.2589358 * float(x[8]))+ (2.791053 * float(x[9]))+ (1.4004267 * float(x[10]))+ (0.19994853 * float(x[11]))+ (0.4140555 * float(x[12]))+ (2.2592838 * float(x[13]))+ (-1.3171972 * float(x[14]))+ (1.8573309 * float(x[15]))+ (-0.85627514 * float(x[16]))+ (1.9764013 * float(x[17]))+ (2.6192477 * float(x[18]))+ (0.21593827 * float(x[19]))+ (2.009214 * float(x[20]))+ (-0.50416446 * float(x[21]))+ (0.56075954 * float(x[22]))+ (-2.8583946 * float(x[23]))+ (-0.157213 * float(x[24]))+ (0.69257814 * float(x[25]))+ (1.9014295 * float(x[26]))+ (2.2426531 * float(x[27]))+ (-0.019179087 * float(x[28]))+ (-1.6009874 * float(x[29]))+ (0.40615615 * float(x[30]))+ (2.590603 * float(x[31]))+ (-3.122484 * float(x[32]))+ (-2.040018 * float(x[33]))+ (-0.31362683 * float(x[34]))+ (2.666945 * float(x[35]))+ (1.8549023 * float(x[36]))+ (0.30677825 * float(x[37]))+ (-0.99012566 * float(x[38]))+ (-0.47429895 * float(x[39]))+ (0.63005656 * float(x[40]))+ (-0.18889907 * float(x[41]))+ (0.8512504 * float(x[42]))+ (0.23787944 * float(x[43]))+ (-0.88755435 * float(x[44]))+ (0.78752 * float(x[45]))+ (1.4149677 * float(x[46]))+ (1.4179008 * float(x[47]))+ (-0.5675035 * float(x[48]))+ (1.4767342 * float(x[49])))+ ((1.7509637 * float(x[50]))+ (-1.6058111 * float(x[51]))+ (-1.6698345 * float(x[52]))+ (-0.59642595 * float(x[53]))+ (2.7444918 * float(x[54]))+ (2.7627199 * float(x[55]))+ (-1.0974896 * float(x[56]))+ (-1.1543242 * float(x[57]))+ (-0.7744973 * float(x[58]))+ (0.056580376 * float(x[59]))+ (-1.5984141 * float(x[60]))+ (-0.051219624 * float(x[61]))+ (-1.0522568 * float(x[62]))+ (1.8978817 * float(x[63]))+ (1.6542687 * float(x[64]))+ (-0.26915833 * float(x[65]))+ (0.028998611 * float(x[66]))+ (-1.2090507 * float(x[67]))+ (-2.4817939 * float(x[68]))+ (-0.51745045 * float(x[69]))+ (1.2213917 * float(x[70]))+ (-0.24224429 * float(x[71]))+ (-0.6386912 * float(x[72]))+ (0.23866199 * float(x[73]))+ (1.9058518 * float(x[74]))+ (-0.034087423 * float(x[75]))+ (0.70081717 * float(x[76]))+ (0.71440595 * float(x[77]))+ (1.7590817 * float(x[78]))+ (-1.2403377 * float(x[79]))+ (0.8931852 * float(x[80]))+ (1.8702229 * float(x[81]))+ (-1.8049835 * float(x[82]))+ (-1.5166572 * float(x[83]))+ (2.673698 * float(x[84]))+ (0.73369634 * float(x[85]))+ (-0.80273926 * float(x[86]))+ (-2.7791016 * float(x[87]))+ (0.82849354 * float(x[88]))+ (0.96398383 * float(x[89]))+ (-0.57298195 * float(x[90]))+ (-1.7570738 * float(x[91]))+ (-0.9332443 * float(x[92]))+ (0.37254083 * float(x[93]))+ (-3.674557 * float(x[94]))+ (-0.81527126 * float(x[95]))+ (-1.0037242 * float(x[96]))+ (0.46520627 * float(x[97]))+ (0.047578674 * float(x[98]))+ (2.174296 * float(x[99]))) + 3.476194), 0)
    h_4 = max((((-0.40690765 * float(x[0]))+ (-0.6397425 * float(x[1]))+ (-0.4579177 * float(x[2]))+ (1.7527673 * float(x[3]))+ (-0.31474212 * float(x[4]))+ (0.32112175 * float(x[5]))+ (0.77004844 * float(x[6]))+ (-0.6698892 * float(x[7]))+ (0.43259463 * float(x[8]))+ (0.8753661 * float(x[9]))+ (-0.8557915 * float(x[10]))+ (-1.2043742 * float(x[11]))+ (0.0014430594 * float(x[12]))+ (0.77451926 * float(x[13]))+ (0.010819288 * float(x[14]))+ (0.58443034 * float(x[15]))+ (-0.2751301 * float(x[16]))+ (-0.33555892 * float(x[17]))+ (-0.4594113 * float(x[18]))+ (-0.6188041 * float(x[19]))+ (-0.7021433 * float(x[20]))+ (0.57490003 * float(x[21]))+ (0.023696119 * float(x[22]))+ (-1.4312195 * float(x[23]))+ (-2.481525 * float(x[24]))+ (-1.381824 * float(x[25]))+ (-0.575313 * float(x[26]))+ (0.19001727 * float(x[27]))+ (-0.5831565 * float(x[28]))+ (-0.61854243 * float(x[29]))+ (-1.4630924 * float(x[30]))+ (-1.0334847 * float(x[31]))+ (0.3496556 * float(x[32]))+ (-1.2961279 * float(x[33]))+ (0.08248454 * float(x[34]))+ (-1.1602455 * float(x[35]))+ (0.41827014 * float(x[36]))+ (1.3096097 * float(x[37]))+ (0.5327373 * float(x[38]))+ (-0.108187795 * float(x[39]))+ (-0.73341095 * float(x[40]))+ (0.98813057 * float(x[41]))+ (-0.4929089 * float(x[42]))+ (-1.3211496 * float(x[43]))+ (-0.782744 * float(x[44]))+ (-0.12711589 * float(x[45]))+ (0.3234881 * float(x[46]))+ (0.013795812 * float(x[47]))+ (2.0529609 * float(x[48]))+ (1.7225633 * float(x[49])))+ ((0.2985683 * float(x[50]))+ (1.8992488 * float(x[51]))+ (-0.6565486 * float(x[52]))+ (-1.4514236 * float(x[53]))+ (-2.126312 * float(x[54]))+ (-0.4261765 * float(x[55]))+ (0.032144424 * float(x[56]))+ (-0.09288099 * float(x[57]))+ (1.2756206 * float(x[58]))+ (0.40783367 * float(x[59]))+ (0.23966046 * float(x[60]))+ (-0.048772622 * float(x[61]))+ (-0.4998152 * float(x[62]))+ (-0.415432 * float(x[63]))+ (1.2790023 * float(x[64]))+ (-0.9570451 * float(x[65]))+ (-0.6473294 * float(x[66]))+ (-1.113784 * float(x[67]))+ (1.6103886 * float(x[68]))+ (-0.858989 * float(x[69]))+ (-1.1563176 * float(x[70]))+ (0.54757243 * float(x[71]))+ (1.3705659 * float(x[72]))+ (-0.64213663 * float(x[73]))+ (-1.0234898 * float(x[74]))+ (0.9838638 * float(x[75]))+ (0.89919406 * float(x[76]))+ (0.9937983 * float(x[77]))+ (-0.9550917 * float(x[78]))+ (-0.2079766 * float(x[79]))+ (0.049922872 * float(x[80]))+ (-0.30770946 * float(x[81]))+ (-1.4815311 * float(x[82]))+ (-0.8275813 * float(x[83]))+ (0.86725074 * float(x[84]))+ (0.7746109 * float(x[85]))+ (-0.0063378233 * float(x[86]))+ (0.6779244 * float(x[87]))+ (0.8900688 * float(x[88]))+ (1.233301 * float(x[89]))+ (0.016531765 * float(x[90]))+ (0.51626843 * float(x[91]))+ (0.67749643 * float(x[92]))+ (-0.13184474 * float(x[93]))+ (-0.4556701 * float(x[94]))+ (1.6094338 * float(x[95]))+ (-0.51975673 * float(x[96]))+ (-0.37631556 * float(x[97]))+ (0.538733 * float(x[98]))+ (1.0116904 * float(x[99]))) + -1.0332586), 0)
    h_5 = max((((-0.6059887 * float(x[0]))+ (0.37342647 * float(x[1]))+ (-0.81503224 * float(x[2]))+ (0.2082174 * float(x[3]))+ (0.34758878 * float(x[4]))+ (-0.88353896 * float(x[5]))+ (-1.0735781 * float(x[6]))+ (0.06153976 * float(x[7]))+ (0.15303305 * float(x[8]))+ (0.7657345 * float(x[9]))+ (-0.1433397 * float(x[10]))+ (-0.124410026 * float(x[11]))+ (1.0308496 * float(x[12]))+ (-0.21838662 * float(x[13]))+ (-1.2549838 * float(x[14]))+ (-0.4557102 * float(x[15]))+ (0.76498497 * float(x[16]))+ (-0.6651842 * float(x[17]))+ (0.41607323 * float(x[18]))+ (0.7885867 * float(x[19]))+ (0.23906267 * float(x[20]))+ (-0.36111867 * float(x[21]))+ (1.0789897 * float(x[22]))+ (-0.7332763 * float(x[23]))+ (0.53236836 * float(x[24]))+ (0.13452448 * float(x[25]))+ (0.4106561 * float(x[26]))+ (-1.233894 * float(x[27]))+ (-0.7248726 * float(x[28]))+ (0.88713235 * float(x[29]))+ (-0.68209785 * float(x[30]))+ (0.094473585 * float(x[31]))+ (0.16068164 * float(x[32]))+ (-0.41120607 * float(x[33]))+ (-0.814591 * float(x[34]))+ (0.10857876 * float(x[35]))+ (-1.5343119 * float(x[36]))+ (-0.88389534 * float(x[37]))+ (1.2990061 * float(x[38]))+ (-0.25680542 * float(x[39]))+ (0.6940198 * float(x[40]))+ (-0.94148576 * float(x[41]))+ (0.8204077 * float(x[42]))+ (-0.4410676 * float(x[43]))+ (0.90619284 * float(x[44]))+ (0.049207266 * float(x[45]))+ (0.42048085 * float(x[46]))+ (0.5231684 * float(x[47]))+ (-0.33104128 * float(x[48]))+ (-0.5736358 * float(x[49])))+ ((-0.21714886 * float(x[50]))+ (-1.2220615 * float(x[51]))+ (-0.39482316 * float(x[52]))+ (0.13891801 * float(x[53]))+ (-0.39687368 * float(x[54]))+ (0.13744882 * float(x[55]))+ (0.22309561 * float(x[56]))+ (-0.016795605 * float(x[57]))+ (-0.1637784 * float(x[58]))+ (1.5555623 * float(x[59]))+ (-0.36648333 * float(x[60]))+ (-1.0407404 * float(x[61]))+ (-0.14149056 * float(x[62]))+ (0.3994823 * float(x[63]))+ (0.6765309 * float(x[64]))+ (0.85845804 * float(x[65]))+ (0.098617874 * float(x[66]))+ (-0.0302326 * float(x[67]))+ (-0.3034395 * float(x[68]))+ (0.44725612 * float(x[69]))+ (0.29651645 * float(x[70]))+ (-0.34452555 * float(x[71]))+ (-0.14868595 * float(x[72]))+ (-0.62708503 * float(x[73]))+ (-0.013866007 * float(x[74]))+ (-0.7619838 * float(x[75]))+ (0.82303894 * float(x[76]))+ (-0.09308509 * float(x[77]))+ (-0.39706272 * float(x[78]))+ (-0.61352867 * float(x[79]))+ (-0.43077695 * float(x[80]))+ (0.17946722 * float(x[81]))+ (1.16111 * float(x[82]))+ (0.23849334 * float(x[83]))+ (-1.0912194 * float(x[84]))+ (0.07795843 * float(x[85]))+ (-0.6041912 * float(x[86]))+ (-0.19783828 * float(x[87]))+ (1.4654667 * float(x[88]))+ (-1.4261767 * float(x[89]))+ (-0.5630655 * float(x[90]))+ (0.884111 * float(x[91]))+ (0.42503595 * float(x[92]))+ (1.109508 * float(x[93]))+ (0.09127182 * float(x[94]))+ (-0.21964195 * float(x[95]))+ (-0.15466468 * float(x[96]))+ (-0.37576294 * float(x[97]))+ (-0.723197 * float(x[98]))+ (0.032323636 * float(x[99]))) + -0.115115814), 0)
    o_0 = (2.8592958 * h_0)+ (0.6744296 * h_1)+ (-2.4403534 * h_2)+ (-2.3972359 * h_3)+ (2.2375998 * h_4)+ (3.1436439 * h_5) + -3.9419208
             
    if num_output_logits==1:
        return o_0>=0
    else:
        return argmax([eval('o'+str(i)) for i in range(num_output_logits)])

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

        model_cap=613

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


        os.remove(cleanvalfile)

