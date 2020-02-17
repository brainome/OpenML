#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 03:27:28
# Invocation: btc -v fri_c4_500_100-4.csv -o fri_c4_500_100-4.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                56.60%
Model accuracy:                     89.60% (448/500 correct)
Improvement over best guess:        33.00% (of possible 43.4%)
Model capacity (MEC):               613 bits
Generalization ratio:               0.73 bits/bit
Model efficiency:                   0.05%/parameter
System behavior
True Negatives:                     52.00% (260/500)
True Positives:                     37.60% (188/500)
False Negatives:                    5.80% (29/500)
False Positives:                    4.60% (23/500)
True Pos. Rate/Sensitivity/Recall:  0.87
True Neg. Rate/Specificity:         0.92
Precision:                          0.89
F-1 Measure:                        0.88
False Negative Rate/Miss Rate:      0.13
Critical Success Index:             0.78
Model bias:                         0.37% higher chance to pick class 0
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="fri_c4_500_100-4.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 100

# Preprocessor for CSV files
def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist=[]
    clean.testfile=testfile
    
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
            raise ValueError("All cells in the target column need to contain a class label.")
        try:
            result=int(value)
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                if (not str(result) in clean.classlist):
                    clean.classlist=clean.classlist+[str(result)]
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                if (result in clean.classlist):
                    result=clean.classlist.index(result)
                else:
                    clean.classlist=clean.classlist+[result]
                    result=clean.classlist.index(result)
                return result
    rowcount=0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f=open(outfile,"w+")
        if (headerless==False):
            next(reader,None)
        outbuf=[]
        for row in reader:
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
    h_0 = max((((8.436645 * float(x[0]))+ (11.614343 * float(x[1]))+ (-3.0123682 * float(x[2]))+ (-9.590998 * float(x[3]))+ (-0.76300687 * float(x[4]))+ (-0.47109038 * float(x[5]))+ (0.7284226 * float(x[6]))+ (-0.49145365 * float(x[7]))+ (-4.9282703 * float(x[8]))+ (3.4599953 * float(x[9]))+ (-6.084374 * float(x[10]))+ (0.11765119 * float(x[11]))+ (0.17032681 * float(x[12]))+ (-5.0203032 * float(x[13]))+ (0.12903324 * float(x[14]))+ (-2.1385942 * float(x[15]))+ (-5.3487315 * float(x[16]))+ (-3.218702 * float(x[17]))+ (2.0236034 * float(x[18]))+ (-7.341338 * float(x[19]))+ (5.5710073 * float(x[20]))+ (-1.4148806 * float(x[21]))+ (-1.0181466 * float(x[22]))+ (2.6519341 * float(x[23]))+ (0.788585 * float(x[24]))+ (3.0517263 * float(x[25]))+ (-6.605786 * float(x[26]))+ (2.3484583 * float(x[27]))+ (-4.2436643 * float(x[28]))+ (-2.7830422 * float(x[29]))+ (5.5074744 * float(x[30]))+ (2.3768895 * float(x[31]))+ (2.8750634 * float(x[32]))+ (-1.8818231 * float(x[33]))+ (0.92942303 * float(x[34]))+ (3.19078 * float(x[35]))+ (3.260999 * float(x[36]))+ (-0.15466404 * float(x[37]))+ (1.0885121 * float(x[38]))+ (4.7691016 * float(x[39]))+ (-2.645485 * float(x[40]))+ (2.7781022 * float(x[41]))+ (-0.16920783 * float(x[42]))+ (-0.66902906 * float(x[43]))+ (0.93457705 * float(x[44]))+ (7.3615303 * float(x[45]))+ (0.37025157 * float(x[46]))+ (-1.3641462 * float(x[47]))+ (-1.0382447 * float(x[48]))+ (-0.52964073 * float(x[49])))+ ((-0.8628506 * float(x[50]))+ (1.8254813 * float(x[51]))+ (-2.9555366 * float(x[52]))+ (-1.1338066 * float(x[53]))+ (2.4354696 * float(x[54]))+ (0.6079065 * float(x[55]))+ (-7.477061 * float(x[56]))+ (4.7736564 * float(x[57]))+ (-2.703337 * float(x[58]))+ (0.041134484 * float(x[59]))+ (-0.091324806 * float(x[60]))+ (0.86995053 * float(x[61]))+ (-1.8054047 * float(x[62]))+ (2.563534 * float(x[63]))+ (0.90299785 * float(x[64]))+ (0.79463714 * float(x[65]))+ (5.6032557 * float(x[66]))+ (-3.9506552 * float(x[67]))+ (-1.6279341 * float(x[68]))+ (0.020761866 * float(x[69]))+ (2.5291328 * float(x[70]))+ (-7.6606154 * float(x[71]))+ (0.9269009 * float(x[72]))+ (6.79794 * float(x[73]))+ (-5.644323 * float(x[74]))+ (1.4982516 * float(x[75]))+ (-0.59591985 * float(x[76]))+ (1.2830099 * float(x[77]))+ (-5.6949654 * float(x[78]))+ (0.70139354 * float(x[79]))+ (2.564126 * float(x[80]))+ (1.1150602 * float(x[81]))+ (1.8022736 * float(x[82]))+ (0.67172617 * float(x[83]))+ (-2.107957 * float(x[84]))+ (-0.16359334 * float(x[85]))+ (1.0353442 * float(x[86]))+ (-0.6701112 * float(x[87]))+ (1.4339494 * float(x[88]))+ (0.5598557 * float(x[89]))+ (-3.2852228 * float(x[90]))+ (2.2544863 * float(x[91]))+ (-2.3720508 * float(x[92]))+ (1.6843885 * float(x[93]))+ (-10.913015 * float(x[94]))+ (4.3449597 * float(x[95]))+ (-4.3242154 * float(x[96]))+ (-1.0035626 * float(x[97]))+ (-1.6838737 * float(x[98]))+ (-1.2047665 * float(x[99]))) + -2.477725), 0)
    h_1 = max((((-1.6194706 * float(x[0]))+ (-1.3281626 * float(x[1]))+ (0.65834415 * float(x[2]))+ (-2.8103983 * float(x[3]))+ (-0.7769023 * float(x[4]))+ (-3.948988 * float(x[5]))+ (-3.610385 * float(x[6]))+ (-0.98842424 * float(x[7]))+ (0.20053661 * float(x[8]))+ (-3.9214945 * float(x[9]))+ (-0.66994494 * float(x[10]))+ (-6.669332 * float(x[11]))+ (0.059164867 * float(x[12]))+ (-3.5556548 * float(x[13]))+ (0.8277972 * float(x[14]))+ (-2.179421 * float(x[15]))+ (2.7551644 * float(x[16]))+ (2.5361583 * float(x[17]))+ (1.0314513 * float(x[18]))+ (1.3344182 * float(x[19]))+ (-1.1237402 * float(x[20]))+ (-1.2671406 * float(x[21]))+ (-2.7578976 * float(x[22]))+ (1.5225562 * float(x[23]))+ (-1.1733011 * float(x[24]))+ (2.0912793 * float(x[25]))+ (-0.3114839 * float(x[26]))+ (0.59981537 * float(x[27]))+ (1.8949308 * float(x[28]))+ (1.1862434 * float(x[29]))+ (-1.7671661 * float(x[30]))+ (-1.0750558 * float(x[31]))+ (-1.1056423 * float(x[32]))+ (1.3639758 * float(x[33]))+ (1.0689024 * float(x[34]))+ (1.4504486 * float(x[35]))+ (2.2966843 * float(x[36]))+ (1.3191153 * float(x[37]))+ (-1.0186436 * float(x[38]))+ (-0.85754436 * float(x[39]))+ (3.164386 * float(x[40]))+ (2.0737314 * float(x[41]))+ (2.1545923 * float(x[42]))+ (-1.5609444 * float(x[43]))+ (0.60503435 * float(x[44]))+ (-4.3805504 * float(x[45]))+ (-1.3622878 * float(x[46]))+ (3.4633803 * float(x[47]))+ (0.8550982 * float(x[48]))+ (-2.0727124 * float(x[49])))+ ((-2.3505423 * float(x[50]))+ (-1.2274911 * float(x[51]))+ (3.7636583 * float(x[52]))+ (-3.7717402 * float(x[53]))+ (-1.671748 * float(x[54]))+ (-1.4781058 * float(x[55]))+ (1.0556562 * float(x[56]))+ (-0.26188564 * float(x[57]))+ (-0.08220605 * float(x[58]))+ (0.28732514 * float(x[59]))+ (-2.270214 * float(x[60]))+ (2.226843 * float(x[61]))+ (2.596708 * float(x[62]))+ (-3.2284603 * float(x[63]))+ (-1.0475197 * float(x[64]))+ (-1.9676337 * float(x[65]))+ (-5.880414 * float(x[66]))+ (0.7788007 * float(x[67]))+ (-0.015409211 * float(x[68]))+ (0.03676941 * float(x[69]))+ (3.3362365 * float(x[70]))+ (5.0649834 * float(x[71]))+ (0.7726201 * float(x[72]))+ (0.22587962 * float(x[73]))+ (2.4095745 * float(x[74]))+ (-1.2640065 * float(x[75]))+ (-0.34237337 * float(x[76]))+ (-3.2305396 * float(x[77]))+ (0.6548914 * float(x[78]))+ (-3.3715563 * float(x[79]))+ (-0.07661421 * float(x[80]))+ (2.3724415 * float(x[81]))+ (-0.90744394 * float(x[82]))+ (-5.09326 * float(x[83]))+ (5.0901127 * float(x[84]))+ (-3.999462 * float(x[85]))+ (-1.9006151 * float(x[86]))+ (0.9899884 * float(x[87]))+ (-4.249314 * float(x[88]))+ (-0.9373778 * float(x[89]))+ (-1.0329564 * float(x[90]))+ (2.0068743 * float(x[91]))+ (4.9930115 * float(x[92]))+ (1.70845 * float(x[93]))+ (3.4646966 * float(x[94]))+ (-0.8755619 * float(x[95]))+ (-3.0633247 * float(x[96]))+ (0.21357726 * float(x[97]))+ (-3.0332558 * float(x[98]))+ (3.0679917 * float(x[99]))) + -1.1057239), 0)
    h_2 = max((((0.54043835 * float(x[0]))+ (2.2998564 * float(x[1]))+ (1.1065618 * float(x[2]))+ (-0.3532886 * float(x[3]))+ (2.5053234 * float(x[4]))+ (-0.34861338 * float(x[5]))+ (2.2514899 * float(x[6]))+ (-1.1611769 * float(x[7]))+ (-1.1701072 * float(x[8]))+ (0.68060905 * float(x[9]))+ (-1.4591714 * float(x[10]))+ (0.24848047 * float(x[11]))+ (-1.5546628 * float(x[12]))+ (2.3900523 * float(x[13]))+ (-0.8064239 * float(x[14]))+ (-1.2088002 * float(x[15]))+ (-0.65801746 * float(x[16]))+ (-1.4083639 * float(x[17]))+ (-1.283912 * float(x[18]))+ (-4.399426 * float(x[19]))+ (0.72105753 * float(x[20]))+ (2.3762023 * float(x[21]))+ (-1.8345475 * float(x[22]))+ (0.10106555 * float(x[23]))+ (1.3735962 * float(x[24]))+ (-0.62005603 * float(x[25]))+ (1.0749083 * float(x[26]))+ (1.6592791 * float(x[27]))+ (0.15688397 * float(x[28]))+ (-0.816813 * float(x[29]))+ (1.126249 * float(x[30]))+ (-0.7387847 * float(x[31]))+ (-0.66370714 * float(x[32]))+ (-2.5163553 * float(x[33]))+ (0.48288614 * float(x[34]))+ (-2.2904274 * float(x[35]))+ (2.311574 * float(x[36]))+ (2.4239964 * float(x[37]))+ (0.21304512 * float(x[38]))+ (1.7534008 * float(x[39]))+ (-0.94608176 * float(x[40]))+ (0.81145227 * float(x[41]))+ (-0.24259675 * float(x[42]))+ (2.3147292 * float(x[43]))+ (0.90928066 * float(x[44]))+ (2.6730752 * float(x[45]))+ (1.0692687 * float(x[46]))+ (-0.5925319 * float(x[47]))+ (-1.3333803 * float(x[48]))+ (2.1220903 * float(x[49])))+ ((-0.28486538 * float(x[50]))+ (-1.5925632 * float(x[51]))+ (1.808185 * float(x[52]))+ (2.1267014 * float(x[53]))+ (-0.30183706 * float(x[54]))+ (-1.1823127 * float(x[55]))+ (-1.2798023 * float(x[56]))+ (-0.028353272 * float(x[57]))+ (-1.2507516 * float(x[58]))+ (0.835 * float(x[59]))+ (-1.4253918 * float(x[60]))+ (1.7738909 * float(x[61]))+ (0.75375104 * float(x[62]))+ (1.9990611 * float(x[63]))+ (0.9995294 * float(x[64]))+ (-0.38117296 * float(x[65]))+ (1.0688504 * float(x[66]))+ (-1.6910644 * float(x[67]))+ (-2.6726315 * float(x[68]))+ (-0.55259794 * float(x[69]))+ (-0.71445894 * float(x[70]))+ (-1.1240842 * float(x[71]))+ (0.80744034 * float(x[72]))+ (1.6950384 * float(x[73]))+ (-2.3196495 * float(x[74]))+ (2.7207215 * float(x[75]))+ (3.3871968 * float(x[76]))+ (-0.5567163 * float(x[77]))+ (-1.060753 * float(x[78]))+ (0.48094532 * float(x[79]))+ (-2.0601087 * float(x[80]))+ (-0.68642926 * float(x[81]))+ (1.1467785 * float(x[82]))+ (0.55794775 * float(x[83]))+ (-2.1989384 * float(x[84]))+ (-0.49652517 * float(x[85]))+ (2.5672834 * float(x[86]))+ (2.7401178 * float(x[87]))+ (0.32039726 * float(x[88]))+ (1.3156917 * float(x[89]))+ (2.448345 * float(x[90]))+ (2.5185187 * float(x[91]))+ (-1.3448687 * float(x[92]))+ (-1.1248337 * float(x[93]))+ (-0.20061505 * float(x[94]))+ (0.43386954 * float(x[95]))+ (-0.98250216 * float(x[96]))+ (-1.9266378 * float(x[97]))+ (-1.6874089 * float(x[98]))+ (-2.0910816 * float(x[99]))) + -0.36729154), 0)
    h_3 = max((((1.8091221 * float(x[0]))+ (1.7761401 * float(x[1]))+ (1.2353313 * float(x[2]))+ (-0.29173067 * float(x[3]))+ (0.13591982 * float(x[4]))+ (0.6986499 * float(x[5]))+ (-0.2849862 * float(x[6]))+ (-0.6800308 * float(x[7]))+ (-0.88179696 * float(x[8]))+ (0.6700961 * float(x[9]))+ (0.92393184 * float(x[10]))+ (-0.98681194 * float(x[11]))+ (1.022935 * float(x[12]))+ (0.030928258 * float(x[13]))+ (1.6307014 * float(x[14]))+ (1.1413474 * float(x[15]))+ (2.3815708 * float(x[16]))+ (-0.52789587 * float(x[17]))+ (-1.3026146 * float(x[18]))+ (-2.463204 * float(x[19]))+ (-0.49653685 * float(x[20]))+ (0.09433835 * float(x[21]))+ (1.4814359 * float(x[22]))+ (-0.11640386 * float(x[23]))+ (-0.073075764 * float(x[24]))+ (-1.0711651 * float(x[25]))+ (1.3959602 * float(x[26]))+ (0.8635666 * float(x[27]))+ (-0.40544835 * float(x[28]))+ (0.3467771 * float(x[29]))+ (0.23572454 * float(x[30]))+ (-0.6980798 * float(x[31]))+ (-0.82833016 * float(x[32]))+ (0.578302 * float(x[33]))+ (-0.26743767 * float(x[34]))+ (0.4373099 * float(x[35]))+ (-1.4499037 * float(x[36]))+ (0.6235519 * float(x[37]))+ (0.631737 * float(x[38]))+ (-0.25989363 * float(x[39]))+ (1.0823467 * float(x[40]))+ (-0.9164267 * float(x[41]))+ (0.6623928 * float(x[42]))+ (0.0029564195 * float(x[43]))+ (-1.4994272 * float(x[44]))+ (-0.47998014 * float(x[45]))+ (0.20336622 * float(x[46]))+ (-0.4977969 * float(x[47]))+ (1.6377379 * float(x[48]))+ (0.25078174 * float(x[49])))+ ((2.1301901 * float(x[50]))+ (0.67989546 * float(x[51]))+ (1.886905 * float(x[52]))+ (0.7013578 * float(x[53]))+ (1.5312378 * float(x[54]))+ (1.2752283 * float(x[55]))+ (0.16413806 * float(x[56]))+ (0.12330216 * float(x[57]))+ (-0.30465358 * float(x[58]))+ (-0.0141712045 * float(x[59]))+ (-0.45424137 * float(x[60]))+ (1.0573266 * float(x[61]))+ (1.1312623 * float(x[62]))+ (-1.3597991 * float(x[63]))+ (1.9094684 * float(x[64]))+ (-0.06859248 * float(x[65]))+ (1.5824109 * float(x[66]))+ (0.6851376 * float(x[67]))+ (-1.7005615 * float(x[68]))+ (0.0029244875 * float(x[69]))+ (-0.41011214 * float(x[70]))+ (-0.310757 * float(x[71]))+ (-0.8586292 * float(x[72]))+ (-1.0642656 * float(x[73]))+ (-0.29357442 * float(x[74]))+ (0.40764835 * float(x[75]))+ (1.0562972 * float(x[76]))+ (1.2396564 * float(x[77]))+ (1.6252984 * float(x[78]))+ (-1.8647987 * float(x[79]))+ (-1.5158383 * float(x[80]))+ (0.2112308 * float(x[81]))+ (-0.8211685 * float(x[82]))+ (1.1850802 * float(x[83]))+ (0.9810107 * float(x[84]))+ (-0.3934495 * float(x[85]))+ (-0.24184583 * float(x[86]))+ (-2.1372333 * float(x[87]))+ (0.8676284 * float(x[88]))+ (-0.96551514 * float(x[89]))+ (0.2308264 * float(x[90]))+ (0.09961928 * float(x[91]))+ (1.3200008 * float(x[92]))+ (0.42651156 * float(x[93]))+ (1.493187 * float(x[94]))+ (1.2221323 * float(x[95]))+ (1.4523185 * float(x[96]))+ (0.77112716 * float(x[97]))+ (1.1993215 * float(x[98]))+ (0.025561241 * float(x[99]))) + -1.9352598), 0)
    h_4 = max((((1.4866854 * float(x[0]))+ (0.9448952 * float(x[1]))+ (-0.8218578 * float(x[2]))+ (-0.39057288 * float(x[3]))+ (2.4049182 * float(x[4]))+ (-1.501856 * float(x[5]))+ (-1.3618851 * float(x[6]))+ (-0.80057824 * float(x[7]))+ (-3.5831447 * float(x[8]))+ (1.68456 * float(x[9]))+ (-1.7343717 * float(x[10]))+ (-2.0785806 * float(x[11]))+ (-1.3439714 * float(x[12]))+ (-1.4979587 * float(x[13]))+ (-0.16095747 * float(x[14]))+ (-1.1986743 * float(x[15]))+ (-0.5389259 * float(x[16]))+ (0.8906201 * float(x[17]))+ (3.164406 * float(x[18]))+ (1.1170632 * float(x[19]))+ (0.6053902 * float(x[20]))+ (-2.0452778 * float(x[21]))+ (-2.1058328 * float(x[22]))+ (2.0197277 * float(x[23]))+ (0.10026878 * float(x[24]))+ (0.49382997 * float(x[25]))+ (-2.7780852 * float(x[26]))+ (1.2175294 * float(x[27]))+ (-2.743967 * float(x[28]))+ (0.16248858 * float(x[29]))+ (1.0797002 * float(x[30]))+ (2.9405723 * float(x[31]))+ (1.240353 * float(x[32]))+ (0.68257654 * float(x[33]))+ (0.1637556 * float(x[34]))+ (-1.4614706 * float(x[35]))+ (0.25441176 * float(x[36]))+ (0.060008314 * float(x[37]))+ (-0.34191945 * float(x[38]))+ (1.4679979 * float(x[39]))+ (1.217169 * float(x[40]))+ (-0.9576208 * float(x[41]))+ (0.10558243 * float(x[42]))+ (-0.13506639 * float(x[43]))+ (-0.41958612 * float(x[44]))+ (0.76037735 * float(x[45]))+ (0.4507151 * float(x[46]))+ (0.9114287 * float(x[47]))+ (1.1221836 * float(x[48]))+ (0.117398135 * float(x[49])))+ ((0.6988776 * float(x[50]))+ (1.5703708 * float(x[51]))+ (-1.8863155 * float(x[52]))+ (-1.098016 * float(x[53]))+ (1.7028261 * float(x[54]))+ (-1.1325563 * float(x[55]))+ (1.191167 * float(x[56]))+ (-0.9988267 * float(x[57]))+ (-0.4453163 * float(x[58]))+ (-1.360733 * float(x[59]))+ (1.9393463 * float(x[60]))+ (-0.8143885 * float(x[61]))+ (0.12296769 * float(x[62]))+ (-1.9576056 * float(x[63]))+ (-0.6264476 * float(x[64]))+ (0.8193066 * float(x[65]))+ (0.16926235 * float(x[66]))+ (-1.4454792 * float(x[67]))+ (-0.42407924 * float(x[68]))+ (-1.5069834 * float(x[69]))+ (-0.38098907 * float(x[70]))+ (-1.2141947 * float(x[71]))+ (-1.4062393 * float(x[72]))+ (-0.5009896 * float(x[73]))+ (1.4660761 * float(x[74]))+ (-1.3293524 * float(x[75]))+ (-0.24318336 * float(x[76]))+ (1.5473357 * float(x[77]))+ (0.18250248 * float(x[78]))+ (0.0085385805 * float(x[79]))+ (1.0391243 * float(x[80]))+ (3.8213444 * float(x[81]))+ (1.7451704 * float(x[82]))+ (0.906015 * float(x[83]))+ (0.12962441 * float(x[84]))+ (-1.9357636 * float(x[85]))+ (-0.5854313 * float(x[86]))+ (-0.9272655 * float(x[87]))+ (-2.654569 * float(x[88]))+ (-2.3555284 * float(x[89]))+ (-2.2456818 * float(x[90]))+ (-1.186809 * float(x[91]))+ (-2.5264008 * float(x[92]))+ (0.13222054 * float(x[93]))+ (-3.104256 * float(x[94]))+ (2.6276171 * float(x[95]))+ (-1.3776985 * float(x[96]))+ (-0.70565534 * float(x[97]))+ (-1.3927068 * float(x[98]))+ (0.40344143 * float(x[99]))) + 0.82221514), 0)
    h_5 = max((((0.41133872 * float(x[0]))+ (1.6463039 * float(x[1]))+ (-1.3924334 * float(x[2]))+ (0.9758735 * float(x[3]))+ (-1.7521673 * float(x[4]))+ (1.2710633 * float(x[5]))+ (-1.0038257 * float(x[6]))+ (1.1573732 * float(x[7]))+ (-1.0545676 * float(x[8]))+ (2.5436473 * float(x[9]))+ (0.19523464 * float(x[10]))+ (-0.69790584 * float(x[11]))+ (1.7193673 * float(x[12]))+ (-0.28705496 * float(x[13]))+ (2.2333245 * float(x[14]))+ (1.4558575 * float(x[15]))+ (-1.6432703 * float(x[16]))+ (-0.77863234 * float(x[17]))+ (-2.9680502 * float(x[18]))+ (0.63225144 * float(x[19]))+ (-0.84270555 * float(x[20]))+ (0.7518021 * float(x[21]))+ (2.3852715 * float(x[22]))+ (3.2595918 * float(x[23]))+ (-0.7666516 * float(x[24]))+ (2.485183 * float(x[25]))+ (0.6684335 * float(x[26]))+ (-1.9559475 * float(x[27]))+ (0.5785344 * float(x[28]))+ (-2.5431423 * float(x[29]))+ (0.061916642 * float(x[30]))+ (-1.594682 * float(x[31]))+ (1.1859949 * float(x[32]))+ (0.29832247 * float(x[33]))+ (-0.42888442 * float(x[34]))+ (0.5590531 * float(x[35]))+ (1.9085325 * float(x[36]))+ (-2.0572429 * float(x[37]))+ (-1.6026138 * float(x[38]))+ (0.3692677 * float(x[39]))+ (-0.47944087 * float(x[40]))+ (0.34093064 * float(x[41]))+ (-1.359329 * float(x[42]))+ (-0.7275892 * float(x[43]))+ (-0.96976596 * float(x[44]))+ (-0.4250822 * float(x[45]))+ (-1.2675602 * float(x[46]))+ (-1.5849568 * float(x[47]))+ (1.3700824 * float(x[48]))+ (0.07359627 * float(x[49])))+ ((-0.1935148 * float(x[50]))+ (1.0900548 * float(x[51]))+ (-0.5605488 * float(x[52]))+ (0.21266954 * float(x[53]))+ (-1.669555 * float(x[54]))+ (0.34052864 * float(x[55]))+ (-1.236626 * float(x[56]))+ (-0.52257264 * float(x[57]))+ (-0.44719088 * float(x[58]))+ (2.7002559 * float(x[59]))+ (0.15203795 * float(x[60]))+ (-0.23335522 * float(x[61]))+ (-0.83661765 * float(x[62]))+ (-0.8798565 * float(x[63]))+ (2.178163 * float(x[64]))+ (0.41104373 * float(x[65]))+ (1.380441 * float(x[66]))+ (-0.23668912 * float(x[67]))+ (-1.1551604 * float(x[68]))+ (1.4012089 * float(x[69]))+ (-0.00032455413 * float(x[70]))+ (-1.9924613 * float(x[71]))+ (0.28747094 * float(x[72]))+ (-1.0429631 * float(x[73]))+ (1.3179537 * float(x[74]))+ (1.1238278 * float(x[75]))+ (2.3221078 * float(x[76]))+ (-0.55307794 * float(x[77]))+ (-2.2185898 * float(x[78]))+ (0.96881634 * float(x[79]))+ (-0.88414496 * float(x[80]))+ (-1.4045727 * float(x[81]))+ (-1.3412083 * float(x[82]))+ (1.0372986 * float(x[83]))+ (3.2372363 * float(x[84]))+ (-0.38424593 * float(x[85]))+ (-1.8150504 * float(x[86]))+ (-2.8840244 * float(x[87]))+ (2.0547078 * float(x[88]))+ (-0.89156294 * float(x[89]))+ (1.4572258 * float(x[90]))+ (-0.1754916 * float(x[91]))+ (-1.4877892 * float(x[92]))+ (2.6139622 * float(x[93]))+ (1.2151364 * float(x[94]))+ (1.5579137 * float(x[95]))+ (-2.6111755 * float(x[96]))+ (2.1211748 * float(x[97]))+ (-1.2630922 * float(x[98]))+ (0.064889774 * float(x[99]))) + 0.16355345), 0)
    o_0 = (2.1407206 * h_0)+ (0.53327805 * h_1)+ (-2.0777504 * h_2)+ (1.5112617 * h_3)+ (-2.3508337 * h_4)+ (-2.0284069 * h_5) + -5.024289
             
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

