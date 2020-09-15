#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Utility eucalyptus-multi.csv -o eucalyptus-multi.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:19.60. Finished on: Sep-08-2020 14:45:43.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         5-way classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 29.08%
Overall Model accuracy:              83.15% (612/736 correct)
Overall Improvement over best guess: 54.07% (of possible 70.92%)
Model capacity (MEC):                331 bits
Generalization ratio:                1.84 bits/bit
Model efficiency:                    0.16%/parameter
Confusion Matrix:
 [23.64% 1.90% 0.82% 1.90% 0.82%]
 [1.90% 11.41% 0.14% 0.41% 0.41%]
 [0.54% 0.14% 12.77% 0.68% 0.41%]
 [2.04% 0.27% 0.68% 14.13% 0.54%]
 [1.09% 0.27% 1.36% 0.54% 21.20%]
Overfitting:                         No
Note: Labels have been remapped to 'good'=0, 'best'=1, 'low'=2, 'average'=3, 'none'=4.
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

# Imports -- external
try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "eucalyptus-multi.csv"


#Number of attributes
num_attr = 19
n_classes = 5


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="Utility"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="Utility"
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
    clean.mapping={'good': 0, 'best': 1, 'low': 2, 'average': 3, 'none': 4}

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


# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([5189394638.35, 5189395227.785, 5308799295.91, 5428203826.195, 5428204298.405001, 5493849745.655001, 5591237056.38, 5622978967.945, 5639909085.14, 5750848080.82, 5861787083.995001, 5861787579.895, 5927433489.72, 6033285196.675, 6073491751.775, 6073492406.385, 6073492786.865, 6094363817.565, 6262951971.59, 6410669110.025, 6546974725.495001, 6785784366.48, 6801554091.375, 6817323374.69, 6817323407.095, 6830787577.375, 6844252198.210001, 6844252683.245001, 6844252715.22, 6844252771.435, 6849554217.334999, 6885965638.73, 6928845453.095, 6940615250.455, 6969052048.425, 6997488818.225, 6997489216.7, 6997489626.065001, 7105499857.305, 7213510104.85, 7213510134.450001, 7213510677.950001, 7213511254.4, 7232208597.664999, 7250905914.535, 7250906509.210001, 7250907092.005001, 7250907120.585, 7250907135.96, 7250907141.790001, 7250907149.96, 7250907160.285, 7263879932.48, 7276852744.4, 7276852792.85, 7430231492.65, 7583610212.45, 7583610260.85, 7584942658.414999, 7584942691.639999, 7600361249.259999, 7639902124.26, 7674961931.014999, 7685899881.37, 7685900435.52, 7727074538.415001, 7768249092.93, 7768249632.625, 7769904175.335, 7818085396.3949995, 7907131686.08, 7990355045.674999, 8088735722.65, 8128942281.695001, 8136413955.9, 8147747743.370001, 8166992500.57, 8174903888.389999, 8198902479.88, 8307551643.4, 8392202295.715, 8392202789.965, 8413493692.139999, 8434784159.7, 8434784449.75, 8438975656.93, 8443166616.0, 8443166629.68, 8443166637.975, 8467368787.25, 8491570941.200001, 8512022440.725, 8531146662.950001, 8531146696.1, 8575183469.235001, 8589545121.764999, 8603906991.145, 8603907250.105, 8603907625.369999, 8619641838.605, 8777153728.27, 8841438750.9, 8841438796.15, 8841439074.3, 8841439383.45, 8841439416.5, 8841439419.3, 8841439433.099998, 8856045067.45, 8888911999.105, 8907173323.52, 8907173838.654999, 8907174350.21, 8907174404.245, 8942523226.355, 8942523236.544998, 8942523256.649998, 8943903336.195, 8945283410.984999, 8948174018.22, 8951064638.0, 8977300712.72, 9014617049.07, 9025697336.02, 9039461448.490002, 9053225564.39, 9057032203.369999, 9060838865.15, 9060838901.9, 9080727066.505001, 9100615250.73, 9100615278.175, 9110161827.955002, 9119708382.490002, 9119944992.935001, 9120181595.25, 9163246940.0, 9215069229.395, 9268827926.67, 9313828653.61, 9313828680.71, 9313828710.195, 9344248359.755001, 9406887665.025, 9468947724.94, 9507724872.35, 9516661632.745, 9516661671.850002, 9657410881.974998, 9657410909.82, 9657410942.185, 9657410960.849998, 9719366635.825, 9781322315.650002, 9781322331.565, 9781322348.27, 9818187108.849998, 9846227549.145, 9874269490.2, 9883544653.08, 9935021072.060001, 9977222349.29, 10005459912.115, 10033697486.35, 10033697565.6, 10052073416.95, 10075948600.480001, 10081448001.724998, 10113821522.43, 10208339413.99, 10270484317.565, 10270484878.485, 10270484914.25, 10302155658.77, 10337243362.74, 10406305754.914999, 10471951192.02, 10476902523.82, 10499436459.4, 10517019039.04, 10534691661.605, 10552364291.515, 10552364704.625, 10552365139.465, 10566885930.09, 10581406687.349998, 10581407173.035, 10581407685.114998, 10581428256.135, 10611016315.59, 10642170338.189999, 10643756978.150002, 10643757198.95, 10643757528.099998, 10649672106.485, 10655586420.82, 10720998052.119999, 10737157131.285, 10753316220.710001, 10758027935.865, 10762739633.394999, 10897945925.78, 10899793532.61, 10899794072.64, 10972156672.43, 11044864476.929998, 11116147091.734999, 11127374650.885, 11173987332.395, 11215576704.75, 11221779400.05, 11234860891.9, 11258918189.18, 11269894025.545, 11269894033.630001, 11296509497.380001, 11323125031.185, 11323125123.27, 11399260041.18, 11448212137.130001, 11456506265.220001, 11464800468.0, 11478279130.189999, 11491757801.38, 11491758051.84, 11521737926.351002, 11551717588.961, 11551717602.795, 11567679163.71, 11585880539.380001, 11654225909.625, 11720332432.55, 11725055848.099998, 11729779272.8, 11729779401.169998, 11898412376.815, 11898412709.300001, 11898413032.825, 11914605210.3, 12005430018.165, 12014727167.545, 12024024311.4, 12098656962.785, 12108234722.8, 12117812490.415, 12119439701.845, 12121066919.185, 12121067435.869999, 12121067956.904999, 12138485484.329998, 12155903000.98, 12171864066.05, 12187825132.82, 12187825627.724998, 12187826131.11, 12187826160.82, 12225556040.125, 12263285904.48, 12288897366.68, 12314508846.009998, 12314508874.435001, 12359596077.060001, 12404683973.310001, 12466203472.96, 12527722268.849998, 12655694358.515, 12786090133.175, 12870407041.045002, 13090671174.244999, 13193006355.71, 13295341553.470001, 13460333165.495, 13663027582.14, 13700730394.48, 13700730628.775, 13700730871.11, 13700731204.525, 13785105758.555, 14002881068.560001, 14136282177.61, 14136282707.525002, 14207518108.085, 14291834483.075, 14317478315.585, 14330040676.864998, 14330040692.025, 14472520848.175, 14727999536.48, 14934225021.060001, 14934225048.560001, 14934225506.26, 14934225990.075, 15573041833.45, 16243529537.9, 16428579008.0, 16581957764.65, 16581957786.599998, 16622164844.6, 16662371916.45, 16704113465.349998, 16704113478.25, 17433131646.05, 17433131954.15, 17481313223.9, 17529494241.95, 17583379727.71, 17738525761.71, 17839786926.8, 17839786951.4, 17959201349.675003, 18098572839.055, 18118529170.8, 18161594460.15, 18204659754.4, 18204659776.55, 18297632759.85, 18553349081.785004, 18789849207.63, 18921913005.45, 19032045049.1, 19032045052.95, 19094744040.245, 19608426682.05, 19642104501.6, 19642105056.3, 19931116134.85, 20233208361.9, 20246289805.1, 20354718894.1, 20463147960.15, 20692538797.14])
labels = array([3.0, 1.0, 0.0, 3.0, 0.0, 4.0, 3.0, 0.0, 1.0, 3.0, 0.0, 3.0, 4.0, 3.0, 4.0, 2.0, 3.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 3.0, 1.0, 0.0, 3.0, 0.0, 2.0, 4.0, 3.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 1.0, 4.0, 0.0, 2.0, 0.0, 4.0, 3.0, 1.0, 3.0, 0.0, 1.0, 0.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 0.0, 3.0, 4.0, 0.0, 4.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 4.0, 1.0, 2.0, 0.0, 3.0, 4.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 4.0, 3.0, 0.0, 1.0, 3.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0, 4.0, 2.0, 0.0, 4.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 2.0, 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0, 3.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 4.0, 2.0, 3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 2.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 2.0, 0.0, 4.0, 3.0, 0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 0.0, 3.0, 2.0, 4.0, 2.0, 3.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 1.0, 3.0, 0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 4.0, 2.0, 3.0, 4.0, 0.0, 2.0, 4.0, 3.0, 0.0, 4.0, 2.0, 3.0, 1.0, 0.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 4.0, 1.0, 3.0, 4.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 3.0, 1.0, 0.0, 3.0, 0.0, 4.0, 1.0, 0.0, 1.0, 4.0, 0.0, 1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.0, 2.0, 4.0, 2.0, 4.0, 0.0, 1.0, 0.0, 4.0, 3.0, 1.0, 4.0, 2.0, 4.0])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)

    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = labels[numers[indys]]
        outputs[defaultindys] = 0.0
        return outputs
    return thresh_search(energys)

numthresholds = 331



# Main method
model_cap = numthresholds


def Validate(file):
    #Load Array
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')


    if n_classes == 2:
        #note that classification is a single line of code
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        correct_count = int(np.sum(outputs.reshape(-1) == cleanarr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(cleanarr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(cleanarr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs, cleanarr[:, -1]


    else:
        #validation
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count = 0, 0
        numeachclass = {}
        for k, o in enumerate(outputs):
            if int(o) == int(float(cleanarr[k, -1])):
                correct_count += 1
            if int(float(cleanarr[k, -1])) in numeachclass.keys():
                numeachclass[int(float(cleanarr[k, -1]))] += 1
            else:
                numeachclass[int(float(cleanarr[k, -1]))] = 1
            count += 1
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    cleanarr = cleanarr.reshape(cleanarr.shape[0], -1)
    with open(preprocessedfile, 'r') as csvinput:
        dirtyreader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(dirtyreader, None) + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            print(str(','.join(str(j) for j in (['"' + i + '"' if ',' in i else i for i in row]))) + ',' + str(get_key(int(outputs[k]), classmapping)))



#Main
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
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}

    #Predict or Validate?
    if not args.validate:
        Predict(cleanfile, get_key, args.headerless, preprocessedfile, classmapping)


    else:
        classifier_type = 'DT'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
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


    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)



