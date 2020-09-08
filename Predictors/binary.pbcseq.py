#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target binaryClass pbcseq.csv -o pbcseq.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:31.97. Finished on: Sep-03-2020 17:41:44.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 50.02%
Training accuracy:                   100.00% (1167/1167 correct)
Validation accuracy:                 64.78% (504/778 correct)
Overall Model accuracy:              85.91% (1671/1945 correct)
Overall Improvement over best guess: 35.89% (of possible 49.98%)
Model capacity (MEC):                470 bits
Generalization ratio:                3.55 bits/bit
Model efficiency:                    0.07%/parameter
System behavior
True Negatives:                      42.98% (836/1945)
True Positives:                      42.93% (835/1945)
False Negatives:                     7.10% (138/1945)
False Positives:                     6.99% (136/1945)
True Pos. Rate/Sensitivity/Recall:   0.86
True Neg. Rate/Specificity:          0.86
Precision:                           0.86
F-1 Measure:                         0.86
False Negative Rate/Miss Rate:       0.14
Critical Success Index:              0.75
Confusion Matrix:
 [42.98% 6.99%]
 [7.10% 42.93%]
Overfitting:                         No
Note: Labels have been remapped to 'N'=0, 'P'=1.
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
TRAINFILE = "pbcseq.csv"


#Number of attributes
num_attr = 18
n_classes = 2


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="binaryClass"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"
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
    clean.mapping={'N': 0, 'P': 1}

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
energy_thresholds = array([4700143925.36, 4727577823.049999, 4727578826.705, 4727579174.030001, 4727579347.67, 4727579430.705, 4727579649.475, 4727580251.015, 4727580456.950001, 4727580546.955, 4727580723.344999, 4727581126.83, 4727581517.395, 4727581871.77, 4727582092.695, 4727582112.62, 4727582145.945, 4727582471.309999, 4727582640.005, 4727583301.495001, 4727583521.3, 4727583711.280001, 4727583904.34, 4727584091.434999, 4727584135.44, 4727584335.33, 4727584468.725, 4727584761.905001, 4727585591.895, 4727585657.1, 4727585671.66, 4727585683.65, 4727585746.985001, 4727585765.924999, 4727585781.724999, 4727585815.2, 4727585859.83, 4727585938.66, 4727586093.16, 4727586298.995, 4727586319.620001, 4727586385.325, 4727586489.695, 4727586537.254999, 4727586584.15, 4727586732.4, 4727586791.014999, 4727586891.780001, 4727586979.26, 4727587120.195, 4727587222.784999, 4727587278.459999, 4727587367.41, 4727587519.12, 4727588044.08, 4727588170.889999, 4727592381.6, 4913796520.05, 4913796666.33, 4913798157.335, 4913798458.684999, 4940129175.83, 4966459747.110001, 4966459894.844999, 4966460682.2, 4966460840.065001, 4966461097.78, 4966461179.285, 4966462767.530001, 4966463253.95, 4966463905.12, 4966464125.85, 4966465107.33, 4966465190.955, 4966465466.95, 4966465523.655, 4966465690.305, 4966465860.385, 4966466667.950001, 4966466735.56, 4966466858.395, 4966466999.955, 4966467470.115, 4966467872.26, 4966468155.75, 4966468339.865, 4966468660.825001, 4966468739.495001, 4966469088.255, 4966469117.094999, 4966469859.055, 4966470022.95, 4966471011.459999, 4966471090.974999, 4966471482.985001, 4966471833.125, 4966472086.115, 4966472680.495, 4966473674.925, 5059575121.5199995, 5152678741.485001, 5152679515.735001, 5781194841.14, 6411903843.595, 6411904256.42, 6411905421.349999, 6411905688.315, 6411906871.095, 6411907015.985, 6411907257.184999, 6411907289.92, 6411907431.95, 6411907491.549999, 6411907675.25, 6411907818.11, 6411907898.96, 6411908024.610001, 6411909087.195, 6411909278.065001, 6411909510.094999, 6411909739.935, 6411910396.455, 6411910465.07, 6411910872.265, 6411911367.175, 6411911459.93, 6411911544.955, 6411911553.994999, 6411911775.334999, 6411912169.225, 6411912295.360001, 6411912327.190001, 6411912393.799999, 6411912608.51, 6411912672.84, 6411912806.49, 6411913907.205, 6466788351.735001, 6466788575.235001, 6598120623.799999, 6598120958.17, 6598122716.375, 6650784238.849999, 6650784728.21, 6650785036.6, 6650785398.97, 6650785853.82, 6650786527.095, 6650787372.17, 6650787542.575, 6650787876.195, 6650788028.92, 6650788383.225, 6650788580.719999, 6650788892.6449995, 6650789206.594999, 6650789320.77, 6650789606.455, 6650790216.795, 6650790270.12, 6650790584.135, 6650790926.655001, 6650791131.77, 6650791163.450001, 6650791264.46, 6650791636.165, 6650791950.725, 6650792285.460001, 6650792392.145, 6650792606.775, 6650793074.045, 6650793087.315001, 6650793143.07, 6650793285.125, 6650793524.06, 6650793602.59, 6650794374.860001, 6650794474.695, 6650794599.934999, 6650795100.415, 6650796164.18, 6650796326.77, 6651899013.690001, 6705664229.585001, 6705664332.674999, 6705664552.719999, 6705665354.035, 6705666161.165001, 6705667744.03, 6705667996.98, 6705668264.145, 6705668353.530001, 6705668476.710001, 6705668721.66, 6705669782.690001, 6705669981.53, 6705670158.75, 6705670701.125, 6705671361.8, 6705672297.24, 6705672585.0, 6705672759.73, 6705672900.764999, 6705673308.924999, 6705673605.09, 6705673675.435, 6705676586.605, 6705679099.085, 6771339056.82, 6836997696.07, 6836999592.48, 6837000735.065001, 6889670908.825001, 6889671328.99, 6891879142.065001, 6891883449.85, 7489647288.695, 7903411490.45, 7903411632.875, 7903411920.485001, 7903412101.365, 7903412211.035, 7903412353.725, 7903413466.38, 7903413816.985001, 7903414070.85, 7903414102.58, 7903414225.125, 7903414460.255001, 7903414571.91, 7903414678.775, 7903414819.13, 7903415389.485001, 7903416521.195001, 7903416750.1, 7903417072.125, 7903417180.99, 7903417499.02, 7903417673.610001, 7903418226.754999, 7903418331.34, 7903419786.360001, 7903419827.695001, 7903420100.035, 7903420247.049999, 7903420420.434999, 7903420621.765, 7903420693.900001, 7903420840.825001, 7903420873.695, 7903421027.865, 7903421218.96, 7903421455.190001, 7903421538.23, 7903421894.050001, 7903421941.42, 7903422220.25, 7903422306.059999, 7903422500.925, 7903422646.559999, 7903422892.46, 7903423104.17, 7903423878.245, 7903423995.794999, 7903424064.709999, 7903424192.215, 7903424686.275, 7903424808.74, 7903425927.799999, 7996523427.78, 8089622805.365, 8089626998.68, 8089628548.405001, 8089628682.84, 8089628776.389999, 8089628913.52, 8089628990.535, 8089629566.41, 8089630772.51, 8089631624.755, 8092929017.39, 8119262559.045, 8142293577.469999, 8142295402.050001, 8142296745.74, 8142297486.99, 8142297587.7, 8142297664.865, 8142297744.690001, 8142297819.965, 8142297861.155, 8142297943.440001, 8142298263.205, 8142298698.610001, 8142300827.465, 8142301627.93, 8142302063.905, 8142302133.795, 8142302157.870001, 8142302373.75, 8142302665.28, 8142302810.174999, 8142303030.18, 8142303250.42, 8142303413.690001, 8142303625.925001, 8142303825.825001, 8142303871.27, 8142303908.865, 8142304086.4, 8142304300.839999, 8142304408.045, 8142304671.875, 8142304820.98, 8142304865.66, 8142304915.055, 8142305354.925, 8142305453.46, 8142306352.26, 8142306584.01, 8142307033.91, 8239808648.045, 8328504455.325001, 8328508700.369999, 8328510065.849999, 8328511532.094999, 8381181571.440001, 8381181972.68, 8381182364.764999, 8381182794.264999, 8381183068.595, 8381183448.01, 8381187824.905001, 8381188777.985001, 8455662174.5, 8567390002.765, 8567392787.705001, 8567397309.015001, 9201410750.13, 9587734801.95, 9587735471.82, 9587739446.465, 9587739523.84, 9587739951.455, 9587740008.33, 9587742282.68, 9587742477.465, 9587742601.16, 9587742947.245, 9587744081.735, 9587744116.77, 9587744627.505001, 9587744725.689999, 9587744841.189999, 9587744893.945, 9587745175.625, 9587745235.869999, 9587745597.43, 9587745767.45, 9587745901.39, 9587746616.765, 9587747015.240002, 9587748143.970001, 9587748297.805, 9587748322.015, 9587749010.415, 9587749699.029999, 9642617977.07, 9642618780.755001, 9642624190.245, 9642624482.98, 9773952358.24, 9773952549.755001, 9773952906.175, 9773953639.205002, 9773953996.095001, 9800285605.915, 9826617119.505001, 9826618300.735, 9826618728.625, 9826618862.24, 9826620280.56, 9826620759.619999, 9826621068.8, 9826621370.490002, 9826622025.849998, 9826622182.349998, 9826622938.255001, 9826623005.915, 9826623276.144999, 9826624425.735, 9826624687.53, 9826625065.68, 9826625662.135, 9826626317.705, 9826626599.925, 9826626902.86, 9826626914.18, 9826626935.09, 9826626999.115002, 9826627165.060001, 9826627272.945, 9826627293.82, 9826627343.91, 9826627451.57, 9826628031.504997, 9826629156.869999, 9826629724.779999, 9826629810.09, 9826629958.9, 9826630071.009998, 9826630272.595001, 9826630718.74, 9826630914.105, 9827728357.82, 9881497634.96, 9881497889.325, 9881500197.815, 9881502665.455, 9881502826.919998, 9881502999.429998, 9881503232.17, 9881503434.805, 9881503647.215, 9881504059.485, 9881504441.724998, 9881505754.28, 9881506172.179998, 9881506718.395, 9881507295.845001, 9881511774.235, 9881512767.220001, 9881513814.420002, 9947171730.605001, 10012829941.45, 10012834232.205002, 10012835071.335, 10012835385.16, 10012838391.385, 10012838996.855, 10012839833.380001, 10039169670.900002, 10065500223.16, 10065501769.54, 10065505554.715, 10065506569.225, 10065507069.425, 10065507094.66, 10065512260.61, 10066613050.53, 10067714291.525, 10067714673.85, 10067716723.695, 10067718744.43, 10067720076.24, 10094052687.055, 10734450398.855, 11272064228.84, 11326946286.895, 11326951120.83, 11355118687.490002, 11355119553.675, 11355120095.61, 11355121513.2, 11565827684.765, 12325677853.855001, 14717167494.05, 14717169612.15, 14717171236.385002, 14717171429.8])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)
    start_label = 1
    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = (numers[indys] + start_label) % 2
        outputs[defaultindys]=0
        return outputs
    return thresh_search(energys)

numthresholds = 470



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


