#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Class volcanoes-d2.csv -o volcanoes-d2.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:12.42. Finished on: Sep-04-2020 12:18:54.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         5-way classifier
Best-guess accuracy:                 94.53%
Overall Model accuracy:              100.00% (9172/9172 correct)
Overall Improvement over best guess: 5.47% (of possible 5.47%)
Model capacity (MEC):                948 bits
Generalization ratio:                9.67 bits/bit
Model efficiency:                    0.00%/parameter
Confusion Matrix:
 [2.61% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.98% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 94.53% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.61% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 1.28%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to '4'=0, '5'=1, '1'=2, '2'=3, '3'=4.
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
TRAINFILE = "volcanoes-d2.csv"


#Number of attributes
num_attr = 3
n_classes = 5


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="Class"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="Class"
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
    clean.mapping={'4': 0, '5': 1, '1': 2, '2': 3, '3': 4}

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
energy_thresholds = array([54.935327, 57.435171, 88.5154285, 90.0332995, 113.3845965, 113.4072195, 134.451642, 134.958936, 157.12422099999998, 158.0411015, 180.4085495, 180.923162, 182.5554655, 184.5025035, 186.461973, 186.4881945, 222.5538495, 223.101776, 235.47761500000001, 236.51595650000002, 249.5392385, 250.004325, 253.03601300000003, 254.54880350000002, 255.4081395, 255.9766665, 259.420219, 259.464902, 267.444162, 267.9246905, 273.5725085, 274.0143515, 279.52717199999995, 280.0014165, 302.0598875, 303.029432, 321.59427600000004, 322.0571745, 322.684229, 323.06536800000003, 326.4754335, 326.982883, 330.4279935, 331.3655485, 348.539338, 348.59932249999997, 360.5217405, 360.593486, 360.9986745, 365.62675950000005, 366.05124550000005, 366.52617150000003, 366.99283, 367.49127899999996, 368.0041485, 368.41688999999997, 368.45957599999997, 369.546498, 369.6897505, 370.058458, 379.4798605, 380.01624449999997, 380.6091785, 381.0762395, 382.5922325, 383.032954, 384.460011, 384.496597, 404.5071125, 404.546571, 404.673788, 405.055708, 407.5895925, 408.032204, 408.53747, 408.974566, 414.54609500000004, 414.97698549999996, 415.4882145, 415.54551100000003, 416.5173745, 417.03267300000005, 427.596739, 428.01416800000004, 446.41153499999996, 446.4595955, 457.480791, 457.9579345, 460.5585785, 461.0470405, 467.47664050000003, 467.9352185, 479.5440795, 479.9904725, 480.53216050000003, 480.623065, 485.5118175, 485.949563, 486.43312149999997, 486.4554275, 486.538853, 487.16535450000003, 488.03794400000004, 488.89925949999997, 489.392353, 489.402638, 496.5282995, 497.014579, 500.3957325, 500.4320985, 500.497752, 501.052353, 501.59023, 501.9726805, 502.35247300000003, 503.5041665, 503.9565425, 505.5418115, 506.082445, 511.94806700000004, 512.420689, 513.3926185, 513.475272, 513.9774635, 515.4392459999999, 515.47298, 517.494449, 518.0534335, 518.6418135, 519.081638, 521.459006, 521.9283045, 524.6209965, 525.0537925, 526.427032, 526.441325, 538.6489825, 539.0181915000001, 539.5113994999999, 539.9556545, 540.382988, 540.425492, 544.6689650000001, 545.0581415, 555.607123, 556.018916, 557.497466, 557.6215804999999, 558.0911475, 558.6066555, 559.0151435, 560.446887, 560.9351615, 567.5499665, 567.597353, 569.5019004999999, 569.9725915, 570.7716075000001, 571.0742775, 572.4332125000001, 572.4499495, 573.483842, 573.936321, 575.0000025, 575.3681770000001, 586.4668575000001, 586.5411055, 588.4521394999999, 588.93773, 590.662304, 591.0510925, 598.376621, 598.419611, 601.54889, 601.5982265, 602.446964, 602.914399, 603.3764335, 603.3945475, 608.5250085, 608.9639455, 615.4935015, 615.9376950000001, 616.3581455000001, 619.567635, 620.0129440000001, 623.4089925, 623.8971730000001, 624.584167, 624.9913994999999, 629.5614674999999, 629.5952455, 631.5826335, 631.6870650000001, 632.022721, 633.5274830000001, 633.573716, 634.7067844999999, 635.0652235, 635.377784, 635.4030225, 635.6484575, 636.05783, 637.4755935000001, 637.4989350000001, 643.6526855, 644.0528019999999, 644.401437, 644.4094635, 645.664515, 646.038879, 650.5634230000001, 650.996044, 655.410451, 655.4252055, 658.5559324999999, 658.6870275, 670.494603, 670.952315, 674.561915, 675.0069675, 675.4006939999999, 675.4024495, 677.5976479999999, 678.016018, 680.602847, 681.02851, 683.494804, 683.6225790000001, 692.478923, 692.539273, 692.5647945000001, 692.9702425, 693.4320955, 693.4583435, 694.6396935, 695.0559109999999, 696.4537005, 696.4925785, 700.993157, 701.370392, 704.5036399999999, 704.6007195, 704.639994, 705.0094845, 706.509106, 707.045845, 711.6114205, 712.0071825, 712.9475084999999, 713.391975, 716.450067, 716.5050105, 717.632535, 717.6504765, 719.5996044999999, 720.017857, 721.3944564999999, 721.4000685, 723.5591119999999, 724.009152, 725.4928279999999, 725.9718885, 729.6045245, 730.0058065000001, 730.5101035, 730.586497, 733.6685175, 734.0356784999999, 736.5880565, 737.0060020000001, 743.6484185, 744.0045425000001, 749.5428704999999, 749.5907745, 750.655754, 750.775934, 751.5005315000001, 751.5152720000001, 753.5995355, 753.9868765, 754.5535514999999, 754.9902605, 761.7180814999999, 762.0927899999999, 763.5766235, 763.998965, 766.6014634999999, 767.037734, 769.6089925, 770.05524, 771.47192, 771.9381565, 778.0937329999999, 778.556703, 778.640999, 782.5525594999999, 782.9677879999999, 783.642329, 784.037455, 785.5365465, 785.610143, 786.5492405, 786.6077375, 787.575464, 788.0334849999999, 788.6247285, 789.0282735000001, 789.5230735, 789.9565165, 790.661022, 791.027872, 792.5107625, 792.6075754999999, 794.524397, 794.6554135, 795.5588705, 795.692111, 796.051963, 800.4252895, 800.4975045, 801.44029, 801.457724, 802.567091, 802.677059, 804.6271859999999, 804.9969289999999, 809.618293, 810.003475, 815.573345, 816.0311905000001, 820.5318195, 820.9717355, 823.5848814999999, 823.9841275, 824.556958, 824.6294075000001, 825.4790005, 825.5021265, 830.6798865000001, 831.0685455, 832.362762, 832.3721365, 834.4968385, 834.4984205000001, 838.720552, 839.051258, 843.4697914999999, 843.945659, 845.6482785000001, 846.04766, 850.6276875, 851.0041745, 852.5384955, 852.9960595, 857.5667215, 857.633486, 858.017897, 859.5927035, 860.009088, 861.5714235, 862.0019495, 862.4468664999999, 862.4706565, 863.537693, 863.6221505, 864.4841664999999, 864.605694, 866.3553425, 866.359843, 879.5103695, 879.5152410000001, 879.9344355000001, 884.5714685, 885.051875, 885.6918595, 886.0264675, 886.555971, 887.006997, 887.5328665, 887.963833, 890.59403, 890.614397, 893.6194175, 894.020273, 894.515764, 894.9462940000001, 901.4298395, 901.4678429999999, 904.601347, 904.999228, 908.575204, 909.0052479999999, 911.6146885, 911.754583, 912.0941620000001, 913.5915345000001, 914.0628925000001, 916.516149, 916.5482475, 917.5797355, 918.019501, 920.4828775000001, 920.4954, 920.5296695, 920.6808169999999, 921.5159835, 921.9788135, 924.582409, 925.0293449999999, 925.6466105, 926.0313695, 926.6101424999999, 927.0520915, 928.5651385, 929.022916, 929.688605, 930.0809535, 932.5021039999999, 932.9385555, 936.4985895, 936.5449209999999, 936.5943855, 937.00704, 937.5962215, 938.021784, 940.5587195, 940.5749985, 941.365929, 941.3806589999999, 941.4701285, 941.4950175, 943.5558625, 943.6133625, 943.6587905, 944.0157775, 944.628179, 945.0290525, 945.3751325000001, 945.3783175, 947.468898, 947.9310915, 949.51216, 949.575349, 953.5319975, 953.9678695, 956.6069835000001, 956.6318555, 956.6479495, 956.734504, 957.088257, 959.5451185, 959.628717, 961.6303754999999, 962.0115125, 962.6055919999999, 962.6354719999999, 962.6962855, 962.7366315, 963.0501025000001, 963.533138, 963.5812005, 963.6349809999999, 963.6662645, 965.6388115, 966.034552, 966.5247549999999, 966.9719315, 971.4135385, 971.4288650000001, 971.5180809999999, 971.598222, 973.501764, 973.5781870000001, 975.548295, 975.58142, 975.619047, 976.0033195000001, 977.048455, 977.3983225, 979.4464905, 979.483725, 981.6287155, 981.6857875000001, 984.3939654999999, 984.50036, 989.5662520000001, 989.9893545, 991.5977614999999, 992.0515775, 993.6412235, 994.0391855, 994.737657, 995.110567, 995.621573, 995.652235, 995.7088590000001, 1000.6180959999999, 1000.9937835000001, 1001.5855805, 1001.999541, 1004.677526, 1005.044699, 1007.385051, 1007.39381, 1007.4967664999999, 1008.0378025, 1008.6176800000001, 1009.0016350000001, 1010.564033, 1010.6373505, 1011.1295975, 1019.5439034999999, 1019.5696295, 1019.6621195, 1020.0359655, 1021.6840950000001, 1022.045967, 1023.4816905, 1023.517519, 1024.6098175, 1024.687819, 1025.4878865, 1025.5776645, 1029.4013224999999, 1029.425668, 1029.510806, 1029.9566384999998, 1032.974872, 1033.3923180000002, 1037.5226240000002, 1037.5745655, 1037.6560725, 1042.4455515, 1042.489989, 1043.6509925, 1044.0219550000002, 1046.3630560000001, 1046.364404, 1047.5888725, 1047.590636, 1048.365589, 1048.3901, 1049.371568, 1049.3803415000002, 1051.6808540000002, 1052.077256, 1053.6226740000002, 1054.0071990000001, 1056.6557444999999, 1056.6974405, 1057.028386, 1057.5227015, 1057.9554045, 1059.6655055, 1059.7508575000002, 1061.5771365, 1062.0029964999999, 1066.373918, 1066.3903015, 1068.6371685, 1068.6917585, 1073.4183535, 1073.4501165000001, 1079.5747565000001, 1079.991324, 1083.585263, 1084.011816, 1084.3954555, 1084.4401825, 1085.656195, 1086.0504019999998, 1086.4935405, 1087.054254, 1100.4586095, 1100.935565, 1106.467596, 1106.5132085, 1106.6254545000002, 1107.0457864999998, 1108.512928, 1108.5829695, 1109.3926675, 1109.447173, 1109.6038239999998, 1110.0369034999999, 1110.4842975000001, 1110.5237849999999, 1116.614436, 1117.044325, 1122.3949969999999, 1122.43201, 1125.464798, 1125.55484, 1128.484754, 1128.9792365, 1131.56358, 1131.618917, 1132.7219835, 1133.1494834999999, 1134.467918, 1134.919048, 1136.491266, 1136.510689, 1140.677185, 1141.0370575, 1145.4519894999999, 1145.488556, 1146.578122, 1146.6840095, 1146.7626085000002, 1150.4408715, 1150.4703975, 1151.3646505000002, 1151.387757, 1153.5408665, 1153.6280465, 1156.4092135, 1156.4965164999999, 1160.482208, 1160.4842250000002, 1164.5700275, 1164.6020254999999, 1170.5863064999999, 1171.0030394999999, 1174.4908995, 1174.5420605, 1180.598457, 1180.9963355, 1183.0255865, 1183.394545, 1184.395673, 1184.414812, 1186.673195, 1187.0749095, 1187.559508, 1187.652028, 1187.678993, 1190.5154275, 1190.621034, 1197.385644, 1197.4046925, 1199.6447640000001, 1200.0552595, 1200.5412430000001, 1200.625396, 1200.6986725, 1201.088104, 1211.631318, 1212.0558270000001, 1214.4555045000002, 1214.484227, 1215.6087499999999, 1216.012933, 1216.5924245, 1216.98343, 1220.4072995000001, 1220.4935525, 1225.5414515, 1225.996356, 1227.469642, 1227.489866, 1233.4646735000001, 1233.6079479999999, 1234.4857069999998, 1234.643766, 1235.0651655, 1237.616158, 1238.0682975, 1238.59638, 1239.036228, 1240.4656005000002, 1240.4901585, 1248.580375, 1248.9972524999998, 1249.504647, 1249.9522705, 1258.557284, 1258.9792845000002, 1259.545451, 1259.975275, 1260.370367, 1261.3996419999999, 1261.436923, 1264.627615, 1265.023224, 1265.504862, 1265.534322, 1277.6468805, 1278.0342945000002, 1279.556845, 1279.9719635000001, 1280.3843725, 1280.4588475, 1280.590341, 1281.0254475000002, 1281.6080285, 1281.6762925, 1283.4883530000002, 1283.5526185, 1284.430335, 1284.4696505, 1284.530837, 1284.9686510000001, 1285.421728, 1285.4421535000001, 1291.548918, 1291.976188, 1292.5534200000002, 1293.0294764999999, 1293.3981079999999, 1293.4867585000002, 1294.5718345, 1294.602384, 1300.001082, 1300.4005200000001, 1303.4737485, 1303.9255535, 1312.445464, 1312.5084295000001, 1312.587735, 1313.018939, 1317.582887, 1317.9820925, 1319.5396145, 1319.965358, 1324.6135375, 1325.115746, 1326.6033674999999, 1327.0149005, 1327.660001, 1328.040375, 1329.6763609999998, 1329.7273335, 1330.5222130000002, 1330.6734645000001, 1333.474452, 1333.5110535, 1333.5814285000001, 1339.5480455, 1340.0084594999998, 1345.359481, 1345.3691880000001, 1348.4971795000001, 1348.5098130000001, 1357.375062, 1357.4095915, 1359.4620395, 1359.4795829999998, 1362.6501195, 1362.700882, 1364.585422, 1364.9919009999999, 1368.6011975000001, 1368.748387, 1369.5728875, 1370.0003875, 1372.37234, 1372.3983830000002, 1373.750456, 1374.0831615000002, 1374.603165, 1375.047689, 1377.552546, 1377.99822, 1379.5113915000002, 1379.9885125, 1386.457939, 1386.92834, 1388.4449805, 1388.92423, 1392.6359750000001, 1393.0262725, 1393.3751805, 1393.4078685, 1395.5175255, 1395.607715, 1403.6321825, 1404.0163465, 1408.551855, 1408.9868510000001, 1412.456318, 1412.4955034999998, 1412.5660725, 1412.97768, 1417.567607, 1418.003979, 1418.6389414999999, 1419.0556714999998, 1421.5843519999999, 1421.6262004999999, 1423.4912239999999, 1423.5339060000001, 1423.601096, 1424.3966424999999, 1424.4291549999998, 1424.537275, 1424.6024825, 1425.511392, 1425.9519445, 1429.5032125, 1429.9705155000001, 1440.3591215000001, 1440.404439, 1441.6928, 1442.0481555000001, 1444.5274264999998, 1444.61937, 1446.3607980000002, 1446.366402, 1450.4587535, 1450.490653, 1458.4373424999999, 1458.928292, 1463.4100075000001, 1463.50578, 1472.5014035, 1472.5355875, 1477.475082, 1477.5196505, 1479.5986755, 1479.989873, 1481.014706, 1481.8985725, 1482.381353, 1482.394342, 1485.6391835, 1486.019468, 1489.4588079999999, 1489.48923, 1493.452613, 1493.5432030000002, 1493.6281755, 1493.9904895, 1497.5426670000002, 1497.962121, 1501.3788595, 1501.4002875, 1508.5218825000002, 1508.640581, 1508.713952, 1509.0557385, 1513.521593, 1513.5848475, 1514.610556, 1515.0191545, 1515.585971, 1515.975786, 1516.4448645, 1516.95383, 1522.583378, 1523.045518, 1529.5940665, 1530.0502235, 1532.5555239999999, 1532.989896, 1534.4885345, 1534.532474, 1545.5399475, 1546.007828, 1546.390032, 1546.4067805, 1550.4156215, 1550.5543834999999, 1552.6392110000002, 1553.065934, 1554.469075, 1554.4892974999998, 1554.521185, 1554.6148665, 1554.682615, 1555.0268655, 1555.6776365, 1556.0735639999998, 1572.4118005, 1572.4824254999999, 1574.5632305, 1574.6560774999998, 1595.5155675, 1595.98308, 1600.3699575, 1600.3923264999999, 1604.5563095, 1604.9995685, 1610.5415965000002, 1611.4898985, 1617.682568, 1618.050481, 1619.8828055, 1620.445199, 1643.5776495, 1643.983264, 1644.4403005, 1644.4888835, 1664.488934, 1664.946625, 1668.9525155000001, 1669.421347, 1672.5336335000002, 1672.9998074999999, 1687.524471, 1687.9746875, 1696.3926805, 1696.4413365, 1697.5794165, 1697.9847025, 1702.5186804999998, 1702.608638, 1722.5773399999998, 1722.9975829999998, 1724.8851909999998, 1725.4483595000002, 1742.5695375, 1743.1071905, 1750.5580089999999, 1751.0327435, 1752.6124915, 1753.0431915, 1767.3869505, 1767.4320105, 1769.526934, 1769.9631975, 1771.569522, 1771.9814285, 1772.624809, 1773.0207735, 1780.4787419999998, 1780.9409165, 1796.897343, 1797.3647, 1810.410404, 1810.9146045, 1815.033002, 1816.4369459999998, 1851.466006, 1852.4471859999999, 1877.6655705, 1878.0995485, 1879.4404585000002, 1879.4608185000002, 1924.0235605, 1925.017093, 1926.1135175, 1927.535949, 1958.5998865000001, 1959.6399105, 1962.526175, 1963.0124685])
labels = array([0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.0, 1.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 4.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0])
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
        outputs[defaultindys] = 2.0
        return outputs
    return thresh_search(energys)

numthresholds = 948



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



