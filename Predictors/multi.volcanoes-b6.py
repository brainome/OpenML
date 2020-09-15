#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Class volcanoes-b6.csv -o volcanoes-b6.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:12.61. Finished on: Sep-04-2020 12:18:40.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         5-way classifier
Best-guess accuracy:                 96.21%
Overall Model accuracy:              100.00% (10130/10130 correct)
Overall Improvement over best guess: 3.79% (of possible 3.79%)
Model capacity (MEC):                719 bits
Generalization ratio:                14.08 bits/bit
Model efficiency:                    0.00%/parameter
Confusion Matrix:
 [0.72% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.73% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 2.08% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 96.21% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.26%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to '3'=0, '4'=1, '5'=2, '1'=3, '2'=4.
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
TRAINFILE = "volcanoes-b6.csv"


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
    clean.mapping={'3': 0, '4': 1, '5': 2, '1': 3, '2': 4}

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
energy_thresholds = array([124.530486, 124.97234399999999, 130.956462, 131.946846, 150.45886000000002, 150.920706, 169.374433, 169.39705750000002, 209.5682025, 210.0015855, 211.0616345, 212.032791, 223.39538, 223.42748699999999, 248.90315049999998, 249.8898595, 261.936551, 262.926133, 270.5661285, 271.493408, 305.59132850000003, 306.061054, 320.4275785, 321.89114, 324.519112, 324.98821350000003, 343.5635315, 344.010809, 344.4148, 344.50849700000003, 344.611542, 345.37982850000003, 362.553168, 363.516084, 365.620039, 366.0563195, 383.427323, 383.906569, 391.53926850000005, 391.983898, 405.396277, 405.4164425, 413.4927715, 413.971833, 415.36465450000003, 415.39432899999997, 434.4726945, 434.5317445, 448.48535649999997, 448.9341685, 453.496397, 453.945279, 461.57208549999996, 462.02854349999996, 477.568406, 478.018516, 483.4266615, 483.91929700000003, 487.517223, 488.04715250000004, 501.582487, 502.02324699999997, 502.54404450000004, 502.99416, 514.4449545, 514.9182914999999, 524.5743965, 524.983864, 544.500054, 544.9637325, 549.5050209999999, 549.9676274999999, 563.4329660000001, 563.9134025000001, 567.4423899999999, 567.9064539999999, 574.530334, 574.975606, 591.6202685000001, 592.0445165, 601.5087345, 601.964641, 606.4757145, 606.9484305000001, 610.36396, 610.379937, 619.5634435, 619.9911035, 622.5260324999999, 622.9848669999999, 627.4349385, 627.455105, 633.4486294999999, 633.502076, 633.950924, 636.4996404999999, 636.9521885, 647.9206755, 648.403307, 652.62771, 653.0693945, 654.876714, 655.370097, 673.5310354999999, 673.9919285, 678.42889, 678.501006, 683.4758214999999, 683.9627264999999, 684.5519105, 684.9907734999999, 686.4619005, 686.740947, 688.4910755000001, 688.9235715, 696.509125, 696.9797914999999, 701.5215029999999, 701.9684135, 706.5479825, 707.0000044999999, 708.498175, 708.9495015, 718.5965125, 719.0528865, 727.523938, 727.982246, 732.562471, 733.0022194999999, 741.6001815, 742.038328, 746.498497, 746.958177, 749.4782525, 749.4997559999999, 761.5410105000001, 761.978436, 767.5312695, 767.9723435000001, 769.4835195, 769.6292425, 770.0597235, 770.419756, 770.42944, 771.4993305, 771.963758, 773.535881, 773.9696105, 775.4342115, 775.904113, 778.393009, 778.404706, 780.6062875, 781.023534, 781.4757294999999, 781.937608, 782.538062, 782.9876465, 788.460079, 788.5010935, 788.952569, 797.5179575, 797.959756, 798.597233, 799.0629994999999, 805.451063, 805.915648, 806.5553864999999, 807.0012340000001, 808.4448110000001, 808.9155020000001, 826.5345514999999, 827.001483, 827.5208005, 827.9393525, 832.5015705000001, 832.938658, 842.5251505, 842.95719, 854.5586995, 855.03045, 857.5954665, 858.047528, 858.613207, 859.0484345, 859.3809034999999, 859.3919435, 860.4189624999999, 860.503416, 860.969927, 869.455442, 869.91009, 871.4749664999999, 871.541235, 876.437678, 876.9033525, 879.3875375, 879.403754, 879.4595735, 879.9166385, 880.3867075, 880.4134285, 884.51841, 884.615193, 885.007983, 887.4382745, 887.4444475, 887.531598, 887.9870905, 890.5227245, 890.9544065, 899.476938, 899.9365265, 903.4708615, 903.930741, 914.4029235, 914.4115015, 914.595648, 915.0394355000001, 915.4276725, 915.453226, 916.4575785, 916.4789519999999, 916.5041014999999, 920.5720954999999, 921.0230770000001, 922.3538255000001, 922.358753, 926.5003899999999, 926.9605409999999, 929.5563864999999, 929.9643135, 935.5237635000001, 935.984418, 940.4500029999999, 940.4669255, 940.913104, 941.501128, 941.9555829999999, 947.4807435, 947.5224545000001, 948.476999, 948.4957535, 951.6515975, 952.051002, 952.499329, 952.946647, 955.5920785000001, 956.025709, 956.5801495000001, 956.994387, 958.487447, 958.96398, 963.5247345, 963.9561795, 964.4637680000001, 964.510653, 964.529137, 964.9568254999999, 966.617686, 967.0485985, 969.5529779999999, 969.988985, 971.4384474999999, 971.470703, 976.5164705, 976.9648835, 980.50498, 980.9600265, 983.410993, 983.4345995, 984.4251690000001, 984.455286, 985.6364490000001, 986.0760645, 989.460945, 989.9214045, 991.4803085, 991.9712185000001, 993.587219, 994.042593, 994.4555760000001, 994.477108, 998.5132265, 998.991942, 999.4458775, 999.4652639999999, 1000.4698285, 1000.9396624999999, 1003.6022455, 1003.9784709999999, 1005.5287725000001, 1005.9774605, 1013.4087645, 1013.4332565, 1023.3706125, 1023.3725039999999, 1025.5722705, 1026.0099209999998, 1027.4112145, 1027.4621025000001, 1035.483765, 1035.9350475, 1038.3634865, 1038.3680749999999, 1042.4430265, 1042.949325, 1043.4574899999998, 1043.906569, 1045.3988920000002, 1045.4150960000002, 1051.4407999999999, 1051.5190555, 1059.519663, 1059.9430275, 1061.6147824999998, 1062.0280645, 1063.4273505, 1063.4514275000001, 1064.558173, 1065.008243, 1068.3971510000001, 1068.4301755000001, 1074.5362335, 1075.0038325, 1082.4870855, 1082.5803035, 1082.9824855000002, 1085.5119555, 1085.979237, 1087.4841689999998, 1087.5255485, 1091.4884379999999, 1091.9789515, 1094.5229, 1094.5912735000002, 1096.4266885, 1096.4354425, 1097.3872835000002, 1097.4057665, 1103.514963, 1103.9790655000002, 1104.501146, 1104.948927, 1105.5638635, 1106.0156120000001, 1106.613683, 1107.0247985, 1111.582518, 1112.0373785, 1112.5345795, 1112.9995285, 1116.505205, 1116.932612, 1123.4070885, 1123.41492, 1124.492401, 1124.9289855000002, 1125.394535, 1125.3967605, 1125.583719, 1126.035742, 1128.4738284999999, 1128.9333245, 1130.941319, 1131.3670265, 1132.4759414999999, 1132.9656865, 1134.3990885, 1134.4066805, 1135.5968805, 1135.991854, 1138.5229020000002, 1138.957867, 1143.4866175000002, 1143.6315085, 1144.3636405, 1144.3671064999999, 1147.3564155, 1147.3610955, 1169.4337525, 1169.4481355, 1172.4319894999999, 1172.907291, 1174.4677385, 1174.5779725, 1175.021049, 1176.4688525, 1176.4951525, 1176.534407, 1176.9596385, 1178.3610635, 1178.3655575, 1178.4764015, 1178.9417509999998, 1180.378262, 1180.39375, 1182.4969215, 1182.936936, 1185.4899215, 1185.936808, 1190.4362345, 1190.4679965, 1191.541014, 1191.953096, 1200.361429, 1200.364926, 1203.403609, 1203.4248505, 1204.429212, 1204.8965965, 1207.4882585, 1207.940734, 1209.4691189999999, 1209.511708, 1209.552264, 1209.9652515, 1212.5579275, 1212.9707985, 1213.450345, 1213.5446775, 1213.9774115, 1220.5269455, 1220.9593135, 1221.6152615, 1222.036209, 1224.4981765, 1224.8150765, 1226.5925815, 1227.051817, 1229.5140195, 1229.967185, 1232.5636335, 1233.003654, 1233.5906304999999, 1234.0297875, 1234.4871235, 1234.933552, 1237.431802, 1237.4475095, 1238.5088759999999, 1238.979213, 1239.466549, 1239.9174425, 1240.449222, 1240.5427515000001, 1241.0512165, 1251.3567600000001, 1251.364116, 1254.4213235, 1254.4473775000001, 1258.3698869999998, 1258.374265, 1258.5708614999999, 1258.9779405, 1259.5294315, 1259.9821044999999, 1261.4869290000001, 1261.9545515, 1262.5148335, 1262.9803965, 1263.514686, 1263.9723629999999, 1266.579918, 1266.996619, 1267.4440020000002, 1267.9107835, 1275.5274915, 1275.9881415, 1277.585556, 1278.0483505000002, 1282.434681, 1282.9143960000001, 1285.390927, 1285.4680585, 1288.552944, 1289.0324235, 1295.5357760000002, 1295.988264, 1301.4328555, 1301.4874869999999, 1301.521163, 1301.544084, 1306.43302, 1306.443829, 1308.5741724999998, 1309.061182, 1309.5033015, 1309.7768515, 1310.4905724999999, 1310.942366, 1313.4842395, 1313.9230155, 1321.582441, 1322.0191205, 1322.524592, 1322.579122, 1322.633684, 1323.0053475, 1326.4473245, 1326.497411, 1326.9585470000002, 1327.500241, 1327.9665645, 1330.4114075, 1330.440279, 1333.4953799999998, 1333.961663, 1336.491539, 1336.9442595, 1339.5218785000002, 1339.9675995, 1340.4673954999998, 1340.9250164999999, 1343.4286459999998, 1343.4483945, 1343.904714, 1344.360162, 1349.3991555, 1349.4215995, 1350.5721254999999, 1350.9981035, 1356.512353, 1356.94416, 1357.546534, 1357.9959815, 1358.4933154999999, 1358.955478, 1360.6041344999999, 1361.0066705, 1365.5023704999999, 1365.9649395000001, 1368.5016110000001, 1368.9354885, 1370.4076555000001, 1370.4129145000002, 1375.45347, 1375.4957845, 1375.5853515, 1376.009008, 1379.50569, 1379.9369915, 1396.3945805, 1396.398449, 1396.452297, 1396.9395045, 1398.5874665000001, 1399.011197, 1400.411759, 1400.4708345, 1404.4022909999999, 1404.885292, 1408.5834025, 1409.007905, 1409.3712185, 1410.3588395000002, 1410.38654, 1410.4542955000002, 1410.9298005, 1411.4787430000001, 1411.9339015, 1415.47515, 1415.6233834999998, 1422.5307075, 1422.9599440000002, 1423.5550605, 1424.023893, 1424.609099, 1425.0493765, 1425.3811824999998, 1425.6001875, 1426.052684, 1427.498473, 1427.9434215, 1429.4924270000001, 1429.9353825, 1433.576223, 1434.031364, 1437.3997675, 1437.406819, 1443.6402535, 1444.0400555, 1444.5635470000002, 1445.0322055000001, 1445.4653245, 1445.9129045, 1450.4214809999999, 1450.9050455, 1453.4001505, 1453.8773035, 1457.462628, 1457.4811610000002, 1457.546771, 1457.9804185, 1458.416692, 1458.4471655, 1462.40315, 1462.9037985, 1465.454756, 1465.4884885000001, 1465.518362, 1465.6467225000001, 1466.0623845, 1470.575488, 1470.9907365, 1475.509063, 1475.940713, 1482.492138, 1482.6061065, 1483.018419, 1486.5758700000001, 1487.009298, 1490.4474915, 1490.4735695, 1498.4311135, 1498.5633315, 1499.0272639999998, 1499.3759485, 1507.4799335, 1507.934305, 1509.531283, 1509.976429, 1510.9231835, 1511.3691605, 1513.4717385, 1513.9201014999999, 1515.5724169999999, 1515.9910610000002, 1516.5145585, 1516.9432255000002, 1528.3607080000002, 1528.384248, 1533.4300520000002, 1533.5941790000002, 1534.045059, 1535.5816095, 1536.012408, 1536.4385745, 1536.9170525, 1543.610603, 1544.0350564999999, 1548.574737, 1549.034556, 1554.4182255, 1554.8931935, 1556.407842, 1556.453899, 1559.4816925, 1559.5380805, 1566.5498345, 1566.975524, 1573.4598815, 1573.9085985000002, 1577.372326, 1577.3909024999998, 1579.509192, 1579.5582020000002, 1589.4931775, 1589.95329, 1595.51741, 1595.963988, 1596.4187634999998, 1596.4514869999998, 1604.3795165000001, 1604.4089035000002, 1611.4263265, 1611.6160955, 1612.0593175, 1626.4854605, 1626.9729775, 1628.5086345, 1628.976126, 1642.458421, 1642.523059, 1642.946533, 1673.391536, 1673.4168574999999, 1674.449364, 1674.9159985000001, 1675.5523845, 1675.987456, 1676.461984, 1676.512079, 1687.3701995000001, 1687.3807345, 1701.0366669999999, 1702.0110835, 1710.4343365, 1710.4533335, 1710.915324, 1711.3725749999999, 1722.6271715, 1723.0719319999998, 1758.5724645, 1759.011446, 1790.5350375, 1791.009783, 1808.452154, 1808.961831, 1826.3743625, 1826.388332, 1837.4871, 1838.9382365000001, 1841.4064195, 1842.865818, 1844.5268645, 1845.0145564999998, 1850.5994575, 1851.0585775, 1855.5681595, 1856.0450385, 1856.4953369999998, 1857.008795, 1857.505684, 1857.9937615, 1864.0186575, 1865.0202105, 1882.456627, 1883.997144])
labels = array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 1.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 4.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0])
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
        outputs[defaultindys] = 3.0
        return outputs
    return thresh_search(energys)

numthresholds = 719



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



