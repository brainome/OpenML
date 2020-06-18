#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593735/php8MUpOC -o Predictors/volcanoes-d2_QC.py -target Class -stopat 94.54 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:04.21. Finished on: May-22-2020 23:55:06.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                94.57%
Model accuracy:                     95.24% (8736/9172 correct)
Improvement over best guess:        0.67% (of possible 5.43%)
Model capacity (MEC):               471 bits
Generalization ratio:               18.54 bits/bit
Confusion Matrix:
 [1.34% 0.01% 1.20% 0.02% 0.03%]
 [0.02% 0.41% 0.52% 0.00% 0.02%]
 [0.96% 0.27% 92.46% 0.29% 0.55%]
 [0.00% 0.00% 0.25% 0.35% 0.01%]
 [0.03% 0.01% 0.52% 0.02% 0.69%]

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
TRAINFILE = "php8MUpOC.csv"


#Number of attributes
num_attr = 3
n_classes = 5


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="Class"


    if (testfile):
        target = ''
    
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

# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([87.130861, 91.0922715, 156.569245, 158.0411015, 182.5554655, 184.5025035, 222.1879555, 223.101776, 235.47761500000001, 236.51595650000002, 249.5392385, 250.004325, 255.4081395, 256.424643, 273.5725085, 275.1646625, 321.110909, 322.057274, 326.4739145, 327.036637, 330.4279935, 331.3655485, 360.5217405, 361.4649645, 366.9957875, 368.0041485, 368.41688999999997, 368.45957599999997, 378.9741975, 380.0365865, 380.6091785, 381.0762395, 382.56935150000004, 383.032954, 414.543068, 415.016462, 416.103293, 417.04109900000003, 446.41153499999996, 446.4595955, 457.4762405, 457.9985385, 466.954279, 467.9352185, 486.4295565, 486.4554275, 488.89925949999997, 489.392353, 489.402638, 500.497752, 501.052353, 501.59023, 501.9726805, 502.354251, 505.5151785, 506.088573, 513.3848245, 513.8959475, 517.494449, 518.497845, 521.459006, 521.928627, 526.4215885, 526.4505845, 539.5113994999999, 539.9556545, 555.6069875000001, 556.018916, 557.5214975, 558.16112, 559.0461170000001, 566.993201, 568.0185695, 570.7716075000001, 571.0742775, 572.4332125000001, 572.4499495, 573.483842, 573.940525, 588.083904, 588.9409905, 598.376621, 598.419611, 615.4935015, 615.9376950000001, 616.3581455000001, 619.52284, 620.0129440000001, 622.965078, 623.8971730000001, 624.5230315, 625.0011770000001, 631.5826335, 632.058691, 635.596532, 636.5748665, 643.939982, 644.440593, 645.664515, 646.0415345, 650.506799, 650.996044, 655.3915445, 655.4509794999999, 658.5559324999999, 658.6870275, 675.4006939999999, 675.532209, 682.9581865, 683.9884655000001, 692.478923, 692.9447210000001, 693.4297075, 693.498426, 694.5524175, 695.138235, 700.993157, 701.3912235, 704.4888, 704.620459, 705.0094845, 706.509106, 707.1271045, 711.5320019999999, 712.0071825, 716.419954, 716.9203435, 717.632535, 717.9998304999999, 730.5101035, 730.610837, 736.5692265, 737.0060020000001, 750.655754, 751.5750204999999, 766.6014634999999, 767.0652654999999, 771.120916, 771.9381565, 778.002731, 778.9895265, 782.5525594999999, 782.9677879999999, 786.5492405, 786.6077375, 788.5996505, 789.0312695, 795.5588705, 795.692111, 796.0566655, 802.567091, 802.6657865, 823.5451635, 823.9841275, 824.514284, 824.6502370000001, 834.4968385, 834.4984205000001, 838.720552, 839.051258, 852.51875, 852.9960595, 859.582064, 860.0094320000001, 862.429558, 862.4706565, 863.4837835, 863.9687445, 864.4841664999999, 864.9712665, 879.5103695, 879.5152410000001, 879.9344355000001, 884.5714685, 885.051875, 885.6918595, 886.0264675, 886.50712, 887.006997, 887.5328665, 887.963833, 900.951654, 901.5022675, 904.5429805, 904.999228, 908.575204, 909.032421, 911.6542245, 912.574958, 916.4872765, 916.5482475, 920.468796, 920.5231425, 921.4895805, 925.6466105, 926.036784, 928.557308, 929.036324, 943.5558625, 943.634531, 944.03282, 953.5319975, 953.9678695, 956.6069835000001, 956.6318555, 956.6479495, 957.0094755, 959.4827165, 959.628717, 961.6303754999999, 962.0409715, 962.5572675, 963.050523, 963.533138, 963.584359, 963.6349809999999, 964.0016215, 965.6388115, 966.034552, 971.5180809999999, 971.974068, 975.548295, 975.618786, 976.0033195000001, 977.0403295000001, 977.4143405, 979.443365, 979.511538, 993.6054545, 994.0391855, 994.65137, 995.110567, 995.5317445000001, 995.652235, 995.7088590000001, 1001.572075, 1002.5030005000001, 1004.65299, 1005.044699, 1010.5546469999999, 1011.4655885, 1024.5293895, 1025.022728, 1029.0174455000001, 1029.4322835, 1037.4829610000002, 1037.6225840000002, 1042.4261625, 1042.489989, 1046.3630560000001, 1046.3667055, 1047.5199714999999, 1047.590636, 1049.371568, 1049.3803415000002, 1056.5930564999999, 1057.0302000000001, 1057.5227015, 1057.9554045, 1059.6267364999999, 1059.7508575000002, 1061.5449315, 1062.0029964999999, 1073.4183535, 1073.4501165000001, 1084.3954555, 1084.4401825, 1085.64636, 1086.0871185, 1086.601125, 1087.054254, 1100.442435, 1101.0284025, 1106.4579665, 1106.5132085, 1108.472785, 1108.971129, 1109.5448895, 1110.0369034999999, 1110.477427, 1110.5237849999999, 1116.614436, 1117.1058885, 1122.3943410000002, 1122.469203, 1125.464144, 1125.55484, 1128.081671, 1128.9792365, 1144.9817764999998, 1145.913906, 1146.572552, 1146.6840095, 1146.7626085000002, 1151.3646505000002, 1151.387757, 1153.4818135, 1153.6280465, 1160.459934, 1160.9398895, 1170.5863064999999, 1171.0113695, 1180.5642115, 1180.9963355, 1184.395673, 1184.4897065, 1186.618391, 1187.0749095, 1187.5685130000002, 1187.678993, 1197.3793445000001, 1197.4046925, 1200.5407365, 1200.6859924999999, 1201.5601394999999, 1215.6087029999999, 1216.012933, 1216.561882, 1216.9969645, 1227.469642, 1227.489866, 1234.5924289999998, 1235.0651655, 1238.59638, 1239.0678395, 1240.4656005000002, 1240.4901585, 1248.570681, 1249.0237769999999, 1249.504647, 1250.00248, 1259.920889, 1260.370367, 1261.3996419999999, 1261.5789865000002, 1264.627615, 1265.0253050000001, 1265.504862, 1265.5365969999998, 1277.6129865, 1278.0342945000002, 1279.9601834999999, 1281.0254475000002, 1284.430335, 1284.4696505, 1292.5534200000002, 1293.0294764999999, 1294.5718345, 1294.602384, 1312.430308, 1312.5084295000001, 1312.587735, 1313.0267525, 1317.582887, 1317.9820925, 1329.6763609999998, 1330.0322350000001, 1333.4798765, 1333.972626, 1348.4971795000001, 1348.5098130000001, 1359.4620395, 1359.9308904999998, 1364.5008745, 1365.0105330000001, 1368.6011975000001, 1369.041216, 1369.5728875, 1370.0003875, 1372.37234, 1372.8692310000001, 1373.750456, 1374.136646, 1388.436582, 1388.9446619999999, 1392.5450535, 1393.415483, 1403.575718, 1404.0163465, 1408.551855, 1408.9868510000001, 1418.6389414999999, 1419.0556714999998, 1421.5605885, 1421.6262004999999, 1423.062994, 1424.0004509999999, 1424.5296945, 1424.6024825, 1425.4741315, 1425.9519445, 1439.994967, 1440.4369885, 1446.3604775, 1446.371737, 1457.9540605, 1458.928292, 1463.4100075000001, 1463.895482, 1481.014706, 1481.9136680000001, 1482.381353, 1482.394342, 1489.4320195, 1489.48923, 1501.3788595, 1501.902994, 1508.5218825000002, 1508.640581, 1508.713952, 1509.06136, 1515.585971, 1515.9886855, 1529.5510455, 1530.0502235, 1532.5555239999999, 1532.989896, 1534.0246695, 1535.542609, 1545.5399475, 1546.021617, 1546.4155150000001, 1552.134264, 1553.072512, 1555.6776365, 1556.0947895, 1572.4109684999999, 1572.4824254999999, 1574.5632305, 1575.040152, 1595.5155675, 1595.99555, 1617.5356809999998, 1618.050481, 1643.0035635, 1643.983264, 1663.4820774999998, 1664.9940015, 1672.4837585, 1672.9998074999999, 1687.4738005, 1687.9746875, 1697.0872884999999, 1698.009707, 1702.5186804999998, 1702.991766, 1742.5695375, 1744.2009305000001, 1750.5580089999999, 1751.0327435, 1752.6124915, 1753.0431915, 1769.526934, 1769.9631975, 1771.5335175, 1771.9814285, 1772.624809, 1773.0207735, 1796.448495, 1797.3647, 1815.033002, 1816.4369459999998, 1877.6655705, 1878.0995485, 1878.973489, 1880.444702, 1924.0235605, 1925.6670709999999, 1927.535949, 1958.5998865000001, 1959.649445])
labels = array([1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 4.0, 2.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0])
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

numthresholds = 471



# Main method
model_cap = numthresholds


def Validate(file):
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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0


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
                numeachclass[int(float(cleanarr[k, -1]))] = 0
            count += 1
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    with open(preprocessedfile, 'r') as csvinput:
        dirtyreader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(dirtyreader, None) + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            print(str(','.join(str(j) for j in ([i for i in row]))) + ',' + str(get_key(int(outputs[k]), classmapping)))



#Main
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
        print("Classifier Type: Quick Clustering")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
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



    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)


