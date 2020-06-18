#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593737/phpdYYRrH -o Predictors/volcanoes-d4_QC.py -target Class -stopat 94.34 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:03.41. Finished on: May-28-2020 00:11:08.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                94.37%
Model accuracy:                     95.10% (8230/8654 correct)
Improvement over best guess:        0.73% (of possible 5.63%)
Model capacity (MEC):               487 bits
Generalization ratio:               16.89 bits/bit
Confusion Matrix:
 [1.53% 0.01% 1.13% 0.03% 0.02%]
 [0.01% 0.53% 0.39% 0.00% 0.03%]
 [1.12% 0.52% 92.10% 0.15% 0.44%]
 [0.01% 0.02% 0.31% 0.29% 0.01%]
 [0.05% 0.01% 0.61% 0.00% 0.66%]

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
TRAINFILE = "phpdYYRrH.csv"


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
energy_thresholds = array([53.9386425, 66.98437, 88.6273435, 91.09114550000001, 107.410923, 113.3919325, 126.4863405, 132.522513, 158.5900385, 159.0959695, 182.024814, 184.545568, 185.46736900000002, 249.562677, 250.03823699999998, 257.496553, 258.45669399999997, 271.52377, 274.539804, 323.69815700000004, 324.04231100000004, 329.9469425, 330.38953649999996, 359.495453, 359.9698255, 366.554188, 366.7412645, 367.195847, 368.498425, 378.029718, 379.0957795, 379.538586, 379.994081, 382.584738, 383.062736, 383.431731, 383.506763, 404.627798, 405.1380405, 416.514551, 417.016523, 426.55895899999996, 427.04254449999996, 457.4636585, 458.0197985, 460.51229950000004, 461.024921, 481.50215000000003, 481.9789585, 484.483431, 484.633491, 485.48974499999997, 485.9692325, 487.520388, 487.99546, 488.443761, 500.4502695, 500.5636125, 504.551896, 504.9661135, 506.5758435, 507.0941735, 512.9295569999999, 513.454916, 519.5453585, 520.0238895, 520.5904125, 521.9215695, 526.9390545, 527.406637, 538.6188175, 539.523653, 555.6092235, 556.0203945000001, 558.583568, 558.6519814999999, 559.0121389999999, 569.187544, 570.093925, 573.435925, 573.5655155, 586.4270595, 586.4654525000001, 590.518619, 591.04219, 602.447272, 602.496151, 608.520183, 608.9554685, 619.5093445, 620.1056034999999, 623.405541, 623.9301385, 632.6597005000001, 633.021405, 634.6745695, 635.0702855, 637.9823715, 638.508631, 643.570789, 644.0717405, 646.7417065, 647.1189395, 651.4970045, 651.98028, 655.4139964999999, 655.9022335, 670.495926, 671.4583805, 679.6121865, 680.034041, 683.4417820000001, 684.1234245, 692.463319, 692.963923, 695.648404, 696.0713765, 704.5595075, 704.617657, 707.5201415, 708.105325, 711.544544, 711.991631, 712.9581865, 713.4039339999999, 719.5415969999999, 720.0360845, 725.5904350000001, 726.003678, 726.8845369999999, 727.4454045, 728.5772065, 729.049618, 730.5208395, 731.457157, 738.538935, 738.6578030000001, 745.562518, 746.0035594999999, 751.5522625, 751.7960445, 763.572277, 764.0506825, 771.534046, 771.5420035, 778.46404, 778.9856084999999, 783.506893, 783.5463689999999, 783.6304415, 784.0249080000001, 788.6241195, 789.020099, 796.558243, 797.0499130000001, 800.4048614999999, 800.4412990000001, 802.6616355, 803.0259355000001, 814.5492429999999, 815.518506, 821.5098674999999, 822.012796, 826.4480175, 826.9076534999999, 830.9736909999999, 831.445027, 838.6568465, 839.0789265, 845.617211, 846.0472315, 856.5883895, 857.0356995, 860.9763869999999, 861.9678515, 862.6180704999999, 862.995952, 864.3537385, 864.367668, 884.5670365000001, 885.09013, 886.630817, 887.059887, 887.5812785, 887.9778615, 891.5350060000001, 891.9917009999999, 901.4052485, 901.410622, 913.664788, 914.0599605, 916.4810595, 916.951221, 918.6121619999999, 919.026524, 920.4624994999999, 920.4873695, 923.664561, 924.0417195, 925.574932, 926.0602269999999, 928.6190194999999, 929.061614, 936.5042834999999, 936.945059, 937.506646, 937.999302, 939.5207809999999, 939.9589149999999, 941.4467685, 941.9029795, 942.5355374999999, 943.133919, 943.6697555000001, 948.4269039999999, 948.4774265, 953.5171435, 953.9687125, 956.686342, 957.0893715, 957.509592, 957.972443, 961.6826285, 962.035042, 962.538051, 962.6060745, 962.993076, 966.4766905, 967.02403, 971.5446325, 971.9726045, 976.6107025, 977.0032739999999, 977.376803, 977.3985009999999, 984.6105474999999, 984.6310825, 994.5476185, 994.7882055, 995.1393725, 995.5831874999999, 996.046656, 996.53477, 996.9967825, 999.5361350000001, 999.9855285, 1001.5860540000001, 1002.0102165000001, 1006.894082, 1007.3745265, 1010.534388, 1011.0216069999999, 1011.491389, 1011.9825109999999, 1018.5621305, 1018.9743425, 1020.6134795, 1021.067301, 1030.887734, 1031.3650954999998, 1032.402999, 1032.4092825, 1037.5714094999998, 1037.9914195000001, 1043.6772740000001, 1044.0199269999998, 1047.5031085, 1047.993841, 1048.3970264999998, 1049.93276, 1050.355335, 1053.583102, 1054.0500539999998, 1057.526468, 1058.000994, 1058.5203625, 1058.567496, 1060.6996064999998, 1061.0550595, 1090.603691, 1090.996459, 1106.634707, 1107.05483, 1108.4652445000002, 1108.6328065, 1109.0483525, 1109.4581429999998, 1109.9504285, 1111.5234169999999, 1111.5498615000001, 1114.385976, 1114.3884715, 1116.6125075, 1117.0579935, 1125.451464, 1125.537727, 1128.5528454999999, 1128.9671655, 1133.5706705, 1133.974738, 1134.4408445, 1134.5025855, 1147.6750195, 1148.1313085, 1150.375353, 1150.3841415000002, 1150.4210065, 1150.4479155, 1154.5590105, 1154.9810875, 1158.4608094999999, 1159.0407639999999, 1164.5694760000001, 1164.5959775000001, 1174.475605, 1174.5263995, 1184.3535495, 1184.3783525, 1186.5450675, 1187.0201754999998, 1197.3794765, 1197.405678, 1199.660254, 1200.0620315, 1200.5685545000001, 1201.036536, 1214.6616290000002, 1215.05955, 1215.569481, 1215.9721905, 1232.9960185, 1233.9555315, 1235.4556189999998, 1236.0559105, 1237.591157, 1238.039241, 1239.62644, 1240.039373, 1241.464326, 1241.5025879999998, 1259.0197779999999, 1259.5692075, 1259.987608, 1260.4029175, 1260.422137, 1262.5513329999999, 1263.168377, 1279.3796790000001, 1279.3870255, 1279.5472835, 1279.9743225, 1281.5770475, 1281.6966255, 1282.570962, 1282.999252, 1283.9227465, 1285.5637055, 1285.9938809999999, 1290.4008414999998, 1290.898139, 1293.5241715, 1294.0149455, 1303.460959, 1303.4891705, 1313.5103315000001, 1313.993754, 1316.4806275, 1316.9836785, 1319.4819, 1319.959423, 1328.6100195, 1329.0477274999998, 1329.600977, 1329.7381894999999, 1335.4804450000001, 1335.9490345, 1344.364152, 1344.4066874999999, 1348.503403, 1348.9352294999999, 1349.4949470000001, 1349.5748705, 1351.6309545, 1352.060072, 1355.376793, 1355.8703435, 1364.5435550000002, 1364.9801915, 1369.096661, 1369.9829345, 1373.6268905000002, 1374.170245, 1375.643692, 1376.0568560000002, 1377.5634555, 1377.990576, 1380.0648405, 1381.5069455, 1385.4684590000002, 1385.9311360000002, 1390.3827765, 1390.399202, 1392.558231, 1393.040907, 1402.0994445000001, 1403.0793305, 1412.5343345, 1413.29131, 1417.5336745, 1417.985958, 1421.4855895, 1421.6102584999999, 1423.5332885, 1423.9710034999998, 1425.5693419999998, 1426.0988645, 1428.4784015, 1428.9822475, 1440.362273, 1440.376745, 1446.8808485, 1447.8648210000001, 1471.450899, 1471.9814165, 1477.468351, 1477.5107395, 1489.945526, 1490.5091619999998, 1491.4167045, 1491.5012069999998, 1494.450659, 1494.5330815, 1494.9924365000002, 1513.5249330000001, 1514.0167265, 1518.4408485, 1518.940945, 1538.5388185000002, 1539.002868, 1544.9347105000002, 1545.8687295, 1550.512442, 1550.5597050000001, 1552.9830984999999, 1553.578066, 1554.2247765000002, 1555.062773, 1572.410392, 1572.506773, 1575.5408175, 1576.031145, 1618.6481125, 1619.1208955000002, 1644.4176185000001, 1645.4216485000002, 1665.435672, 1666.4424915, 1668.9007465, 1669.4589045, 1673.4949235, 1673.979854, 1694.5369719999999, 1694.993716, 1695.4266604999998, 1695.5122904999998, 1701.58405, 1702.104147, 1743.5536835, 1744.706105, 1747.0138415000001, 1747.6887345, 1750.6149745, 1751.0505640000001, 1752.595495, 1753.0364734999998, 1778.9819715, 1779.5404085, 1812.3812779999998, 1812.908536, 1849.9646475, 1851.9119925, 1922.5228550000002, 1925.0659329999999, 1926.1001219999998, 1927.533331, 1957.1530265000001, 1959.108766])
labels = array([0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0])
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

numthresholds = 487



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


