#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593731/php0PPrNB -o Predictors/volcanoes-b5_QC.py -target Class -stopat 96.51 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:03.57. Finished on: May-21-2020 20:22:29.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                96.13%
Model accuracy:                     100.00% (9989/9989 correct)
Improvement over best guess:        3.87% (of possible 3.87%)
Model capacity (MEC):               729 bits
Generalization ratio:               13.70 bits/bit
Confusion Matrix:
 [0.75% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.75% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 2.14% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 96.10% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.26%]

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
TRAINFILE = "php0PPrNB.csv"


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
energy_thresholds = array([123.4716535, 124.97350750000001, 131.4483745, 131.936242, 149.44518649999998, 150.91873099999998, 168.901412, 169.881484, 209.5424285, 209.99743949999998, 211.030753, 212.041065, 223.39741099999998, 223.4245795, 249.388091, 249.8950595, 260.433627, 260.49372, 271.511772, 271.9911155, 305.674206, 306.10724949999997, 319.927823, 320.4102755, 323.4933945, 323.9844975, 343.5131575, 344.0097095, 344.604641, 345.016209, 345.382839, 345.90796850000004, 362.0768345, 363.01660449999997, 365.5486975, 366.01727700000004, 383.396305, 383.90676699999995, 391.51901899999996, 391.980105, 405.3920375, 405.4114865, 413.4823185, 413.97306100000003, 415.362904, 415.36734950000005, 434.46457050000004, 434.949135, 448.48634849999996, 448.9327085, 451.4502725, 451.9427855, 461.6107755, 462.5510875, 477.57741050000004, 478.02787850000004, 483.4631385, 483.92546600000003, 487.5412035, 488.04199900000003, 501.57348249999995, 502.0233025, 502.561888, 502.982858, 512.4243240000001, 512.910502, 522.554229, 522.9923249999999, 543.522925, 543.950918, 549.513445, 549.9620239999999, 563.4415355000001, 564.424374, 566.4413824999999, 566.9246955, 574.4705220000001, 574.9697960000001, 591.6199295, 592.0374280000001, 601.505446, 601.9699834999999, 607.4938044999999, 607.5439664999999, 610.359062, 610.371324, 619.5257225, 619.9980295, 622.5157715, 622.9808785, 627.440381, 627.9099545, 633.447698, 633.506179, 633.953385, 636.4711275, 636.955744, 649.4000785000001, 649.4055895, 652.6277375, 653.071807, 655.3746759999999, 655.3954515, 672.486273, 672.9850054999999, 678.41783, 678.4381985, 683.5286655, 683.9644975, 684.5512865000001, 684.9829965, 686.4714455000001, 686.741471, 689.4833854999999, 689.5352245, 697.4884904999999, 697.9803345, 701.52666, 701.9721285, 706.545652, 706.9901525, 708.4742249999999, 708.94415, 717.5692375, 718.0413905, 727.51198, 727.9747935, 732.5468900000001, 732.996718, 741.560363, 742.042053, 748.4907820000001, 748.953591, 749.4655375, 749.9174995000001, 760.5312759999999, 760.9814934999999, 767.532007, 767.9762754999999, 769.4812675000001, 769.962468, 770.3937635, 770.4132565, 770.5757249999999, 771.0430635, 771.5295025, 771.970518, 773.529295, 773.97465, 775.4327265, 775.9016565, 778.4096005, 778.418672, 780.5695925, 781.0295035, 781.4619399999999, 781.52103, 783.5462144999999, 783.9734229999999, 787.4351875, 787.483822, 788.490971, 788.95402, 796.6243440000001, 797.06675, 797.519086, 797.9646150000001, 804.4671579999999, 804.9195749999999, 806.5689295, 806.9984115, 808.465281, 808.9152939999999, 825.532454, 825.9948745, 826.472667, 826.9491645, 831.5037374999999, 831.944135, 842.518717, 842.9670185, 853.5855985000001, 854.027311, 857.381525, 857.4038015, 857.6086740000001, 858.0432265, 858.611131, 859.050439, 859.405321, 859.4250705, 860.485091, 860.9636324999999, 869.4582065, 869.913701, 870.489516, 870.5380620000001, 878.3992345, 878.4133065000001, 878.4264069999999, 878.4380835, 879.4475395, 879.4776165000001, 880.4213325000001, 880.426766, 884.51965, 884.618154, 884.9992179999999, 887.446149, 887.989472, 889.5074225000001, 889.946299, 899.4857095, 899.931529, 903.4821205000001, 903.930801, 914.4118385, 914.552813, 915.02239, 915.446644, 915.486639, 916.4597705, 916.4671384999999, 916.4849495, 916.931112, 920.554963, 921.0308155, 922.352543, 922.3584800000001, 925.5349395000001, 925.9572235, 929.5566195, 929.968532, 935.5245255, 935.9851954999999, 940.448343, 940.4590475, 940.9069895, 942.5207055000001, 942.9586895, 947.4854455, 947.5204695, 948.455127, 948.4883050000001, 951.6006785, 952.0432625, 952.5045605, 952.5506359999999, 955.6068740000001, 956.023151, 956.5520875, 957.0008290000001, 958.5154494999999, 958.9598455, 962.5332395, 962.9571344999999, 964.5087665, 964.5335315, 964.9545365, 966.6188259999999, 967.0534835, 969.541837, 969.9829159999999, 971.4659220000001, 971.9215389999999, 977.5067859999999, 977.9509775, 980.508319, 980.9624655, 983.4179935, 983.8913415, 985.4329785, 985.4533624999999, 985.6305649999999, 986.0731785, 989.4634205, 989.9175789999999, 991.47543, 991.916829, 992.5354185, 992.9668505, 993.580472, 994.021527, 994.4540999999999, 994.9159245, 998.5424075000001, 998.9823325, 999.4414435000001, 999.9077265000001, 1000.465404, 1000.937984, 1003.594104, 1003.9733235, 1005.5343075000001, 1005.9760995, 1013.4302795, 1013.4391430000001, 1023.3687105, 1023.3709564999999, 1025.568066, 1026.015674, 1029.3537425, 1029.3731085, 1032.35933, 1032.3604295, 1035.473398, 1035.942078, 1041.5030649999999, 1041.5462045, 1042.4440345, 1042.9022145, 1043.4086085, 1043.9071975000002, 1059.5119439999999, 1059.9387980000001, 1060.5705665, 1061.0346035, 1063.425712, 1063.45668, 1063.9163565, 1064.3575985, 1064.373212, 1064.5700725000002, 1065.0037665, 1067.4267055, 1067.4481475, 1074.54427, 1075.0023395, 1076.353868, 1076.3568965, 1081.5058645, 1081.5754474999999, 1081.9801785, 1083.551987, 1083.9697355, 1087.4817699999999, 1087.932206, 1091.5569675, 1091.9716535, 1093.5101730000001, 1093.965105, 1095.4285220000002, 1095.9029745, 1098.3779359999999, 1098.386191, 1103.5195370000001, 1103.982769, 1104.505815, 1104.948216, 1105.5497625, 1106.0118935, 1106.5637230000002, 1107.0263805, 1111.5811015, 1112.041929, 1112.535883, 1112.9991655, 1116.3568395, 1116.3608515, 1116.5128180000002, 1116.9399615, 1123.3961615, 1123.4002265, 1124.3995375, 1124.415351, 1124.494796, 1124.9286895, 1125.6068465, 1126.034041, 1128.4710175, 1128.9408665, 1130.370481, 1130.3895775, 1132.5037659999998, 1132.968366, 1134.404372, 1134.409194, 1135.5977905, 1135.9941815, 1137.4901785000002, 1137.5202920000002, 1144.3641005, 1144.3669049999999, 1144.5150789999998, 1144.983362, 1148.3606829999999, 1148.3691655, 1168.424722, 1168.466416, 1172.431685, 1172.911121, 1174.5619755, 1175.011209, 1175.4311925, 1175.4634435, 1175.5023035, 1175.9409154999998, 1176.472755, 1176.4991255, 1178.358044, 1178.3601215, 1178.4784855, 1178.9449915, 1180.3786180000002, 1180.3844450000001, 1182.496599, 1182.5143309999999, 1185.4849015, 1185.9351980000001, 1191.4333035, 1191.4637305, 1191.5306715, 1191.944261, 1198.903466, 1199.3765895000001, 1203.39805, 1203.4143745000001, 1205.443142, 1205.4547465, 1207.493951, 1207.947877, 1209.46507, 1209.5276585, 1209.9574545, 1212.555235, 1212.9792445, 1213.4442365, 1213.480474, 1213.5499845, 1213.985607, 1219.496519, 1219.9483605, 1221.6134265, 1222.0353405, 1224.4898134999999, 1224.814018, 1225.5746720000002, 1226.0427635, 1229.5026755, 1229.9541625, 1231.5422429999999, 1231.993172, 1233.590547, 1234.029391, 1234.464461, 1234.9285515, 1236.4661125, 1236.479724, 1238.5200675, 1238.9807775, 1240.433788, 1240.46992, 1240.5371095, 1240.6581259999998, 1250.3732774999999, 1250.3829999999998, 1253.4032269999998, 1253.4280425, 1258.3545960000001, 1258.356245, 1258.360294, 1258.367932, 1258.554353, 1258.9783575000001, 1259.5691390000002, 1259.9768205, 1261.4780475, 1261.9526925, 1262.517958, 1263.0172280000002, 1263.5120495, 1263.972765, 1266.5745325, 1266.987943, 1268.454982, 1268.9094049999999, 1275.5203365, 1275.978883, 1277.5768285, 1278.044069, 1282.438684, 1282.9411734999999, 1286.396424, 1286.4719975, 1288.5971555, 1289.0236645, 1295.529641, 1295.9847635, 1301.4563229999999, 1301.4835555, 1301.525232, 1301.5353575, 1306.4545575, 1306.5286635, 1308.450879, 1308.6256565, 1309.0577555, 1309.49774, 1309.7739000000001, 1312.3518695, 1312.3552815, 1313.477114, 1313.9191495, 1321.6281865, 1321.6625035, 1322.0193614999998, 1323.4808495, 1323.9508505, 1326.4996919999999, 1326.9528599999999, 1327.5357589999999, 1327.970362, 1328.446591, 1328.9209205000002, 1330.4142259999999, 1330.8877215, 1333.5041985, 1333.590597, 1335.5020365, 1335.9417515, 1336.4795245, 1336.935986, 1339.476893, 1339.962172, 1341.4542820000001, 1341.9107955, 1343.3759545, 1343.3895545, 1343.4279065, 1343.4520175, 1349.3915885000001, 1349.4119885, 1350.5589275, 1350.984579, 1355.553549, 1355.9915314999998, 1356.526652, 1356.975862, 1358.5257845, 1358.9597130000002, 1359.5350855000002, 1359.999385, 1365.5347040000001, 1365.9534265, 1369.4066025, 1369.437089, 1369.4901060000002, 1369.941745, 1375.4363395, 1375.5636905000001, 1376.002109, 1379.4668325, 1379.9265504999998, 1387.9114335, 1388.3576134999998, 1396.3970225, 1396.521873, 1399.527268, 1400.0055619999998, 1403.4078425, 1403.42471, 1408.3677795, 1408.3694895, 1408.5812775, 1409.0034890000002, 1409.36382, 1409.381061, 1411.495071, 1411.933333, 1414.5844615, 1415.0044560000001, 1422.514438, 1422.6142275, 1423.0237175, 1424.604787, 1425.0420435, 1425.3741125000001, 1425.3870339999999, 1425.617066, 1426.04954, 1427.498827, 1427.941527, 1428.483224, 1428.940621, 1432.539385, 1433.041594, 1437.4032379999999, 1437.4155475, 1442.6388955, 1443.053275, 1444.5604604999999, 1445.008029, 1445.442458, 1445.91068, 1450.4148384999999, 1450.896021, 1453.3919839999999, 1453.4424205, 1457.5462635, 1457.983464, 1458.419499, 1458.445769, 1459.468897, 1459.485837, 1462.4190965, 1462.9048954999998, 1465.473684, 1465.5029245, 1465.9498055, 1466.653538, 1467.0591985, 1469.5521800000001, 1469.9827930000001, 1475.5016065, 1475.9412065000001, 1481.5661495, 1482.3628265, 1483.5101049999998, 1483.9467245, 1486.5681305, 1487.008153, 1490.420036, 1490.903906, 1497.3742805, 1497.3860894999998, 1497.5692085, 1498.0295820000001, 1498.4268160000001, 1498.9198995000002, 1507.5279405, 1507.9408035000001, 1508.5099344999999, 1508.9834935, 1512.3626330000002, 1512.371376, 1513.4731335000001, 1513.5674625000001, 1513.9993725, 1516.4932895, 1516.5143130000001, 1527.908622, 1528.3530595, 1533.4249455, 1533.5605125, 1533.6994719999998, 1534.0460214999998, 1536.438882, 1536.4736705, 1543.606599, 1544.0357589999999, 1547.5846065, 1548.0212865, 1554.395362, 1554.889157, 1556.4458355000002, 1556.9188100000001, 1559.4839235, 1559.5384945, 1566.5535665, 1566.9799469999998, 1573.46349, 1573.9104539999998, 1577.3785635, 1577.3916175, 1579.515921, 1579.9482899999998, 1589.4593745, 1589.956109, 1595.5218595000001, 1595.968679, 1596.4316345, 1596.456972, 1603.3821874999999, 1603.4006425, 1611.4261219999999, 1611.6157899999998, 1612.061737, 1628.503515, 1628.9695080000001, 1636.551174, 1636.56595, 1642.4641895, 1642.946317, 1643.4631055, 1643.951246, 1673.0392425, 1673.3904505, 1674.4557915, 1674.92099, 1675.5586645, 1675.995756, 1676.4550375, 1676.9490715000002, 1687.372858, 1687.384314, 1701.467881, 1702.010346, 1710.459015, 1710.911652, 1711.3802085, 1711.8685375, 1722.6205315, 1723.0626505, 1758.5723005, 1759.007071, 1789.5198295, 1790.022729, 1808.4629730000001, 1809.4459545, 1824.9072325000002, 1825.3821715, 1836.997132, 1838.4569685000001, 1842.355831, 1842.86247, 1844.522641, 1845.002601, 1850.550366, 1851.040301, 1855.5682495, 1856.0402924999999, 1856.4947685, 1857.024954, 1857.5012715, 1857.9791155, 1864.0405034999999, 1865.0074034999998, 1882.4562515, 1883.4546015])
labels = array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 0.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 3.0, 1.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 4.0, 3.0, 0.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 4.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0])
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

numthresholds = 729



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


