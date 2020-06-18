#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593724/phpe7BPV1 -o Predictors/volcanoes-a2_QC.py -target Class -stopat 91.19 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:02.75. Finished on: May-22-2020 05:44:30.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                90.85%
Model accuracy:                     100.00% (1623/1623 correct)
Improvement over best guess:        9.15% (of possible 9.15%)
Model capacity (MEC):               281 bits
Generalization ratio:               5.77 bits/bit
Confusion Matrix:
 [2.16% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 1.79% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 2.71% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 2.71% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 90.63%]

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
TRAINFILE = "phpe7BPV1.csv"


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
    clean.mapping={'2': 0, '3': 1, '4': 2, '5': 3, '1': 4}

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
energy_thresholds = array([260.94920649999995, 263.5979635, 267.03700249999997, 315.4698535, 319.0241995, 350.9759955, 351.9685815, 416.5388295, 417.0493075, 422.455204, 423.02127700000005, 438.57099700000003, 439.5643625, 460.476321, 462.015911, 492.588441, 495.047412, 499.0512445, 500.1957445, 501.526196, 505.9875845, 507.47805100000005, 511.5073205, 514.4910394999999, 549.078741, 550.027415, 575.888625, 576.434842, 583.558073, 584.0608905, 609.4671209999999, 609.5944905, 610.0671985, 622.6194505, 623.050175, 626.4421175, 627.044478, 647.3704, 648.876923, 706.025886, 707.0249530000001, 708.50118, 709.0392805, 709.6058955, 711.0847134999999, 726.0640249999999, 727.0570475, 756.455325, 756.5573254999999, 757.0365425, 760.3655785000001, 760.868699, 768.096677, 769.0568425, 772.5083325, 774.483228, 779.5497485000001, 780.015669, 780.506734, 782.5082635, 783.5146179999999, 784.4944025, 784.969327, 796.4558455, 797.4566805, 817.4054005, 817.5160900000001, 817.9967835, 828.102138, 829.5550995000001, 830.5696995000001, 831.174138, 831.9829675, 850.6005305, 852.087046, 854.0739935, 855.0080555, 858.5657960000001, 859.5104865000001, 878.5288355, 879.027493, 879.7494005000001, 880.062184, 891.5876055, 891.9948234999999, 894.5851505, 894.9990849999999, 901.494719, 902.0787435, 902.9793400000001, 905.6245905, 906.0541599999999, 919.524424, 920.0253605, 925.544312, 926.527668, 942.5328635000001, 943.0316425, 951.520356, 952.0077194999999, 954.4971694999999, 954.6058345, 954.66754, 955.0653565, 958.0393525, 959.0523310000001, 975.981813, 976.92908, 980.166728, 981.0657590000001, 1005.4708390000001, 1007.0637730000001, 1011.513647, 1012.0172205, 1015.6591535, 1017.050966, 1022.4153670000001, 1022.93866, 1024.5150135, 1025.0211359999998, 1025.9658915, 1026.6447, 1027.0605845, 1029.4243675, 1029.931221, 1031.0300335000002, 1032.0379675, 1037.5247745000001, 1038.07707, 1042.987236, 1043.9989475, 1049.007528, 1050.0619015, 1055.509361, 1055.5644645, 1061.4468474999999, 1061.940244, 1074.4062804999999, 1074.9153270000002, 1075.5270885, 1075.6886255, 1077.140824, 1078.9679295, 1082.0410695, 1083.461274, 1092.023607, 1093.054723, 1094.514059, 1095.4565475, 1099.59094, 1100.204957, 1105.5684565000001, 1106.0742075, 1109.446228, 1110.1081215, 1111.0427399999999, 1113.506394, 1113.9674975, 1114.521057, 1115.970168, 1122.4345955, 1122.574773, 1123.012796, 1123.489298, 1123.959658, 1134.4029505, 1135.0508425, 1136.1148440000002, 1137.0990955, 1145.0996839999998, 1147.023066, 1193.50938, 1193.9885815, 1197.041127, 1198.022393, 1198.9652845, 1199.9593810000001, 1206.5356135, 1208.005418, 1223.018919, 1224.009468, 1224.528338, 1225.0503045, 1242.6279614999999, 1243.0332939999998, 1243.5618100000002, 1245.02779, 1250.0617625, 1253.0500195, 1254.5792059999999, 1255.5000645, 1260.5378609999998, 1261.0215975, 1261.4164085, 1261.910566, 1265.5108625, 1265.9659625, 1277.521913, 1277.9562775, 1283.0216615, 1284.016053, 1285.5352775000001, 1286.995313, 1308.3838375, 1308.4577215, 1309.4524595, 1309.5300124999999, 1309.9676435000001, 1315.4593665, 1316.9425145, 1317.9778685, 1318.925884, 1328.0806714999999, 1329.0588914999998, 1332.539788, 1332.9807270000001, 1335.4775545, 1336.0143065, 1338.0566735, 1339.0998155, 1347.464096, 1347.9774790000001, 1352.4517985, 1352.9868545, 1354.9517879999999, 1355.964497, 1369.603039, 1370.570493, 1378.9476875, 1380.6254255, 1381.568742, 1391.4592830000001, 1391.949522, 1398.5989220000001, 1399.591073, 1402.5548555, 1404.5044415, 1405.6127465, 1406.1343590000001, 1417.5224165, 1418.1240195, 1419.4773525, 1453.931344, 1454.5067715, 1455.598512, 1458.0300255, 1481.994499, 1483.4288824999999, 1489.0632095, 1491.0650065, 1501.574337, 1504.5313795, 1505.4446335, 1505.5968685, 1521.0859464999999, 1522.0828755, 1525.613204, 1527.5454129999998, 1530.44205, 1530.4591194999998, 1556.5496125, 1557.0909434999999, 1580.543984, 1582.075992, 1585.518842, 1586.506354, 1649.4437075, 1652.9307095, 1699.50248, 1702.951507, 1764.0565735, 1765.6665795, 1786.5082364999998, 1789.555503, 1799.0340935, 1799.962814, 1812.562841, 1815.0345404999998, 1824.5234845, 1827.0224295, 1996.9723995])
labels = array([2.0, 0.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0, 4.0, 0.0, 4.0, 0.0, 2.0, 4.0, 2.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 0.0, 4.0, 3.0, 4.0, 3.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 1.0, 2.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 1.0, 4.0, 1.0, 4.0, 2.0, 4.0, 0.0, 4.0, 2.0, 1.0, 4.0, 2.0, 4.0, 1.0, 3.0, 4.0, 0.0, 4.0, 3.0, 4.0, 3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 1.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 0.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.0, 4.0, 0.0, 4.0, 1.0, 4.0, 0.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 4.0, 0.0, 4.0, 1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 4.0, 2.0, 4.0, 2.0, 1.0, 3.0, 4.0, 0.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 3.0, 4.0, 2.0, 3.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 1.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 4.0, 2.0, 4.0, 1.0, 4.0, 0.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 3.0, 4.0, 3.0, 4.0, 0.0, 4.0, 3.0, 4.0, 0.0, 4.0, 0.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 2.0, 4.0, 2.0, 3.0, 4.0, 1.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 0.0, 4.0, 0.0, 4.0, 1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0, 4.0, 2.0, 1.0, 4.0, 1.0, 4.0, 0.0, 4.0, 1.0, 4.0, 1.0, 4.0, 2.0, 1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.0, 4.0, 3.0, 4.0, 2.0, 4.0, 2.0, 4.0, 1.0, 4.0, 3.0, 4.0, 3.0, 4.0, 2.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 4.0, 3.0])
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
        outputs[defaultindys] = 4.0
        return outputs
    return thresh_search(energys)

numthresholds = 281



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


