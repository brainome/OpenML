#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/30/dataset_30_page-blocks.arff -o Predictors/page-blocks_QC.py -target class -stopat 97.62 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:04.56. Finished on: May-28-2020 00:11:03.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                89.83%
Model accuracy:                     99.41% (5441/5473 correct)
Improvement over best guess:        9.58% (of possible 10.17%)
Model capacity (MEC):               875 bits
Generalization ratio:               6.21 bits/bit
Confusion Matrix:
 [89.64% 0.05% 0.07% 0.00% 0.00%]
 [0.16% 5.85% 0.00% 0.00% 0.00%]
 [0.15% 0.11% 1.35% 0.00% 0.00%]
 [0.02% 0.00% 0.02% 2.06% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.51%]

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
TRAINFILE = "dataset_30_page-blocks.csv"


#Number of attributes
num_attr = 10
n_classes = 5


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="class"


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
    clean.mapping={'1': 0, '2': 1, '4': 2, '5': 3, '3': 4}

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
energy_thresholds = array([38.25, 38.6965, 39.143, 39.3495, 40.8, 42.0, 42.388999999999996, 43.9445, 44.0, 44.125, 45.0625, 46.0, 46.085499999999996, 46.1835, 46.9335, 47.538, 49.0135, 49.111000000000004, 49.111000000000004, 49.472, 49.854, 50.1875, 50.5835, 50.8335, 52.0, 52.0, 52.3575, 53.605000000000004, 53.710499999999996, 53.778, 53.790499999999994, 54.091499999999996, 54.1165, 55.8015, 56.028, 56.8, 56.900499999999994, 58.0, 58.285, 58.8955, 59.212, 59.3665, 59.506, 59.67100000000001, 60.228500000000004, 60.7385, 61.575, 61.95, 63.6665, 64.0415, 64.3335, 64.6255, 65.76599999999999, 66.2775, 66.611, 67.8295, 67.9345, 68.225, 68.44999999999999, 68.8135, 69.1845, 69.6585, 70.24600000000001, 70.815, 70.875, 72.406, 72.7855, 73.22, 73.4225, 73.571, 73.7365, 73.994, 74.107, 74.1965, 74.3495, 74.50999999999999, 74.76050000000001, 74.9385, 75.869, 76.5375, 77.0595, 77.212, 78.819, 79.552, 80.2585, 81.1335, 81.96000000000001, 82.005, 82.22200000000001, 82.616, 83.0095, 83.417, 83.8245, 83.8745, 83.8955, 83.95150000000001, 84.02000000000001, 84.0495, 84.1985, 84.35050000000001, 84.483, 85.799, 86.1315, 86.3965, 86.638, 87.532, 87.76599999999999, 88.132, 89.0295, 89.1795, 89.3755, 89.459, 89.48349999999999, 89.9435, 90.4875, 90.985, 91.11500000000001, 91.3075, 91.4005, 92.107, 92.3155, 92.467, 92.93050000000001, 93.5815, 94.02799999999999, 94.4215, 95.30600000000001, 95.613, 96.0465, 96.256, 96.3535, 96.644, 97.4535, 98.1695, 98.7075, 99.04499999999999, 99.887, 100.0375, 100.09299999999999, 100.259, 100.58800000000001, 100.85650000000001, 101.2415, 101.82, 102.8335, 103.2295, 103.735, 103.99549999999999, 104.1325, 105.6, 106.014, 106.9495, 107.17349999999999, 108.2655, 108.9965, 109.315, 109.565, 109.8005, 109.8725, 110.459, 113.90899999999999, 114.191, 114.62049999999999, 114.7385, 116.031, 116.6505, 117.20400000000001, 117.324, 118.8315, 119.061, 119.27550000000001, 119.5835, 123.642, 123.768, 125.6455, 125.929, 127.5155, 127.643, 129.0195, 129.05450000000002, 129.15300000000002, 129.4135, 130.054, 130.151, 133.393, 133.9965, 134.144, 135.5865, 135.7855, 135.9375, 135.992, 136.188, 136.608, 136.708, 137.454, 137.688, 138.76100000000002, 139.10649999999998, 140.344, 140.3785, 141.746, 141.973, 145.352, 145.98250000000002, 147.089, 147.6825, 148.211, 148.6435, 148.882, 150.01999999999998, 150.1565, 150.478, 150.6625, 150.89749999999998, 151.08100000000002, 153.719, 154.0, 154.675, 154.7375, 155.243, 155.657, 158.9795, 159.04749999999999, 159.2115, 159.974, 160.008, 160.2935, 160.817, 161.3225, 161.4845, 162.2475, 162.328, 163.353, 163.7475, 164.01, 164.121, 165.1035, 165.137, 166.2815, 166.36149999999998, 167.01100000000002, 167.364, 167.4325, 167.7305, 169.8645, 169.9855, 170.274, 170.5635, 170.6055, 170.626, 173.8725, 174.22, 176.7615, 176.89600000000002, 177.925, 178.122, 178.99450000000002, 179.1475, 179.41750000000002, 179.755, 179.966, 180.482, 180.4975, 180.7775, 184.315, 184.4065, 184.45999999999998, 184.5255, 184.7455, 184.8305, 185.57299999999998, 185.8785, 189.01749999999998, 189.027, 189.1035, 190.5215, 190.9545, 191.76100000000002, 191.8175, 193.869, 194.092, 196.3695, 196.474, 198.743, 199.10750000000002, 200.84, 200.88850000000002, 201.934, 202.1735, 202.5375, 202.625, 206.60199999999998, 206.7405, 210.184, 210.44349999999997, 212.4665, 212.555, 213.982, 214.1225, 214.309, 214.4115, 215.67399999999998, 216.054, 220.512, 220.603, 221.79449999999997, 222.0695, 227.62849999999997, 227.73649999999998, 228.12099999999998, 228.531, 237.63, 237.92700000000002, 238.503, 238.6655, 238.9205, 239.111, 240.8865, 240.94549999999998, 241.29749999999999, 241.552, 242.789, 242.9805, 243.9875, 244.16899999999998, 247.58499999999998, 247.965, 248.14800000000002, 248.401, 251.039, 251.495, 251.95749999999998, 252.04149999999998, 254.3035, 254.675, 254.87400000000002, 255.4665, 255.765, 264.346, 264.807, 266.0125, 266.048, 266.801, 267.2875, 268.971, 269.0895, 273.998, 274.052, 280.56550000000004, 280.751, 280.811, 280.84299999999996, 281.214, 281.303, 283.621, 283.705, 285.952, 286.07550000000003, 287.9365, 288.121, 289.7615, 290.082, 293.23, 293.46299999999997, 293.99199999999996, 294.0345, 298.6385, 298.6835, 302.62149999999997, 303.20799999999997, 306.119, 307.241, 309.315, 309.404, 309.759, 309.9015, 310.0135, 310.119, 313.821, 314.1305, 317.4155, 317.847, 321.39549999999997, 321.66200000000003, 325.395, 325.801, 328.2335, 328.336, 337.9065, 338.04449999999997, 339.793, 339.9935, 341.653, 342.0205, 344.70399999999995, 345.153, 348.1495, 348.4865, 351.844, 352.1355, 352.4955, 352.8945, 354.964, 355.579, 362.198, 362.32, 369.959, 370.0585, 373.807, 374.1385, 376.9135, 377.2455, 380.198, 380.375, 380.4275, 380.495, 381.911, 382.20550000000003, 383.91150000000005, 384.66450000000003, 389.3065, 389.543, 391.033, 391.12350000000004, 396.28, 396.4275, 403.946, 404.1055, 404.9555, 405.29049999999995, 406.4935, 406.9075, 414.586, 414.932, 416.24, 416.884, 418.2905, 418.5615, 419.443, 420.1495, 422.75350000000003, 423.01, 423.97749999999996, 424.006, 424.2525, 427.19, 427.2855, 436.317, 436.505, 439.055, 439.572, 450.41, 450.967, 451.7415, 451.9285, 468.89700000000005, 469.1305, 474.719, 474.79650000000004, 493.77, 493.9755, 494.703, 494.815, 495.867, 496.169, 512.441, 512.6714999999999, 514.8685, 515.2665000000001, 518.387, 518.53, 520.2205, 520.835, 529.8, 530.267, 536.8385000000001, 537.002, 537.348, 537.7760000000001, 538.5595000000001, 539.1469999999999, 547.1855, 547.527, 551.938, 552.0615, 552.187, 552.2475, 565.4110000000001, 565.9475, 568.266, 568.425, 572.4695, 572.495, 573.4745, 574.078, 576.9285, 577.0875000000001, 587.4314999999999, 588.271, 589.1775, 592.297, 592.3534999999999, 593.9725, 594.3309999999999, 601.433, 601.5419999999999, 612.2545, 612.636, 625.538, 626.001, 630.7105, 631.226, 636.5775, 636.812, 652.7570000000001, 653.8795, 671.726, 673.0464999999999, 676.472, 676.609, 676.876, 681.9775, 682.062, 714.4085, 715.454, 720.662, 721.2345, 729.3295, 729.341, 740.3235, 740.5475, 757.5519999999999, 758.059, 758.154, 758.3054999999999, 762.6355, 763.4055, 764.012, 764.3165, 776.6445, 778.5745, 786.7195, 787.4165, 803.9825000000001, 805.5535, 815.947, 816.2085, 816.44, 816.6615, 821.604, 822.003, 823.232, 823.698, 831.7090000000001, 832.0945, 836.963, 837.3715, 840.9815, 841.69, 847.9815, 848.637, 849.053, 849.909, 850.194, 873.347, 873.6995, 892.6315, 893.3005, 916.886, 917.338, 924.53, 924.6044999999999, 926.6465000000001, 927.652, 946.8745, 947.489, 955.799, 956.004, 963.3795, 963.742, 965.4395, 965.8725, 990.133, 990.971, 1026.058, 1027.4425, 1028.6835, 1029.566, 1045.5765000000001, 1047.768, 1054.7334999999998, 1056.238, 1057.413, 1057.7035, 1064.379, 1065.126, 1132.3725, 1132.7215, 1153.263, 1153.772, 1188.8735, 1190.1025, 1193.632, 1195.0945, 1235.8575, 1239.0140000000001, 1285.9560000000001, 1288.165, 1290.3065000000001, 1290.7605, 1303.4575, 1303.9234999999999, 1327.4279999999999, 1328.062, 1348.501, 1349.5569999999998, 1397.4355, 1397.777, 1398.4155, 1400.6864999999998, 1415.4160000000002, 1416.5900000000001, 1420.6745, 1421.2134999999998, 1433.004, 1434.229, 1457.0425, 1458.6575, 1477.923, 1481.102, 1525.437, 1525.7514999999999, 1545.8445, 1546.33, 1558.4454999999998, 1559.6315, 1648.051, 1648.9854999999998, 1662.9859999999999, 1666.0515, 1689.1895, 1691.9565, 1757.6385, 1758.716, 1768.1264999999999, 1769.225, 1843.4679999999998, 1844.6060000000002, 1855.8995, 1857.037, 1939.922, 1942.013, 1950.7285000000002, 1950.813, 2042.2855, 2043.163, 2047.6965, 2048.57, 2066.773, 2068.2165, 2085.2715, 2086.348, 2096.896, 2097.4184999999998, 2101.3365, 2103.816, 2117.9325, 2120.572, 2129.859, 2130.9915, 2153.173, 2155.4309999999996, 2168.5665, 2169.5895, 2176.194, 2178.863, 2339.8140000000003, 2341.648, 2396.0095, 2398.8725, 2477.4375, 2481.415, 2507.615, 2509.8085, 2533.859, 2534.8885, 2711.9705, 2719.3125, 2762.2645, 2765.699, 2885.9624999999996, 2889.1255, 2958.489, 2959.8685, 2995.2250000000004, 2995.8195, 3037.7565, 3039.958, 3063.8685, 3065.452, 3095.8360000000002, 3097.105, 3114.417, 3115.8685, 3118.25, 3120.8805, 3124.0515, 3126.3835, 3130.3775, 3136.335, 3144.9835000000003, 3145.812, 3149.362, 3151.407, 3153.2025000000003, 3154.6345, 3156.664, 3161.398, 3162.1255, 3168.5460000000003, 3171.2574999999997, 3175.669, 3183.3835, 3190.282, 3192.715, 3213.6375, 3217.089, 3224.687, 3226.3775, 3230.6155, 3233.5789999999997, 3250.312, 3251.9015, 3269.5605, 3275.248, 3292.0025, 3295.699, 3350.69, 3351.6724999999997, 3379.7635, 3380.3625, 3405.502, 3409.9350000000004, 3434.8985, 3436.48, 3472.2415, 3478.139, 3506.8955, 3514.9449999999997, 3522.8225, 3524.5145, 3562.694, 3567.5634999999997, 3581.4755, 3582.7394999999997, 3584.7599999999998, 3586.328, 3595.4264999999996, 3597.7934999999998, 3603.1695, 3604.822, 3609.7205, 3615.001, 3620.282, 3629.623, 3642.2070000000003, 3645.3900000000003, 3655.295, 3668.639, 3790.044, 3792.5950000000003, 3809.8485, 3813.6155, 3824.2794999999996, 3826.346, 3829.911, 3831.542, 4095.173, 4102.171, 4150.6345, 4162.4595, 4169.3265, 4172.404500000001, 4364.292, 4364.943, 4528.554, 4557.7055, 4686.6855, 4694.531, 4749.504, 4751.2775, 4810.994500000001, 4828.4735, 5069.278, 5071.550499999999, 5223.705, 5224.7245, 5544.7665, 5550.276, 5617.204, 5629.5615, 5695.1595, 5699.351500000001, 5821.82, 5845.5185, 6324.812, 6332.851999999999, 6669.763, 6679.6115, 7723.0355, 7740.611999999999, 8715.6295, 8724.254, 8859.529499999999, 8869.086, 8876.648000000001, 8885.2915, 8901.5215, 8909.3035, 9452.7805, 9466.164499999999, 9515.376499999998, 9573.169999999998, 9908.377, 9922.585500000001, 10251.126, 10266.0455, 10706.6915, 10716.6735, 10798.871, 10804.846, 12650.272, 12670.412499999999, 13491.8875, 13613.1015, 14688.1325, 14718.304500000002, 14736.446500000002, 14800.3285, 15632.283, 15749.9585, 17337.744, 17465.12, 17505.158000000003, 17805.323, 18063.858, 18130.381, 19909.680999999997, 20135.6625, 20614.331, 21103.3425, 21365.122, 21557.957000000002, 22870.242, 23093.192499999997, 23369.578, 23624.251, 24589.074, 24621.7315, 33925.413499999995, 34827.509, 37118.781, 38235.7765, 39107.864, 39292.781, 43384.633, 49131.957500000004, 54450.823, 63288.6415, 64389.174, 65729.72150000001, 67519.65950000001, 90416.1175, 123601.90400000001, 169161.168])
labels = array([0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 3.0, 0.0, 4.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 4.0, 0.0, 3.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 1.0, 3.0, 4.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 4.0, 0.0, 1.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0])
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

numthresholds = 875



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


