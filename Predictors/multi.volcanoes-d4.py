#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Class volcanoes-d4.csv -o volcanoes-d4.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:00:12.26. Finished on: Sep-04-2020 12:19:08.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         5-way classifier
Best-guess accuracy:                 94.33%
Overall Model accuracy:              100.00% (8654/8654 correct)
Overall Improvement over best guess: 5.67% (of possible 5.67%)
Model capacity (MEC):                920 bits
Generalization ratio:                9.40 bits/bit
Model efficiency:                    0.00%/parameter
Confusion Matrix:
 [2.73% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.97% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 94.33% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.65% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 1.33%]
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
TRAINFILE = "volcanoes-d4.csv"


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
energy_thresholds = array([55.43564550000001, 57.930362, 90.518136, 91.09114550000001, 111.9192405, 113.3919325, 129.5007085, 129.98458549999998, 158.5900385, 159.058009, 178.94359, 179.881662, 182.57502449999998, 183.99800499999998, 184.898105, 185.46736900000002, 222.189668, 223.0465235, 235.9452215, 236.9978395, 249.61913099999998, 250.0360645, 253.0109005, 255.008759, 257.429002, 258.45669399999997, 268.4327725, 268.9098085, 273.58446000000004, 274.53249900000003, 279.52516049999997, 279.9966505, 301.60201900000004, 302.0361795, 321.621601, 322.05471850000004, 323.69815700000004, 324.04231100000004, 326.4853825, 326.9799755, 330.3516535, 330.367382, 348.4786315, 348.9984425, 359.495453, 359.9698255, 360.527204, 360.978427, 366.4976865, 366.6644125, 366.7412645, 367.104371, 367.538603, 368.001729, 369.6083555, 370.06593399999997, 370.53983900000003, 370.979539, 378.538805, 379.029952, 379.538586, 380.004767, 382.578672, 382.730629, 383.062736, 383.4502105, 383.506763, 404.456583, 404.5253065, 404.627798, 405.09799999999996, 406.5545565, 407.53058350000003, 409.52160200000003, 409.971112, 413.53679450000004, 413.96357, 416.619931, 417.016523, 426.619676, 427.01847599999996, 427.8895665, 428.384252, 446.471007, 446.4768425, 457.4814045, 457.998046, 460.5584295, 460.68356900000003, 467.506104, 467.947908, 481.5522015, 481.9789585, 484.48735899999997, 484.633491, 485.4924385, 485.9451135, 487.520388, 487.661927, 488.052024, 488.385557, 488.4221, 488.475886, 495.57136049999997, 495.99539400000003, 500.40687649999995, 500.441734, 500.5071995, 500.5636125, 504.551896, 504.5772825, 506.5758435, 507.0941735, 512.9147125, 513.4039535, 513.454916, 513.5079355, 513.9492305, 515.407097, 515.470684, 519.5453585, 520.0238895, 520.5904125, 521.0645374999999, 521.446448, 521.9116655, 523.6945235000001, 524.0531805, 527.4006999999999, 527.406637, 538.6633875, 539.0344115, 539.491062, 539.9490915, 543.402421, 543.8868259999999, 544.6672555, 545.057416, 555.6092235, 556.0203945000001, 557.5608755000001, 557.9971985, 558.583568, 558.6519814999999, 559.0121389999999, 560.4586099999999, 560.9303084999999, 566.532981, 566.9634034999999, 569.5754870000001, 570.0783369999999, 570.496895, 570.9634155000001, 572.8797099999999, 573.3848325, 573.4670265, 573.5060945, 573.598029, 584.5023695, 584.5791865000001, 586.433466, 586.4654525000001, 590.522397, 591.04219, 602.3773735, 602.4040575, 602.447272, 602.4833404999999, 602.5447825, 602.9861395, 608.537819, 608.9554685, 616.4528215, 616.499315, 619.5772910000001, 619.972647, 623.412576, 623.4456945, 624.5818815, 624.971393, 629.5102335, 629.9698324999999, 631.613822, 632.0238535, 632.6597005000001, 633.021405, 633.5184775, 633.5793115, 634.0677375, 634.3662935, 634.6745695, 635.0702855, 636.607595, 637.1037285, 638.4452249999999, 638.4829575, 643.6440895000001, 644.0717405, 646.7417065, 647.0776255000001, 651.5967305, 651.98028, 655.424358, 655.4823845000001, 659.5317785, 660.0008455, 670.4980095, 670.9673594999999, 675.4249595, 675.43223, 675.5710795, 675.998185, 677.5535015, 677.6719085, 679.6121865, 680.034041, 683.476442, 683.539409, 692.4748725, 692.963923, 694.5511875, 695.465465, 695.595384, 696.068381, 697.4479775, 697.495013, 704.5322595, 704.586013, 704.617657, 707.5325789999999, 708.004091, 711.562255, 711.9889625, 712.9581865, 713.4039339999999, 716.430436, 716.4899215, 716.666211, 717.025943, 719.5415969999999, 720.0360845, 725.5904350000001, 725.9838414999999, 726.942955, 727.3679685, 728.5772065, 729.018818, 730.5208395, 730.9584259999999, 732.6177769999999, 733.0495205, 738.538935, 738.6578030000001, 745.562518, 745.6888405, 750.4708734999999, 750.5526299999999, 751.4350415, 751.48395, 751.601171, 751.7960445, 752.562198, 752.9769775, 755.5518265000001, 755.9879155000001, 762.7139745, 763.0891565, 763.572277, 764.0506825, 765.6344185, 766.0354175, 769.597488, 770.132441, 771.534046, 771.5387475, 778.46404, 778.4915755, 778.546168, 778.970734, 783.506893, 783.5463689999999, 783.6304415, 784.0249080000001, 786.5483905000001, 786.6139575, 787.5181115, 787.655673, 788.039955, 788.6241195, 789.005744, 789.516511, 789.579692, 791.477655, 791.546132, 795.549974, 795.991794, 796.6318269999999, 796.6672309999999, 796.7056915000001, 797.0499130000001, 800.436335, 800.4412990000001, 801.43015, 801.471901, 802.6616355, 802.7167435, 803.6458865, 803.6743495000001, 804.610801, 804.6733145000001, 809.5581970000001, 809.994308, 814.5492429999999, 815.0407144999999, 821.5098674999999, 821.9600975, 825.572971, 825.9871855, 826.4480175, 826.4845025, 829.6623735000001, 830.0920115, 830.9736909999999, 831.445027, 835.4595365, 835.9088995, 838.6658605, 839.0496075, 845.6526154999999, 846.0344245, 852.5446505, 852.9831320000001, 856.5810575, 856.6645765000001, 857.0356995, 859.6075815, 859.989609, 861.4115265, 861.575616, 862.6180704999999, 862.674446, 864.3537385, 864.3653375, 864.4818889999999, 864.5724745, 869.4962674999999, 869.5497745, 879.4454525000001, 879.511368, 884.5670365000001, 885.0437724999999, 886.630817, 887.059887, 887.5812785, 887.9778615, 891.5350060000001, 891.9739015, 893.627347, 894.024609, 896.4566434999999, 896.5110540000001, 901.4052485, 901.410622, 905.5996250000001, 906.0057995, 908.593743, 909.044208, 911.665532, 912.0702085, 913.6457674999999, 914.0599605, 916.5218075, 916.951221, 918.4958295, 918.5552305, 918.614068, 919.026524, 920.4624994999999, 920.4873695, 920.557871, 920.6687870000001, 921.499039, 921.5554804999999, 922.5201535, 923.0170929999999, 923.664561, 924.0380375, 925.6312735, 925.7204825, 926.0748595, 928.6190194999999, 929.0260304999999, 929.618164, 930.0754360000001, 932.4932805000001, 932.9334265, 935.555689, 935.9762470000001, 936.5042834999999, 936.5405965, 937.5543535, 937.999302, 939.5207809999999, 939.5415925, 941.3677225, 941.383631, 941.4467685, 941.4741215, 942.666017, 943.027936, 943.481732, 943.7222675, 945.3748929999999, 945.3806325, 948.4269039999999, 948.4774265, 948.4910665, 948.535856, 953.5236755, 953.9687125, 956.605624, 956.7261249999999, 957.0893715, 957.576636, 957.5962910000001, 960.5740635, 960.604343, 961.6576635, 961.6694095, 962.035042, 962.557374, 962.6060745, 962.993076, 963.4753195000001, 963.552144, 966.5496740000001, 966.5688729999999, 966.7131870000001, 967.049122, 971.388723, 971.4036189999999, 971.5446325, 971.9726045, 975.5070815, 975.554811, 976.6107025, 977.0032739999999, 977.3899445, 977.3985009999999, 978.4530065, 978.4798040000001, 982.574006, 982.675068, 984.6105474999999, 984.6310825, 989.5594945, 989.9953250000001, 994.6517925, 994.7882055, 995.1187150000001, 995.630402, 995.6988325, 996.046656, 996.53477, 996.987627, 999.5684625, 999.9855285, 1001.6472835, 1002.008476, 1004.6712605, 1005.0448134999999, 1006.4717780000001, 1006.4851635, 1007.0575200000001, 1007.3679565, 1007.673684, 1008.0547254999999, 1008.5627704999999, 1009.0100815000001, 1010.606138, 1011.0216069999999, 1011.597119, 1011.9825109999999, 1018.5791505, 1018.970809, 1020.623092, 1021.0413105, 1021.7461975, 1022.060111, 1024.555817, 1025.0012315, 1025.5414485000001, 1025.644107, 1028.540187, 1028.6207555, 1030.9875215, 1031.3650954999998, 1031.4831445, 1031.9271095, 1032.402999, 1032.4092825, 1036.5493145, 1036.9602985, 1037.5714094999998, 1037.975317, 1042.4505239999999, 1042.4978935, 1043.6772740000001, 1044.0199269999998, 1047.5347510000001, 1047.9811495, 1048.3721565, 1048.3970264999998, 1049.9557355, 1050.351827, 1051.6474345000001, 1052.07909, 1053.6187015, 1054.0083344999998, 1057.5716745, 1057.6789315, 1058.0298895, 1058.5203625, 1058.5507545, 1060.6996064999998, 1061.0550595, 1061.5758165000002, 1061.9950735, 1068.627941, 1068.6556095, 1075.384649, 1075.386942, 1079.6187005, 1080.006193, 1083.4189685000001, 1083.4298675, 1085.586607, 1086.034241, 1086.494189, 1086.5208355, 1086.6736150000002, 1087.051956, 1090.603691, 1090.996459, 1100.438547, 1100.9195745, 1106.635086, 1107.051562, 1108.4652445000002, 1108.5719669999999, 1108.6673175, 1109.0483525, 1109.529074, 1109.9504285, 1111.5234169999999, 1111.5498615000001, 1114.385976, 1114.3872740000002, 1116.6179835, 1117.0579935, 1122.3670769999999, 1122.387812, 1125.4706435, 1125.5241529999998, 1128.5528454999999, 1128.612341, 1131.7071775, 1132.083009, 1133.5706705, 1133.9668265, 1134.4556655000001, 1134.5025855, 1135.47741, 1135.4959685, 1140.669083, 1141.0337909999998, 1144.445449, 1144.463181, 1147.6750195, 1148.1065795, 1148.527519, 1148.973043, 1150.375353, 1150.3836145, 1150.4210065, 1150.4479155, 1154.572073, 1154.978459, 1157.4062035, 1157.4516775, 1158.547172, 1158.572106, 1159.9492725, 1160.3517365, 1160.480417, 1160.488735, 1164.5694760000001, 1164.5959775000001, 1170.5906375, 1171.0430645000001, 1174.487846, 1174.5263995, 1180.5851295, 1180.9848670000001, 1184.3535495, 1184.3554395, 1185.3950785, 1185.4084285, 1186.6038174999999, 1186.741102, 1187.0846315, 1188.5775745, 1189.0296045, 1190.5918255000001, 1191.025733, 1197.3794765, 1197.394797, 1199.660254, 1200.0531805, 1200.671135, 1201.036536, 1201.5685595, 1201.6661330000002, 1213.46271, 1213.488347, 1214.6616290000002, 1215.05955, 1215.579892, 1215.631741, 1216.0104995, 1220.4059255, 1220.425005, 1224.6276855, 1225.0053010000001, 1227.471863, 1227.5011985, 1233.5157450000002, 1233.9555315, 1235.471821, 1235.64482, 1236.080215, 1236.6696889999998, 1237.0509545, 1237.5915060000002, 1238.039241, 1239.62644, 1240.034648, 1241.464326, 1241.5025879999998, 1248.5735915, 1248.9966785, 1249.5022745, 1249.9482524999999, 1259.5205495, 1259.5692075, 1259.9663795, 1260.356519, 1260.363725, 1260.406446, 1260.422137, 1262.5679575, 1262.684855, 1265.35526, 1265.3667894999999, 1266.5147115, 1266.537792, 1278.623828, 1279.0305835, 1279.3796790000001, 1279.383972, 1279.5561255, 1279.6001335, 1281.6289565, 1281.6966255, 1282.570962, 1282.999252, 1283.4461535, 1283.509346, 1285.5637055, 1285.9938809999999, 1287.522982, 1287.9615065, 1290.4307184999998, 1290.4393615, 1290.517402, 1290.9761795, 1293.6062729999999, 1294.0149455, 1295.5451509999998, 1295.6007205, 1300.403565, 1300.4465765, 1303.460959, 1303.4891705, 1312.4479205, 1312.4997185000002, 1313.571725, 1313.9657865, 1316.5596569999998, 1316.9836785, 1319.5266935, 1319.95683, 1325.6514065000001, 1326.0540784999998, 1328.6326199999999, 1329.0477274999998, 1329.6239775, 1329.7381894999999, 1330.523308, 1330.657148, 1333.461352, 1333.922627, 1335.4804450000001, 1335.936556, 1339.6263720000002, 1340.0068635, 1344.3713465, 1344.4066874999999, 1348.503403, 1348.9352294999999, 1349.5362975, 1349.5748705, 1351.6967125, 1352.0298675, 1355.376793, 1355.4738965, 1360.517487, 1360.5606635, 1364.605524, 1364.9801915, 1368.625518, 1369.042453, 1369.5243945, 1369.9820235, 1372.9567459999998, 1373.3647325, 1373.7901000000002, 1374.0883370000001, 1375.643692, 1376.0568560000002, 1377.5634555, 1377.6091474999998, 1380.5537614999998, 1380.983657, 1385.4684590000002, 1385.5178385, 1388.4504635, 1388.5606195, 1390.3827765, 1390.394635, 1392.587487, 1393.040907, 1396.5484105, 1396.6343419999998, 1402.6222659999999, 1403.0793305, 1403.580188, 1404.0045665, 1409.5112709999999, 1410.002373, 1412.404122, 1412.4624055, 1412.568212, 1412.9911444999998, 1417.5651845, 1417.985958, 1418.645169, 1419.0217995, 1421.539831, 1421.6102584999999, 1422.9093185, 1423.382283, 1423.521975, 1423.5494355, 1423.567992, 1425.5693419999998, 1426.069703, 1426.529601, 1428.4784015, 1428.9700015, 1440.362273, 1440.376745, 1442.5979779999998, 1443.0663095, 1444.5169500000002, 1444.6117975000002, 1446.984246, 1447.8648210000001, 1448.987035, 1449.8807935, 1458.4490999999998, 1458.4802685, 1462.4164959999998, 1462.4216635, 1471.450899, 1471.5343644999998, 1477.468351, 1477.5107395, 1478.954255, 1479.5273670000001, 1479.987586, 1481.4648095, 1482.3616835, 1485.6044900000002, 1486.0177375, 1490.4644094999999, 1490.5091619999998, 1491.433634, 1491.5012069999998, 1494.450659, 1494.476101, 1494.5584735, 1494.9924365000002, 1497.4589225, 1497.9550895, 1500.3915525, 1500.4625230000001, 1508.548118, 1508.56674, 1509.620652, 1510.5487545, 1513.5249330000001, 1513.6008285, 1515.5417155, 1515.5784845, 1518.4786330000002, 1518.5485755, 1520.6096125, 1521.005054, 1529.5529895, 1530.0621225, 1532.5328325, 1532.9898035, 1535.4965275, 1535.564418, 1538.639202, 1539.002868, 1544.9666590000002, 1545.3985265000001, 1546.5703174999999, 1547.133865, 1550.512442, 1550.5597050000001, 1552.9830984999999, 1553.4985465, 1554.5857205000002, 1555.062286, 1555.586421, 1556.0226014999998, 1556.6665045, 1557.0602494999998, 1572.410392, 1572.506773, 1575.511448, 1575.5578194999998, 1576.031145, 1597.3611555, 1597.3694839999998, 1605.558393, 1605.9955610000002, 1612.497395, 1612.5554725000002, 1618.6481125, 1619.053052, 1643.584886, 1643.5963305, 1644.4444965, 1644.4862054999999, 1665.493581, 1665.994247, 1669.390671, 1669.412731, 1673.5384785, 1673.979854, 1688.5079535, 1688.960429, 1694.5369719999999, 1694.993716, 1695.4266604999998, 1695.5122904999998, 1698.6268625, 1699.0607245, 1701.58405, 1701.67326, 1722.588466, 1723.034083, 1743.6077115, 1744.706105, 1747.0251115, 1747.6887345, 1750.6149745, 1751.0332899999999, 1752.595495, 1753.0364734999998, 1767.935477, 1768.9463465, 1771.5675310000001, 1772.021256, 1773.5236985, 1774.0279679999999, 1779.442757, 1779.5404085, 1812.4008515, 1812.492378, 1815.0240925, 1815.492301, 1851.438677, 1851.9119925, 1877.6840120000002, 1878.055342, 1878.4115875, 1878.464692, 1924.0180985, 1925.0659329999999, 1926.1001219999998, 1927.533331, 1957.1530265000001, 1958.080565, 1961.598912, 1962.0154674999999])
labels = array([0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 4.0, 2.0, 1.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 0.0, 4.0, 1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 3.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 4.0, 2.0])
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

numthresholds = 920



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



