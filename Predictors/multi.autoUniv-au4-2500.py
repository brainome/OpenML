#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target Class autoUniv-au4-2500.csv -o autoUniv-au4-2500.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:01:23.42. Finished on: Sep-04-2020 10:20:57.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         3-way classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 46.92%
Overall Model accuracy:              78.32% (1958/2500 correct)
Overall Improvement over best guess: 31.40% (of possible 53.08%)
Model capacity (MEC):                881 bits
Generalization ratio:                2.22 bits/bit
Model efficiency:                    0.03%/parameter
Confusion Matrix:
 [37.32% 8.12% 1.48%]
 [8.24% 35.88% 1.12%]
 [1.32% 1.40% 5.12%]
Overfitting:                         No
Note: Labels have been remapped to 'class2'=0, 'class1'=1, 'class3'=2.
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
TRAINFILE = "autoUniv-au4-2500.csv"


#Number of attributes
num_attr = 100
n_classes = 3


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
    clean.mapping={'class2': 0, 'class1': 1, 'class3': 2}

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
energy_thresholds = array([79339532283.656, 80975513405.87799, 81080265381.861, 81328048159.2795, 81824133270.1575, 82215426586.2095, 82508997429.057, 83223899235.09149, 83595543584.099, 84094041733.0615, 84460811630.7525, 84602043609.04349, 84690250966.4325, 84732753805.884, 84885446194.681, 85044718472.7795, 85162525610.9995, 85358188821.04, 85558947184.92252, 85561553507.62302, 85625236057.20851, 85703614120.876, 86403160353.306, 86537350410.54102, 86659931362.531, 87028048822.207, 87276725081.688, 87423062241.245, 87617605131.4905, 87724348670.3005, 87773711183.9455, 87958828267.8525, 88127741064.138, 88166640782.092, 88311712653.72452, 88345183676.518, 88418110334.959, 88483795204.21799, 88539187998.2815, 88671269726.405, 88733449603.577, 88765396417.7195, 88794329940.323, 88840771067.79901, 88919331406.702, 89046471917.448, 89278031472.272, 89300519358.47351, 89325369457.631, 89400273939.895, 89495467402.106, 89715219692.87999, 89756994625.621, 89784661557.0585, 89900228447.638, 89990758317.9345, 90045994409.2945, 90064180555.4925, 90068258427.79001, 90121108343.15302, 90198926549.668, 90241273970.921, 90258570780.544, 90268696576.7955, 90340536513.168, 90370150696.5455, 90469933749.63199, 90569647960.0745, 90655010485.496, 90677517756.9075, 90685698103.4965, 90707416259.1225, 90775668446.55649, 90824997564.10501, 90829015149.844, 90842445324.23499, 90859214996.4395, 90907056737.66751, 90973299199.4035, 91054329928.3625, 91120002752.97949, 91231220536.952, 91406237020.5785, 91422442769.79001, 91583197320.778, 91675578951.3215, 91744773886.6705, 91765049467.44049, 91806812152.59349, 91828326115.50699, 91835875537.65302, 91872343753.7125, 91890373533.5665, 91906717169.888, 91916254953.047, 91925302228.0605, 91954394037.069, 92139187256.09601, 92206035141.996, 92258850361.83002, 92269876260.2185, 92512086640.233, 92604009918.52151, 92636428998.267, 92660525161.2185, 92691047572.617, 92711491307.423, 92759335095.22202, 92792451808.464, 92853923142.139, 92924200120.8735, 92928243418.224, 93020036652.8645, 93038830806.0455, 93061089210.193, 93075558496.36101, 93076828791.397, 93079404727.248, 93096362333.651, 93315791713.45349, 93466679229.315, 93493571068.7115, 93496756771.81851, 93508878184.25949, 93530548199.748, 93562009711.6365, 93585603854.445, 93606961865.461, 93635902157.37451, 93662355961.46701, 93711688702.27501, 93740758012.848, 93757129090.5625, 93783433189.3495, 93791507180.7735, 93797587512.9975, 93809384640.2555, 93870159929.8215, 93954451656.7025, 94044660588.992, 94087342584.54651, 94104270426.50299, 94162819691.867, 94212181321.5105, 94222319751.108, 94237029466.6965, 94268309385.739, 94272967141.689, 94294632039.5875, 94360667621.21599, 94386840415.01099, 94406744921.121, 94421253185.8995, 94452027119.1805, 94478960662.17749, 94501387286.69849, 94521775434.5625, 94531899566.99149, 94553286768.74901, 94575549168.886, 94614750572.6365, 94649513146.4725, 94667516708.708, 94683795121.68451, 94701074401.62851, 94754209855.95801, 94814625596.8575, 94847764428.695, 94869445525.49551, 94930240764.8775, 94972787000.7455, 94994160658.7785, 95030909893.14499, 95047105579.479, 95085385698.3735, 95184769581.73851, 95196069963.62201, 95212269012.05649, 95236295343.26651, 95261857519.2685, 95279371798.10751, 95281949886.956, 95353804848.358, 95376093391.3435, 95388251973.147, 95399160192.5895, 95402346534.0045, 95436964100.20001, 95457023057.104, 95463118502.401, 95486095709.25299, 95515154606.3695, 95515181617.0815, 95541511241.388, 95580699998.845, 95676228175.1995, 95681550987.92151, 95697772902.239, 95726665978.381, 95748233380.8185, 95774382253.578, 95798066712.546, 95808194109.0545, 95875972736.964, 95904893945.8665, 95983268061.67, 96029265563.81299, 96039368174.4065, 96044524223.003, 96092784851.1745, 96114165214.34549, 96127040406.3505, 96148160463.74951, 96184447925.19751, 96222791451.6635, 96225150515.7765, 96246504329.5345, 96288485811.6365, 96389145198.4445, 96391723794.617, 96396732435.77, 96412169232.814, 96426168058.7995, 96440490717.116, 96490538772.4235, 96536961957.5155, 96543630730.275, 96562714566.04199, 96574274012.648, 96583007978.468, 96650930421.3295, 96676835129.165, 96695242313.5235, 96716611870.8815, 96733740970.3205, 96737803234.392, 96759182646.819, 96795031130.4765, 96811249814.5475, 96835162374.7845, 96870845036.594, 96883570660.5175, 96887047687.153, 96895684740.52899, 96904731817.11801, 96936946102.916, 96963108932.1925, 97064621776.87799, 97111281569.7365, 97124875396.8515, 97137581778.6375, 97146238777.638, 97158392780.63, 97183985745.6185, 97224645421.9655, 97238272573.994, 97290323616.4515, 97294404137.651, 97300451288.3145, 97311663207.28601, 97315734030.3715, 97323293024.162, 97335440123.127, 97379780253.347, 97396150500.7015, 97427866153.5195, 97452866493.6295, 97469090549.1965, 97507167982.46649, 97521866730.1105, 97534585802.1865, 97551513454.611, 97570652414.5185, 97672516972.2315, 97705635166.13, 97732116954.0295, 97745730344.2285, 97783264581.37851, 97798345443.752, 97819127706.351, 97841409775.5115, 97851562002.8925, 97870498846.20749, 97955408824.84201, 97968830045.9935, 97978985613.1615, 97999383631.6005, 98021072672.69699, 98036342101.824, 98050684640.097, 98063613929.441, 98074845479.323, 98076277477.874, 98087839151.5425, 98100035065.359, 98108896857.4745, 98132959176.569, 98143693925.742, 98161315691.5495, 98173516468.9335, 98222434567.849, 98265400343.0935, 98300526970.636, 98301922526.36801, 98441865640.53099, 98546401707.75, 98549351960.869, 98552811775.005, 98553966655.4315, 98568845692.1795, 98576739290.35251, 98578825271.22299, 98601246050.45898, 98638448143.4365, 98642518402.255, 98654652574.50351, 98660712638.4855, 98683587117.21149, 98696421968.824, 98716834115.1005, 98736527848.9155, 98745190850.28, 98752745587.1665, 98768954654.732, 98780186123.65851, 98786266372.4195, 98787690251.6875, 98795831723.9875, 98814609726.604, 98836890404.845, 98874048517.5515, 98910648991.8465, 98967193998.1275, 98970678666.8385, 98972004278.3535, 98988226806.9775, 99007329463.4685, 99011371965.51599, 99030301670.46051, 99034953498.29752, 99036792311.96051, 99047808640.534, 99054454776.541, 99101939208.822, 99103081430.6615, 99109862292.927, 99116644614.55899, 99121411940.396, 99132116304.88, 99143544926.974, 99149046819.27951, 99149068504.28448, 99155141082.2255, 99164684522.313, 99175909192.0565, 99225242734.4805, 99271986961.1015, 99308637625.2405, 99337685213.0515, 99351539150.905, 99362980202.703, 99380066332.90399, 99387989640.9085, 99414339424.335, 99437323299.37552, 99513932239.66199, 99525152485.2355, 99567613340.8785, 99583291781.96051, 99618283504.2465, 99644745244.1915, 99655575747.8865, 99691482073.22601, 99717827325.9725, 99718556279.81699, 99740630381.08101, 99814269823.4365, 99841319441.557, 99859593873.149, 99876533445.711, 99889950281.725, 99906124164.76501, 99911635420.6055, 99936509625.52249, 99989174751.5155, 99997584671.32, 100012886049.8215, 100054832526.06801, 100058313695.02301, 100078561289.817, 100081171642.443, 100087263362.60999, 100096818256.86299, 100105462572.6585, 100112106772.795, 100119642719.8445, 100135974647.02449, 100149396281.4975, 100179256453.712, 100198954137.56601, 100235553050.22699, 100267227794.5765, 100272390613.71149, 100284383796.0965, 100301875531.8175, 100369747645.42151, 100393351224.47299, 100395928162.833, 100413633646.8075, 100422286507.91049, 100443940736.171, 100483759869.34549, 100483902163.28049, 100491436316.7285, 100550938184.6955, 100585150943.567, 100605975512.0975, 100631146048.30249, 100648675200.9725, 100660073988.66751, 100665542210.746, 100672173230.64099, 100694446686.90051, 100729736475.239, 100745167819.629, 100759860405.433, 100776090741.248, 100787635710.2205, 100797905110.67401, 100816930406.3645, 100830541393.3895, 100855386891.943, 100928790728.6465, 100934861869.85, 100934879261.6375, 100934895919.589, 100998536254.7005, 101010372569.34201, 101023403227.377, 101069558730.8745, 101119139028.86151, 101134788521.943, 101171791878.6145, 101191499279.228, 101373175275.07501, 101446401076.0455, 101485980796.4305, 101502193642.0555, 101511747957.172, 101519279862.82251, 101528842035.31201, 101537495886.3905, 101558482074.0725, 101563280238.809, 101570286176.27351, 101575748136.645, 101602253116.6655, 101624525756.60449, 101629187484.97, 101667151869.16049, 101681263908.0135, 101706493355.99149, 101738874704.6075, 101771844001.827, 101781971885.9845, 101823917189.16751, 101830565733.789, 101836646764.5745, 101846207769.8875, 101852243253.666, 101852258984.7415, 101868459345.932, 101895520205.803, 101920344722.0025, 101940774150.1105, 101946859453.6265, 101960499339.1395, 101963086928.1965, 101967276240.376, 101983451704.358, 102081947567.3195, 102157463524.2025, 102196845737.99799, 102202024740.8665, 102216206958.459, 102264201010.913, 102282963442.77249, 102316114550.984, 102328968185.0385, 102330506210.9735, 102340781729.706, 102356246426.3345, 102366379488.1815, 102377213861.434, 102383264191.82751, 102391204346.14151, 102426942101.50299, 102468928330.6935, 102545166442.6195, 102551222709.8215, 102571649085.4235, 102589296725.4615, 102639913949.1745, 102662922228.21051, 102670455464.2095, 102690636403.4055, 102739818076.4985, 102782372023.8895, 102817490960.6695, 102851214771.4495, 102865338337.38751, 102865363554.0615, 102868843155.28351, 102890396716.01251, 102908483273.79001, 103040414306.193, 103089564037.3835, 103103740078.6355, 103123989163.328, 103131508782.52951, 103137080500.39099, 103158031636.8215, 103177712703.01, 103216575608.2225, 103255190700.522, 103274118271.9545, 103278767503.03601, 103286313468.27, 103341413874.0235, 103347811203.0115, 103376885774.964, 103387003357.906, 103403324617.61551, 103411280922.2055, 103415036448.64499, 103435256313.6225, 103480672569.5975, 103492145152.793, 103499524416.9975, 103511961460.0375, 103534242284.8015, 103547560666.55501, 103588935518.6635, 103610295670.1195, 103618984092.18051, 103620465842.3045, 103623941578.538, 103636795141.6885, 103651142713.57199, 103662566883.36499, 103672541395.24399, 103705476810.9865, 103729999428.58249, 103793607759.857, 103819915599.6135, 103832815699.518, 103850327026.347, 103865756146.77899, 103913221629.761, 103931290659.075, 103975105678.64899, 104017656459.45651, 104023732407.183, 104029810179.77798, 104066423383.501, 104075040415.3675, 104080969492.0905, 104085194344.92499, 104086491785.30649, 104093712721.844, 104108768046.70251, 104117762539.3945, 104119179760.137, 104121747894.4525, 104140028208.579, 104155456075.711, 104182508198.69899, 104203373584.05399, 104228954697.62201, 104293328234.67801, 104306043628.0015, 104323563886.161, 104353788727.66501, 104365401012.539, 104371451713.235, 104428101440.9365, 104437687919.0525, 104448724479.94151, 104452798900.0115, 104468990204.863, 104491814511.423, 104537825230.02951, 104560829639.7445, 104585851479.6855, 104611599491.41449, 104624282666.29349, 104645278386.559, 104674333543.064, 104689115762.28351, 104696347817.0605, 104711361640.17151, 104742510433.653, 104791840273.5215, 104820756561.55951, 104822167191.55899, 104841155514.4055, 104864168318.78351, 104881794251.823, 104907791629.375, 104945689590.288, 104961889034.8135, 104972041300.3205, 104992486371.9025, 105015455378.7095, 105030709763.5025, 105044345754.43799, 105057972217.491, 105060566121.38199, 105072739849.095, 105113532492.8, 105119626936.3605, 105138399900.8885, 105162536538.253, 105177984234.1525, 105213887727.9395, 105218565974.8945, 105280728418.291, 105296937916.0115, 105344780494.192, 105375318273.32849, 105405874187.396, 105445082887.345, 105462376745.02951, 105472509660.31851, 105476023047.1185, 105482106033.8935, 105493699270.383, 105518676307.28299, 105550320919.5375, 105575777210.45099, 105597482447.7585, 105608775308.913, 105621241153.6575, 105642353982.317, 105655077311.22801, 105658578592.33151, 105688807785.42648, 105800926319.5685, 105855149918.7985, 105866575294.699, 105909664282.413, 105992671416.3035, 105996187299.2845, 106003710636.259, 106010367717.9435, 106026419268.40698, 106042600856.63348, 106073825589.01, 106125990587.76001, 106204212621.7005, 106214341426.5375, 106238259033.81601, 106284788831.3245, 106301378377.82501, 106339109912.3085, 106375723837.3565, 106383281831.8905, 106385848227.44499, 106389904825.90948, 106487675500.633, 106531791907.66599, 106688294936.25, 106717378662.0775, 106729541195.7295, 106809396263.6525, 106813434926.812, 106821381904.9735, 106827461092.62799, 106852105934.06151, 106857393664.901, 106894994275.745, 106905102342.465, 106916149886.202, 106928269912.804, 107006971440.27051, 107011159753.024, 107037481291.676, 107059923590.7015, 107072621253.80301, 107076112821.0705, 107079032034.6115, 107089998999.664, 107098394622.9675, 107155683189.244, 107210112227.357, 107236468519.9985, 107247114001.7485, 107259972333.65051, 107290183747.09201, 107305823780.0815, 107322039845.37201, 107335638150.66399, 107358617784.3105, 107371488088.996, 107387692280.506, 107435388753.0105, 107452895926.9195, 107471886825.413, 107474461115.41, 107480536811.5055, 107489506503.384, 107501066885.607, 107544764404.87799, 107587007469.6095, 107594385606.20401, 107624654726.42749, 107650968540.66699, 107674676500.39749, 107700461380.944, 107715230887.29599, 107784067564.456, 107889934855.656, 107937759538.3275, 107974409687.961, 107990584855.5135, 108050181732.1335, 108069691856.2065, 108096171269.8865, 108119746734.8185, 108141299849.91, 108170835615.93051, 108231843011.426, 108253002674.57999, 108284651640.9065, 108366218991.47449, 108374365522.1065, 108428319930.3465, 108462181338.014, 108474461163.25, 108482675407.51901, 108483830896.65952, 108542730865.1035, 108562830834.21701, 108585664038.022, 108601875763.993, 108661458961.262, 108679137792.3175, 108701811556.863, 108723314614.9275, 108818855809.51498, 108878323361.0135, 108904466551.35849, 108988949843.74, 109002602656.128, 109022985436.923, 109033131983.45749, 109070256832.2025, 109086463692.861, 109123516550.4225, 109142456516.54199, 109171503490.5885, 109201719431.975, 109228218492.811, 109245839987.1715, 109252477479.2695, 109278243980.768, 109350457612.5835, 109383587858.974, 109402368351.0865, 109402387555.56, 109466417842.19699, 109521831033.7355, 109541945581.22449, 109613104918.55249, 109645884899.7365, 109824466184.4965, 109834015190.116, 109879278937.371, 109941495342.876, 110019146536.5305, 110068459325.63501, 110107666433.4265, 110113750403.90251, 110170442778.1055, 110275566481.9685, 110341118185.732, 110374236118.8865, 110532338392.4925, 110581640834.718, 110637050014.0505, 110666036026.95499, 110734285633.0965, 110770756123.4725, 110805900674.65051, 110824842439.8945, 110878362549.3705, 110891071170.435, 110897007250.22699, 110932877321.8695, 111184112084.8775, 111187633167.72, 111414427366.25, 111430932529.537, 111453197782.8595, 111561835716.0475, 111569178993.991, 111583182340.0565, 111617626439.91751, 111682602498.011, 111738129768.271, 111794866479.9025, 111840146536.556, 112103598038.2015, 112129151414.75, 112145374480.87149, 112170213775.734, 112252097153.9585, 112381662863.26599, 112398618335.599, 112428987858.104, 112518283747.01349, 112626469157.47101, 112675794556.1605, 112772949284.137, 112863538623.437, 112904796187.424, 112934988885.553, 112947740494.86899, 112968156180.317, 113013440335.0065, 113110620216.90051, 113218080047.935, 113292822195.166, 113296908063.9635, 113394810170.9255, 113612954667.9245, 113678632856.3875, 113782097406.293, 113905745115.0275, 114119660062.2915, 114429051843.84549, 114630411582.88199, 114682685992.614, 114792574936.918, 114870235300.0675, 114901193708.738, 114951307842.4165, 115004531244.37949, 115128109177.02449, 115174558484.8095, 115225941732.7545, 115581500834.6865, 115746246052.934, 115921318007.5595, 115952814615.88, 115984464302.317, 116208456680.745, 116310796175.228, 116569753396.14151, 116750681171.19, 116999181352.2615, 117317266743.18149, 117352406897.969, 117519340297.7915, 117774666913.9575, 117915834767.2625, 118001616553.4865, 118082606707.9855, 118304210427.807, 118474294383.539, 118653411584.956, 119363899979.449, 120184256877.01901, 120482660113.504, 120880284612.944, 121028998806.70001, 121188225974.974, 121808147783.344, 122523762115.253, 122832473361.149, 123057261196.295])
labels = array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0])
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
        outputs[defaultindys] = 1.0
        return outputs
    return thresh_search(energys)

numthresholds = 881



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



