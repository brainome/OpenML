#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593744/phpiubDlf -o Predictors/autoUniv-au4-2500_QC.py -target Class -stopat 73.0 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:05.27. Finished on: May-22-2020 20:42:39.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        3-way classifier
Best-guess accuracy:                46.94%
Model accuracy:                     78.16% (1954/2500 correct)
Improvement over best guess:        31.22% (of possible 53.06%)
Model capacity (MEC):               851 bits
Generalization ratio:               2.29 bits/bit
Confusion Matrix:
 [38.16% 7.52% 1.24%]
 [8.84% 35.32% 1.08%]
 [1.72% 1.44% 4.68%]

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
TRAINFILE = "phpiubDlf.csv"


#Number of attributes
num_attr = 100
n_classes = 3


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
energy_thresholds = array([79218472887.2005, 80807994631.096, 81100701943.5755, 81388360881.043, 81714321589.414, 82045302183.7025, 82507519029.126, 83263117778.9465, 83861701170.0885, 84094041733.0615, 84521593452.045, 84643825155.2205, 84690250966.4325, 84732753805.884, 84885446194.681, 85113756645.785, 85541846665.979, 85561553507.62302, 85583475915.869, 85606443761.67099, 85644720308.30151, 85703614120.876, 86161619718.718, 86272979362.70801, 86403883534.8465, 86581461696.6275, 86638408780.4935, 87037593318.57199, 87269186533.6365, 87423062241.245, 87970261359.69351, 88053650638.5735, 88174779405.9065, 88177171805.1445, 88255578567.71051, 88477737480.993, 88483795204.21799, 88493908205.7755, 88562622426.932, 88633491397.28299, 88721335398.1795, 88738934147.73349, 88815329578.435, 88919331406.702, 89046471917.448, 89285034654.41849, 89300519358.47351, 89398284400.4095, 89730328301.62051, 89752930607.341, 89780597538.7785, 89890665451.11499, 90006808725.7215, 90049471756.265, 90064180555.4925, 90192826518.324, 90241273970.921, 90267287731.9545, 90371091999.1165, 90598258799.3265, 90637908267.52551, 90656110766.9385, 90680107373.949, 90684775509.0615, 90733567043.426, 90786871868.7445, 90824997564.10501, 90837693258.02301, 90851123432.414, 90859214996.4395, 90907056737.66751, 90986946725.732, 91054329928.3625, 91120002752.97949, 91184684385.7865, 91378100190.74649, 91422442293.3605, 91487954943.215, 91557004162.822, 91623525996.486, 91678144561.12001, 91718271531.67001, 91781255366.315, 91806812152.59349, 91828326115.50699, 91871009599.517, 91916254953.047, 91923827769.7875, 91940031161.901, 91982537123.3265, 92018407573.4195, 92151390000.343, 92321042331.4615, 92411454859.055, 92453202870.535, 92526130867.89252, 92582852040.98651, 92636428998.267, 92664554857.9615, 92691047572.617, 92711491307.423, 92853923142.139, 92974322375.888, 93061089210.193, 93071360302.23999, 93076828791.397, 93079404727.248, 93101522255.10999, 93181416492.81601, 93195018181.797, 93201092696.8935, 93203702301.40298, 93218756973.89899, 93282077192.6055, 93292208028.551, 93374680765.142, 93426213395.1495, 93469864932.422, 93505382173.1555, 93544799198.5195, 93567170918.401, 93639927628.5965, 93655389875.626, 93681888375.59799, 93713186502.84851, 93731958347.585, 93740758012.848, 93757127174.31, 93789513521.5735, 93809384640.2555, 93830981127.4245, 93952960752.126, 93957625237.95349, 94063786458.7905, 94087342584.54651, 94104270426.50299, 94172996118.72949, 94218261702.21649, 94266250682.0505, 94272967141.689, 94294632039.5875, 94378181203.172, 94388913640.844, 94406744921.121, 94433237371.98401, 94464763625.2215, 94489228770.65201, 94553286768.74901, 94578827493.2455, 94593192703.72101, 94623027011.9765, 94649513146.4725, 94671097572.88199, 94680071572.5065, 94691875440.3535, 94754765014.525, 94805968543.0385, 94845750682.4875, 94851829619.502, 94865988585.06299, 94893728279.34349, 94915298243.024, 94920691175.951, 94986603497.274, 95029399674.467, 95045610031.84549, 95059796401.2605, 95069954071.886, 95092960598.366, 95128077592.4745, 95166544723.637, 95192183654.375, 95211123952.92651, 95281949886.956, 95359883393.9595, 95381253161.5065, 95389646207.7715, 95399160192.5895, 95402346534.0045, 95452968272.1135, 95486095709.25299, 95521086100.1505, 95533361885.2015, 95568357376.52249, 95598015796.8685, 95707882557.3595, 95748242073.6925, 95759684533.309, 95808194109.0545, 95863123876.5115, 95904893945.8665, 95939096585.7945, 95959514660.056, 95983268061.67, 96044524223.003, 96092784851.1745, 96106823894.625, 96112014744.0205, 96119356063.741, 96179812526.5485, 96210983613.84052, 96217623830.647, 96238443316.6795, 96251879170.25049, 96260133790.435, 96263601238.909, 96263622786.22049, 96296020883.90651, 96319964978.36101, 96339671892.9425, 96372767092.6685, 96391723794.617, 96403893989.769, 96430366195.74051, 96466972727.4265, 96552479644.0585, 96624790558.81549, 96659334530.3295, 96676835138.1095, 96695242313.5235, 96711960833.1045, 96716941856.681, 96721592894.458, 96721621565.25749, 96732815666.47049, 96759182646.819, 96792444977.9715, 96805150477.245, 96807736629.75, 96811210953.1355, 96835123513.3725, 96864055514.15201, 96892207713.8935, 96904731817.11801, 97085623826.261, 97111253601.2995, 97115351877.6405, 97143662048.0565, 97158408336.74301, 97203467940.88501, 97211022908.4755, 97285342112.14801, 97292890181.0155, 97298970457.9915, 97305027065.2895, 97311663207.28601, 97314246988.2045, 97323293024.162, 97335440123.127, 97377377513.25198, 97379951584.39499, 97396150500.7015, 97427866153.5195, 97452884940.514, 97469090549.1965, 97499618502.109, 97521866730.1105, 97534585802.1865, 97551513454.611, 97565719484.8125, 97567154696.2, 97619450866.5935, 97653732633.6495, 97672516972.2315, 97721057065.3295, 97726044348.1265, 97742233214.5305, 97790821170.5545, 97870498846.20749, 97893470256.259, 97925199847.93, 97968826713.4375, 97978992220.352, 98006056248.432, 98035782627.8695, 98050684640.097, 98100035065.359, 98108896857.4745, 98132959176.569, 98166875991.0285, 98203649043.4725, 98206263647.18149, 98215788058.558, 98248787257.16501, 98450491264.698, 98476070996.474, 98522847811.1055, 98543859477.698, 98550480880.241, 98570270601.7875, 98611247306.26651, 98616619398.085, 98622808172.8095, 98651149892.4935, 98660698616.2445, 98670859701.64949, 98683587117.21149, 98693869514.39099, 98704119134.619, 98736529640.0065, 98759739967.4655, 98794938269.757, 98819261079.729, 98836890404.845, 98878257066.32399, 98927583658.95099, 98965924089.7485, 98969408758.4595, 98972004278.3535, 99018855422.8595, 99036792311.96051, 99047099166.9325, 99101939208.822, 99103081430.6615, 99104499196.91049, 99111281518.5425, 99121411940.396, 99132116304.88, 99143544926.974, 99152188667.891, 99158280598.9845, 99164684522.313, 99297053575.51099, 99321510539.543, 99337685213.0515, 99351539150.905, 99371633821.8915, 99407113215.4625, 99437323299.37552, 99458630865.1335, 99472295097.219, 99476340639.457, 99487548229.2045, 99506350354.72299, 99531193330.33301, 99571117731.455, 99583291781.96051, 99661661569.968, 99672687589.8715, 99682833539.09, 99704364897.466, 99738046275.38, 99761743508.2495, 99767846388.5815, 99771905704.5795, 99786089788.0415, 99806360851.77399, 99824538163.73999, 99836673953.034, 99841319441.557, 99853514820.41751, 99876533445.711, 99889950281.725, 99906124164.76501, 99918277329.1035, 99989174751.5155, 99995228541.203, 100012886049.8215, 100054832526.06801, 100058313695.02301, 100062973480.567, 100081171642.443, 100089231860.781, 100101366263.56349, 100112106772.795, 100118730532.1745, 100122201787.617, 100151500137.7845, 100179256453.712, 100198954137.56601, 100235553050.22699, 100267227794.5765, 100272390613.71149, 100273536572.70999, 100336877328.768, 100362946079.9995, 100370187759.8725, 100376250595.988, 100380469041.2085, 100389150297.8885, 100395928162.833, 100435882379.69601, 100483759869.34549, 100487959908.368, 100495494061.81601, 100501563907.1425, 100505330482.653, 100537035808.705, 100541692468.82199, 100550938184.6955, 100583702158.475, 100610523000.5345, 100631146048.30249, 100650680008.34799, 100660073988.66751, 100671597426.28299, 100680315735.117, 100696533975.83951, 100771408315.2355, 100792508323.2175, 100801406233.9245, 100816930406.3645, 100828130460.4175, 100852975958.971, 100892188587.255, 100915180008.037, 100928790728.6465, 100934861869.85, 100934879261.6375, 100934895725.295, 100949224159.6565, 100966135927.323, 100978856588.04651, 100990587697.53049, 100998536254.7005, 101002986479.4, 101009083392.90451, 101023403227.377, 101069558730.8745, 101089084565.7915, 101134788521.943, 101185413111.832, 101191501597.16101, 101373175275.07501, 101386789887.82199, 101393558761.016, 101460019450.759, 101487461625.7615, 101503674471.3865, 101513117036.84601, 101522453563.06, 101537504962.46951, 101555029386.41351, 101582670607.6625, 101601427016.164, 101612960073.963, 101625693095.0335, 101644145427.34, 101670609638.699, 101684721677.552, 101709971940.6175, 101738898837.8075, 101744028384.122, 101787317604.25299, 101836646764.5745, 101846207769.8875, 101852243253.666, 101852258984.7415, 101862387891.135, 101894775905.3345, 101927161637.2215, 101935789830.42749, 101948022405.396, 102089194963.815, 102154913168.477, 102174558479.2075, 102187280718.47949, 102194629366.805, 102206065661.846, 102216206958.459, 102231668676.464, 102246012814.5515, 102265687743.6915, 102280392175.2325, 102282963442.77249, 102316114550.984, 102328968185.0385, 102329743713.791, 102340781729.706, 102366368890.58101, 102377232138.823, 102385865924.754, 102411298775.731, 102468928330.6935, 102571649085.4235, 102606800972.80049, 102617420537.9945, 102619476820.34851, 102639913949.1745, 102662922228.21051, 102670455464.2095, 102685160639.26001, 102701870664.6865, 102711996630.0775, 102726930608.1145, 102739842504.87201, 102795977315.41049, 102822157324.24548, 102851216761.38449, 102866241768.42499, 102886519751.043, 102908483273.79001, 103049063193.865, 103090134678.6355, 103105743637.78549, 103123455859.4075, 103158031636.8215, 103189129083.479, 103207925058.2755, 103255447230.6165, 103274118271.9545, 103278767503.03601, 103286313468.27, 103341413874.0235, 103350367942.32599, 103366724150.7485, 103378877354.6145, 103388994937.5565, 103403324617.61551, 103441387308.5705, 103450352396.70349, 103456430621.6955, 103499518873.736, 103509629258.72049, 103522332344.78, 103537981484.0975, 103583263881.12, 103618984092.18051, 103620465842.3045, 103623786352.3375, 103640143463.073, 103663310699.413, 103672541395.24399, 103676584504.9465, 103682053791.13751, 103685529774.469, 103689621604.853, 103708378872.327, 103732577155.4845, 103750785746.17, 103762342470.17749, 103779245824.62549, 103855831518.975, 103871260639.40701, 103880507407.5705, 103894145900.889, 103969033541.285, 103992011009.976, 104011041245.6815, 104017656459.45651, 104023732407.183, 104029810179.77798, 104039924180.89499, 104056112983.617, 104066423383.501, 104075070984.9425, 104082297502.047, 104089853728.9505, 104117762539.3945, 104119179760.137, 104121747894.4525, 104140015398.8855, 104143333285.886, 104153480311.6195, 104165628643.625, 104182533740.87949, 104203373584.05399, 104225659702.802, 104473070684.1335, 104495894990.69351, 104537825230.02951, 104608105033.001, 104620788207.88, 104643780815.599, 104681565597.841, 104712109440.54851, 104734423225.724, 104748452470.8065, 104749134905.1705, 104749812108.67401, 104754040963.3085, 104791840273.5215, 104807133231.006, 104822167191.55899, 104837075876.0145, 104907791629.375, 104945703533.3115, 104961889034.8135, 105015455378.7095, 105030709763.5025, 105036787206.3785, 105040308775.405, 105047867323.46451, 105057972217.491, 105060561039.186, 105072739849.095, 105101943395.81349, 105106568239.741, 105113532492.8, 105119606740.2755, 105177984234.1525, 105184053806.69449, 105211309753.466, 105286806881.72351, 105296937916.0115, 105349756848.9865, 105367770348.418, 105431064133.93701, 105445082887.345, 105462376745.02951, 105472509660.31851, 105493673076.943, 105509110737.452, 105655077311.22801, 105658578592.33151, 105676097425.47449, 105712930307.341, 105755633402.8785, 105800926319.5685, 105829828248.8615, 105838799007.6635, 105907054510.596, 105930066952.39, 105968452849.20499, 105992671416.3035, 106002814437.22101, 106010367717.9435, 106026419268.40698, 106042600856.63348, 106045480068.433, 106050807470.7675, 106079152991.3445, 106106778168.563, 106111564479.1035, 106123928148.189, 106158943694.8345, 106262174975.5625, 106273726072.29599, 106320494830.584, 106349550934.9955, 106375723837.3565, 106383281831.8905, 106459314300.763, 106543083586.8915, 106661970784.43399, 106717386156.15451, 106732445663.57901, 106811428371.02151, 106831697915.685, 106852105934.06151, 106857393664.901, 106882404227.7645, 106951859167.931, 106961194832.0945, 107038914826.4515, 107057328231.90001, 107069709937.05151, 107076112821.0705, 107083357761.4765, 107089998999.664, 107098394622.9675, 107155070833.41348, 107181426346.3695, 107202385450.1405, 107209173676.75601, 107221323692.3505, 107228895888.201, 107236468519.9985, 107245107823.1055, 107249683235.763, 107259816075.4495, 107283583341.23999, 107292956352.75, 107335638150.66399, 107358654353.1185, 107438907012.14, 107460452142.54199, 107474081821.4385, 107480162357.835, 107494438596.1875, 107501066885.607, 107544764404.87799, 107607102344.7215, 107619267242.8725, 107624654726.42749, 107631273995.89299, 107715230887.29599, 107768023926.563, 107869672514.1065, 107954694629.469, 107974409687.961, 107984552024.6405, 108046301154.7115, 108049079146.3575, 108076539464.90999, 108094172158.93799, 108111089916.54501, 108121044185.51651, 108133197850.217, 108138866060.15851, 108142937374.60901, 108150900848.6145, 108231843011.426, 108252084225.09299, 108261651388.584, 108284651640.9065, 108310250184.354, 108351518329.565, 108411365592.41049, 108445227000.078, 108477192474.016, 108555423885.0015, 108585664038.022, 108601875763.993, 108661458961.262, 108675087937.536, 108710784929.8335, 108742434001.5715, 108779845969.66, 108818855809.51498, 108868181005.138, 108899313798.53299, 108924163716.2865, 108955819292.636, 108988949843.74, 108999096806.8475, 109012714210.3665, 109035715696.3475, 109067343411.821, 109086463692.861, 109107313157.0025, 109127401335.67151, 109129979455.012, 109145034635.8825, 109165442461.692, 109219340926.33551, 109252477479.2695, 109278243980.768, 109373403163.6935, 109388355820.17749, 109402387555.56, 109404950352.90201, 109423718538.5815, 109466417842.19699, 109511710702.07849, 109550930928.89551, 109581468915.93149, 109619937512.068, 109652717493.252, 109821868176.1995, 109836894900.716, 109879278937.371, 109937980306.81299, 109993386681.198, 110079901123.5925, 110095520269.3075, 110101589016.7365, 110107677415.90851, 110113750403.90251, 110170442778.1055, 110304245993.6755, 110341118185.732, 110366685847.548, 110503989003.7285, 110520211217.94, 110532338392.4925, 110653257061.06299, 110657306607.9235, 110666036026.95499, 110805900674.65051, 110841023454.3865, 110848569749.4925, 110875624757.0235, 110897007250.22699, 110932877321.8695, 111061352332.5145, 111124144781.4825, 111168485304.452, 111182812881.3395, 111309693971.1925, 111502552997.0105, 111545657752.50949, 111564431710.2415, 111588549701.8005, 111617626439.91751, 111690910522.4155, 111738008916.7135, 111778695929.877, 111840146536.556, 111906814875.39249, 111914907295.169, 112034664477.3725, 112043504229.192, 112092706447.55, 112153990709.61249, 112274933386.283, 112432472038.47601, 112483995105.06601, 112514224343.5045, 112552713194.9545, 112576602085.8755, 112611752798.09451, 112778150217.9495, 112863538623.437, 112904796187.424, 112934988885.553, 112947708661.93549, 113103806902.544, 113276608355.433, 113292822195.166, 113308447482.17201, 113397705296.5635, 113612954667.9245, 113783044859.03949, 114154829566.0495, 114300932285.8295, 114344919079.31151, 114501432324.62401, 114534547552.6855, 114567657029.2365, 114662267210.255, 114685093942.7155, 114782283226.0315, 114820219404.4075, 114833080654.547, 114851850211.3865, 114870235300.0675, 114901193708.738, 115128109177.02449, 115217238311.07349, 115376797996.6995, 115516549230.15, 115733387614.9125, 115921318007.5595, 115952814615.88, 115954236993.2635, 115956974749.218, 116195526486.9, 116390102112.983, 116586694156.6185, 116750681171.19, 116912113646.485, 117519340297.7915, 117692919300.3235, 117924083127.851, 118061796189.4485, 118154216988.5515, 118474294383.539, 118772881891.9195, 119431468678.07849, 120435922971.3795, 120701194419.755, 120735809522.7955, 121060508694.698, 121881800230.43399, 122518272570.2915, 122684647532.21451, 122832473361.149, 123057261196.295, 125784038210.01099, 128656025575.17099])
labels = array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0])
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

numthresholds = 851



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


