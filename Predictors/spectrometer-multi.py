#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/52217/phpHNcI2h -o Predictors/spectrometer_QC.py -target LRS-class -stopat 71.0 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:04.01. Finished on: May-22-2020 20:42:33.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        48-way classifier
Best-guess accuracy:                11.18%
Model accuracy:                     100.00% (531/531 correct)
Improvement over best guess:        88.82% (of possible 88.82%)
Model capacity (MEC):               506 bits
Generalization ratio:               1.04 bits/bit
Confusion Matrix:
 [7.91% 0.00% 0.00% ... 0.00% 0.00% 0.00%]
 [0.00% 0.19% 0.00% ... 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.19% ... 0.00% 0.00% 0.00%]
 ...
 [0.00% 0.00% 0.00% ... 0.56% 0.00% 0.00%]
 [0.00% 0.00% 0.00% ... 0.00% 0.19% 0.00%]
 [0.00% 0.00% 0.00% ... 0.00% 0.00% 0.19%]

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
TRAINFILE = "phpHNcI2h.csv"


#Number of attributes
num_attr = 102
n_classes = 48


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="LRS-class"


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
    clean.mapping={'28': 0, '85': 1, '95': 2, '18': 3, '42': 4, '4': 5, '23': 6, '29': 7, '15': 8, '43': 9, '17': 10, '24': 11, '44': 12, '26': 13, '21': 14, '14': 15, '31': 16, '25': 17, '34': 18, '22': 19, '33': 20, '27': 21, '2': 22, '35': 23, '32': 24, '73': 25, '37': 26, '16': 27, '80': 28, '91': 29, '13': 30, '71': 31, '41': 32, '92': 33, '81': 34, '36': 35, '96': 36, '79': 37, '45': 38, '69': 39, '39': 40, '12': 41, '3': 42, '5': 43, '50': 44, '38': 45, '82': 46, '72': 47}

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
energy_thresholds = array([12685549.722980002, 24921599.359300006, 32276583.075129993, 38798763.75117999, 62194769.10064, 86281442.49439001, 89361987.46474001, 99249856.84759, 108394108.563195, 110001779.53167501, 114661985.13775998, 119674193.398055, 123116660.202215, 130135284.70666, 143887230.82279503, 157341821.20587498, 161509737.56147492, 165207801.34187496, 170322879.72781998, 173463096.444585, 178402206.29359505, 183422393.59848505, 185041884.677025, 186817398.48752004, 192920911.45600003, 198848597.35113496, 204870160.98741496, 214128127.13517994, 220240390.27534997, 227678939.63191503, 248101701.87796503, 268009165.09042493, 285457876.20957494, 297945438.25017, 299160253.57769006, 317150145.84596, 339336178.46744, 345057881.6857, 350521653.29958, 357306588.926915, 361393322.45245993, 363762925.9342251, 364529806.460675, 366004260.469805, 373130706.47655, 381686788.84141505, 384583735.56584513, 385007091.10644495, 385184727.84107494, 400243413.82237995, 416908100.769135, 421389910.863225, 425137472.6525899, 428518355.87591004, 439926361.05716014, 456512084.913635, 476314686.4099351, 490279793.85594505, 501724818.668, 523529575.17946506, 538594694.963365, 541494085.9579048, 543953035.24644, 546860785.4962802, 547385378.4180901, 547465023.38245, 547766868.0115349, 553478081.878085, 559166280.641225, 565778175.02808, 586486493.1570151, 600682237.5302502, 613782595.9075801, 618168893.4455049, 626174128.20068, 632641867.683005, 635549593.4978399, 640944163.149725, 647507036.492185, 658841386.5217751, 676813877.2759001, 700178059.9492551, 710975458.94663, 718684492.405815, 728506199.61676, 752961019.162765, 756785753.4430598, 767728688.8634, 778025438.2971251, 783960878.9506199, 794291155.8539249, 801978684.13801, 803736415.833185, 815185348.1779599, 828510147.3355649, 831951503.337405, 837007961.7760699, 842110364.25279, 846008621.9534202, 852144801.8475502, 858768633.71995, 869362236.4249501, 895669362.738, 912762985.6418549, 915205820.891215, 919657059.1064252, 950071388.247515, 970767007.048215, 990283283.89235, 1006231220.966295, 1010729076.167195, 1013773454.5689499, 1028651117.3874497, 1061688031.06395, 1090756675.829615, 1100675592.0616798, 1104008329.2394, 1118988542.942285, 1132610475.86129, 1137747329.6417198, 1143045774.74223, 1146036224.8441503, 1149119393.1996698, 1153574767.860805, 1178609525.9110303, 1205251997.299375, 1212241524.60368, 1219791021.31645, 1229808689.35158, 1235151138.09748, 1239855530.6436949, 1244065441.2460618, 1244954789.7234173, 1252418072.8207803, 1260265019.0198202, 1263074476.9947252, 1265302038.8068304, 1267047292.474185, 1269302364.4177551, 1272717728.621255, 1283184990.74704, 1293658553.4934502, 1297311667.022695, 1299806254.0355804, 1308577445.3404107, 1317400604.1176, 1324991844.5227747, 1333850412.0033503, 1348106290.68295, 1359891614.9557996, 1376158447.9765596, 1392830618.40231, 1393754303.4434052, 1395127641.9399898, 1401192278.2347102, 1407337187.689845, 1408929338.9386, 1411467375.0735998, 1417039529.2920249, 1422861860.7374253, 1426726452.1013403, 1429910642.1414747, 1440732343.1357703, 1451176996.1483655, 1455322817.3652203, 1459702968.0735655, 1463288262.635225, 1468604492.6057, 1472233891.990875, 1475654611.237875, 1480095701.5202901, 1483402887.1099, 1494008938.78241, 1505691596.1451, 1509729748.3752398, 1511185417.26964, 1529395707.1550899, 1548061166.1679602, 1557639334.8837404, 1562389712.5184147, 1579368378.9826045, 1599776613.5867298, 1603629007.4358048, 1607050945.38908, 1613289944.3659296, 1616231845.8031247, 1623234163.8938403, 1652816610.4256, 1677551621.04945, 1680020700.3618999, 1680877800.75846, 1683340362.39941, 1686704056.143375, 1693611022.9592247, 1707849563.61582, 1718180009.8575249, 1720653941.0757046, 1725497240.7359, 1736461117.8946953, 1748027324.018445, 1759612715.6147249, 1772586765.3723102, 1782485402.4421554, 1785858280.5331264, 1790545131.6259415, 1803092254.2793698, 1814400760.102015, 1819415374.3164248, 1822186115.2604403, 1828461326.1727853, 1833321695.0350199, 1834716419.4869351, 1835117462.830935, 1845945172.0141149, 1856711529.0004048, 1860403293.3073397, 1873022109.8478746, 1882464107.2636251, 1884743691.3049002, 1887495283.5673, 1891618330.0530996, 1895621726.338099, 1898368190.7044992, 1902102767.986885, 1905040271.3622053, 1916366105.5213, 1927152693.5722249, 1928863480.017045, 1934025850.3371997, 1943897614.6061401, 1950173321.89813, 1952681638.9551897, 1968587241.5026698, 1983636791.3838701, 1989147771.2905302, 1995303749.0380301, 2000391480.2504153, 2005879113.8628502, 2009048670.736835, 2016604721.5487003, 2026203236.9657302, 2028985181.458635, 2034516767.1713252, 2039854607.74319, 2042189290.085475, 2042785452.081295, 2046992372.5847352, 2060279651.89119, 2073291164.3511925, 2079867776.0501924, 2086758454.1881251, 2091661128.43649, 2096902309.2639098, 2117226574.04237, 2138560612.594595, 2150091880.1187057, 2163505557.573251, 2175196249.4155, 2176270124.14975, 2183941220.4787498, 2193115088.68489, 2222018107.353999, 2226834357.399844, 2240828819.926895, 2272201223.7644, 2276640372.9477253, 2280743698.4869547, 2282388996.0909653, 2287159334.7123604, 2297081659.687475, 2313814599.8686004, 2327349728.7274294, 2337152082.3521295, 2345486853.3796344, 2354214348.6043944, 2362146465.73201, 2362334857.90707, 2364993639.5266705, 2367796497.3845706, 2382768427.7846203, 2397698110.895385, 2407359104.7796993, 2416787181.1688895, 2417975899.973445, 2421985155.1014395, 2424924150.20061, 2430834087.12449, 2437890533.6541348, 2440510571.781035, 2443170139.4918003, 2456652781.6834, 2469794915.7324495, 2475967136.79834, 2491883872.58904, 2502877722.47989, 2505423307.7052903, 2507961213.1556597, 2509861700.6122994, 2513837925.46, 2526436214.0755253, 2538893266.670665, 2545854732.05303, 2567323621.7321196, 2584930756.6521506, 2585911921.538696, 2586864263.790635, 2594270734.258651, 2604964312.418151, 2614589708.4352007, 2623476012.74555, 2631197491.3804007, 2637456534.6860447, 2646589205.371394, 2654610464.176565, 2668063589.6657305, 2684404930.8070498, 2690501180.36492, 2695961038.2934904, 2712986927.2023554, 2718967261.2376504, 2724651250.07755, 2731229361.2518005, 2743537468.2131004, 2761580886.2113943, 2769866833.1442804, 2771957045.150051, 2780181650.4271154, 2786779597.3788505, 2788601130.8025, 2794357907.9774094, 2809962895.85406, 2823701839.9022894, 2827160288.026, 2830847261.2919445, 2844238399.8849497, 2856686515.7517047, 2857458146.718815, 2860889432.5371003, 2880280272.3934507, 2901209203.8212805, 2907577946.3564653, 2915016955.445735, 2922448591.5217447, 2930713703.28426, 2944536100.8109007, 2952796671.025651, 2959520071.9629655, 2967508216.40578, 2970195112.4572353, 2971097588.002985, 2975779406.432334, 2982031400.1664543, 2984715173.688289, 2986764619.7540846, 2997436578.12748, 3006960971.6610403, 3009498584.248725, 3014626460.378434, 3023960765.3401995, 3030516995.3856754, 3032873433.374075, 3037386419.038464, 3042789480.163665, 3065304468.5144606, 3086171887.67961, 3088320475.171, 3090315410.76377, 3102615095.5335197, 3119422338.0978246, 3128581371.5064754, 3145242921.7073503, 3178327370.7011356, 3216904800.11512, 3231460268.867584, 3232448995.858385, 3248098341.6106653, 3265076765.8250103, 3268704982.90555, 3276147820.8153543, 3290761311.0731344, 3308268950.3450346, 3317035517.5584445, 3318029972.0139046, 3319353283.3520193, 3322735907.9945297, 3329463916.922805, 3334869170.7917147, 3338619652.717514, 3356960298.349021, 3364826285.35867, 3368102951.58633, 3372966101.5783644, 3378934528.7498837, 3381525586.7720695, 3385682962.83012, 3389230820.984745, 3390864846.5887156, 3398502820.7343597, 3408453250.58247, 3413435383.2385144, 3416431419.7249346, 3418239214.2698994, 3422532594.07748, 3430981216.3247147, 3435766765.9681845, 3440818023.249785, 3453731718.3282843, 3463832737.9785995, 3471307501.718095, 3477450590.041045, 3485389648.6943007, 3506570351.5823994, 3524002136.5350046, 3528842467.7011604, 3535859994.266905, 3541944329.417499, 3554793328.3990593, 3570941461.527564, 3590212897.4458547, 3607054075.1847005, 3617559563.990016, 3631739215.397951, 3644077494.0626903, 3651870460.3948555, 3656146015.7333846, 3667071684.3605833, 3673965471.6989803, 3674377566.5890503, 3678760941.2094746, 3683879898.7664557, 3693094778.673545, 3702350177.5134, 3708532178.239065, 3713671860.308625, 3717986450.02697, 3723717186.2743297, 3726559620.537685, 3727822509.4431505, 3730484602.5306797, 3737976494.0112944, 3744012808.587949, 3745185160.4403496, 3748393453.407805, 3751065965.596575, 3753148276.0229597, 3755740027.436715, 3760477726.4948254, 3774175736.4069004, 3789178078.1986, 3811066319.342205, 3830125238.8650002, 3841082052.5017853, 3850092914.82469, 3851280402.9347, 3852150228.852955, 3852569736.072655, 3857544790.1734247, 3873695148.5198746, 3903018326.51285, 3922237638.149495, 3928498958.32341, 3934069692.91249, 3935438362.483875, 3940565913.5738, 3944850003.4388757, 3955462534.4147205, 3972797783.6221447, 3980034856.5286546, 3981639868.95961, 3985215420.5813055, 3994281726.226805, 4003123278.999201, 4007908082.8562007, 4012889746.9119, 4016986271.5438347, 4020959575.8816895, 4030730800.3893995, 4039111887.7522693, 4049260306.5958695, 4065665355.7471, 4080295295.4497795, 4090428842.97861, 4095386262.23295, 4103404491.754605, 4112387473.0537753, 4116582279.7456245, 4127120343.9082847, 4136302349.3576393, 4141708695.114156, 4147110411.769701, 4155644598.7363453, 4165776070.4829006, 4169591297.5864544, 4173760880.5295396, 4181733520.612015, 4193635701.3738747, 4203168720.047045, 4209161477.790945, 4221555865.5955496, 4236767832.1709003, 4248022678.6114197, 4259343384.9285345, 4271450999.5015154, 4280246237.5130506, 4286065951.3685, 4291019948.3332696])
labels = array([0.0, 11.0, 27.0, 16.0, 10.0, 21.0, 0.0, 21.0, 13.0, 3.0, 13.0, 8.0, 27.0, 4.0, 7.0, 6.0, 4.0, 34.0, 4.0, 19.0, 32.0, 4.0, 19.0, 0.0, 16.0, 8.0, 0.0, 13.0, 4.0, 6.0, 46.0, 6.0, 15.0, 3.0, 6.0, 4.0, 6.0, 7.0, 4.0, 32.0, 9.0, 17.0, 0.0, 3.0, 21.0, 4.0, 16.0, 0.0, 13.0, 21.0, 13.0, 4.0, 17.0, 0.0, 21.0, 3.0, 30.0, 4.0, 12.0, 6.0, 19.0, 4.0, 13.0, 8.0, 20.0, 14.0, 8.0, 6.0, 8.0, 3.0, 21.0, 28.0, 9.0, 12.0, 19.0, 13.0, 27.0, 5.0, 7.0, 21.0, 4.0, 9.0, 10.0, 9.0, 3.0, 22.0, 6.0, 39.0, 7.0, 21.0, 13.0, 4.0, 9.0, 38.0, 13.0, 7.0, 8.0, 27.0, 0.0, 7.0, 21.0, 11.0, 0.0, 9.0, 12.0, 0.0, 13.0, 5.0, 4.0, 15.0, 11.0, 45.0, 7.0, 0.0, 4.0, 14.0, 3.0, 33.0, 13.0, 7.0, 4.0, 42.0, 6.0, 0.0, 12.0, 9.0, 17.0, 18.0, 0.0, 4.0, 10.0, 37.0, 7.0, 0.0, 14.0, 17.0, 19.0, 8.0, 14.0, 10.0, 8.0, 14.0, 0.0, 14.0, 4.0, 7.0, 0.0, 17.0, 0.0, 21.0, 7.0, 21.0, 11.0, 13.0, 19.0, 7.0, 24.0, 20.0, 19.0, 6.0, 24.0, 12.0, 15.0, 21.0, 29.0, 6.0, 7.0, 4.0, 43.0, 0.0, 19.0, 8.0, 21.0, 19.0, 4.0, 7.0, 0.0, 19.0, 3.0, 27.0, 0.0, 8.0, 6.0, 9.0, 7.0, 3.0, 14.0, 11.0, 7.0, 23.0, 38.0, 12.0, 38.0, 16.0, 8.0, 9.0, 13.0, 0.0, 4.0, 14.0, 4.0, 0.0, 10.0, 37.0, 19.0, 44.0, 4.0, 7.0, 3.0, 15.0, 11.0, 27.0, 6.0, 19.0, 12.0, 11.0, 15.0, 32.0, 21.0, 7.0, 6.0, 12.0, 5.0, 9.0, 11.0, 4.0, 12.0, 11.0, 24.0, 6.0, 12.0, 32.0, 13.0, 21.0, 7.0, 40.0, 8.0, 3.0, 34.0, 0.0, 9.0, 4.0, 7.0, 40.0, 6.0, 21.0, 9.0, 21.0, 10.0, 12.0, 7.0, 17.0, 12.0, 8.0, 6.0, 24.0, 10.0, 7.0, 13.0, 10.0, 19.0, 20.0, 17.0, 4.0, 9.0, 19.0, 16.0, 8.0, 3.0, 17.0, 23.0, 8.0, 41.0, 7.0, 3.0, 9.0, 7.0, 15.0, 6.0, 2.0, 24.0, 7.0, 4.0, 7.0, 12.0, 4.0, 8.0, 9.0, 13.0, 12.0, 16.0, 25.0, 0.0, 21.0, 27.0, 28.0, 30.0, 19.0, 27.0, 13.0, 7.0, 15.0, 19.0, 8.0, 19.0, 17.0, 21.0, 13.0, 16.0, 47.0, 7.0, 21.0, 1.0, 10.0, 9.0, 16.0, 36.0, 14.0, 4.0, 6.0, 13.0, 6.0, 45.0, 17.0, 6.0, 4.0, 14.0, 32.0, 7.0, 19.0, 4.0, 21.0, 8.0, 29.0, 3.0, 11.0, 0.0, 6.0, 14.0, 27.0, 29.0, 4.0, 19.0, 14.0, 21.0, 35.0, 11.0, 40.0, 13.0, 7.0, 0.0, 4.0, 6.0, 3.0, 5.0, 7.0, 21.0, 0.0, 41.0, 13.0, 28.0, 21.0, 24.0, 12.0, 21.0, 32.0, 6.0, 8.0, 11.0, 0.0, 11.0, 3.0, 7.0, 27.0, 7.0, 19.0, 7.0, 38.0, 27.0, 19.0, 0.0, 7.0, 12.0, 0.0, 8.0, 13.0, 7.0, 13.0, 7.0, 14.0, 7.0, 3.0, 21.0, 17.0, 0.0, 7.0, 9.0, 3.0, 13.0, 7.0, 21.0, 30.0, 0.0, 21.0, 4.0, 17.0, 4.0, 13.0, 3.0, 24.0, 13.0, 39.0, 13.0, 32.0, 6.0, 45.0, 19.0, 7.0, 4.0, 0.0, 37.0, 12.0, 17.0, 21.0, 32.0, 21.0, 4.0, 9.0, 4.0, 0.0, 19.0, 4.0, 19.0, 9.0, 13.0, 11.0, 3.0, 15.0, 22.0, 26.0, 13.0, 17.0, 10.0, 27.0, 8.0, 10.0, 3.0, 23.0, 22.0, 17.0, 4.0, 21.0, 6.0, 18.0, 5.0, 31.0, 8.0, 17.0, 38.0, 21.0, 7.0, 3.0, 7.0, 24.0, 17.0, 7.0, 18.0, 21.0, 6.0, 16.0, 19.0, 5.0, 27.0, 8.0, 6.0, 21.0, 6.0, 15.0, 21.0, 0.0, 7.0, 10.0, 5.0, 11.0, 14.0, 20.0, 7.0, 8.0, 11.0, 0.0, 11.0, 9.0, 24.0, 3.0, 17.0, 7.0, 0.0, 24.0, 7.0, 6.0, 3.0, 12.0, 18.0, 7.0, 4.0, 15.0])
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
        outputs[defaultindys] = 7.0
        return outputs
    return thresh_search(energys)

numthresholds = 506



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


