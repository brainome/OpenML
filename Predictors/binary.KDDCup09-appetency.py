#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target APPETENCY -cm {'-1':0,'1':1} KDDCup09-appetency.csv -o KDDCup09-appetency.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 1:21:06.83. Finished on: Sep-03-2020 11:05:28.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 98.22%
Training accuracy:                   100.00% (30000/30000 correct)
Validation accuracy:                 96.61% (19323/20000 correct)
Overall Model accuracy:              98.64% (49323/50000 correct)
Overall Improvement over best guess: 0.42% (of possible 1.78%)
Model capacity (MEC):                1014 bits
Generalization ratio:                48.64 bits/bit
Model efficiency:                    0.00%/parameter
System behavior
True Negatives:                      97.61% (48803/50000)
True Positives:                      1.04% (520/50000)
False Negatives:                     0.74% (370/50000)
False Positives:                     0.61% (307/50000)
True Pos. Rate/Sensitivity/Recall:   0.58
True Neg. Rate/Specificity:          0.99
Precision:                           0.63
F-1 Measure:                         0.61
False Negative Rate/Miss Rate:       0.42
Critical Success Index:              0.43
Confusion Matrix:
 [97.61% 0.61%]
 [0.74% 1.04%]
Overfitting:                         No
Note: Labels have been remapped to '-1'=0, '1'=1.
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
TRAINFILE = "KDDCup09-appetency.csv"


#Number of attributes
num_attr = 230
n_classes = 2


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="APPETENCY"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="APPETENCY"
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
    clean.mapping={'-1':0,'1':1}

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
energy_thresholds = array([326133423926.8772, 326136389265.3712, 327525535398.8774, 327540141893.1026, 327559176620.567, 327564079292.7034, 328144970991.5046, 328148538141.64856, 328459916996.3701, 328469932749.06116, 329175544882.7792, 329179349904.7978, 329318885900.1722, 329332983969.8396, 329335572464.0052, 329336573359.72, 329420025786.1211, 329426342034.60693, 329539370777.599, 329548192558.6334, 329884886713.48126, 329899694968.0152, 330132515154.5841, 330150109273.1494, 330166196337.6424, 330169790567.8516, 330179729321.0596, 330181163232.442, 330250993073.07556, 330253542753.0232, 330539833638.9575, 330541683243.76135, 330630118460.3921, 330632417547.8825, 330711917773.7105, 330722512416.6992, 330919987852.74194, 330920748512.08923, 330982839957.48975, 330984152669.9444, 331169744771.5864, 331170282174.405, 331375997451.713, 331376774447.1556, 331604312582.75085, 331609317017.88745, 331729007491.29956, 331730234931.89514, 331784490263.5272, 331785141856.7333, 331821959367.834, 331826036514.33105, 332115769537.2672, 332117081901.30676, 332128042743.70056, 332128688282.31573, 332130203698.0116, 332131286266.9071, 332366303888.4427, 332367591161.62354, 332436298264.0209, 332445966080.29285, 332652019468.2076, 332652967400.1248, 332670296412.57745, 332672502369.615, 332687666980.9696, 332689810490.8839, 332761103871.6195, 332763578841.04, 333053003955.27045, 333053323234.8793, 333060000463.801, 333060953863.949, 333072322843.5297, 333073586926.89197, 333142408329.67633, 333143461502.4292, 333151988889.2848, 333153264882.85547, 333212733986.8142, 333213529050.9325, 333288163848.56433, 333289559007.4552, 333479089564.16534, 333482827100.22797, 333490410507.50354, 333491957538.5138, 333510885909.10474, 333512476610.20496, 333520935674.95447, 333522356589.8923, 333768747796.9206, 333769672356.74866, 333837945659.699, 333841891887.6201, 333863150813.18384, 333863261048.6405, 333900271951.8172, 333902982536.1107, 333955358593.40845, 333956932090.01917, 334079035734.9911, 334079532701.43835, 334195873080.29095, 334196908755.5227, 334232482169.3378, 334232601947.2251, 334320049009.60425, 334320456731.94495, 334334499884.74524, 334335868337.31116, 334494769924.2694, 334495550855.6547, 334603872346.9854, 334604048980.303, 334773339013.21045, 334773535864.62524, 334870103162.66296, 334871577827.08716, 334876620603.68726, 334877352247.6474, 334909452421.27795, 334909797037.03424, 334984322998.5414, 334984474765.33826, 335012446150.37085, 335013180908.1817, 335056627786.47205, 335057405769.9818, 335068591080.2157, 335070027863.6643, 335112780355.098, 335113178141.72876, 335150053217.994, 335150728863.8491, 335172296095.23065, 335172718097.89777, 335339909781.4132, 335341725203.40607, 335454889040.0579, 335455088104.84326, 335519927203.98645, 335522960501.86084, 335613727720.8983, 335615206449.276, 335676503714.62213, 335677981427.3541, 335758828240.43494, 335760105384.6803, 335787022006.8944, 335787629908.8418, 335937244386.98065, 335937597939.6312, 336000027391.5071, 336000834515.79724, 336090168881.0453, 336090440609.73694, 336177027536.419, 336177333927.7854, 336191612668.8983, 336191994405.0427, 336277752031.0111, 336278095907.0762, 336383954307.3833, 336384651929.7295, 336397511689.5767, 336397845282.2508, 336487984585.87726, 336491295225.8214, 336506414860.6915, 336508874535.1441, 336572486964.7672, 336573238962.15027, 336630168568.7499, 336631375272.09595, 336660659570.6004, 336660960943.78345, 336713140599.0652, 336713967173.55164, 336842062684.0426, 336843178551.79846, 336897312965.2605, 336897902699.0819, 336955752460.6228, 336957321910.7272, 336960929866.1406, 336961117152.98956, 336969673147.5397, 336970210239.28876, 336974019524.8552, 336975006246.3782, 336997489399.04315, 336997954631.48315, 337014272652.154, 337014486894.32336, 337031363677.8789, 337031709677.3325, 337070459423.0601, 337071409915.5057, 337121572507.9866, 337123544920.35754, 337130558712.50244, 337131578144.6282, 337133030262.44385, 337133352708.2539, 337192887178.56537, 337193966888.64954, 337255479537.33014, 337256236007.0458, 337461875769.0571, 337462454382.2313, 337474833228.2733, 337476593406.8611, 337513369495.5065, 337514142040.47485, 337521971426.7703, 337522219070.2876, 337564857675.2684, 337565280389.40894, 337566291933.3036, 337566824951.6758, 337574613419.9179, 337575824969.3686, 337591249404.9095, 337591494052.4136, 337685788111.23676, 337686151552.05835, 337711811922.874, 337712803363.972, 337803307496.2889, 337803591244.86426, 337858742346.26184, 337859353148.6404, 337866586834.021, 337866908834.5541, 337958358106.725, 337958455566.5011, 338015873582.2772, 338016055992.71155, 338028573133.96716, 338029270513.9298, 338163550085.4691, 338163989681.51697, 338164762513.47876, 338166046168.9467, 338188609889.329, 338188798652.00397, 338190439114.67236, 338190832842.39734, 338191227769.73035, 338193027285.13544, 338211871666.0657, 338212285451.37024, 338225112257.4728, 338225421433.5208, 338226226142.15076, 338227005230.56213, 338554198265.6279, 338556343414.2889, 338576958564.50134, 338577641921.43945, 338587082836.52106, 338588241290.08105, 338700651093.8838, 338700938540.4631, 338704632350.21814, 338705522922.0576, 338755512697.0952, 338755985201.11005, 338808751318.4755, 338809501972.73694, 338871264760.9563, 338871811390.15424, 338912268708.7173, 338912505976.0076, 338914444621.6407, 338915624200.5719, 338927917739.96216, 338928770581.69617, 339047486023.6642, 339048050256.55383, 339088355184.38574, 339089113813.2838, 339132955950.46484, 339133498253.6743, 339192423086.4516, 339192857091.5219, 339278252790.0568, 339278860689.2355, 339313307216.97565, 339313526474.853, 339410366075.7854, 339410764021.8534, 339514135046.19147, 339514249566.2657, 339533588737.32385, 339533743477.6221, 339533776105.7301, 339533818195.62103, 339537610958.77264, 339537847829.7492, 339637933988.238, 339638370254.9193, 339660931287.11676, 339661674883.5241, 339672596024.4163, 339672874170.43024, 339724870510.36365, 339725654199.8297, 339729861831.71686, 339730159177.95447, 339803842389.85394, 339804215321.94995, 339812084456.9889, 339812645877.3974, 339826936876.8967, 339828724505.4365, 339832541481.07733, 339833889710.83923, 339845820457.56537, 339846357402.15173, 339859996578.48, 339860694590.23596, 339865166414.62115, 339865537095.3474, 339900343943.09906, 339900710368.7019, 339969332019.91864, 339970373871.87744, 339971696512.5172, 339972105086.5096, 340020471246.7411, 340020930076.22034, 340024310385.757, 340024888074.11316, 340047542758.33777, 340048019827.04407, 340054449194.9142, 340054718089.42395, 340059012965.7954, 340059371834.7068, 340075830947.40295, 340076212846.31946, 340217252144.2529, 340217527887.7107, 340225565598.06067, 340226193152.71155, 340263588379.24036, 340263917398.19495, 340408059224.3259, 340408492791.8003, 340448243384.2136, 340448703088.202, 340491531250.38464, 340492360772.3563, 340522828061.6591, 340523639590.9446, 340584969217.1529, 340587102676.4153, 340591383068.9708, 340591684537.6633, 340598126413.51416, 340599216683.19446, 340613453306.60626, 340614451678.1086, 340616496558.1262, 340616780985.8053, 340636657726.79944, 340636919131.0916, 340671067974.1022, 340671771240.60754, 340854009517.16724, 340854543532.61163, 340945784259.8719, 340946962600.37683, 340996089127.16003, 340996814683.9967, 341008608315.4564, 341009003571.19257, 341014603166.0811, 341015042907.6849, 341035135086.14197, 341035313218.6653, 341065068309.96, 341066132946.0142, 341069935711.13574, 341070358193.31665, 341090167053.105, 341092378438.26074, 341096889820.41144, 341097182697.051, 341117648946.7526, 341118088986.84235, 341159147046.12585, 341160021787.3957, 341205591315.6707, 341205714264.11414, 341235689074.2709, 341237123605.1295, 341302281946.7584, 341302955491.47754, 341308676710.2393, 341309318296.9702, 341344209150.1113, 341344528821.9127, 341394670053.619, 341394968110.7279, 341425482814.06946, 341426219318.50244, 341443813313.84937, 341445851853.7079, 341502944636.3875, 341503033655.64075, 341522962940.5113, 341523242991.9226, 341583276255.1793, 341583904387.7711, 341643976562.6776, 341644266772.66846, 341663204429.48315, 341663322512.8241, 341681793486.407, 341682086987.00507, 341747515205.39307, 341747654855.00555, 341757679461.1002, 341758150603.93585, 341812701830.9164, 341813229934.07825, 341821987179.797, 341822614435.0973, 341843419513.421, 341844028243.91174, 341855606969.4637, 341856053353.55884, 341889321067.8304, 341889951928.625, 341893388189.4867, 341894402035.5984, 341954098815.943, 341954529407.2378, 342036312739.16785, 342037350745.00165, 342052195346.857, 342052409988.5073, 342173301501.26685, 342173528979.167, 342201526861.9396, 342201802265.1742, 342263142076.78375, 342263439973.65497, 342312367344.2772, 342312862667.3528, 342319477687.9543, 342320048111.92017, 342423221861.18134, 342423642790.8435, 342499074793.6478, 342499634087.723, 342538488068.32666, 342540680334.6628, 342550120198.7467, 342551007254.48, 342563956878.8382, 342564236014.7904, 342628747041.50244, 342628895897.3672, 342673769655.8185, 342674392198.86646, 342684629616.9056, 342685040572.1433, 342710129205.00494, 342710380617.6715, 342720056276.3561, 342720114900.4171, 342794813748.6019, 342795207704.4353, 342814310738.62915, 342815422368.5973, 342892840430.9365, 342893239472.22394, 342902590574.0685, 342902825124.4639, 343024814584.40356, 343026952819.50916, 343028618932.4016, 343029345411.4049, 343031915384.3992, 343032814672.3628, 343094977972.7042, 343095267416.58575, 343105736424.21265, 343105984473.0469, 343111130285.4529, 343111966314.74945, 343120887264.8999, 343121032937.06165, 343133551460.15015, 343133700075.3773, 343173992872.58203, 343174067082.8351, 343187342314.3722, 343188017329.83044, 343227080370.72516, 343228582233.2991, 343235164288.69165, 343235346491.3667, 343239832352.8625, 343240952253.4983, 343300782478.4732, 343301110394.8538, 343309461654.50305, 343309563148.82794, 343311693577.12744, 343312131767.3816, 343418772935.03253, 343419079700.2715, 343488974786.676, 343489726651.63684, 343502880026.33124, 343503193348.20447, 343511416132.01263, 343511684624.0093, 343538384382.1411, 343538683776.20056, 343541424996.8909, 343542069226.54803, 343563630926.1475, 343563972625.2873, 343590931195.7189, 343591187794.1571, 343670384593.65247, 343671273264.889, 343672123146.2991, 343672387722.5924, 343762573454.376, 343763919331.67017, 343850464556.50964, 343851059103.27466, 343881990958.8095, 343882340285.6406, 343898196335.8417, 343898620390.77185, 343912329742.5684, 343912859337.8049, 343916740105.6472, 343917234358.2612, 343929406022.0846, 343929695818.9565, 343959016856.6652, 343959252854.22833, 344002900767.62994, 344003655609.0605, 344019747430.80896, 344020004160.9929, 344020202127.2412, 344020335123.93884, 344032886259.9791, 344033065266.4868, 344057480565.05493, 344057668678.54614, 344107603014.42236, 344107801775.48206, 344181386457.5503, 344182134547.59296, 344199890920.60986, 344200040300.95447, 344256800345.78064, 344257652329.8828, 344283687750.5784, 344283879630.83014, 344320044414.01697, 344320408311.0226, 344326525224.2755, 344326945740.3356, 344359965220.30414, 344360635627.2669, 344460610705.8716, 344461128194.5619, 344470729575.8312, 344471054622.99677, 344513438467.04803, 344514858204.3423, 344567540692.17957, 344568711603.1189, 344707507141.26355, 344707953894.739, 344831112803.7564, 344831946016.7117, 344839001673.19507, 344839060531.57904, 344880828984.0698, 344881618753.8647, 344888008128.59784, 344888270169.55396, 344901364406.448, 344902439532.38367, 344963580582.5989, 344963789685.87744, 344974343795.9796, 344975540188.1483, 345039015062.3428, 345040175229.23126, 345161527111.3356, 345161932523.4094, 345162172258.36743, 345162433699.62244, 345190894712.0571, 345191996032.4364, 345226705147.2661, 345228090382.8498, 345277602914.8998, 345278765258.62836, 345287459849.31226, 345287658723.0736, 345322258346.8321, 345323271510.644, 345324130817.1204, 345324406392.2226, 345397613259.2783, 345397947056.40454, 345403858053.8722, 345404729252.5707, 345406251735.0504, 345407463077.38745, 345431995628.0125, 345432242128.4969, 345496442104.7386, 345497684925.14594, 345527938192.1714, 345528040080.38794, 345600846700.20776, 345600927597.3577, 345601733753.8307, 345602382032.59845, 345626922391.1732, 345627450271.7174, 345662000921.89136, 345663020665.29114, 345695044163.3783, 345695238371.0857, 345760834615.83545, 345761903407.62524, 345770948792.34625, 345771561160.31165, 345782686561.8203, 345784212907.70404, 345786225611.1476, 345786954819.419, 345788010089.47076, 345789164016.11633, 345844705551.61755, 345844794149.9929, 345844953611.425, 345845356232.41736, 345875941572.14276, 345876397481.8274, 345901222446.75385, 345901569924.1664, 345985879127.8888, 345986140005.5805, 346071073527.82837, 346072718179.2946, 346094000222.74164, 346095605141.5303, 346144615543.34436, 346144979369.95593, 346151612641.2907, 346151962846.4063, 346207983524.08093, 346208302958.04663, 346214159364.2409, 346214423292.1155, 346282941169.7528, 346283520043.98865, 346341191935.496, 346341564061.4839, 346365912828.4985, 346366230399.55774, 346404108281.95166, 346405027567.01135, 346421774308.53577, 346422133793.8345, 346425364347.69995, 346425757058.7865, 346492513045.5272, 346492665440.1223, 346649136336.6288, 346650622083.1628, 346658730951.28455, 346658929231.3161, 346728898848.9707, 346730217717.1516, 346732729068.6614, 346733941248.9768, 346796482225.9418, 346796991867.33997, 346806520609.66187, 346807004309.31433, 346977043239.48535, 346977772503.84717, 347004519379.2013, 347004893533.2545, 347055431751.9229, 347055854962.6706, 347059816079.19104, 347060114543.66956, 347061053075.2433, 347061828463.75635, 347089291218.4823, 347090723794.9286, 347152393894.5714, 347153045440.41547, 347200502917.11707, 347201312695.62036, 347263498836.77673, 347264047419.38696, 347309276634.2217, 347310341004.0012, 347316353616.55475, 347316509854.44617, 347394255210.2867, 347395735994.27795, 347420215076.8221, 347420603972.6786, 347449059160.46375, 347449524666.48584, 347472981934.43115, 347473484201.74023, 347534513100.27454, 347535424629.93494, 347555971491.7223, 347557168643.65295, 347610497281.48535, 347611174750.9442, 347618753021.86597, 347619567803.2622, 347625959674.0036, 347626570792.45886, 347740630498.4236, 347740991681.31464, 347879988916.4012, 347880662594.5348, 347924577422.7968, 347925451362.1911, 347955682444.8791, 347956757508.3331, 348010505303.4729, 348010708866.9955, 348014133038.3572, 348014842329.052, 348024285498.5294, 348024771044.597, 348028681821.72156, 348029946580.246, 348069130069.8981, 348070198856.95874, 348078800957.192, 348079066816.83966, 348296424903.5817, 348299292760.2732, 348391937118.505, 348392708047.87854, 348413452283.3751, 348414583113.1486, 348422607381.59766, 348422811579.662, 348543004613.77496, 348544019003.4476, 348575985317.1225, 348576077678.6155, 348620725305.66077, 348620814472.4134, 348628775143.18854, 348629749978.4798, 348806466588.0556, 348808082602.9452, 348816956945.4229, 348818615111.8423, 348820201920.0388, 348820404107.47705, 348829576057.0131, 348831748216.76685, 348834478591.37634, 348835663068.0185, 348837444724.6535, 348838011417.63544, 348911797416.82434, 348914114636.9098, 348960965218.55853, 348962331173.55273, 349093778498.71204, 349095176044.8567, 349102619370.91797, 349103053527.11017, 349103546646.7314, 349104090331.3152, 349196369965.6392, 349197640515.9503, 349222445967.312, 349223917166.8921, 349291801511.6665, 349293084503.896, 349347858491.0117, 349348046189.6537, 349384151871.31647, 349384723345.28455, 349402534047.1176, 349403304317.7211, 349406553454.87506, 349407230257.6577, 349417250453.61005, 349418020372.06946, 349434666366.8915, 349435618665.6351, 349441742457.76746, 349443055584.59656, 349653307601.75037, 349655036172.172, 349917148391.25195, 349917353493.38696, 349944201615.94104, 349944417863.39636, 349953537305.52, 349954526948.8673, 350008403744.52075, 350009131191.6919, 350099701826.451, 350100585446.0128, 350240762704.2455, 350241168705.4192, 350252738382.8883, 350253019602.9825, 350297889284.3178, 350298718510.68335, 350313918325.8451, 350315671980.1051, 350482534314.4832, 350482879108.978, 350485865621.9752, 350486918814.97644, 350638623426.0043, 350639435825.84863, 350641738043.5367, 350642694780.8625, 350702006750.28394, 350703683108.34827, 350720881126.4226, 350723403155.24805, 350766284309.0168, 350767269122.1696, 350836362708.49603, 350837801062.76953, 350881040368.4669, 350883520753.9957, 350937885653.4547, 350939393701.1099, 350942944059.03125, 350943695086.5396, 351023707091.61646, 351024430790.63477, 351094234280.11426, 351096087239.9288, 351287387123.6642, 351288894516.1564, 351363410022.90564, 351365624441.7268, 351475749566.83813, 351476300594.3846, 351564357891.9829, 351567216407.4752, 351580095652.7096, 351581469860.7328, 351610031864.8273, 351610929470.0397, 351643063510.83136, 351643575454.1532, 351724372391.9646, 351725815205.00977, 351741466462.222, 351742726754.3495, 351779522503.3706, 351780284010.3009, 351800798547.8009, 351801023549.55786, 351853822544.6592, 351854286101.196, 351953947597.10236, 351955443690.4998, 351961213200.6692, 351964602533.68036, 352138451479.12714, 352140473966.17053, 352152467437.64435, 352154037513.33966, 352179638185.25824, 352182262244.61, 352296331709.79236, 352297260961.70624, 352381818863.26685, 352383978860.9309, 352669334119.575, 352670049443.67395, 352720937126.95575, 352721490827.9409, 352745850739.53394, 352746391976.5569, 352791018755.251, 352795011518.2318, 353063737827.40295, 353064551777.0731, 353068307106.8224, 353069784875.41833, 353086990767.12415, 353088922449.2676, 353137963109.6278, 353139659925.58624, 353473319044.41724, 353474967618.708, 353513113603.61334, 353514300147.3823, 353555760611.64526, 353557537036.2583, 353646184798.7506, 353647959979.7123, 353757270744.89325, 353757837682.9391, 354060376704.0735, 354061668102.5618, 354131731740.86633, 354137739695.37286, 354152657579.8473, 354154007040.0845, 354258793084.0308, 354259636299.79205, 354419459517.8585, 354420484287.59863, 354624858623.3684, 354626927564.92456, 354693903428.2216, 354695993538.64, 354834241084.08887, 354836523406.61426, 354907293730.1282, 354911743001.27936, 355131035077.85876, 355132952021.397, 355359953423.34576, 355364167614.14215, 355480778676.9824, 355482491082.0969, 355555150233.095, 355558474844.82684, 356041615920.99365, 356045420660.8178, 356091457695.7593, 356097608420.38696, 356262919909.69415, 356264936341.801, 356301420466.7422, 356303652959.5959, 356334510658.59595, 356335759822.0181, 356886647858.7902, 356894920067.38007, 357167813339.7523, 357172800172.73694, 357282278929.8374, 357289216503.202, 357379688099.595, 357380136918.04724, 357575943028.78265, 357579508179.2837, 357592106344.25, 357594343592.7242, 357939955569.19196, 357943675473.21686, 358550598692.70825, 358551917298.9764, 358556892326.568, 358561129918.90784, 358594910762.70337, 358602755857.46484, 358635823837.7936, 358647824884.3276, 359024085291.0371, 359029625339.4159, 359033048674.473, 359034063403.8197, 359049753614.2611, 359053000422.19885, 359065390741.38416, 359066900090.6919, 359243007783.3818, 359249066786.77625, 359871622590.88995, 359877420694.1782, 360343145461.75714, 360346404985.01587, 360651396508.2412, 360655143134.21313, 361225578946.9204, 361238032669.33484, 361470977450.9243, 361490280749.62805, 363103450595.1867, 363109597227.034, 365336313721.5884, 365356813975.0697, 367523390670.57245, 367577569681.9608, 370924235713.4254, 371110740708.96564])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)
    start_label = 1
    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = (numers[indys] + start_label) % 2
        outputs[defaultindys]=0
        return outputs
    return thresh_search(energys)

numthresholds = 1014



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


