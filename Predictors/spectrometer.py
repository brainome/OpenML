#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53488/spectrometer.arff -o Predictors/spectrometer_NN.py -target binaryClass -stopat 97.93 -f NN -e 20 --yes
# Total compiler execution time: 0:00:36.61. Finished on: Apr-21-2020 21:31:28.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                89.64%
Model accuracy:                     99.43% (528/531 correct)
Improvement over best guess:        9.79% (of possible 10.36%)
Model capacity (MEC):               521 bits
Generalization ratio:               1.01 bits/bit
Model efficiency:                   0.01%/parameter
System behavior
True Negatives:                     89.45% (475/531)
True Positives:                     9.98% (53/531)
False Negatives:                    0.38% (2/531)
False Positives:                    0.19% (1/531)
True Pos. Rate/Sensitivity/Recall:  0.96
True Neg. Rate/Specificity:         1.00
Precision:                          0.98
F-1 Measure:                        0.97
False Negative Rate/Miss Rate:      0.04
Critical Success Index:             0.95

Warning: The prediction model overfits the training data.
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
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "spectrometer.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 102
n_classes = 2

mappings = [{1085196.0: 0, 25559033.0: 1, 38061252.0: 2, 91012968.0: 3, 106552395.0: 4, 109300583.0: 5, 109768202.0: 6, 125509343.0: 7, 168132081.0: 8, 184695789.0: 9, 188006130.0: 10, 218378534.0: 11, 233250084.0: 12, 262013015.0: 13, 334022802.0: 14, 343713860.0: 15, 345464906.0: 16, 354641221.0: 17, 359038278.0: 18, 362813769.0: 19, 363773262.0: 20, 364349170.0: 21, 378598744.0: 22, 383836606.0: 23, 384681946.0: 24, 448390488.0: 25, 463700242.0: 26, 491629898.0: 27, 535239784.0: 28, 541017236.0: 29, 541035355.0: 30, 546982200.0: 31, 558963494.0: 32, 571659272.0: 33, 583397464.0: 34, 620562669.0: 35, 630853198.0: 36, 633491979.0: 37, 636665188.0: 38, 644286260.0: 39, 649794698.0: 40, 654412227.0: 41, 662336163.0: 42, 709066325.0: 43, 711951533.0: 44, 751524220.0: 45, 759173473.0: 46, 775346484.0: 47, 779766362.0: 48, 780198305.0: 49, 786791059.0: 50, 804373186.0: 51, 831022265.0: 52, 841137832.0: 53, 842151872.0: 54, 862180365.0: 55, 875610277.0: 56, 879679011.0: 57, 915608694.0: 58, 922764325.0: 59, 936496254.0: 60, 1001744452.0: 61, 1098960550.0: 62, 1105624585.0: 63, 1131417354.0: 64, 1132867658.0: 65, 1141691946.0: 66, 1143463705.0: 67, 1149631284.0: 68, 1156580910.0: 69, 1199699668.0: 70, 1213678071.0: 71, 1224969054.0: 72, 1243125259.0: 73, 1244904046.0: 74, 1264614336.0: 75, 1265048915.0: 76, 1268105659.0: 77, 1274939019.0: 78, 1297804077.0: 79, 1300873656.0: 80, 1318518149.0: 81, 1330520959.0: 82, 1395902443.0: 83, 1408191589.0: 84, 1413277611.0: 85, 1419865583.0: 86, 1424922869.0: 87, 1427595713.0: 88, 1449238608.0: 89, 1460941593.0: 90, 1464700339.0: 91, 1471572028.0: 92, 1471951232.0: 93, 1508396632.0: 94, 1510126952.0: 95, 1552744342.0: 96, 1603057064.0: 97, 1603258338.0: 98, 1609905507.0: 99, 1615788900.0: 100, 1629745460.0: 101, 1679215821.0: 102, 1679892458.0: 103, 1687658496.0: 104, 1698629485.0: 105, 1716134778.0: 106, 1732350810.0: 107, 1739640145.0: 108, 1762808275.0: 109, 1788175167.0: 110, 1814597409.0: 111, 1819090709.0: 112, 1834434176.0: 113, 1834866438.0: 114, 1882352747.0: 115, 1886195044.0: 116, 1887859267.0: 117, 1895863148.0: 118, 1903336100.0: 119, 1927380194.0: 120, 1929408257.0: 121, 1937706868.0: 122, 1984268439.0: 123, 2007554972.0: 124, 2009607175.0: 125, 2022666289.0: 126, 2025116234.0: 127, 2030680114.0: 128, 2037419976.0: 129, 2050503892.0: 130, 2069122429.0: 131, 2076523148.0: 132, 2082277058.0: 133, 2090307474.0: 134, 2098253417.0: 135, 2147525791.0: 136, 2174355332.0: 137, 2176499031.0: 138, 2190445080.0: 139, 2219240384.0: 140, 2223856877.0: 141, 2251851080.0: 142, 2270076545.0: 143, 2273387291.0: 144, 2282247827.0: 145, 2291138732.0: 146, 2361664330.0: 147, 2362040188.0: 148, 2396958102.0: 149, 2397504735.0: 150, 2416279450.0: 151, 2418657568.0: 152, 2424380733.0: 153, 2441440671.0: 154, 2443967942.0: 155, 2468405095.0: 156, 2507175545.0: 157, 2507810812.0: 158, 2515765850.0: 159, 2540679764.0: 160, 2585309694.0: 161, 2587214501.0: 162, 2608602717.0: 163, 2619642019.0: 164, 2635085564.0: 165, 2654935236.0: 166, 2687615507.0: 167, 2716998122.0: 168, 2719997088.0: 169, 2733154694.0: 170, 2752984677.0: 171, 2769241875.0: 172, 2769554661.0: 173, 2789649932.0: 174, 2798130732.0: 175, 2820857143.0: 176, 2824560688.0: 177, 2831932330.0: 178, 2855606440.0: 179, 2856833130.0: 180, 2863695910.0: 181, 2908668711.0: 182, 2920429991.0: 183, 2951175841.0: 184, 2953483848.0: 185, 2964625084.0: 186, 2969996456.0: 187, 2971263201.0: 188, 2979361307.0: 189, 2983769086.0: 190, 2984729252.0: 191, 2987866105.0: 192, 3006915604.0: 193, 3017169328.0: 194, 3030283117.0: 195, 3088637367.0: 196, 3091056213.0: 197, 3124675403.0: 198, 3131553774.0: 199, 3153408193.0: 200, 3202314838.0: 201, 3232540693.0: 202, 3269980854.0: 203, 3316397423.0: 204, 3325152634.0: 205, 3332840706.0: 206, 3335961797.0: 207, 3367012336.0: 208, 3388509122.0: 209, 3391781255.0: 210, 3411681748.0: 211, 3417670682.0: 212, 3417868334.0: 213, 3492484198.0: 214, 3519722214.0: 215, 3527347559.0: 216, 3541378818.0: 217, 3541570877.0: 218, 3567083521.0: 219, 3573867197.0: 220, 3651495614.0: 221, 3659864761.0: 222, 3673344594.0: 223, 3673652709.0: 224, 3684409129.0: 225, 3687922345.0: 226, 3712367430.0: 227, 3726682671.0: 228, 3728028160.0: 229, 3732007525.0: 230, 3744073227.0: 231, 3754660787.0: 232, 3755886702.0: 233, 3794140985.0: 234, 3848971698.0: 235, 3850278977.0: 236, 3922925967.0: 237, 3944441814.0: 238, 3979114470.0: 239, 3987170506.0: 240, 4004856071.0: 241, 4010025716.0: 242, 4014819284.0: 243, 4018219678.0: 244, 4022765042.0: 245, 4072337212.0: 246, 4087318976.0: 247, 4092606073.0: 248, 4097233706.0: 249, 4108641434.0: 250, 4135130014.0: 251, 4136539668.0: 252, 4147344872.0: 253, 4163010534.0: 254, 4167606728.0: 255, 4186584856.0: 256, 4199748617.0: 257, 4211735132.0: 258, 4242160080.0: 259, 4252952229.0: 260, 4264802653.0: 261, 4277166646.0: 262, 4282390830.0: 263, 4292300670.0: 264, 2698538993.0: 265, 1755482099.0: 266, 3376738937.0: 267, 4230439124.0: 268, 1894440665.0: 269, 2692450117.0: 270, 545932207.0: 271, 547587037.0: 272, 2936962036.0: 273, 366727047.0: 274, 4000454467.0: 275, 3230561416.0: 276, 3965548273.0: 277, 1813264444.0: 278, 414801568.0: 279, 1359037266.0: 280, 1684816005.0: 281, 1258999759.0: 282, 2438648499.0: 283, 3832257131.0: 284, 2175103328.0: 285, 174415665.0: 286, 3933136696.0: 287, 3920616253.0: 288, 3477357573.0: 289, 3262722028.0: 290, 3299207448.0: 291, 119792225.0: 292, 3319384568.0: 293, 3743009523.0: 294, 1452183213.0: 295, 3674169666.0: 296, 3444870002.0: 297, 1782605130.0: 298, 4145942959.0: 299, 2281594658.0: 300, 1511306530.0: 301, 2263632572.0: 302, 1260595919.0: 303, 3404292542.0: 304, 3750496083.0: 305, 2278958379.0: 306, 1040496559.0: 307, 731596317.0: 308, 3764136716.0: 309, 3039310747.0: 310, 2786005050.0: 311, 1315346733.0: 312, 1040819615.0: 313, 2895931572.0: 314, 2502085655.0: 315, 1954166276.0: 316, 3136145483.0: 317, 1393147811.0: 318, 1680930698.0: 319, 3703763090.0: 320, 1269561068.0: 321, 3827057765.0: 322, 1391571512.0: 323, 3861968832.0: 324, 753464870.0: 325, 161349050.0: 326, 910723369.0: 327, 4205653237.0: 328, 1542443272.0: 329, 3725504054.0: 330, 3368260257.0: 331, 1233709753.0: 332, 2653350074.0: 333, 2828820373.0: 334, 1081618286.0: 335, 3944322266.0: 336, 3461658625.0: 337, 299341624.0: 338, 1457526258.0: 339, 3852185910.0: 340, 2424533487.0: 341, 1856400329.0: 342, 724481309.0: 343, 2228873448.0: 344, 2728369254.0: 345, 1010739497.0: 346, 38602410.0: 347, 1719291999.0: 348, 4114051570.0: 349, 1561602912.0: 350, 2324601323.0: 351, 3720994225.0: 352, 2786617479.0: 353, 2510976615.0: 354, 962711257.0: 355, 2367641050.0: 356, 1818802479.0: 357, 3029812027.0: 358, 4037755128.0: 359, 1478417691.0: 360, 3697331230.0: 361, 1824345750.0: 362, 3034530140.0: 363, 3934068655.0: 364, 1235652190.0: 365, 3980022793.0: 366, 2135261105.0: 367, 3113233047.0: 368, 2151719954.0: 369, 2905552483.0: 370, 181454204.0: 371, 2857146073.0: 372, 3435832325.0: 373, 2480749744.0: 374, 3851346818.0: 375, 2092076078.0: 376, 848931938.0: 377, 3231424525.0: 378, 1336234720.0: 379, 977888143.0: 380, 1996580163.0: 381, 2140923543.0: 382, 1905811453.0: 383, 2094614699.0: 384, 3635906240.0: 385, 2626371972.0: 386, 3381918623.0: 387, 913866012.0: 388, 2600393076.0: 389, 1863470125.0: 390, 2194846467.0: 391, 425571129.0: 392, 3529401203.0: 393, 4118175972.0: 394, 1101459911.0: 395, 273067914.0: 396, 1595558148.0: 397, 1949148063.0: 398, 3389010798.0: 399, 4288806133.0: 400, 1856086907.0: 401, 296913859.0: 402, 2502736717.0: 403, 2003270807.0: 404, 4039526604.0: 405, 3087066200.0: 406, 3607546831.0: 407, 1950260940.0: 408, 384751539.0: 409, 3414256357.0: 410, 1244067738.0: 411, 1408720125.0: 412, 3045336044.0: 413, 2470243115.0: 414, 1015874009.0: 415, 3006068448.0: 416, 1147670791.0: 417, 546854341.0: 418, 2042547590.0: 419, 2345832027.0: 420, 1781428909.0: 421, 2969458129.0: 422, 487995803.0: 423, 3700004288.0: 424, 1485031896.0: 425, 2302091470.0: 426, 4058056337.0: 427, 2708039104.0: 428, 3380194261.0: 429, 3318383810.0: 430, 2923524600.0: 431, 3476609215.0: 432, 298045331.0: 433, 854421628.0: 434, 1562243179.0: 435, 800857581.0: 436, 171574187.0: 437, 3884485244.0: 438, 384396485.0: 439, 614842720.0: 440, 3626635319.0: 441, 690353904.0: 442, 160734547.0: 443, 2042088898.0: 444, 825063393.0: 445, 831946241.0: 446, 802166244.0: 447, 3714043866.0: 448, 3266496511.0: 449, 3011145050.0: 450, 1834061884.0: 451, 3982323801.0: 452, 1515413251.0: 453, 1881635922.0: 454, 4170640359.0: 455, 2680256784.0: 456, 2585578819.0: 457, 430526275.0: 458, 418076049.0: 459, 4175946824.0: 460, 1408204714.0: 461, 1295885240.0: 462, 1791979456.0: 463, 611786515.0: 464, 3783278189.0: 465, 84852501.0: 466, 558431427.0: 467, 1359805545.0: 468, 2821907415.0: 469, 3935874256.0: 470, 1431292245.0: 471, 1728978383.0: 472, 3605624243.0: 473, 208944628.0: 474, 3434765236.0: 475, 2367013849.0: 476, 221170312.0: 477, 3852017345.0: 478, 423762791.0: 479, 588640262.0: 480, 196898627.0: 481, 1925988044.0: 482, 1615741628.0: 483, 1393421002.0: 484, 510886413.0: 485, 2550096478.0: 486, 2536171673.0: 487, 1502048324.0: 488, 3750703687.0: 489, 3351278157.0: 490, 3361705054.0: 491, 2361688722.0: 492, 1831645480.0: 493, 118621575.0: 494, 86774930.0: 495, 3745353841.0: 496, 3084341057.0: 497, 4109785748.0: 498, 3426259749.0: 499, 547010295.0: 500, 184456274.0: 501, 2329159818.0: 502, 1290498000.0: 503, 3651309323.0: 504, 3340344274.0: 505, 2773419789.0: 506, 2583614056.0: 507, 2416361653.0: 508, 2041355748.0: 509, 1209866743.0: 510, 1480841914.0: 511, 1982069323.0: 512, 1405550051.0: 513, 3281381822.0: 514, 1721079684.0: 515, 133828950.0: 516, 1674949354.0: 517, 199859779.0: 518, 3465069838.0: 519, 1899934940.0: 520, 2344207782.0: 521, 1009782162.0: 522, 1993089788.0: 523, 153011272.0: 524, 3316738094.0: 525, 23348331.0: 526, 2638892969.0: 527, 2436199156.0: 528, 2026356848.0: 529, 3682419490.0: 530}]
list_of_cols_to_normalize = [0]

transform_true = False

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values()))+1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize,mappings):
            if i>=data_arr.shape[1]:
                break
            col = data_arr[:,i]
            normcol = column_norm(col,mapping)
            data_arr[:,i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([])
        components = np.array([])
        whiten = None
        explained_variance = np.array([])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"


    if (testfile):
        target=''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless==False):
                header=next(reader, None)
                try:
                    if (target!=''): 
                        hc=header.index(target)
                    else:
                        hc=len(header)-1
                        target=header[hc]
                except:
                    raise NameError("Target '"+target+"' not found! Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=header.index(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute '"+ignorecolumns[i]+"' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '"+ignorecolumns[i]+"' not found in header. Header must be same as in file passed to btc.")
                for i in range(0,len(header)):      
                    if (i==hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i]+",", end = '', file=outputfile)
                print(header[hc],file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"'+row[name]+'"'+",",end = '', file=outputfile)
                        else:
                            print (row[name]+",",end = '', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target!=""): 
                        hc=int(target)
                    else:
                        hc=-1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=int(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute "+str(col)+" is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    if (hc==-1):
                        hc=len(row)-1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0,len(row)):
                        if (i in il):
                            continue
                        if (i==hc):
                            continue
                        if (',' in row[i]):
                            print ('"'+row[i]+'"'+",",end = '', file=outputfile)
                        else:
                            print(row[i]+",",end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    clean.mapping={'N': 0, 'P': 1}

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

# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def classify(row):
    #inits
    x=row
    o=[0]*num_output_logits


    #Nueron Equations
    h_0 = max((((0.4311939 * float(x[0]))+ (0.19837768 * float(x[1]))+ (0.07877358 * float(x[2]))+ (-0.14218085 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.13444138 * float(x[5]))+ (0.77237433 * float(x[6]))+ (0.91416305 * float(x[7]))+ (-0.24548115 * float(x[8]))+ (0.57564604 * float(x[9]))+ (0.04866012 * float(x[10]))+ (0.12687762 * float(x[11]))+ (0.8429451 * float(x[12]))+ (-0.8662067 * float(x[13]))+ (-0.83440673 * float(x[14]))+ (-0.96776706 * float(x[15]))+ (0.6573018 * float(x[16]))+ (0.5482345 * float(x[17]))+ (0.73242116 * float(x[18]))+ (0.95013636 * float(x[19]))+ (0.59139794 * float(x[20]))+ (-0.08386715 * float(x[21]))+ (0.55444247 * float(x[22]))+ (-0.7703683 * float(x[23]))+ (0.27277258 * float(x[24]))+ (-0.72045743 * float(x[25]))+ (0.882087 * float(x[26]))+ (0.036350254 * float(x[27]))+ (-0.17818534 * float(x[28]))+ (-0.47874257 * float(x[29]))+ (0.5406451 * float(x[30]))+ (-0.09556831 * float(x[31]))+ (0.12892713 * float(x[32]))+ (-0.97032565 * float(x[33]))+ (0.22722684 * float(x[34]))+ (0.21526773 * float(x[35]))+ (0.22524853 * float(x[36]))+ (0.87850773 * float(x[37]))+ (0.354576 * float(x[38]))+ (-0.29053396 * float(x[39]))+ (-0.13515285 * float(x[40]))+ (0.3850043 * float(x[41]))+ (-0.8887445 * float(x[42]))+ (0.32452822 * float(x[43]))+ (0.33134022 * float(x[44]))+ (-0.5892877 * float(x[45]))+ (-0.751253 * float(x[46]))+ (-0.377651 * float(x[47]))+ (-0.28196397 * float(x[48]))+ (0.13074106 * float(x[49])))+ ((-0.1323011 * float(x[50]))+ (0.96741277 * float(x[51]))+ (-0.80476713 * float(x[52]))+ (-0.5904809 * float(x[53]))+ (-0.6853724 * float(x[54]))+ (0.2977374 * float(x[55]))+ (-0.5023721 * float(x[56]))+ (-0.07621841 * float(x[57]))+ (-0.520551 * float(x[58]))+ (-0.69206166 * float(x[59]))+ (-0.78915167 * float(x[60]))+ (0.30308056 * float(x[61]))+ (-0.7332277 * float(x[62]))+ (-0.61680806 * float(x[63]))+ (-0.2727873 * float(x[64]))+ (0.6324013 * float(x[65]))+ (-0.81569225 * float(x[66]))+ (0.6671159 * float(x[67]))+ (-0.8165891 * float(x[68]))+ (0.9441044 * float(x[69]))+ (-0.06996915 * float(x[70]))+ (0.9454815 * float(x[71]))+ (0.20236613 * float(x[72]))+ (0.47086123 * float(x[73]))+ (-0.9296265 * float(x[74]))+ (-0.44137022 * float(x[75]))+ (-0.7662828 * float(x[76]))+ (-0.4159815 * float(x[77]))+ (-0.7693289 * float(x[78]))+ (-0.3703751 * float(x[79]))+ (-0.17911491 * float(x[80]))+ (-0.877851 * float(x[81]))+ (0.37823957 * float(x[82]))+ (0.12585647 * float(x[83]))+ (-0.47583517 * float(x[84]))+ (0.040381983 * float(x[85]))+ (-0.8178148 * float(x[86]))+ (0.14383341 * float(x[87]))+ (0.85190994 * float(x[88]))+ (-0.36967602 * float(x[89]))+ (0.32771796 * float(x[90]))+ (-0.743842 * float(x[91]))+ (0.42469075 * float(x[92]))+ (-0.42782575 * float(x[93]))+ (-0.64058125 * float(x[94]))+ (0.1670342 * float(x[95]))+ (-0.9677696 * float(x[96]))+ (0.6486343 * float(x[97]))+ (-0.9989281 * float(x[98]))+ (0.34575832 * float(x[99])))+ ((-0.46762684 * float(x[100]))+ (0.46520182 * float(x[101]))) + 0.08951957), 0)
    h_1 = max((((0.71470225 * float(x[0]))+ (-0.64227486 * float(x[1]))+ (0.4304543 * float(x[2]))+ (1.2880145 * float(x[3]))+ (-0.660361 * float(x[4]))+ (-0.91868937 * float(x[5]))+ (0.47888845 * float(x[6]))+ (0.37217814 * float(x[7]))+ (1.1456162 * float(x[8]))+ (1.0659192 * float(x[9]))+ (1.074418 * float(x[10]))+ (-0.6145741 * float(x[11]))+ (0.8962811 * float(x[12]))+ (-0.49605325 * float(x[13]))+ (-0.04603353 * float(x[14]))+ (0.46691552 * float(x[15]))+ (-0.6263388 * float(x[16]))+ (0.08042546 * float(x[17]))+ (-0.6613098 * float(x[18]))+ (-0.7883973 * float(x[19]))+ (-0.7467887 * float(x[20]))+ (-0.25715172 * float(x[21]))+ (-0.90968984 * float(x[22]))+ (-0.01156035 * float(x[23]))+ (0.4665478 * float(x[24]))+ (-0.07055079 * float(x[25]))+ (-0.73332304 * float(x[26]))+ (-0.76416135 * float(x[27]))+ (-0.38471204 * float(x[28]))+ (-0.5701806 * float(x[29]))+ (1.047766 * float(x[30]))+ (-0.63791096 * float(x[31]))+ (0.6621385 * float(x[32]))+ (0.20209411 * float(x[33]))+ (0.18628888 * float(x[34]))+ (0.0447081 * float(x[35]))+ (0.9265692 * float(x[36]))+ (-0.21031974 * float(x[37]))+ (0.6417628 * float(x[38]))+ (0.089143425 * float(x[39]))+ (-0.19536963 * float(x[40]))+ (0.9504942 * float(x[41]))+ (0.18718645 * float(x[42]))+ (-0.76743317 * float(x[43]))+ (-0.21552326 * float(x[44]))+ (0.51601565 * float(x[45]))+ (0.2824987 * float(x[46]))+ (0.74221545 * float(x[47]))+ (-0.459808 * float(x[48]))+ (0.49873525 * float(x[49])))+ ((-0.112677224 * float(x[50]))+ (0.12918359 * float(x[51]))+ (0.9713275 * float(x[52]))+ (0.92221504 * float(x[53]))+ (0.76908845 * float(x[54]))+ (-0.24512832 * float(x[55]))+ (0.82465553 * float(x[56]))+ (0.326966 * float(x[57]))+ (-0.115933836 * float(x[58]))+ (-1.0093659 * float(x[59]))+ (0.4608475 * float(x[60]))+ (-0.07706306 * float(x[61]))+ (0.4790261 * float(x[62]))+ (0.36860883 * float(x[63]))+ (0.5211038 * float(x[64]))+ (-0.4590959 * float(x[65]))+ (0.54114217 * float(x[66]))+ (0.003415544 * float(x[67]))+ (-0.11730464 * float(x[68]))+ (0.3053843 * float(x[69]))+ (-1.0104511 * float(x[70]))+ (-0.7882212 * float(x[71]))+ (-0.48244464 * float(x[72]))+ (0.921183 * float(x[73]))+ (-0.7007767 * float(x[74]))+ (-0.42187876 * float(x[75]))+ (-0.85967135 * float(x[76]))+ (-0.33587813 * float(x[77]))+ (-0.3297818 * float(x[78]))+ (0.9158597 * float(x[79]))+ (0.8188286 * float(x[80]))+ (-0.13557741 * float(x[81]))+ (-0.9840016 * float(x[82]))+ (0.68966043 * float(x[83]))+ (0.4135147 * float(x[84]))+ (-0.5783444 * float(x[85]))+ (0.9246707 * float(x[86]))+ (-0.62235767 * float(x[87]))+ (0.8601555 * float(x[88]))+ (-0.41141513 * float(x[89]))+ (-0.0082784565 * float(x[90]))+ (0.7826313 * float(x[91]))+ (0.5299038 * float(x[92]))+ (0.8513173 * float(x[93]))+ (0.39731988 * float(x[94]))+ (-0.6997354 * float(x[95]))+ (0.8851524 * float(x[96]))+ (0.76458216 * float(x[97]))+ (0.42108622 * float(x[98]))+ (-0.6030035 * float(x[99])))+ ((-0.066407256 * float(x[100]))+ (-0.5557645 * float(x[101]))) + 0.66530776), 0)
    h_2 = max((((-0.10361247 * float(x[0]))+ (0.3360668 * float(x[1]))+ (0.8714864 * float(x[2]))+ (1.8940164 * float(x[3]))+ (-0.89294076 * float(x[4]))+ (-0.7463027 * float(x[5]))+ (-0.79338354 * float(x[6]))+ (-0.4347594 * float(x[7]))+ (-0.17141105 * float(x[8]))+ (0.0738317 * float(x[9]))+ (0.31487396 * float(x[10]))+ (0.54951334 * float(x[11]))+ (0.79668164 * float(x[12]))+ (0.9912336 * float(x[13]))+ (0.35369352 * float(x[14]))+ (0.67171746 * float(x[15]))+ (0.5432491 * float(x[16]))+ (-0.40845245 * float(x[17]))+ (0.73699254 * float(x[18]))+ (-0.90901864 * float(x[19]))+ (-0.98340434 * float(x[20]))+ (-0.43043852 * float(x[21]))+ (-0.022863893 * float(x[22]))+ (0.36391076 * float(x[23]))+ (-0.7084072 * float(x[24]))+ (0.42580798 * float(x[25]))+ (0.6658842 * float(x[26]))+ (-0.5591686 * float(x[27]))+ (-0.13109633 * float(x[28]))+ (0.0995582 * float(x[29]))+ (-0.022110283 * float(x[30]))+ (0.64111835 * float(x[31]))+ (0.9579575 * float(x[32]))+ (0.6219925 * float(x[33]))+ (-0.8094548 * float(x[34]))+ (0.63099563 * float(x[35]))+ (-0.8683767 * float(x[36]))+ (-0.4836836 * float(x[37]))+ (-0.6453234 * float(x[38]))+ (-0.60887194 * float(x[39]))+ (0.0732034 * float(x[40]))+ (0.37624267 * float(x[41]))+ (-0.056223933 * float(x[42]))+ (0.9267838 * float(x[43]))+ (-0.53878736 * float(x[44]))+ (0.5087454 * float(x[45]))+ (0.6872677 * float(x[46]))+ (0.079401456 * float(x[47]))+ (-0.063462965 * float(x[48]))+ (0.1931206 * float(x[49])))+ ((-0.14954104 * float(x[50]))+ (0.1108545 * float(x[51]))+ (-0.122352295 * float(x[52]))+ (-0.21767071 * float(x[53]))+ (0.42269063 * float(x[54]))+ (-0.78440267 * float(x[55]))+ (-0.628708 * float(x[56]))+ (0.74139285 * float(x[57]))+ (0.5671605 * float(x[58]))+ (-0.47612002 * float(x[59]))+ (-0.2838242 * float(x[60]))+ (0.81191635 * float(x[61]))+ (0.5893464 * float(x[62]))+ (0.15813206 * float(x[63]))+ (-0.5883829 * float(x[64]))+ (-0.9415394 * float(x[65]))+ (-0.16677566 * float(x[66]))+ (0.29366726 * float(x[67]))+ (0.62657493 * float(x[68]))+ (0.6802424 * float(x[69]))+ (-0.96265906 * float(x[70]))+ (-0.5039217 * float(x[71]))+ (-0.3604046 * float(x[72]))+ (-0.6479234 * float(x[73]))+ (0.61867934 * float(x[74]))+ (0.48720744 * float(x[75]))+ (0.058250055 * float(x[76]))+ (0.9662255 * float(x[77]))+ (-0.69130665 * float(x[78]))+ (0.81337804 * float(x[79]))+ (-0.6873364 * float(x[80]))+ (-0.6110509 * float(x[81]))+ (0.3761408 * float(x[82]))+ (-0.7617589 * float(x[83]))+ (0.19156761 * float(x[84]))+ (-0.08647221 * float(x[85]))+ (0.42023897 * float(x[86]))+ (0.91449183 * float(x[87]))+ (0.6169666 * float(x[88]))+ (0.7037049 * float(x[89]))+ (-0.8587028 * float(x[90]))+ (-0.26049078 * float(x[91]))+ (0.14198917 * float(x[92]))+ (-0.17844714 * float(x[93]))+ (-0.44281343 * float(x[94]))+ (-0.32226244 * float(x[95]))+ (0.964257 * float(x[96]))+ (-0.26606998 * float(x[97]))+ (-0.29384786 * float(x[98]))+ (-0.8074941 * float(x[99])))+ ((0.7817541 * float(x[100]))+ (-0.61752933 * float(x[101]))) + -0.20865032), 0)
    h_3 = max((((0.75039923 * float(x[0]))+ (-0.9446937 * float(x[1]))+ (0.3206772 * float(x[2]))+ (-0.17112225 * float(x[3]))+ (0.5825631 * float(x[4]))+ (0.44239622 * float(x[5]))+ (-0.039784387 * float(x[6]))+ (0.28772807 * float(x[7]))+ (0.0035462615 * float(x[8]))+ (0.6230369 * float(x[9]))+ (-0.047832027 * float(x[10]))+ (0.04631198 * float(x[11]))+ (-0.49895883 * float(x[12]))+ (0.21008603 * float(x[13]))+ (-0.39419037 * float(x[14]))+ (0.15456803 * float(x[15]))+ (-0.66064376 * float(x[16]))+ (-0.6810618 * float(x[17]))+ (-0.16594052 * float(x[18]))+ (-0.14636096 * float(x[19]))+ (-0.46378148 * float(x[20]))+ (-0.7368063 * float(x[21]))+ (-0.92157894 * float(x[22]))+ (-0.9495363 * float(x[23]))+ (-0.45689943 * float(x[24]))+ (-0.07629312 * float(x[25]))+ (0.45248657 * float(x[26]))+ (-0.0502566 * float(x[27]))+ (0.80810165 * float(x[28]))+ (-0.92956036 * float(x[29]))+ (-0.6386787 * float(x[30]))+ (-0.32297102 * float(x[31]))+ (0.15499237 * float(x[32]))+ (0.7054723 * float(x[33]))+ (-0.2995961 * float(x[34]))+ (-0.46402264 * float(x[35]))+ (-0.87622166 * float(x[36]))+ (0.642607 * float(x[37]))+ (-0.24066712 * float(x[38]))+ (0.1431004 * float(x[39]))+ (0.9671108 * float(x[40]))+ (-0.99681085 * float(x[41]))+ (-0.7090997 * float(x[42]))+ (0.558222 * float(x[43]))+ (0.61025494 * float(x[44]))+ (0.5384942 * float(x[45]))+ (0.07399778 * float(x[46]))+ (0.95771396 * float(x[47]))+ (-0.20763087 * float(x[48]))+ (0.2038874 * float(x[49])))+ ((-0.873262 * float(x[50]))+ (-0.1802851 * float(x[51]))+ (0.44500017 * float(x[52]))+ (-0.52252233 * float(x[53]))+ (0.8876552 * float(x[54]))+ (0.37356675 * float(x[55]))+ (-0.42484924 * float(x[56]))+ (0.53799784 * float(x[57]))+ (-0.83367044 * float(x[58]))+ (0.94954884 * float(x[59]))+ (-0.9014295 * float(x[60]))+ (0.86691177 * float(x[61]))+ (-0.49429226 * float(x[62]))+ (0.5156482 * float(x[63]))+ (-0.9998526 * float(x[64]))+ (-0.4915198 * float(x[65]))+ (0.49820122 * float(x[66]))+ (0.06467214 * float(x[67]))+ (-0.7700957 * float(x[68]))+ (-0.21274051 * float(x[69]))+ (-0.2489013 * float(x[70]))+ (0.1363245 * float(x[71]))+ (0.33595416 * float(x[72]))+ (0.6816605 * float(x[73]))+ (-0.005537206 * float(x[74]))+ (-0.21595657 * float(x[75]))+ (-0.7120469 * float(x[76]))+ (0.6096459 * float(x[77]))+ (0.42674083 * float(x[78]))+ (-0.1826452 * float(x[79]))+ (0.03686462 * float(x[80]))+ (0.3303657 * float(x[81]))+ (-0.6703888 * float(x[82]))+ (-0.9456044 * float(x[83]))+ (-0.3649926 * float(x[84]))+ (0.19117004 * float(x[85]))+ (-0.026787817 * float(x[86]))+ (0.38510925 * float(x[87]))+ (0.6393796 * float(x[88]))+ (-0.023115067 * float(x[89]))+ (-0.73146594 * float(x[90]))+ (0.701256 * float(x[91]))+ (0.14998065 * float(x[92]))+ (0.47987497 * float(x[93]))+ (0.4093293 * float(x[94]))+ (0.93642354 * float(x[95]))+ (-0.40938535 * float(x[96]))+ (0.41061354 * float(x[97]))+ (-0.26864734 * float(x[98]))+ (-0.20917855 * float(x[99])))+ ((-0.53881073 * float(x[100]))+ (-0.31197965 * float(x[101]))) + -0.6536135), 0)
    h_4 = max((((-0.14120533 * float(x[0]))+ (0.620616 * float(x[1]))+ (-0.76813596 * float(x[2]))+ (-1.5489578 * float(x[3]))+ (-0.056716528 * float(x[4]))+ (-0.62680477 * float(x[5]))+ (-0.2649881 * float(x[6]))+ (-1.1427373 * float(x[7]))+ (0.47181487 * float(x[8]))+ (-0.8590595 * float(x[9]))+ (0.13597436 * float(x[10]))+ (-0.5600607 * float(x[11]))+ (-0.95660514 * float(x[12]))+ (-0.79386264 * float(x[13]))+ (0.37034306 * float(x[14]))+ (0.8002359 * float(x[15]))+ (-0.8785785 * float(x[16]))+ (-0.4145453 * float(x[17]))+ (-0.1894944 * float(x[18]))+ (0.9515178 * float(x[19]))+ (-0.13075005 * float(x[20]))+ (0.62739265 * float(x[21]))+ (-0.49034756 * float(x[22]))+ (-0.14984654 * float(x[23]))+ (0.564683 * float(x[24]))+ (0.45853442 * float(x[25]))+ (-0.58456236 * float(x[26]))+ (-0.56396854 * float(x[27]))+ (0.058576215 * float(x[28]))+ (0.8170831 * float(x[29]))+ (-0.83337647 * float(x[30]))+ (0.502902 * float(x[31]))+ (0.43839937 * float(x[32]))+ (0.67734873 * float(x[33]))+ (0.7439388 * float(x[34]))+ (0.7453654 * float(x[35]))+ (0.8025387 * float(x[36]))+ (0.9173436 * float(x[37]))+ (-0.38334605 * float(x[38]))+ (0.8609703 * float(x[39]))+ (0.8781071 * float(x[40]))+ (0.26345074 * float(x[41]))+ (0.32531396 * float(x[42]))+ (-0.2910885 * float(x[43]))+ (0.5084994 * float(x[44]))+ (0.22127675 * float(x[45]))+ (0.62352765 * float(x[46]))+ (-0.65036416 * float(x[47]))+ (0.60465306 * float(x[48]))+ (0.44499335 * float(x[49])))+ ((0.097796574 * float(x[50]))+ (-0.4821237 * float(x[51]))+ (-0.8479572 * float(x[52]))+ (-0.0125261005 * float(x[53]))+ (-0.74262625 * float(x[54]))+ (0.49353212 * float(x[55]))+ (0.81096244 * float(x[56]))+ (0.60170174 * float(x[57]))+ (-0.36445037 * float(x[58]))+ (-0.67093396 * float(x[59]))+ (-0.9477716 * float(x[60]))+ (0.36554238 * float(x[61]))+ (-0.58018744 * float(x[62]))+ (0.50110936 * float(x[63]))+ (0.1806075 * float(x[64]))+ (-0.031492464 * float(x[65]))+ (0.5626809 * float(x[66]))+ (1.0114117 * float(x[67]))+ (-0.714368 * float(x[68]))+ (0.12495909 * float(x[69]))+ (-0.34685555 * float(x[70]))+ (-0.9522535 * float(x[71]))+ (-0.90453494 * float(x[72]))+ (0.17930786 * float(x[73]))+ (-0.8469417 * float(x[74]))+ (-0.8667347 * float(x[75]))+ (-0.67853105 * float(x[76]))+ (-0.0809821 * float(x[77]))+ (-0.018954247 * float(x[78]))+ (0.81643367 * float(x[79]))+ (-0.23813063 * float(x[80]))+ (-0.8454918 * float(x[81]))+ (-0.5232847 * float(x[82]))+ (0.8717379 * float(x[83]))+ (0.13171634 * float(x[84]))+ (-0.3602281 * float(x[85]))+ (0.81966144 * float(x[86]))+ (-0.509007 * float(x[87]))+ (0.68760914 * float(x[88]))+ (-0.7628482 * float(x[89]))+ (0.80326426 * float(x[90]))+ (0.7601901 * float(x[91]))+ (-0.41695762 * float(x[92]))+ (-0.95609426 * float(x[93]))+ (-0.38485953 * float(x[94]))+ (-0.6982789 * float(x[95]))+ (-0.5390187 * float(x[96]))+ (0.11089577 * float(x[97]))+ (-0.11278694 * float(x[98]))+ (0.29437327 * float(x[99])))+ ((0.18560307 * float(x[100]))+ (-0.46953163 * float(x[101]))) + 0.4302664), 0)
    o[0] = (1.4655473 * h_0)+ (-3.644378 * h_1)+ (-1.3372884 * h_2)+ (-0.7414596 * h_3)+ (1.9208359 * h_4) + 0.8481178

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        writer = csv.writer(sys.stdout, lineterminator=os.linesep)
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            writer.writerow(','.join(next(reader, None) + ["Prediction"]))
        
        
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            writer.writerow(row)


def Validate(arr):
    if n_classes == 2:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        outputs=[]
        for i, row in enumerate(arr):
            outputs.append(int(classify(arr[i, :-1].tolist())))
        outputs=np.array(outputs)
        correct_count = int(np.sum(outputs.reshape(-1) == arr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(arr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(arr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0
    else:
        numeachclass = {}
        count, correct_count = 0, 0
        preds = []
        for i, row in enumerate(arr):
            pred = int(classify(arr[i].tolist()))
            preds.append(pred)
            if pred == int(float(arr[i, -1])):
                correct_count += 1
                if int(float(arr[i, -1])) in numeachclass.keys():
                    numeachclass[int(float(arr[i, -1]))] += 1
                else:
                    numeachclass[int(float(arr[i, -1]))] = 0
            count += 1
        return count, correct_count, numeachclass, preds
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()


    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempdir + os.sep + "clean.csv"
        preprocessedfile = tempdir + os.sep + "prep.csv"
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x,y: x
        classmapping = {}


    #load file
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')


    #Normalize
    cleanarr = Normalize(cleanarr)


    #Transform
    if transform_true:
        if args.validate:
            trans = transform(cleanarr[:, :-1])
            cleanarr = np.concatenate((trans, cleanarr[:, -1].reshape(-1, 1)), axis = 1)
        else:
            cleanarr = transform(cleanarr)


    #Predict
    if not args.validate:
        Predict(cleanarr, args.headerless, preprocessedfile, get_key, classmapping)


    #Validate
    else: 
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
            #Correct Labels
            true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap=521
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





            def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
                #check for numpy/scipy is imported
                try:
                    from scipy.sparse import coo_matrix #required for multiclass metrics
                    try:
                        np.array
                    except:
                        import numpy as np
                except:
                    raise ValueError("Scipy and Numpy Required for Multiclass Metrics")
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


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)
