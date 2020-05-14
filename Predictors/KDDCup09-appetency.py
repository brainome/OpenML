#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53994/KDDCup09_appetency.arff -o Predictors/KDDCup09-appetency_QC.py -target APPETENCY -cm {'-1':0,'1':1} -stopat 98.22 -f QC -e 100 --yes
# Total compiler execution time: 0:01:22.48. Finished on: Apr-22-2020 10:19:15.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                98.22%
Model accuracy:                     98.23% (49117/50000 correct)
Improvement over best guess:        0.01% (of possible 1.78%)
Model capacity (MEC):               896 bits
Generalization ratio:               54.81 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     97.31% (48654/50000)
True Positives:                     0.93% (463/50000)
False Negatives:                    0.85% (427/50000)
False Positives:                    0.91% (456/50000)
True Pos. Rate/Sensitivity/Recall:  0.52
True Neg. Rate/Specificity:         0.99
Precision:                          0.50
F-1 Measure:                        0.51
False Negative Rate/Miss Rate:      0.48
Critical Success Index:             0.34

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

from bisect import bisect_left
# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "KDDCup09_appetency.csv"


#Number of attributes
num_attr = 230
n_classes = 2


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="APPETENCY"


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
energy_thresholds = array([326689051021.8806, 326719819773.798, 327525789698.00824, 327540141893.1026, 327559176620.567, 327572114312.3235, 329321363630.8346, 329332983969.8396, 329421989237.02454, 329426342034.60693, 329801546644.1974, 329805593373.742, 329884886713.48126, 329887671312.0152, 330133033538.7396, 330141182485.02454, 330165548804.7935, 330170785432.78625, 330180798074.2436, 330181636393.10803, 330250561313.4678, 330253542753.0232, 330629530697.1844, 330632521503.55334, 330903747501.1119, 330907046963.5681, 331604295372.0121, 331606019269.575, 331607066029.3671, 331609742147.16455, 331726661226.9388, 331730234931.89514, 331821959367.834, 331822735973.8566, 331824129748.52893, 331827609093.7018, 332039045237.8273, 332040219586.2147, 332652232764.4775, 332656751843.62634, 332670296412.57745, 332674129056.04517, 332687666980.9696, 332688453837.7137, 332751086405.1849, 332752197543.90015, 333071962825.6334, 333073586926.89197, 333142408329.67633, 333143461502.4292, 333151988889.2848, 333153264882.85547, 333212733986.8142, 333214001280.6494, 333261110238.1793, 333262959388.9471, 333481236634.5796, 333481849034.058, 333488528856.8997, 333492350977.666, 333564136159.82434, 333567571610.83923, 333838185977.8569, 333842787827.2501, 333863045937.20135, 333865822853.35486, 333901936368.0886, 333902982536.1107, 333954940385.8648, 333956932090.01917, 334079095428.31445, 334079532701.43835, 334196486146.6655, 334199532615.5191, 334259522734.12646, 334262262177.61584, 334333780412.01514, 334336204994.15344, 334414914726.5971, 334414947055.89307, 334494769924.2694, 334495550855.6547, 334714898385.0645, 334717098032.0574, 334771866231.07043, 334774592971.6946, 334876981528.03796, 334877352247.6474, 334908289437.67944, 334909797037.03424, 335012446150.37085, 335013255435.6638, 335049890203.03516, 335050343729.159, 335056627786.47205, 335058527876.39966, 335112780355.098, 335113178141.72876, 335157659963.5298, 335160069707.59033, 335167725490.2678, 335169607784.95874, 335172296095.23065, 335174159802.59534, 335195907027.06323, 335197448726.42554, 335283309857.13184, 335283962720.4722, 335339909781.4132, 335342051777.59924, 335354157381.6343, 335355148123.0502, 335454775109.24066, 335455269037.3515, 335613433829.5078, 335615206449.276, 335777249686.2987, 335777513019.3165, 335902008146.643, 335902957454.1098, 335934591772.0825, 335937606697.90186, 335954762020.3358, 335955711116.8082, 335999458865.5232, 336001282641.66473, 336065791015.80664, 336068258259.1393, 336090373938.86365, 336090440609.73694, 336155426311.24805, 336156610298.96484, 336210606958.66797, 336212493688.4454, 336277744717.5608, 336278095907.0762, 336370114424.9408, 336371185664.4791, 336377005045.97565, 336377842524.949, 336383954307.3833, 336384651929.7295, 336397530129.55115, 336397686793.22186, 336480517007.1104, 336480704973.92786, 336589416757.7101, 336589590348.9711, 336630163801.806, 336631375272.09595, 336660855498.9319, 336661851746.8032, 336713354617.2571, 336713967173.55164, 336897312965.2605, 336897592635.9715, 336953808992.3898, 336957085697.15454, 336985501557.4093, 336987818357.96924, 336996879161.3465, 336998233204.4744, 337014137004.6376, 337015020551.9574, 337031344294.2935, 337031709677.3325, 337070239520.1123, 337071391620.4269, 337121572507.9866, 337123765131.9155, 337128182195.3626, 337129318874.23474, 337130558712.50244, 337131159627.55914, 337195616932.6538, 337196084754.3918, 337246002012.6402, 337246650096.97565, 337258184053.7459, 337258931668.8075, 337476388401.9786, 337476593406.8611, 337566604669.9221, 337566824951.6758, 337575333939.0822, 337576783820.98804, 337711811922.874, 337712778360.61115, 337718038591.77136, 337718435919.8939, 337749207746.75793, 337749745022.3053, 337817901982.84, 337820295442.58496, 337858742346.26184, 337859388862.08276, 337958358106.725, 337958653883.21826, 337976776198.6072, 337977282757.4978, 337990024519.2728, 337992033041.7407, 338023177447.9773, 338023558760.8644, 338027934245.8562, 338029270513.9298, 338118692304.39514, 338119437525.12854, 338128372673.86523, 338129364888.3973, 338190439114.67236, 338190832842.39734, 338191214364.2003, 338192358369.19543, 338225107917.7302, 338226995530.46533, 338250675118.62103, 338251558438.9852, 338555820501.54626, 338556343526.58325, 338575700726.2531, 338577702619.40845, 338587046607.23737, 338588241290.08105, 338700651093.8838, 338701799709.6594, 338704573582.8255, 338705481927.79944, 338715576175.1903, 338715943073.9332, 338755512697.0952, 338755985201.11005, 338792334204.60583, 338793521514.897, 338808858321.9831, 338809346355.11847, 338871275658.20605, 338871822699.70483, 338880591617.8395, 338881595307.07263, 338912014795.48553, 338912496395.42053, 338999443554.6435, 339000785596.0759, 339047419046.05347, 339048050256.55383, 339131568236.9153, 339133431698.32886, 339192505765.5941, 339192990152.90356, 339278189751.3262, 339278817569.8312, 339410278634.432, 339410578169.89355, 339451400643.5045, 339453247144.3535, 339514056724.6477, 339514249566.2657, 339533588737.32385, 339533818195.62103, 339637933988.238, 339638467701.1521, 339660921233.0029, 339661674883.5241, 339672596024.4163, 339672874170.43024, 339724870510.36365, 339725677545.5713, 339733698461.184, 339734014813.6106, 339826936876.8967, 339828056293.52325, 339831718884.98804, 339833441240.18036, 339864782682.8944, 339865720528.89636, 339900640420.11505, 339900658052.302, 339952431535.84125, 339952523370.7928, 339975046375.4927, 339975228394.4617, 339981236869.67444, 339981739397.8896, 340020354250.6025, 340020950379.77844, 340030814095.26935, 340034194133.7554, 340047487765.7995, 340048905225.1498, 340059012965.7954, 340059371834.7068, 340112499294.3943, 340112773073.64514, 340126390069.2416, 340127717835.4807, 340168757586.64825, 340169060313.9641, 340217113258.1824, 340218552711.604, 340262992867.65424, 340263936778.20483, 340308678776.37115, 340309087417.4193, 340326239451.23145, 340326364967.0702, 340372030087.47754, 340372169469.0298, 340433429647.64966, 340434587006.2644, 340444646174.0517, 340444784549.6819, 340448176065.8185, 340448703088.202, 340455174468.5046, 340456816679.3382, 340494690036.1704, 340495263555.47363, 340524438657.324, 340525025974.51483, 340598091432.03015, 340599428102.38904, 340614144051.5289, 340614297471.2088, 340738653260.8463, 340738750309.20935, 340945851587.1382, 340946962600.37683, 341014603166.0811, 341015452961.4902, 341065710507.1589, 341066132946.0142, 341088303672.8975, 341088589857.35284, 341090411412.55414, 341091250811.6503, 341096702572.5351, 341097182697.051, 341226017302.3951, 341226629755.0159, 341236505151.4607, 341237135066.8157, 341302528538.0515, 341302665076.6332, 341308643407.05756, 341310302261.01495, 341344209150.1113, 341344528821.9127, 341426192298.19775, 341426236996.615, 341444887064.9306, 341445742963.4618, 341710942059.60876, 341711393348.71936, 341791864941.8378, 341792283315.4198, 341810432832.75385, 341810981373.09094, 341821718642.0853, 341822614435.0973, 341843419513.421, 341844028243.91174, 341855606969.4637, 341855933556.9981, 341888976850.4517, 341889951928.625, 341915786602.0255, 341915861628.9652, 342024309268.0375, 342024642239.9853, 342051836059.9903, 342052586937.26416, 342063196896.96906, 342065304745.36554, 342066307616.17285, 342066405653.527, 342072560236.46924, 342072879294.36115, 342134143459.53143, 342134504736.07837, 342173164591.4019, 342173886012.41205, 342201526861.9396, 342201802265.1742, 342248768514.8164, 342249300347.5855, 342257816824.99805, 342259089814.4318, 342262916884.95447, 342263439973.65497, 342267189082.9724, 342269193423.8751, 342423782407.98413, 342424131562.7263, 342482209459.01526, 342482499913.2514, 342495356947.9641, 342495822688.24744, 342529342892.6461, 342529758864.5891, 342549869165.0178, 342551007254.48, 342702392755.007, 342702463278.02747, 342710129205.00494, 342710415337.1603, 342720056276.3561, 342721295094.28986, 342794540914.5696, 342795988163.79614, 342814310738.62915, 342815422368.5973, 342866794118.6804, 342867302343.00134, 342874019650.16833, 342874879210.3075, 342935225642.08124, 342935871304.36646, 343008633005.2935, 343009445643.5638, 343012668092.2962, 343013687616.269, 343022440604.961, 343023519223.3335, 343028750990.458, 343029345411.4049, 343031953490.4691, 343032596783.81555, 343054256036.09, 343054869281.04785, 343089230367.2129, 343089660724.3146, 343094977972.7042, 343095267416.58575, 343104847042.3805, 343105984473.0469, 343120316026.3319, 343121032937.06165, 343133551460.15015, 343134009531.9601, 343134777330.0186, 343135388653.0408, 343173921453.08716, 343174388485.4282, 343187591873.7594, 343188161950.379, 343226131188.9381, 343227700753.23096, 343235155095.97925, 343235376747.21814, 343300216455.1804, 343301376028.6811, 343308761599.0608, 343309563148.82794, 343311673510.8657, 343312081651.53455, 343507820711.1156, 343508128653.2317, 343541529927.70197, 343542002633.3165, 343562835660.1731, 343563972625.2873, 343590886526.14233, 343591187794.1571, 343612564386.54407, 343613259115.146, 343626133054.12305, 343626874848.43756, 343671857108.3163, 343672380255.70154, 343801020929.09143, 343801834819.9881, 343878319915.3052, 343879543094.25244, 343898196335.8417, 343899705706.90857, 343912329742.5684, 343912859337.8049, 343916499969.84705, 343917439000.26556, 343927932164.70337, 343930436315.9428, 343944511250.0419, 343946366130.2633, 343984300945.7057, 343985153335.8911, 344019555755.44617, 344020004160.9929, 344057480565.05493, 344057668678.54614, 344207682601.37244, 344208194972.84753, 344256031087.867, 344257652329.8828, 344283687750.5784, 344283756432.89026, 344305198425.2062, 344305752367.4988, 344317595494.8912, 344317828637.01636, 344320044414.01697, 344320317867.666, 344326473304.6012, 344327837363.02454, 344359965220.30414, 344360635627.2669, 344367970232.67676, 344368561053.08545, 344460610705.8716, 344461445333.4618, 344469451923.5331, 344469948818.5021, 344470661206.84546, 344470838989.88184, 344472424504.55774, 344473329283.1614, 344493925550.65173, 344494935076.73267, 344513085812.199, 344514834630.16003, 344567345627.68317, 344568440507.9548, 344642489286.73193, 344642742243.4523, 344707507141.26355, 344708013198.8036, 344727374127.18726, 344729276440.6054, 344831112803.7564, 344832095724.0553, 344838619138.3125, 344839060531.57904, 344880917299.8154, 344881618753.8647, 344887990301.758, 344888508479.0239, 344899467107.57117, 344900247659.2905, 344963580582.5989, 344963832127.7136, 344974343795.9796, 344975533096.04724, 344975964224.6367, 344976812460.69324, 345161527111.3356, 345161999685.2273, 345225957761.5373, 345228329980.579, 345233695416.8517, 345234760642.29297, 345322258346.8321, 345323154816.29004, 345324130817.1204, 345324406392.2226, 345397667755.02185, 345399118942.5485, 345406684602.7709, 345407580925.0059, 345431948804.5874, 345432242128.4969, 345527765194.9455, 345528040080.38794, 345600204201.11273, 345600999444.3109, 345601438962.7843, 345602382032.59845, 345612388256.5625, 345613363711.2667, 345625861608.0475, 345627450271.7174, 345662000921.89136, 345663425839.65936, 345740191289.89075, 345741304850.4274, 345770222478.38635, 345771547792.1532, 345786429129.5781, 345786606167.74744, 345788010089.47076, 345789164016.11633, 345844464816.3977, 345845196770.98535, 345900994424.28107, 345901616812.34216, 345948320258.802, 345948428991.657, 345985454622.9989, 345986409313.4922, 346094403270.4431, 346095738786.68756, 346134599069.81946, 346135116114.46375, 346152514653.31586, 346154027840.6578, 346207983524.08093, 346208302958.04663, 346214159364.2409, 346214423292.1155, 346281915363.0033, 346283520043.98865, 346340674101.7956, 346341564061.4839, 346348946993.89874, 346350661122.1247, 346365941436.63104, 346366230399.55774, 346367696126.5487, 346368271096.0675, 346402175893.78723, 346402388095.4447, 346406288308.38153, 346407434878.4418, 346425171422.65204, 346425897480.6076, 346561910639.594, 346562662716.3511, 346649403572.19543, 346650582189.6345, 346658730951.28455, 346659841861.8172, 346727331929.45386, 346727655730.15967, 346732655120.7455, 346733325386.72974, 346778786285.5498, 346779349704.4788, 346853197343.3566, 346853702286.2914, 346884344654.4741, 346885545685.88464, 346929024307.438, 346929621079.9301, 346976532765.5775, 346977734501.9975, 347019775937.07996, 347020162965.64795, 347061500967.7714, 347061828463.75635, 347089527572.2738, 347090962637.1334, 347091764471.2347, 347092063074.9979, 347112529793.9645, 347113618508.47546, 347123198159.4785, 347124931432.0039, 347152393894.5714, 347153407411.9253, 347190966039.6145, 347191117439.6542, 347200502917.11707, 347201236936.1899, 347263581616.5831, 347264047419.38696, 347309393834.6348, 347310341004.0012, 347394255210.2867, 347395787301.3802, 347472981934.43115, 347473484201.74023, 347555810345.94946, 347556525631.82825, 347740630498.4236, 347741547405.5069, 347742927649.00745, 347743037949.0397, 347761999654.3682, 347763076229.1306, 347800139340.98584, 347801075856.057, 347839778118.55634, 347840907772.90576, 347924778847.43005, 347925451362.1911, 347955682444.8791, 347956757508.3331, 348010505303.4729, 348010708866.9955, 348014133038.3572, 348014216625.8884, 348078338350.64325, 348079366260.22266, 348196625655.0087, 348197627580.7395, 348290161080.23254, 348291513659.95544, 348310759085.09607, 348313023030.2804, 348392487347.1542, 348392763339.54395, 348543385500.1279, 348543947616.0417, 348559644070.3691, 348560678010.81647, 348575199898.3318, 348576077678.6155, 348628827292.5938, 348629749978.4798, 348722348555.3397, 348724007682.1061, 348807841086.3502, 348808082602.9452, 348820201920.0388, 348820404107.47705, 348829576057.0131, 348831507732.8971, 348834078644.3702, 348835984474.4853, 348878730507.7914, 348880399135.44946, 348902849318.4293, 348905383756.28174, 348913020415.2621, 348914114636.9098, 348959514627.61426, 348960191049.1686, 348960938006.0188, 348962331173.55273, 349102096730.76794, 349103053527.11017, 349103546646.7314, 349103915721.95605, 349110242760.72705, 349112227428.1504, 349196023365.5701, 349196793023.52246, 349224736831.11774, 349225279148.32477, 349228322238.5844, 349230433328.7889, 349292428247.80725, 349293774038.4674, 349402534047.1176, 349403304317.7211, 349406514343.5875, 349406832467.0909, 349435080225.18384, 349435270736.4794, 349723984121.87915, 349726425838.24133, 349936522971.55115, 349936830292.2678, 349943935139.334, 349944417863.39636, 349948548916.4381, 349949670306.23376, 349958791698.95215, 349959646309.1609, 350111634550.0542, 350112130208.5928, 350314553509.3694, 350315671980.1051, 350430168668.4205, 350430881572.2607, 350485865621.9752, 350487489761.81287, 350591945900.7477, 350592363941.05383, 350617389631.5895, 350618483412.08374, 350619106596.04297, 350619521318.6498, 350651643625.55414, 350652645856.98773, 350721189304.8008, 350722026043.93915, 350766284309.0168, 350767289609.73816, 350836206289.98987, 350837801062.76953, 350881599501.4503, 350882679658.7758, 350943533151.01575, 350943695086.5396, 350989709263.1289, 350990264620.1424, 351090280618.182, 351090823413.2626, 351094234280.11426, 351095462238.9116, 351475028913.9216, 351476300594.3846, 351659552775.4037, 351661227510.067, 351723610354.9011, 351724999059.1663, 351741164104.27893, 351743255380.66675, 351780237744.87354, 351780284010.3009, 351800262424.9534, 351801034422.8987, 351853822544.6592, 351854286101.196, 351935808069.2625, 351937751032.00977, 351961213200.6692, 351964602533.68036, 351986448830.98566, 351986852338.40356, 352057019477.9457, 352057560618.07294, 352098352253.30255, 352099674011.1497, 352125582614.8612, 352127425808.56995, 352605993060.9215, 352608355554.46643, 352669202865.3187, 352670862486.823, 352813970736.29987, 352814374294.18835, 353062611163.3384, 353064608974.8987, 353244903297.4205, 353247489268.98315, 353403237996.7335, 353406298481.2803, 353409376560.54706, 353412231923.67834, 353472997529.838, 353477063359.1178, 353646184798.7506, 353647959979.7123, 353661388977.7188, 353667204770.72906, 353920556312.43506, 353924358685.7138, 354259298586.7321, 354259636299.79205, 354289983917.4197, 354294656165.3506, 354528889881.2117, 354529393984.6194, 354620925805.93665, 354629684219.3008, 354695535923.7565, 354696763091.75305, 355131035077.85876, 355132952021.397, 355348544557.9865, 355353361756.2565, 355362527472.4711, 355364167614.14215, 355625694066.0438, 355628515576.6679, 356262919909.69415, 356264629022.02295, 356300011299.8739, 356303652959.5959, 356334445980.6255, 356336998611.7948, 356488655200.7337, 356501123021.9178, 356886647858.7902, 356894920067.38007, 357285200967.8287, 357289216503.202, 357377893437.1661, 357383618764.71814, 357575943028.78265, 357580608086.34216, 358556892326.568, 358561129918.90784, 359024085291.0371, 359029625339.4159, 359035832089.20715, 359038349761.73303, 359065390741.38416, 359066900090.6919, 359871622590.88995, 359877420694.1782, 360038177014.255, 360047523002.1499, 361229204118.6178, 361238032669.33484, 361284578656.04517, 361295447683.5508, 361470485137.23474, 361495803442.16516, 363146838147.1841, 363165203554.80695, 364430223672.02673, 364471060249.6515, 364595681985.38904, 364600034383.204, 365336313721.5884, 365346621466.9719, 365389096442.01886, 365416776098.21985, 365927408955.84174, 365945227557.30493, 367523390670.57245, 367552168739.31555, 368605082875.7377, 368689734619.7571])
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

numthresholds=896



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
        return count, correct_count, numeachclass, outputs, cleanarr[:,-1]


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

    #Predict or Validate?
    if not args.validate:
        Predict(cleanfile, get_key, args.headerless, preprocessedfile, classmapping)


    else:
        
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



    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)


