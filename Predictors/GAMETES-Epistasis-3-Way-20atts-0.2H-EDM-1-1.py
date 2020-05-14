#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/4965217/GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1.arff -o Predictors/GAMETES-Epistasis-3-Way-20atts-0.2H-EDM-1-1_NN.py -target class -stopat 62.82 -f NN -e 20 --yes
# Total compiler execution time: 0:03:45.33. Finished on: Apr-21-2020 14:01:16.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     65.25% (1044/1600 correct)
Improvement over best guess:        15.25% (of possible 50.0%)
Model capacity (MEC):               210 bits
Generalization ratio:               4.97 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     31.56% (505/1600)
True Positives:                     33.69% (539/1600)
False Negatives:                    16.31% (261/1600)
False Positives:                    18.44% (295/1600)
True Pos. Rate/Sensitivity/Recall:  0.67
True Neg. Rate/Specificity:         0.63
Precision:                          0.65
F-1 Measure:                        0.66
False Negative Rate/Miss Rate:      0.33
Critical Success Index:             0.49

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
TRAINFILE = "GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 20
n_classes = 2

mappings = []
list_of_cols_to_normalize = []

transform_true = True

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
        mean = np.array([0.9395833333333333, 0.58125, 0.40625, 0.215625, 0.46458333333333335, 0.23333333333333334, 0.15104166666666666, 0.7020833333333333, 0.309375, 0.27291666666666664, 0.5114583333333333, 0.3885416666666667, 0.11145833333333334, 0.10104166666666667, 0.5833333333333334, 0.5833333333333334, 0.8114583333333333, 0.3885416666666667, 0.3958333333333333, 0.39895833333333336])
        components = np.array([array([ 8.82236634e-01,  1.05510970e-01,  9.78256383e-02,  1.73866861e-02,
        4.01107805e-02, -1.26982785e-02, -1.14721709e-02, -1.48660353e-01,
        4.00913400e-03, -2.88915757e-03, -2.14387959e-01,  3.19675625e-04,
        1.26405981e-02,  3.81696221e-03, -5.64520000e-02,  6.85152300e-02,
        3.44265468e-01,  2.16290210e-02,  3.68897741e-02, -4.75662791e-02]), array([ 0.31875559,  0.10159601,  0.03701101,  0.02634353,  0.00606509,
        0.04803529,  0.01700453,  0.66858333,  0.01367682,  0.03139752,
        0.08542635, -0.1200994 , -0.00375857,  0.00502729,  0.29553115,
        0.21077759, -0.50025876, -0.16877989, -0.03753524, -0.03896565]), array([-0.26065321,  0.08817915,  0.11955124,  0.00320051, -0.02576976,
        0.04073926, -0.03791026,  0.59646543,  0.04015088,  0.02519404,
       -0.2431574 , -0.09694767,  0.01633792, -0.01890775, -0.09950156,
        0.18375825,  0.65220519,  0.00892132,  0.09955159,  0.00496236]), array([ 0.06494918, -0.20826618, -0.01029632,  0.02698444, -0.01575199,
        0.04192587, -0.00905774,  0.15661357, -0.00975645,  0.0036973 ,
        0.05361717,  0.06908719,  0.02786197, -0.02344296,  0.61229295,
       -0.66863354,  0.22976439,  0.1853244 , -0.07943774, -0.02129977]), array([-0.13678112,  0.22512503,  0.14421308,  0.09144393, -0.26119589,
        0.01300666, -0.05501404, -0.32686275, -0.03046539,  0.03382435,
       -0.11568788, -0.14406591, -0.02409155, -0.01314686,  0.64512302,
        0.45301774,  0.04934152,  0.15811984,  0.18094028,  0.06814092]), array([-0.01539015,  0.73951729, -0.18715359,  0.03580112,  0.11406036,
       -0.0537888 ,  0.01107454,  0.02890095, -0.03280578,  0.05905569,
        0.38921069,  0.3424554 ,  0.05139215,  0.01988001,  0.02588166,
       -0.15357599,  0.11076782, -0.0877636 ,  0.28090098, -0.01465044]), array([ 0.14294287, -0.43806032, -0.16645011,  0.01300692, -0.41613494,
       -0.0280515 , -0.03380043,  0.1046553 , -0.00358582,  0.02563174,
        0.55048716,  0.28508206,  0.00661901,  0.03029014, -0.05518222,
        0.2779793 ,  0.15727976,  0.16021838,  0.23144031,  0.05930615]), array([ 0.05083365,  0.10176488,  0.157381  , -0.01302555,  0.30882788,
       -0.01103854, -0.04164278,  0.0024873 ,  0.18202242, -0.01485053,
        0.443191  , -0.42391297,  0.02020248,  0.01702203, -0.05994788,
        0.04203338,  0.08121613,  0.38227485, -0.22064509,  0.49793434]), array([ 0.00828758,  0.02700765,  0.69241527, -0.01527832, -0.20689232,
       -0.01637217, -0.02461329,  0.06250229, -0.296857  ,  0.10747789,
       -0.07625218,  0.18960714, -0.032891  ,  0.01101905, -0.1677163 ,
       -0.22857381, -0.1876805 ,  0.06880402,  0.29472824,  0.34943272]), array([-0.04380148, -0.33203124,  0.24267312,  0.01339678,  0.63327452,
       -0.07452157, -0.02078083, -0.09426854,  0.02449204,  0.09481323,
        0.14022334,  0.17126319,  0.05448739,  0.0090252 ,  0.22325779,
        0.17980718,  0.12248211, -0.44414931,  0.23325383,  0.04870933]), array([-2.76436367e-03, -6.32664147e-02, -2.15038913e-01, -1.52248707e-02,
        3.61727492e-01, -3.86427377e-02, -2.17907085e-02,  1.13177316e-01,
        1.42208208e-01,  3.19737808e-02, -3.58744117e-01,  3.90176842e-01,
        4.49507811e-02,  5.23312560e-04,  6.35234284e-03,  1.24266791e-01,
       -2.06227477e-01,  6.14832181e-01,  2.29290821e-01,  1.05286770e-01]), array([ 0.05327013, -0.05443605, -0.46274763,  0.04874466, -0.11698657,
        0.01021023, -0.02700739,  0.00362472,  0.00109908, -0.07040725,
       -0.23493663, -0.12662937,  0.04316596, -0.02580388,  0.0336593 ,
       -0.11003796, -0.00338267, -0.34334174,  0.23816931,  0.7046817 ]), array([ 0.01356785, -0.05508113,  0.04967381, -0.05527405, -0.0266588 ,
       -0.0932418 , -0.0131131 , -0.01758317,  0.408483  , -0.13350798,
        0.04575861, -0.44486467, -0.00402564, -0.02455052, -0.08271905,
       -0.2088889 , -0.10056358,  0.05167047,  0.67283937, -0.28302052]), array([-1.36411458e-02,  5.74660847e-02,  1.91707075e-01,  6.20181604e-02,
       -2.19402578e-01,  7.39244868e-04, -4.45174744e-02, -3.45490702e-02,
        8.20111831e-01,  2.16207753e-01, -6.88057649e-02,  2.99344426e-01,
       -2.61432994e-03,  1.98991377e-02,  9.16952054e-03, -3.25802278e-02,
       -2.33751998e-02, -1.66214332e-01, -2.12661044e-01,  1.26423967e-01]), array([-0.0084026 ,  0.02215206,  0.12920739,  0.12182435,  0.04025908,
        0.595602  ,  0.04322597, -0.01241205,  0.101104  , -0.74659927,
        0.03855502,  0.17839388, -0.04714926,  0.04857942,  0.00659699,
        0.03545202,  0.00110554, -0.02619557,  0.02422464,  0.03764147]), array([ 0.00856845, -0.03947193, -0.08152827,  0.31430547,  0.05764397,
        0.71483939, -0.00225446, -0.05755134, -0.04109803,  0.56802656,
        0.02650776, -0.13006727,  0.04846547, -0.00424663, -0.09234329,
       -0.04080207, -0.01907949,  0.03973341,  0.12134363, -0.0641006 ]), array([-0.01607301, -0.02922322,  0.02306366,  0.91414636,  0.01646209,
       -0.31801962, -0.16370282,  0.03222484, -0.0337184 , -0.12913883,
       -0.00898138, -0.01989966,  0.05455053,  0.01877499, -0.07601383,
       -0.03372021, -0.02618306,  0.02478951, -0.03676578, -0.04756043])])
        whiten = False
        explained_variance = np.array([0.5499526577328048, 0.5098313724095801, 0.5008406867854533, 0.44570816795628854, 0.42646499517198827, 0.4053071568223997, 0.39005450892661875, 0.34911282352270995, 0.3338547252762126, 0.31627536846945853, 0.31447909305912963, 0.3025804984728883, 0.2706282482280905, 0.24434337829202768, 0.2210645778942676, 0.21363259881075777, 0.18712522005702423])
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
    target="class"


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
    h_0 = max((((-1.6577111 * float(x[0]))+ (-2.704603 * float(x[1]))+ (-0.9006845 * float(x[2]))+ (-1.2574998 * float(x[3]))+ (0.8715662 * float(x[4]))+ (-0.096956186 * float(x[5]))+ (-2.7436888 * float(x[6]))+ (1.3537049 * float(x[7]))+ (-0.94615924 * float(x[8]))+ (0.26509818 * float(x[9]))+ (-2.5938022 * float(x[10]))+ (0.19362585 * float(x[11]))+ (-2.225344 * float(x[12]))+ (1.9643732 * float(x[13]))+ (0.37359038 * float(x[14]))+ (-2.8569477 * float(x[15]))+ (2.74011 * float(x[16]))) + -2.7852192), 0)
    h_1 = max((((7.421384 * float(x[0]))+ (4.087008 * float(x[1]))+ (-2.2702537 * float(x[2]))+ (0.4785121 * float(x[3]))+ (-8.2044325 * float(x[4]))+ (2.7804887 * float(x[5]))+ (-0.83362776 * float(x[6]))+ (-1.3979228 * float(x[7]))+ (4.1708503 * float(x[8]))+ (2.941835 * float(x[9]))+ (1.6504474 * float(x[10]))+ (2.0949728 * float(x[11]))+ (-1.2338556 * float(x[12]))+ (-0.45437747 * float(x[13]))+ (-3.4931505 * float(x[14]))+ (5.341755 * float(x[15]))+ (-3.434721 * float(x[16]))) + 0.06958449), 0)
    h_2 = max((((-0.26930368 * float(x[0]))+ (0.1622675 * float(x[1]))+ (-0.21915418 * float(x[2]))+ (0.051206246 * float(x[3]))+ (-0.0052334685 * float(x[4]))+ (-0.97386867 * float(x[5]))+ (-1.0694906 * float(x[6]))+ (2.1938663 * float(x[7]))+ (-0.85751104 * float(x[8]))+ (-1.5042887 * float(x[9]))+ (-0.7358557 * float(x[10]))+ (-0.061434723 * float(x[11]))+ (-1.4675938 * float(x[12]))+ (-0.00663316 * float(x[13]))+ (-0.18061927 * float(x[14]))+ (-0.7098305 * float(x[15]))+ (-0.32067063 * float(x[16]))) + -1.9551141), 0)
    h_3 = max((((-0.6486297 * float(x[0]))+ (-0.6818927 * float(x[1]))+ (0.49778125 * float(x[2]))+ (0.18921815 * float(x[3]))+ (0.44445175 * float(x[4]))+ (-0.31146374 * float(x[5]))+ (0.55168384 * float(x[6]))+ (-0.46028358 * float(x[7]))+ (0.10364726 * float(x[8]))+ (-0.92210233 * float(x[9]))+ (0.038503353 * float(x[10]))+ (-0.07328653 * float(x[11]))+ (0.92834914 * float(x[12]))+ (-0.4953761 * float(x[13]))+ (0.7273038 * float(x[14]))+ (-0.43668985 * float(x[15]))+ (-0.030627668 * float(x[16]))) + 0.7662381), 0)
    h_4 = max((((1.9788437 * float(x[0]))+ (1.0511026 * float(x[1]))+ (-0.4329556 * float(x[2]))+ (0.1269719 * float(x[3]))+ (-2.2025177 * float(x[4]))+ (0.5662535 * float(x[5]))+ (-0.14831138 * float(x[6]))+ (-0.38190496 * float(x[7]))+ (0.675879 * float(x[8]))+ (0.86627775 * float(x[9]))+ (0.6941096 * float(x[10]))+ (-0.11018073 * float(x[11]))+ (-0.01766189 * float(x[12]))+ (-0.47218752 * float(x[13]))+ (-0.61760026 * float(x[14]))+ (1.0289053 * float(x[15]))+ (-1.0883083 * float(x[16]))) + -0.11173625), 0)
    h_5 = max((((-0.0074136243 * float(x[0]))+ (0.114984155 * float(x[1]))+ (-0.014755664 * float(x[2]))+ (-0.40818247 * float(x[3]))+ (-0.4755734 * float(x[4]))+ (0.054205082 * float(x[5]))+ (-0.6813656 * float(x[6]))+ (0.5013215 * float(x[7]))+ (-0.41355988 * float(x[8]))+ (0.50831336 * float(x[9]))+ (-0.41473156 * float(x[10]))+ (0.76367897 * float(x[11]))+ (-1.1559238 * float(x[12]))+ (0.33961043 * float(x[13]))+ (0.38771248 * float(x[14]))+ (-0.58877057 * float(x[15]))+ (0.08561699 * float(x[16]))) + 0.90765584), 0)
    h_6 = max((((1.6742452 * float(x[0]))+ (0.88231355 * float(x[1]))+ (-0.6057401 * float(x[2]))+ (0.17714044 * float(x[3]))+ (-2.0778332 * float(x[4]))+ (0.68553656 * float(x[5]))+ (-0.25113538 * float(x[6]))+ (-0.44303182 * float(x[7]))+ (1.251445 * float(x[8]))+ (0.39577928 * float(x[9]))+ (0.29251218 * float(x[10]))+ (0.83376956 * float(x[11]))+ (-0.46604636 * float(x[12]))+ (-0.020936558 * float(x[13]))+ (-1.0439135 * float(x[14]))+ (1.308671 * float(x[15]))+ (-0.7476106 * float(x[16]))) + 0.025630457), 0)
    h_7 = max((((0.5050292 * float(x[0]))+ (0.39778668 * float(x[1]))+ (-0.40697226 * float(x[2]))+ (0.08326205 * float(x[3]))+ (0.12465865 * float(x[4]))+ (0.109576136 * float(x[5]))+ (-0.07024665 * float(x[6]))+ (0.015896412 * float(x[7]))+ (0.0833207 * float(x[8]))+ (0.4706299 * float(x[9]))+ (-0.21001329 * float(x[10]))+ (-0.3384519 * float(x[11]))+ (-0.25259337 * float(x[12]))+ (0.058896452 * float(x[13]))+ (-0.46841317 * float(x[14]))+ (0.7992941 * float(x[15]))+ (0.109955356 * float(x[16]))) + 0.6778814), 0)
    h_8 = max((((0.37336048 * float(x[0]))+ (-0.08397763 * float(x[1]))+ (0.07383688 * float(x[2]))+ (-0.17798118 * float(x[3]))+ (-0.2696137 * float(x[4]))+ (0.36367014 * float(x[5]))+ (-0.012905099 * float(x[6]))+ (-0.44032735 * float(x[7]))+ (0.3132479 * float(x[8]))+ (0.6996633 * float(x[9]))+ (0.06474549 * float(x[10]))+ (1.1503185 * float(x[11]))+ (0.45228505 * float(x[12]))+ (0.5951322 * float(x[13]))+ (-0.49680793 * float(x[14]))+ (0.21987739 * float(x[15]))+ (-0.24442866 * float(x[16]))) + -0.41485634), 0)
    h_9 = max((((-1.0177854 * float(x[0]))+ (-1.7478292 * float(x[1]))+ (-0.5736792 * float(x[2]))+ (-0.89407736 * float(x[3]))+ (0.6358248 * float(x[4]))+ (-0.018001083 * float(x[5]))+ (-1.7963898 * float(x[6]))+ (1.0599513 * float(x[7]))+ (-0.5033219 * float(x[8]))+ (0.26117066 * float(x[9]))+ (-1.7747741 * float(x[10]))+ (0.24847996 * float(x[11]))+ (-1.6388159 * float(x[12]))+ (1.306415 * float(x[13]))+ (0.3593794 * float(x[14]))+ (-1.7473379 * float(x[15]))+ (1.7615368 * float(x[16]))) + -1.7229296), 0)
    h_10 = max((((-0.20488276 * float(x[0]))+ (0.02757832 * float(x[1]))+ (0.35277072 * float(x[2]))+ (-0.20770395 * float(x[3]))+ (-0.28648657 * float(x[4]))+ (0.40791774 * float(x[5]))+ (-0.0978198 * float(x[6]))+ (0.39345184 * float(x[7]))+ (0.61983687 * float(x[8]))+ (0.28439882 * float(x[9]))+ (-0.32790068 * float(x[10]))+ (-0.29945454 * float(x[11]))+ (-0.39400405 * float(x[12]))+ (0.5976722 * float(x[13]))+ (0.48791072 * float(x[14]))+ (-0.16982032 * float(x[15]))+ (0.288767 * float(x[16]))) + -0.73564076), 0)
    o[0] = (4.6341724 * h_0)+ (-2.8767555 * h_1)+ (1.9621918 * h_2)+ (1.70291 * h_3)+ (4.569728 * h_4)+ (1.8781054 * h_5)+ (6.7567396 * h_6)+ (2.9393454 * h_7)+ (2.0758967 * h_8)+ (-7.1264606 * h_9)+ (2.7457929 * h_10) + -5.9137197

    

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
        model_cap=210
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
