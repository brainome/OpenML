#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/4965216/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.arff -o Predictors/GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1_NN.py -target class -stopat 80.56 -f NN -e 20 --yes
# Total compiler execution time: 0:35:07.37. Finished on: Apr-21-2020 14:37:42.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     79.93% (1279/1600 correct)
Improvement over best guess:        29.93% (of possible 50.0%)
Model capacity (MEC):               115 bits
Generalization ratio:               11.12 bits/bit
Model efficiency:                   0.26%/parameter
System behavior
True Negatives:                     43.06% (689/1600)
True Positives:                     36.88% (590/1600)
False Negatives:                    13.12% (210/1600)
False Positives:                    6.94% (111/1600)
True Pos. Rate/Sensitivity/Recall:  0.74
True Neg. Rate/Specificity:         0.86
Precision:                          0.84
F-1 Measure:                        0.79
False Negative Rate/Miss Rate:      0.26
Critical Success Index:             0.65

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
TRAINFILE = "GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv"


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
        mean = np.array([0.019791666666666666, 0.265625, 0.7208333333333333, 0.875, 0.35208333333333336, 0.7854166666666667, 0.4979166666666667, 0.7291666666666666, 0.33541666666666664, 0.759375, 0.027083333333333334, 0.8645833333333334, 0.47708333333333336, 0.07083333333333333, 0.18229166666666666, 0.665625, 0.453125, 0.8052083333333333, 0.38333333333333336, 0.4125])
        components = np.array([array([-9.30536072e-03, -3.98882310e-02, -1.61633026e-01,  2.37071992e-01,
        4.76692327e-02, -1.34631720e-01,  2.25826890e-02,  5.65946778e-02,
        5.97911714e-02,  1.70986684e-01, -4.12137341e-04,  6.46940050e-01,
        5.18960780e-02, -3.62560219e-03, -3.30696830e-02,  5.81805538e-02,
       -1.12548371e-01, -6.48085191e-01, -2.38078025e-02,  1.03278546e-02]), array([-4.42854028e-04,  2.00506233e-02, -1.35477645e-01,  1.97203801e-01,
       -3.38161587e-03,  6.97148202e-01, -4.99910294e-02, -5.13555826e-02,
        9.90271360e-03, -5.45194794e-01, -5.49448940e-03,  3.22915008e-01,
        6.76548650e-03, -1.41896579e-02,  5.06722274e-03,  5.06814510e-02,
       -2.61793023e-03,  1.44336543e-01, -1.21866736e-01,  1.04967512e-01]), array([-0.01183056, -0.03062782,  0.00778501,  0.82649838,  0.05636644,
       -0.23457235, -0.09228169, -0.2450081 ,  0.0399512 , -0.18882973,
       -0.0020406 , -0.31985125, -0.11294737,  0.00190237,  0.03288076,
       -0.14168561, -0.11351288, -0.03922478,  0.02866242,  0.01708292]), array([-0.01087334, -0.01189208, -0.23983014,  0.15302758, -0.07740427,
        0.35404144, -0.04788318, -0.35465808,  0.11877661,  0.5976036 ,
        0.00082091, -0.14968391,  0.0785368 ,  0.00703765,  0.03495766,
        0.48899184, -0.09565727,  0.08956759,  0.00350567,  0.04484457]), array([ 0.00242609, -0.06383163,  0.31728427,  0.39379846, -0.06636688,
        0.04254286,  0.23022832,  0.48713374, -0.09260325,  0.33994867,
       -0.0079696 ,  0.27531573,  0.10475742,  0.00549567, -0.01641017,
       -0.05368849, -0.04481032,  0.47514854, -0.04057172, -0.00867932]), array([ 1.98422049e-03,  8.52295317e-02, -4.06155775e-01, -4.75777365e-02,
       -9.67877469e-02, -4.34605022e-01, -1.36361788e-01, -2.75639946e-01,
       -2.68088352e-02, -7.74587083e-02, -2.32155798e-02,  3.72820521e-01,
        2.63211430e-04,  3.43388243e-03,  1.50517746e-02,  3.65651354e-02,
       -8.43209323e-02,  5.02910840e-01, -2.16911953e-01, -2.78884029e-01]), array([-0.00700569, -0.01313643,  0.74148679, -0.09537252, -0.07735786,
       -0.0484574 ,  0.02110698, -0.51239154, -0.03417435, -0.09298899,
        0.01186296,  0.21205256,  0.11043034, -0.01868843,  0.02286101,
        0.16656581, -0.26708881, -0.00497589, -0.02979434, -0.03968826]), array([-5.39871097e-03,  3.08120012e-02,  2.66363096e-02,  5.43986992e-02,
        4.69596629e-04, -2.33101915e-01, -2.23251961e-01,  3.63814693e-01,
       -1.36299833e-01, -3.05438436e-01, -4.89228750e-03, -6.30996394e-02,
        8.95092195e-02, -1.04163222e-02,  1.25444493e-02,  7.59738112e-01,
       -6.60776942e-02,  6.39994757e-03,  2.26111345e-01,  4.38675841e-02]), array([-0.00450715, -0.0901906 , -0.03658533,  0.02166118,  0.03336844,
       -0.07909144,  0.77803463, -0.08458536,  0.10903485, -0.14108903,
        0.0025262 , -0.01327212, -0.41266927, -0.00186013,  0.0146153 ,
        0.313689  ,  0.20552547, -0.01881469, -0.10049359, -0.12880792]), array([ 0.00261608, -0.02106808, -0.07306934,  0.02342376, -0.02633741,
        0.08040261,  0.21612878,  0.12327547, -0.04891429, -0.11069341,
       -0.00488507, -0.26924554,  0.57123773, -0.03463221, -0.0173306 ,
        0.01643341, -0.23436685, -0.17945341, -0.39577499, -0.51534511]), array([ 8.22610214e-04,  7.34552016e-04,  4.43751164e-02,  1.15945751e-01,
        7.28918668e-02, -1.15882225e-01,  4.34875652e-02, -2.08939670e-01,
       -1.08813108e-01, -1.68778158e-02,  8.55805489e-04,  6.28949117e-02,
        5.14144728e-01,  4.12773030e-03, -2.06338078e-02,  5.45175927e-02,
        7.78946811e-01,  8.08274879e-03, -3.51786322e-02,  1.68507948e-01]), array([-1.71086477e-04, -1.50843938e-02, -2.27702546e-01, -6.29945234e-02,
        1.77154530e-03, -8.55757752e-02,  4.15830427e-01, -1.00828736e-01,
        1.31226006e-01, -1.22705487e-01,  2.39618772e-03,  2.55413991e-02,
        4.17429359e-01,  1.86206776e-02,  4.06340312e-02, -1.28017238e-01,
       -3.36103717e-01,  1.26313372e-01,  5.50847469e-01,  3.09234036e-01]), array([ 0.00825483, -0.04556278,  0.08539354, -0.00892228, -0.45355643,
       -0.10630425, -0.09445477,  0.13917013,  0.8132921 , -0.08575201,
        0.01143379, -0.02479351,  0.0857474 , -0.02538657,  0.02480897,
        0.02719102,  0.08056472,  0.00703589, -0.19316357,  0.14977651]), array([-0.01086403, -0.27069028,  0.02209216, -0.08492914,  0.77067865,
       -0.08841697, -0.05242944,  0.03596312,  0.257818  ,  0.01074039,
        0.00425746, -0.00797269,  0.06471414, -0.00776487,  0.04263216,
        0.07085921, -0.14762561,  0.12748124, -0.34709164,  0.27718421]), array([ 0.01722167,  0.17551416,  0.10469904,  0.04460126,  0.37362179,
        0.11922924, -0.10514245,  0.00783083,  0.41217046,  0.01645868,
        0.01806084,  0.0973167 ,  0.01270459,  0.00555086,  0.09397417,
       -0.01421287,  0.17256943,  0.05691281,  0.45994631, -0.59772449]), array([-0.00611414,  0.91643118,  0.04223591,  0.02835556,  0.12802812,
       -0.0344529 ,  0.11990576,  0.04589475,  0.03834562,  0.03829871,
       -0.00426515, -0.02695642,  0.00606306,  0.01894094,  0.17649516,
        0.02150231, -0.06461722, -0.0297702 , -0.20213013,  0.2045923 ]), array([-0.00942175, -0.17144333, -0.01288105, -0.01804218, -0.07650939,
        0.00402163, -0.01713374,  0.03220429, -0.09015718,  0.00386083,
       -0.01636565,  0.02312478,  0.00377389,  0.03379374,  0.9740279 ,
       -0.03004097,  0.03367007, -0.03877336, -0.01851554, -0.01056831])])
        whiten = False
        explained_variance = np.array([0.5660325353430221, 0.545483351126661, 0.5101127880410075, 0.48830180586137406, 0.48460030480511956, 0.4613080598903314, 0.44303683640817665, 0.4264540193234295, 0.38994808838109635, 0.37788941283313005, 0.35901338508952085, 0.3305817631588995, 0.28149095103429045, 0.2698398158000865, 0.2579903810794659, 0.22247881090382898, 0.15877725159864894])
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
    h_0 = max((((0.15416318 * float(x[0]))+ (-0.5807323 * float(x[1]))+ (-0.09049848 * float(x[2]))+ (1.1218123 * float(x[3]))+ (-0.21808246 * float(x[4]))+ (-4.1107593 * float(x[5]))+ (-0.30814755 * float(x[6]))+ (3.51842 * float(x[7]))+ (-3.5461574 * float(x[8]))+ (-11.134345 * float(x[9]))+ (0.6836041 * float(x[10]))+ (10.518534 * float(x[11]))+ (-0.51864517 * float(x[12]))+ (0.72767186 * float(x[13]))+ (-1.6305579 * float(x[14]))+ (0.7154266 * float(x[15]))+ (-0.31217632 * float(x[16]))) + 6.445683), 0)
    h_1 = max((((-0.27764857 * float(x[0]))+ (-4.1592174 * float(x[1]))+ (2.9227178 * float(x[2]))+ (-0.71149963 * float(x[3]))+ (-3.6201897 * float(x[4]))+ (3.7879813 * float(x[5]))+ (-2.0019033 * float(x[6]))+ (-3.8478098 * float(x[7]))+ (3.3647223 * float(x[8]))+ (6.602097 * float(x[9]))+ (-1.1140618 * float(x[10]))+ (0.945525 * float(x[11]))+ (0.92336035 * float(x[12]))+ (-0.707055 * float(x[13]))+ (5.33901 * float(x[14]))+ (-1.2659028 * float(x[15]))+ (1.2089807 * float(x[16]))) + -3.9985611), 0)
    h_2 = max((((-0.08607435 * float(x[0]))+ (0.39019257 * float(x[1]))+ (2.4897616 * float(x[2]))+ (-0.19817246 * float(x[3]))+ (2.9172611 * float(x[4]))+ (-1.4228324 * float(x[5]))+ (-0.99506116 * float(x[6]))+ (0.5935481 * float(x[7]))+ (-1.4537729 * float(x[8]))+ (-3.988321 * float(x[9]))+ (0.031577867 * float(x[10]))+ (-1.4069543 * float(x[11]))+ (0.744809 * float(x[12]))+ (-1.8107835 * float(x[13]))+ (0.60986835 * float(x[14]))+ (2.5551395 * float(x[15]))+ (-2.324208 * float(x[16]))) + -2.5296097), 0)
    h_3 = max((((-0.017052785 * float(x[0]))+ (0.86359936 * float(x[1]))+ (-0.6740535 * float(x[2]))+ (-0.075983204 * float(x[3]))+ (0.0055871797 * float(x[4]))+ (-0.123368666 * float(x[5]))+ (-0.19810404 * float(x[6]))+ (-0.4836178 * float(x[7]))+ (0.78034484 * float(x[8]))+ (-0.32010704 * float(x[9]))+ (-0.25566933 * float(x[10]))+ (-1.3771625 * float(x[11]))+ (0.92864037 * float(x[12]))+ (2.4440763 * float(x[13]))+ (-5.347607 * float(x[14]))+ (2.1359107 * float(x[15]))+ (0.35102138 * float(x[16]))) + -0.3678519), 0)
    h_4 = max((((0.08305313 * float(x[0]))+ (-0.23795792 * float(x[1]))+ (-0.03722388 * float(x[2]))+ (0.460125 * float(x[3]))+ (0.15673551 * float(x[4]))+ (-1.6337733 * float(x[5]))+ (-0.13629884 * float(x[6]))+ (1.4042606 * float(x[7]))+ (-1.5614439 * float(x[8]))+ (-4.7266417 * float(x[9]))+ (0.35188776 * float(x[10]))+ (4.3234954 * float(x[11]))+ (-0.12062813 * float(x[12]))+ (0.48407334 * float(x[13]))+ (-1.1165348 * float(x[14]))+ (0.56352544 * float(x[15]))+ (-0.21287887 * float(x[16]))) + 1.0751932), 0)
    h_5 = max((((0.078019895 * float(x[0]))+ (0.058662932 * float(x[1]))+ (-0.08541301 * float(x[2]))+ (-0.15447797 * float(x[3]))+ (0.18931107 * float(x[4]))+ (1.4615706 * float(x[5]))+ (0.21750651 * float(x[6]))+ (-0.7307578 * float(x[7]))+ (0.55319846 * float(x[8]))+ (2.6616368 * float(x[9]))+ (-0.24209255 * float(x[10]))+ (-2.4750085 * float(x[11]))+ (0.14336263 * float(x[12]))+ (0.10395343 * float(x[13]))+ (0.37128899 * float(x[14]))+ (-0.1964928 * float(x[15]))+ (-0.059977643 * float(x[16]))) + 3.48863), 0)
    o[0] = (2.6584296 * h_0)+ (0.15006591 * h_1)+ (0.46151388 * h_2)+ (0.9923237 * h_3)+ (-5.723321 * h_4)+ (3.0904098 * h_5) + -19.553171

    

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
        model_cap=115
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
