#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/4965218/GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001.arff -o Predictors/GAMETES-Heterogeneity-20atts-1600-Het-0.4-0.2-50-EDM-2-001_NN.py -target class -stopat 100.0 -f NN -e 20 --yes
# Total compiler execution time: 0:35:12.40. Finished on: Apr-21-2020 15:14:13.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     74.50% (1192/1600 correct)
Improvement over best guess:        24.50% (of possible 50.0%)
Model capacity (MEC):               115 bits
Generalization ratio:               10.36 bits/bit
Model efficiency:                   0.21%/parameter
System behavior
True Negatives:                     37.62% (602/1600)
True Positives:                     36.88% (590/1600)
False Negatives:                    13.12% (210/1600)
False Positives:                    12.38% (198/1600)
True Pos. Rate/Sensitivity/Recall:  0.74
True Neg. Rate/Specificity:         0.75
Precision:                          0.75
F-1 Measure:                        0.74
False Negative Rate/Miss Rate:      0.26
Critical Success Index:             0.59

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
TRAINFILE = "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001.csv"


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
        mean = np.array([0.721875, 0.415625, 0.028125, 0.9302083333333333, 0.16979166666666667, 0.48020833333333335, 0.128125, 0.5822916666666667, 0.28854166666666664, 0.16458333333333333, 0.25104166666666666, 0.08854166666666667, 0.6635416666666667, 0.596875, 0.8802083333333334, 0.4166666666666667, 0.38958333333333334, 0.378125, 0.41041666666666665, 0.39895833333333336])
        components = np.array([array([-0.23124084,  0.0406164 ,  0.00517798, -0.37817954,  0.01188646,
       -0.02103773,  0.01099015,  0.14375908,  0.00096211, -0.0194877 ,
       -0.02637873, -0.00565329,  0.45039721, -0.02267263,  0.72711771,
       -0.10971652,  0.13602687, -0.02961181,  0.07646779,  0.10007255]), array([-0.2199585 , -0.03949931, -0.00398538,  0.84792896,  0.02705448,
       -0.10744255,  0.05346451, -0.00663585,  0.03435135,  0.0038072 ,
       -0.02089273, -0.00195418, -0.07744468,  0.16164698,  0.42463298,
       -0.01570055, -0.01900398,  0.00816168,  0.02459467, -0.02292862]), array([ 7.40539575e-01,  1.03865320e-01, -1.09553100e-04,  1.56907650e-01,
        5.53120818e-03,  1.67292765e-01, -2.32601344e-02, -3.61544774e-01,
       -4.46021865e-04, -7.43273979e-03, -8.16647073e-02,  1.61139411e-03,
        3.50700874e-01, -2.80077247e-03,  1.77508815e-01, -1.51295346e-01,
       -1.66096506e-01, -1.84901789e-01, -1.04314682e-01,  1.16593954e-02]), array([ 0.34396556,  0.17055995, -0.00891333,  0.01718193,  0.0158908 ,
        0.07561341, -0.00976787,  0.60089117,  0.04091637,  0.02750588,
       -0.04776385,  0.0329412 , -0.03101501,  0.59097475, -0.03185848,
        0.11922062,  0.1432417 ,  0.00977644,  0.15187632,  0.26489206]), array([-0.31860096, -0.03048049,  0.01884165,  0.22301263,  0.01869339,
        0.22926003,  0.01202442,  0.09005805,  0.1556644 , -0.00244082,
       -0.01941974,  0.01020441,  0.7327326 ,  0.03411192, -0.43920253,
        0.05596068, -0.07551494, -0.0321414 , -0.06228786,  0.12994218]), array([-0.21923245, -0.06437987, -0.00452836, -0.21397781, -0.0348805 ,
        0.09096695,  0.00983336, -0.54502465,  0.07545304,  0.03015082,
        0.05792841,  0.00955154, -0.03324717,  0.74686811,  0.02778103,
        0.08847449, -0.08938136,  0.01318946, -0.06159883, -0.07630008]), array([-0.19280917,  0.21490992, -0.00998473,  0.01099325, -0.02704725,
        0.75999358, -0.01186838,  0.03287524, -0.10117462, -0.03255219,
        0.02604842, -0.00110303, -0.27715085, -0.11419231,  0.11328999,
        0.09554502, -0.06032992, -0.44521363, -0.02764477,  0.10229189]), array([-0.05641913,  0.50675996, -0.00168495,  0.06451717, -0.00920012,
       -0.08875456,  0.01582246, -0.35868107,  0.04015518,  0.02601608,
        0.04115389, -0.00315156,  0.01921479, -0.09404981, -0.11483976,
       -0.01403096,  0.28683922,  0.04400671,  0.64967206,  0.25710283]), array([ 0.1071523 ,  0.13333137,  0.01543273,  0.02045619,  0.01148256,
       -0.02346327,  0.01548609, -0.06916759, -0.00978526, -0.00592476,
        0.09966987, -0.01116226,  0.11468059, -0.12390355,  0.1287751 ,
        0.91199096,  0.14496503,  0.12319445, -0.13511879, -0.1514118 ]), array([-0.11216039,  0.7338761 ,  0.00762709,  0.01609818, -0.00682101,
       -0.05562835,  0.0280371 ,  0.063304  , -0.15653652, -0.02802857,
       -0.05436472,  0.002736  ,  0.02511821,  0.07472915, -0.04810867,
       -0.22782579,  0.06104802,  0.21071803, -0.48104972, -0.26901599]), array([ 0.10196513, -0.19751787, -0.00136461,  0.05638402,  0.00923944,
        0.49124316, -0.0102744 , -0.07614019,  0.19679481,  0.03102449,
        0.12800736, -0.00974281, -0.04909727, -0.06423407,  0.05900731,
       -0.16662996,  0.44874593,  0.62995388, -0.08738597,  0.01714883]), array([-0.05414063,  0.04984108, -0.00567533, -0.02094153,  0.0290858 ,
       -0.14400689, -0.05891341, -0.14572338,  0.07744264, -0.02946918,
       -0.12143609,  0.01563965, -0.13720229, -0.08709103,  0.04395862,
        0.07876159, -0.13833238,  0.13794396, -0.41046116,  0.82751369]), array([ 0.00443211, -0.13612799, -0.01755452,  0.04425106, -0.1097334 ,
       -0.16714072,  0.02212602, -0.0953989 ,  0.03888006,  0.011093  ,
       -0.1048241 ,  0.01690713,  0.00633154,  0.03873869, -0.10046965,
       -0.05287713,  0.75360257, -0.49641992, -0.29520931,  0.02788509]), array([ 0.02103862,  0.17735012, -0.01016375, -0.04851164, -0.04987676,
       -0.07866665, -0.05077307,  0.07449032,  0.90583845, -0.0337261 ,
        0.22607019, -0.05955119, -0.10214983, -0.06701021,  0.05443186,
       -0.03151115, -0.10867632, -0.15811759, -0.06460418, -0.11244473]), array([ 0.0417512 , -0.00548509,  0.02619013,  0.0484316 , -0.10780099,
       -0.08857281, -0.04533519,  0.03014959, -0.2447891 , -0.07097585,
        0.92567173, -0.02297292,  0.07097807,  0.03409348, -0.00732441,
       -0.08991744,  0.00547863, -0.08197288, -0.08871774,  0.14645852]), array([-0.00712712,  0.02433785, -0.00338103, -0.01971475,  0.55138943,
       -0.00966061,  0.0644613 , -0.00288577,  0.00387465,  0.80993012,
        0.11350443, -0.10281527, -0.0071932 , -0.02161491,  0.00202278,
       -0.01964772,  0.01065339, -0.07958057, -0.06119474,  0.01063342]), array([ 0.00608145, -0.01440962, -0.02067341, -0.01930922,  0.80624826,
       -0.00714618,  0.07740919, -0.0330447 ,  0.01541368, -0.57066281,
        0.05478934, -0.01230433, -0.0212357 ,  0.03646148, -0.03140103,
       -0.01764721,  0.07864186, -0.05628726, -0.00173047, -0.01741169])])
        whiten = False
        explained_variance = np.array([0.5094397547159499, 0.49709161469391294, 0.4621466657102033, 0.45236144289471564, 0.4335635819317387, 0.39770280693799215, 0.3830973415351283, 0.3562010184822724, 0.3273631718342066, 0.31718753027551294, 0.3047839114483098, 0.26735669279225077, 0.2639368454382916, 0.24125833139496142, 0.21387400220755357, 0.15899497572347515, 0.14711806694689192])
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
    h_0 = max((((6.2747855 * float(x[0]))+ (0.5942218 * float(x[1]))+ (5.168482 * float(x[2]))+ (-0.3343529 * float(x[3]))+ (-2.9939044 * float(x[4]))+ (-1.4460893 * float(x[5]))+ (0.14850876 * float(x[6]))+ (-0.50406873 * float(x[7]))+ (7.200033 * float(x[8]))+ (-4.7624397 * float(x[9]))+ (2.3836603 * float(x[10]))+ (-0.77892756 * float(x[11]))+ (2.5633144 * float(x[12]))+ (-0.6240152 * float(x[13]))+ (-4.324545 * float(x[14]))+ (-6.010681 * float(x[15]))+ (-1.3841784 * float(x[16]))) + -7.6475563), 0)
    h_1 = max((((-5.716849 * float(x[0]))+ (-0.22718927 * float(x[1]))+ (-4.3703737 * float(x[2]))+ (-0.2595342 * float(x[3]))+ (1.0193808 * float(x[4]))+ (-0.27908102 * float(x[5]))+ (-1.4989299 * float(x[6]))+ (-4.1211557 * float(x[7]))+ (-5.791137 * float(x[8]))+ (0.6179284 * float(x[9]))+ (1.0953337 * float(x[10]))+ (11.714264 * float(x[11]))+ (-6.5002627 * float(x[12]))+ (-0.10121629 * float(x[13]))+ (4.6249876 * float(x[14]))+ (0.8523135 * float(x[15]))+ (0.5497908 * float(x[16]))) + 6.195553), 0)
    h_2 = max((((0.9183426 * float(x[0]))+ (0.11752019 * float(x[1]))+ (0.83759123 * float(x[2]))+ (0.60714746 * float(x[3]))+ (1.9450645 * float(x[4]))+ (-0.37383705 * float(x[5]))+ (0.7857181 * float(x[6]))+ (-2.2982447 * float(x[7]))+ (0.029187078 * float(x[8]))+ (1.264414 * float(x[9]))+ (0.9710487 * float(x[10]))+ (7.3940706 * float(x[11]))+ (1.5620745 * float(x[12]))+ (0.07285382 * float(x[13]))+ (1.3861111 * float(x[14]))+ (1.1264946 * float(x[15]))+ (0.28192377 * float(x[16]))) + -0.81543976), 0)
    h_3 = max((((0.5507869 * float(x[0]))+ (-0.75972664 * float(x[1]))+ (2.5203586 * float(x[2]))+ (-1.0901603 * float(x[3]))+ (0.35629758 * float(x[4]))+ (-0.22628157 * float(x[5]))+ (2.2054937 * float(x[6]))+ (-1.3618091 * float(x[7]))+ (1.3060473 * float(x[8]))+ (1.3612391 * float(x[9]))+ (-0.085268095 * float(x[10]))+ (1.1875126 * float(x[11]))+ (1.4735248 * float(x[12]))+ (-0.3073591 * float(x[13]))+ (-1.3318853 * float(x[14]))+ (3.013952 * float(x[15]))+ (2.530549 * float(x[16]))) + 0.92846227), 0)
    h_4 = max((((-0.8399887 * float(x[0]))+ (0.39968497 * float(x[1]))+ (0.07742532 * float(x[2]))+ (-0.5078764 * float(x[3]))+ (0.18583353 * float(x[4]))+ (-0.0561794 * float(x[5]))+ (-1.8947737 * float(x[6]))+ (-0.47248986 * float(x[7]))+ (-0.5842198 * float(x[8]))+ (0.3700406 * float(x[9]))+ (1.1277741 * float(x[10]))+ (1.5650258 * float(x[11]))+ (-5.4586105 * float(x[12]))+ (0.07007565 * float(x[13]))+ (-0.14171939 * float(x[14]))+ (-0.489464 * float(x[15]))+ (0.2370939 * float(x[16]))) + 0.21934672), 0)
    h_5 = max((((0.44220585 * float(x[0]))+ (0.35941765 * float(x[1]))+ (0.038845196 * float(x[2]))+ (0.49393696 * float(x[3]))+ (0.3009835 * float(x[4]))+ (-0.092392944 * float(x[5]))+ (-1.1463956 * float(x[6]))+ (0.3020025 * float(x[7]))+ (0.05923447 * float(x[8]))+ (-0.10542904 * float(x[9]))+ (1.0839633 * float(x[10]))+ (1.1586308 * float(x[11]))+ (-1.254914 * float(x[12]))+ (-0.051992502 * float(x[13]))+ (0.039867736 * float(x[14]))+ (-1.000258 * float(x[15]))+ (-0.31592375 * float(x[16]))) + 1.2296467), 0)
    o[0] = (0.23200665 * h_0)+ (-0.19340768 * h_1)+ (0.84776175 * h_2)+ (-0.47364733 * h_3)+ (1.2896641 * h_4)+ (-1.9141351 * h_5) + 1.8910552

    

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
