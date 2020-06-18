#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/2184/BayesianNetworkGenerator_zoo.arff -o Predictors/BNG(zoo-nominal-1000000)_NN.py -target type -stopat 93.86 -f NN -e 3 --yes --runlocalonly
# Total compiler execution time: 2:08:37.16. Finished on: Jun-05-2020 22:48:47.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        7-way classifier
Best-guess accuracy:                42.02%
Model accuracy:                     93.91% (939180/999999 correct)
Improvement over best guess:        51.89% (of possible 57.98%)
Model capacity (MEC):               307 bits
Generalization ratio:               3059.21 bits/bit
Confusion Matrix:
 [12.02% 0.04% 0.12% 0.05% 0.01% 0.56% 0.13%]
 [0.02% 39.47% 0.00% 0.01% 0.02% 0.06% 0.05%]
 [0.04% 0.00% 9.16% 0.02% 0.56% 0.23% 0.07%]
 [0.01% 0.01% 0.02% 19.41% 0.03% 0.15% 0.05%]
 [0.01% 0.01% 1.03% 0.03% 6.93% 0.04% 0.05%]
 [0.53% 0.12% 0.26% 0.23% 0.04% 3.70% 0.38%]
 [0.24% 0.08% 0.17% 0.08% 0.07% 0.44% 3.22%]

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
TRAINFILE = "BayesianNetworkGenerator_zoo.csv"


#Number of output logits
num_output_logits = 7

#Number of attributes
num_attr = 17
n_classes = 7

mappings = [{13120529.0: 0, 22038877.0: 1, 117437110.0: 2, 133782293.0: 3, 149060979.0: 4, 189113590.0: 5, 197980915.0: 6, 214846638.0: 7, 293482766.0: 8, 298438918.0: 9, 373135080.0: 10, 584252530.0: 11, 704615212.0: 12, 744212000.0: 13, 752743942.0: 14, 773143157.0: 15, 774942256.0: 16, 822945407.0: 17, 844703930.0: 18, 869147049.0: 19, 878725936.0: 20, 1016767401.0: 21, 1125635993.0: 22, 1125972319.0: 23, 1205411905.0: 24, 1351015120.0: 25, 1401591111.0: 26, 1406814162.0: 27, 1440416042.0: 28, 1458066827.0: 29, 1466808295.0: 30, 1489220549.0: 31, 1506756718.0: 32, 1559989163.0: 33, 1574132893.0: 34, 1585185686.0: 35, 1595133347.0: 36, 1608342591.0: 37, 1744256722.0: 38, 1777040477.0: 39, 1832803961.0: 40, 1852629197.0: 41, 1940301880.0: 42, 1950037758.0: 43, 1957767378.0: 44, 2107455857.0: 45, 2163761501.0: 46, 2203004802.0: 47, 2209811699.0: 48, 2223629247.0: 49, 2233680476.0: 50, 2284240540.0: 51, 2343511109.0: 52, 2383478420.0: 53, 2412105095.0: 54, 2453043442.0: 55, 2468141291.0: 56, 2560465762.0: 57, 2571044658.0: 58, 2579420686.0: 59, 2746820935.0: 60, 2795630676.0: 61, 2922675039.0: 62, 2964036461.0: 63, 3042426822.0: 64, 3054645060.0: 65, 3064768660.0: 66, 3112416541.0: 67, 3123563957.0: 68, 3157691388.0: 69, 3190505002.0: 70, 3198437122.0: 71, 3201480873.0: 72, 3219624110.0: 73, 3256768255.0: 74, 3398724132.0: 75, 3400449319.0: 76, 3461413561.0: 77, 3624607037.0: 78, 3654688985.0: 79, 3679943724.0: 80, 3732236668.0: 81, 3737687316.0: 82, 3751589890.0: 83, 3812981318.0: 84, 3854672160.0: 85, 3890034724.0: 86, 3914382451.0: 87, 3942618020.0: 88, 3956451473.0: 89, 4000454991.0: 90, 4021318520.0: 91, 4023774479.0: 92, 4026676655.0: 93, 4070099558.0: 94, 4071067935.0: 95, 4073228349.0: 96, 4143483097.0: 97, 4184026786.0: 98, 4189989858.0: 99}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {1686921404.0: 0, 3271256840.0: 1, 4078137749.0: 2}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}, {734881840.0: 0, 4261170317.0: 1}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

transform_true = False

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values())) + 1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize, mappings):
            if i >= data_arr.shape[1]:
                break
            col = data_arr[:, i]
            normcol = column_norm(col,mapping)
            data_arr[:, i] = normcol
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
    target="type"


    # if (testfile):
    #     target = ''
    
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
    clean.mapping={'fish': 0, 'mammal': 1, 'invertebrate': 2, 'bird': 3, 'insect': 4, 'reptile': 5, 'amphibian': 6}

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
def single_classify(row):
    #inits
    x = row
    o = [0] * num_output_logits


    #Nueron Equations
    h_0 = max((((29.919601 * float(x[0]))+ (7.764002 * float(x[1]))+ (-4.714985 * float(x[2]))+ (0.8850374 * float(x[3]))+ (3.2253873 * float(x[4]))+ (3.111018 * float(x[5]))+ (0.24666704 * float(x[6]))+ (1.5520961 * float(x[7]))+ (0.4845034 * float(x[8]))+ (-1.4435732 * float(x[9]))+ (2.420115 * float(x[10]))+ (1.4303143 * float(x[11]))+ (2.6645284 * float(x[12]))+ (0.9382948 * float(x[13]))+ (-1.6012286 * float(x[14]))+ (2.9173331 * float(x[15]))+ (1.5844648 * float(x[16]))) + 1.5596874), 0)
    h_1 = max((((31.080969 * float(x[0]))+ (-2.4484475 * float(x[1]))+ (0.518296 * float(x[2]))+ (1.330385 * float(x[3]))+ (-3.0336046 * float(x[4]))+ (-1.9639082 * float(x[5]))+ (0.43266365 * float(x[6]))+ (-0.3398105 * float(x[7]))+ (0.3974213 * float(x[8]))+ (1.1637782 * float(x[9]))+ (-0.90729046 * float(x[10]))+ (0.06713904 * float(x[11]))+ (-1.2025983 * float(x[12]))+ (0.5335511 * float(x[13]))+ (0.24688369 * float(x[14]))+ (-0.2830295 * float(x[15]))+ (-0.79008317 * float(x[16]))) + 0.820368), 0)
    h_2 = max((((16.92296 * float(x[0]))+ (-3.2820325 * float(x[1]))+ (2.2686634 * float(x[2]))+ (1.0493454 * float(x[3]))+ (-2.3859131 * float(x[4]))+ (-1.7454553 * float(x[5]))+ (0.21884353 * float(x[6]))+ (-0.5299275 * float(x[7]))+ (-0.53666186 * float(x[8]))+ (1.4020675 * float(x[9]))+ (-0.037518878 * float(x[10]))+ (-1.0139061 * float(x[11]))+ (-1.3622682 * float(x[12]))+ (1.7764355 * float(x[13]))+ (0.85567504 * float(x[14]))+ (-1.4520808 * float(x[15]))+ (-0.77250993 * float(x[16]))) + 1.1992759), 0)
    h_3 = max((((3.3411815 * float(x[0]))+ (-1.4787501 * float(x[1]))+ (-5.101954 * float(x[2]))+ (4.2834997 * float(x[3]))+ (-1.5947444 * float(x[4]))+ (-4.5994687 * float(x[5]))+ (1.9162526 * float(x[6]))+ (2.594824 * float(x[7]))+ (4.148694 * float(x[8]))+ (-4.8522205 * float(x[9]))+ (-3.707726 * float(x[10]))+ (5.082174 * float(x[11]))+ (1.717937 * float(x[12]))+ (1.275063 * float(x[13]))+ (-1.3405334 * float(x[14]))+ (-0.15636191 * float(x[15]))+ (0.25601375 * float(x[16]))) + 4.9139595), 0)
    h_4 = max((((10.183592 * float(x[0]))+ (-0.98588794 * float(x[1]))+ (-0.49284157 * float(x[2]))+ (-1.3125544 * float(x[3]))+ (0.26052964 * float(x[4]))+ (-1.8860055 * float(x[5]))+ (-0.49974465 * float(x[6]))+ (1.506854 * float(x[7]))+ (-2.1065803 * float(x[8]))+ (-0.15726776 * float(x[9]))+ (-2.3398843 * float(x[10]))+ (-0.30114225 * float(x[11]))+ (0.15174721 * float(x[12]))+ (2.554065 * float(x[13]))+ (2.234232 * float(x[14]))+ (-0.1496085 * float(x[15]))+ (1.7625645 * float(x[16]))) + 1.0608032), 0)
    h_5 = max((((-0.13153586 * float(x[0]))+ (0.8434396 * float(x[1]))+ (2.3213294 * float(x[2]))+ (4.391505 * float(x[3]))+ (0.97964394 * float(x[4]))+ (2.3663018 * float(x[5]))+ (4.2450113 * float(x[6]))+ (4.2418737 * float(x[7]))+ (4.49838 * float(x[8]))+ (3.997499 * float(x[9]))+ (2.2032492 * float(x[10]))+ (4.125912 * float(x[11]))+ (5.101725 * float(x[12]))+ (4.6975718 * float(x[13]))+ (4.3060036 * float(x[14]))+ (3.6587987 * float(x[15]))+ (3.6154594 * float(x[16]))) + 5.7083335), 0)
    h_6 = max((((5.3761687 * float(x[0]))+ (-1.2321938 * float(x[1]))+ (2.3515732 * float(x[2]))+ (4.558859 * float(x[3]))+ (-0.5545768 * float(x[4]))+ (-0.2600698 * float(x[5]))+ (5.6244855 * float(x[6]))+ (2.801304 * float(x[7]))+ (4.8023443 * float(x[8]))+ (4.783461 * float(x[9]))+ (2.1297941 * float(x[10]))+ (2.4175706 * float(x[11]))+ (2.9815414 * float(x[12]))+ (3.549954 * float(x[13]))+ (2.461643 * float(x[14]))+ (1.8257545 * float(x[15]))+ (2.108629 * float(x[16]))) + 4.1594524), 0)
    h_7 = max((((5.5668206 * float(x[0]))+ (-1.1250274 * float(x[1]))+ (1.3090105 * float(x[2]))+ (2.734914 * float(x[3]))+ (0.08888318 * float(x[4]))+ (-0.03949071 * float(x[5]))+ (1.2987531 * float(x[6]))+ (3.272518 * float(x[7]))+ (3.4764767 * float(x[8]))+ (4.082551 * float(x[9]))+ (-0.06347847 * float(x[10]))+ (3.6656587 * float(x[11]))+ (3.4266133 * float(x[12]))+ (4.9179335 * float(x[13]))+ (5.777098 * float(x[14]))+ (2.2408931 * float(x[15]))+ (2.879917 * float(x[16]))) + 4.0010657), 0)
    h_8 = max((((5.456721 * float(x[0]))+ (-1.601136 * float(x[1]))+ (2.1777558 * float(x[2]))+ (4.0647016 * float(x[3]))+ (-1.5001559 * float(x[4]))+ (0.7677582 * float(x[5]))+ (2.2007265 * float(x[6]))+ (2.7595375 * float(x[7]))+ (4.8789988 * float(x[8]))+ (4.8462677 * float(x[9]))+ (0.7918397 * float(x[10]))+ (3.4898534 * float(x[11]))+ (2.988355 * float(x[12]))+ (3.64919 * float(x[13]))+ (4.7811356 * float(x[14]))+ (1.9606248 * float(x[15]))+ (1.9339566 * float(x[16]))) + 3.8191545), 0)
    h_9 = max((((5.4682956 * float(x[0]))+ (-0.34513175 * float(x[1]))+ (2.4382644 * float(x[2]))+ (1.7089938 * float(x[3]))+ (2.4022596 * float(x[4]))+ (-1.0895774 * float(x[5]))+ (4.735426 * float(x[6]))+ (3.31288 * float(x[7]))+ (2.2721858 * float(x[8]))+ (3.5399034 * float(x[9]))+ (0.8338708 * float(x[10]))+ (1.1597954 * float(x[11]))+ (3.296873 * float(x[12]))+ (4.684513 * float(x[13]))+ (3.1908207 * float(x[14]))+ (1.9971008 * float(x[15]))+ (3.6831105 * float(x[16]))) + 3.4084268), 0)
    h_10 = max((((-0.09992709 * float(x[0]))+ (1.4208956 * float(x[1]))+ (0.83279824 * float(x[2]))+ (2.6618185 * float(x[3]))+ (2.2361393 * float(x[4]))+ (0.78790754 * float(x[5]))+ (3.683332 * float(x[6]))+ (3.2996142 * float(x[7]))+ (4.615961 * float(x[8]))+ (4.66278 * float(x[9]))+ (1.9309881 * float(x[10]))+ (2.9125834 * float(x[11]))+ (3.9490466 * float(x[12]))+ (3.7836688 * float(x[13]))+ (3.779663 * float(x[14]))+ (2.801981 * float(x[15]))+ (3.2731013 * float(x[16]))) + 3.0696797), 0)
    h_11 = max((((0.0 * float(x[0]))+ (0.0 * float(x[1]))+ (0.0 * float(x[2]))+ (0.0 * float(x[3]))+ (0.0 * float(x[4]))+ (0.0 * float(x[5]))+ (0.0 * float(x[6]))+ (0.0 * float(x[7]))+ (0.0 * float(x[8]))+ (0.0 * float(x[9]))+ (0.0 * float(x[10]))+ (0.0 * float(x[11]))+ (0.0 * float(x[12]))+ (0.0 * float(x[13]))+ (0.0 * float(x[14]))+ (0.0 * float(x[15]))+ (0.0 * float(x[16]))) + 0.0), 0)
    o[0] = (2.6357982 * h_0)+ (-1.4375248 * h_1)+ (-2.4894166 * h_2)+ (0.3191359 * h_3)+ (-0.5140623 * h_4)+ (-5.658434 * h_5)+ (4.6151233 * h_6)+ (4.5454636 * h_7)+ (4.6000724 * h_8)+ (4.6452093 * h_9)+ (-11.415076 * h_10)+ (0.0 * h_11) + 0.60728955
    o[1] = (-0.90699804 * h_0)+ (3.1415029 * h_1)+ (4.2143946 * h_2)+ (1.0742179 * h_3)+ (2.3104002 * h_4)+ (3.0477874 * h_5)+ (-3.6509266 * h_6)+ (-3.4354706 * h_7)+ (-3.641611 * h_8)+ (-3.4127023 * h_9)+ (8.620768 * h_10)+ (0.0 * h_11) + -6.4267936
    o[2] = (-0.99314946 * h_0)+ (3.028154 * h_1)+ (4.0900245 * h_2)+ (1.6207507 * h_3)+ (2.2910826 * h_4)+ (5.521676 * h_5)+ (-3.390363 * h_6)+ (-3.102414 * h_7)+ (-3.4147482 * h_8)+ (-3.0580242 * h_9)+ (3.9559932 * h_10)+ (0.0 * h_11) + 8.399304
    o[3] = (-1.0837597 * h_0)+ (3.071898 * h_1)+ (4.1945043 * h_2)+ (0.7972549 * h_3)+ (2.2205997 * h_4)+ (6.0379033 * h_5)+ (-3.2153685 * h_6)+ (-2.9981382 * h_7)+ (-3.1384287 * h_8)+ (-3.0496955 * h_9)+ (3.4854448 * h_10)+ (0.0 * h_11) + -7.757072
    o[4] = (-0.8326444 * h_0)+ (3.1305203 * h_1)+ (4.1029315 * h_2)+ (1.4282558 * h_3)+ (2.125824 * h_4)+ (6.3449492 * h_5)+ (-3.5464036 * h_6)+ (-3.3993838 * h_7)+ (-3.4715984 * h_8)+ (-3.6000035 * h_9)+ (3.9848235 * h_10)+ (0.0 * h_11) + 4.7681293
    o[5] = (-1.0352255 * h_0)+ (3.075677 * h_1)+ (4.1118145 * h_2)+ (1.3653888 * h_3)+ (2.1653264 * h_4)+ (4.462901 * h_5)+ (-3.3673022 * h_6)+ (-2.8912911 * h_7)+ (-2.9940524 * h_8)+ (-3.4317265 * h_9)+ (5.3874645 * h_10)+ (0.0 * h_11) + 1.9067502
    o[6] = (-0.9909773 * h_0)+ (3.1178043 * h_1)+ (4.1221495 * h_2)+ (1.4369595 * h_3)+ (1.9758148 * h_4)+ (4.6927333 * h_5)+ (-2.9561913 * h_6)+ (-3.391488 * h_7)+ (-3.2076995 * h_8)+ (-3.3139274 * h_9)+ (5.3066564 * h_10)+ (0.0 * h_11) + -0.36190692

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


#for classifying batches
def classify(arr):
    outputs = []
    for row in arr:
        outputs.append(single_classify(row))
    return outputs


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
            pred = str(get_key(int(single_classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            writer.writerow(row)


def Validate(arr):
    if n_classes == 2:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        outputs=[]
        for i, row in enumerate(arr):
            outputs.append(int(single_classify(arr[i, :-1].tolist())))
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
            pred = int(single_classify(arr[i].tolist()))
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
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
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
        print("Classifier Type: Neural Network")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
            #Correct Labels
            true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap = 307
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


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)
