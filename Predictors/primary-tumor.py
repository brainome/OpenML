#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/3594/dataset_113_primary-tumor.arff -o Predictors/primary-tumor_QC.py -target class -stopat 44.57 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:02.71. Finished on: May-23-2020 16:00:05.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        21-way classifier
Best-guess accuracy:                26.10%
Model accuracy:                     51.91% (176/339 correct)
Improvement over best guess:        25.81% (of possible 73.9%)
Model capacity (MEC):               268 bits
Generalization ratio:               0.65 bits/bit
Confusion Matrix:
 [20.35% 0.59% 0.29% 0.00% 0.00% 1.18% 1.18% 0.00% 0.00% 0.29% 0.00%
  0.00% 0.00% 0.88% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.88% 3.24% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 2.36% 0.00% 0.00% 0.00%
  0.29% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [2.06% 0.29% 2.06% 0.00% 0.00% 0.00% 1.18% 0.59% 1.77% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.29% 0.00% 0.29% 0.00% 0.00% 0.00%]
 [0.59% 0.29% 0.00% 1.47% 0.00% 0.00% 0.00% 0.00% 1.47% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00%]
 [0.59% 0.00% 0.00% 0.00% 0.88% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.59%
  0.00% 0.59% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00% 0.00%]
 [0.88% 0.00% 0.29% 0.00% 0.00% 4.72% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.29% 0.29% 0.29% 0.29% 0.00% 0.00% 2.36% 0.00% 0.29% 0.00% 0.00% 0.00%
  0.00% 0.29% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [1.47% 0.59% 0.88% 0.00% 0.00% 0.00% 0.00% 3.83% 0.29% 0.59% 0.00% 0.00%
  0.00% 0.59% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 1.77% 0.29% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.59% 0.00% 0.59% 0.59% 0.00% 0.00% 0.59% 0.29% 0.29% 2.65% 0.00% 0.29%
  0.00% 0.88% 0.00% 0.29% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.59% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.29% 0.00% 0.00% 0.29% 0.00% 0.00% 0.29% 0.00% 0.00% 0.59%
  0.00% 0.29% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.88% 0.00% 0.59% 0.00% 0.00% 0.00% 0.00% 0.29% 1.77% 0.59% 0.00% 0.00%
  0.59% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.59% 0.29% 0.59% 0.59% 0.00% 0.00% 2.06% 0.59% 0.88% 0.88% 0.00% 0.29%
  0.00% 3.83% 0.29% 0.59% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.59% 0.00% 0.00% 0.00% 0.00% 0.29% 0.29% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.29% 1.18% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.29% 0.59% 0.00% 0.00% 0.00% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.59% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.29% 0.00% 0.29% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.29% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.29% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.29% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.29%]

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
TRAINFILE = "dataset_113_primary-tumor.csv"


#Number of attributes
num_attr = 17
n_classes = 21


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="class"


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
    clean.mapping={'lung': 0, 'breast': 1, 'ovary': 2, 'colon': 3, 'prostate': 4, "'head and neck'": 5, 'thyroid': 6, 'pancreas': 7, 'liver': 8, 'kidney': 9, "'salivary glands'": 10, 'rectum': 11, 'gallbladder': 12, 'stomach': 13, 'esophagus': 14, "'corpus uteri'": 15, 'testis': 16, "'cervix uteri'": 17, "'duoden and sm.int'": 18, 'bladder': 19, 'vagina': 20}

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
energy_thresholds = array([27403821378.5, 27513653611.0, 27513653611.0, 27513653611.0, 27832614783.0, 28110859076.0, 28280629618.0, 28388049169.0, 28509465061.0, 28826977301.5, 28937902334.5, 29057343427.5, 29084783227.0, 29084783227.0, 29084783227.0, 29092195654.0, 29153204365.5, 29232679233.0, 29288014367.5, 29311253287.0, 29342213257.0, 29353981821.0, 29442880012.0, 29493338534.0, 29519339762.0, 29519339762.0, 29519339762.0, 29519339762.0, 29519674700.5, 29520009639.0, 29524216374.5, 29528423110.0, 29528423110.0, 29554759276.5, 29581095443.0, 29581095443.0, 29592864007.0, 29609926251.5, 29660384773.5, 29705549615.0, 29705549615.0, 29705549615.0, 29731885781.5, 29758221948.0, 29758221948.0, 29760902127.0, 29767305296.0, 29784367540.5, 29805410026.5, 29809390268.0, 29809390268.0, 29809390268.0, 29809390268.0, 29809725206.5, 29810060145.0, 29815018887.0, 29831746193.0, 29852788679.0, 29862062601.0, 29862237621.0, 29862572559.5, 29903582139.5, 29997104134.0, 30022688294.0, 30048272454.0, 30101279725.5, 30129103446.5, 30156592229.0, 30169953108.0, 30183313987.0, 30222048454.5, 30273968781.0, 30341489547.5, 30369138248.5, 30395474415.0, 30395474415.0, 30408835294.0, 30422196173.0, 30460930640.5, 30543369688.0, 30560702550.0, 30560702550.0, 30571193409.0, 30581684268.0, 30597529575.5, 30623865742.0, 30699637806.5, 30764919012.0, 30764919012.0, 30808253102.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30851587192.0, 30872964616.5, 30925059963.0, 30955777885.0, 30963190312.0, 30987201968.5, 31008450218.0, 31023123631.5, 31037797045.0, 31037797045.0, 31037797045.0, 31037797045.0, 31037797045.0, 31037797045.0, 31037797045.0, 31037797045.0, 31044166117.5, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31090469378.0, 31101295175.5, 31122337661.5, 31236745043.0, 31256712137.0, 31276679231.0, 31276679231.0, 31276679231.0, 31276679231.0, 31276679231.0, 31276679231.0, 31276679231.0, 31277014169.5, 31287839967.0, 31308547514.5, 31324057883.5, 31329351564.0, 31329351564.0, 31329351564.0, 31329351564.0, 31329351564.0, 31329351564.0, 31329351564.0, 31329351564.0, 31340177361.5, 31361219847.5, 31371436536.0, 31411763318.5, 31498563493.5, 31515561417.0, 31521930489.5, 31542972975.5, 31557646389.0, 31557646389.0, 31557981327.5, 31563275008.0, 31568233750.0, 31568233750.0, 31568233750.0, 31568233750.0, 31568233750.0, 31568568688.5, 31599060116.5, 31629216606.0, 31629216606.0, 31629216606.0, 31629216606.0, 31629216606.0, 31650820524.5, 31681698365.0, 31690972287.0, 31712014773.0, 31733057259.0, 31743750431.0, 31754443603.0, 31754443603.0, 31754443603.0, 31754443603.0, 31754443603.0, 31754443603.0, 31780779769.5, 31807115936.0, 31811271197.5, 31815426459.0, 31815426459.0, 31832313683.5, 31849200908.0, 31884234010.0, 31924560792.5, 31950896959.0, 31971939445.0, 31971939445.0, 31982632617.0, 31993325789.0, 32014368275.0, 32040704441.5, 32050153383.5, 32054308645.0, 32071195869.5, 32088083094.0, 32140178440.5, 32238645352.0, 32275674690.5, 32284880308.0, 32284880308.0, 32289035569.5, 32345111157.5, 32477527538.0, 32505351259.0, 32509598134.5, 32518803752.0, 32527917755.5, 32583993343.5, 32690073557.5, 32717897278.5, 32809514650.5, 32918129946.0, 32961464036.0, 32961464036.0, 32961464036.0, 32961464036.0, 32961464036.0, 32961464036.0, 33054568962.5, 33158499686.5, 33184835853.0, 33200346222.0, 33200346222.0, 33200346222.0, 33200346222.0, 33200346222.0, 33293451148.5, 33386556075.0, 33386556075.0, 33386556075.0, 33386556075.0, 33386556075.0, 33433934727.5, 33439228408.0, 33439228408.0, 33439228408.0, 33460270894.0, 33481313380.0, 33481313380.0, 33481313380.0, 33481313380.0, 33579711987.0, 33678110594.0, 33678110594.0, 33699153080.0, 33720195566.0, 33792258006.5, 33864320447.0, 33864320447.0, 33864320447.0, 33885362933.0, 33911699099.5, 33938035266.0, 33959077752.0, 34057476359.0, 34296358545.0])
labels = array([5.0, 11.0, 0.0, 5.0, 10.0, 0.0, 8.0, 6.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 14.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 13.0, 2.0, 9.0, 0.0, 15.0, 2.0, 6.0, 13.0, 9.0, 13.0, 19.0, 2.0, 9.0, 5.0, 0.0, 20.0, 3.0, 6.0, 9.0, 13.0, 6.0, 2.0, 13.0, 0.0, 6.0, 10.0, 0.0, 13.0, 5.0, 0.0, 14.0, 0.0, 7.0, 1.0, 0.0, 13.0, 14.0, 5.0, 0.0, 7.0, 13.0, 0.0, 14.0, 0.0, 15.0, 13.0, 3.0, 9.0, 5.0, 0.0, 4.0, 17.0, 3.0, 2.0, 13.0, 4.0, 0.0, 9.0, 13.0, 9.0, 16.0, 4.0, 1.0, 6.0, 0.0, 7.0, 0.0, 8.0, 3.0, 1.0, 13.0, 1.0, 2.0, 11.0, 1.0, 2.0, 13.0, 1.0, 2.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0, 12.0, 0.0, 13.0, 4.0, 7.0, 4.0, 3.0, 9.0, 0.0, 9.0, 0.0, 1.0, 2.0, 0.0, 2.0, 7.0, 2.0, 1.0, 12.0, 0.0, 2.0, 1.0, 0.0, 5.0, 6.0, 14.0, 13.0, 9.0, 0.0, 13.0, 9.0, 0.0, 9.0, 6.0, 7.0, 1.0, 0.0, 13.0, 7.0, 1.0, 7.0, 15.0, 0.0, 3.0, 1.0, 7.0, 2.0, 0.0, 14.0, 13.0, 0.0, 13.0, 0.0, 7.0, 1.0, 15.0, 1.0, 13.0, 9.0, 17.0, 2.0, 13.0, 2.0, 13.0, 6.0, 11.0, 13.0, 7.0, 14.0, 13.0, 6.0, 2.0, 12.0, 9.0, 8.0, 7.0, 0.0, 13.0, 7.0, 9.0, 1.0, 2.0, 11.0, 9.0, 4.0, 0.0, 7.0, 5.0, 0.0, 6.0, 0.0, 13.0, 0.0, 3.0, 7.0, 13.0, 11.0, 13.0, 1.0, 6.0, 0.0, 7.0, 0.0, 12.0, 1.0, 7.0, 0.0, 12.0, 1.0, 13.0, 1.0, 4.0, 0.0, 8.0, 1.0, 19.0, 8.0, 12.0, 18.0, 3.0, 7.0, 12.0, 9.0, 13.0, 3.0, 8.0, 6.0, 12.0, 3.0, 13.0, 12.0, 13.0, 9.0, 4.0, 9.0, 7.0, 13.0, 7.0, 12.0, 9.0, 7.0, 2.0, 7.0, 15.0, 7.0, 12.0, 8.0, 3.0, 15.0, 9.0, 12.0, 13.0, 0.0, 9.0, 11.0, 7.0, 3.0, 7.0, 2.0, 7.0, 12.0])
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
        outputs[defaultindys] = 9.0
        return outputs
    return thresh_search(energys)

numthresholds = 268



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


