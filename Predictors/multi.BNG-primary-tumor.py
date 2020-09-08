#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target class BNG-primary-tumor.csv -o BNG-primary-tumor.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 2:45:21.48. Finished on: Sep-03-2020 15:50:02.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         22-way classifier
Best-guess accuracy:                 24.07%
Overall Model accuracy:              36.19% (361948/1000000 correct)
Overall Improvement over best guess: 12.12% (of possible 75.93%)
Model capacity (MEC):                522 bits
Generalization ratio:                693.38 bits/bit
Model efficiency:                    0.02%/parameter
Confusion Matrix:
 [0.45% 1.49% 0.45% 0.00% 0.04% 0.55% 0.00% 3.44% 0.00% 0.00% 0.02% 0.01%
  0.43% 0.00% 0.00% 0.10% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01%]
 [0.38% 3.37% 1.11% 0.01% 0.04% 1.62% 0.00% 3.49% 0.00% 0.00% 0.03% 0.03%
  0.94% 0.00% 0.00% 0.22% 0.00% 0.01% 0.01% 0.00% 0.00% 0.01%]
 [0.01% 0.38% 3.73% 0.00% 0.01% 1.56% 0.00% 0.64% 0.00% 0.01% 0.00% 0.01%
  0.58% 0.00% 0.00% 0.06% 0.00% 0.00% 0.00% 0.01% 0.00% 0.00%]
 [0.08% 0.67% 0.65% 0.01% 0.03% 0.39% 0.00% 1.31% 0.00% 0.00% 0.01% 0.02%
  0.73% 0.00% 0.00% 0.25% 0.00% 0.01% 0.01% 0.00% 0.00% 0.00%]
 [0.03% 0.30% 0.04% 0.00% 3.33% 0.07% 0.00% 1.68% 0.00% 0.00% 0.03% 0.01%
  0.07% 0.00% 0.00% 0.25% 0.00% 0.01% 0.00% 0.00% 0.00% 0.05%]
 [0.00% 0.69% 1.59% 0.00% 0.02% 4.83% 0.00% 0.40% 0.00% 0.00% 0.00% 0.04%
  0.83% 0.00% 0.00% 0.01% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.25% 0.58% 0.00% 0.02% 0.29% 0.00% 0.25% 0.00% 0.01% 0.01% 0.03%
  0.39% 0.00% 0.00% 0.02% 0.00% 0.01% 0.00% 0.01% 0.00% 0.00%]
 [0.37% 1.17% 2.46% 0.00% 1.02% 1.33% 0.00% 15.77% 0.00% 0.01% 0.06%
  0.05% 1.27% 0.00% 0.00% 0.39% 0.00% 0.02% 0.02% 0.03% 0.00% 0.09%]
 [0.07% 1.09% 1.22% 0.00% 0.02% 1.07% 0.00% 2.88% 0.00% 0.00% 0.01% 0.01%
  1.61% 0.00% 0.00% 0.16% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.07% 0.28% 0.00% 0.01% 0.04% 0.00% 0.15% 0.00% 0.02% 0.00% 0.02%
  0.05% 0.00% 0.00% 0.03% 0.00% 0.01% 0.01% 0.02% 0.00% 0.00%]
 [0.03% 0.73% 0.04% 0.00% 0.16% 0.08% 0.00% 0.47% 0.00% 0.00% 0.11% 0.03%
  0.11% 0.00% 0.00% 0.08% 0.00% 0.01% 0.00% 0.01% 0.00% 0.03%]
 [0.02% 1.03% 0.64% 0.00% 0.05% 0.65% 0.00% 0.82% 0.00% 0.01% 0.03% 0.12%
  0.56% 0.00% 0.00% 0.16% 0.00% 0.02% 0.01% 0.01% 0.00% 0.01%]
 [0.00% 0.22% 0.38% 0.00% 0.05% 0.23% 0.00% 0.58% 0.00% 0.00% 0.01% 0.01%
  3.23% 0.00% 0.00% 0.01% 0.00% 0.01% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.01% 0.01% 0.00% 0.00% 0.00% 0.00% 0.04% 0.00% 0.00% 0.01% 0.00%
  0.00% 0.02% 0.00% 0.00% 0.00% 0.00% 0.01% 0.01% 0.00% 0.02%]
 [0.03% 0.08% 0.21% 0.00% 0.10% 0.12% 0.00% 2.01% 0.00% 0.00% 0.00% 0.00%
  0.06% 0.00% 0.00% 0.05% 0.00% 0.00% 0.00% 0.01% 0.00% 0.03%]
 [0.05% 0.99% 0.03% 0.00% 0.10% 0.03% 0.00% 0.90% 0.00% 0.00% 0.04% 0.00%
  0.03% 0.00% 0.00% 0.80% 0.00% 0.01% 0.01% 0.00% 0.00% 0.01%]
 [0.00% 0.05% 0.05% 0.00% 0.01% 0.01% 0.00% 0.14% 0.00% 0.00% 0.01% 0.01%
  0.08% 0.00% 0.00% 0.01% 0.01% 0.01% 0.00% 0.01% 0.00% 0.00%]
 [0.07% 0.19% 0.24% 0.01% 0.03% 0.08% 0.00% 0.89% 0.00% 0.01% 0.01% 0.02%
  0.41% 0.00% 0.00% 0.11% 0.00% 0.06% 0.01% 0.00% 0.00% 0.00%]
 [0.00% 0.05% 0.04% 0.00% 0.02% 0.00% 0.00% 0.11% 0.00% 0.01% 0.01% 0.01%
  0.01% 0.00% 0.00% 0.07% 0.00% 0.01% 0.05% 0.01% 0.00% 0.03%]
 [0.00% 0.02% 0.04% 0.00% 0.01% 0.00% 0.00% 0.21% 0.00% 0.01% 0.01% 0.01%
  0.01% 0.00% 0.00% 0.00% 0.00% 0.00% 0.01% 0.08% 0.00% 0.03%]
 [0.01% 0.07% 0.03% 0.00% 0.02% 0.01% 0.00% 0.45% 0.00% 0.00% 0.01% 0.00%
  0.03% 0.00% 0.00% 0.04% 0.00% 0.00% 0.01% 0.00% 0.00% 0.01%]
 [0.00% 0.03% 0.01% 0.00% 0.06% 0.00% 0.00% 0.33% 0.00% 0.00% 0.02% 0.01%
  0.00% 0.00% 0.00% 0.01% 0.00% 0.00% 0.01% 0.03% 0.00% 0.22%]
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to 'kidney'=0, 'stomach'=1, 'breast'=2, 'colon'=3, ''head and neck''=4, 'ovary'=5, ''corpus uteri''=6, 'lung'=7, 'pancreas'=8, ''cervix uteri''=9, 'rectum'=10, 'thyroid'=11, 'gallbladder'=12, 'anus'=13, 'esophagus'=14, 'prostate'=15, ''duoden and sm.int''=16, 'liver'=17, 'testis'=18, 'vagina'=19, 'bladder'=20, ''salivary glands''=21.
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
TRAINFILE = "BNG-primary-tumor.csv"


#Number of attributes
num_attr = 17
n_classes = 22


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="class"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="class"
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
    clean.mapping={'kidney': 0, 'stomach': 1, 'breast': 2, 'colon': 3, "'head and neck'": 4, 'ovary': 5, "'corpus uteri'": 6, 'lung': 7, 'pancreas': 8, "'cervix uteri'": 9, 'rectum': 10, 'thyroid': 11, 'gallbladder': 12, 'anus': 13, 'esophagus': 14, 'prostate': 15, "'duoden and sm.int'": 16, 'liver': 17, 'testis': 18, 'vagina': 19, 'bladder': 20, "'salivary glands'": 21}

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
energy_thresholds = array([25114980532.5, 25181282924.5, 25274387851.0, 25353862718.5, 25395292156.5, 25433795169.5, 25499909158.0, 25553047859.0, 25567871950.5, 25594208117.0, 25634174342.5, 25700476734.5, 25713918390.0, 25791930045.0, 25806754136.5, 25833090303.0, 25858232437.0, 25859695649.5, 25873056528.5, 25886686587.5, 25939358920.5, 25952800576.0, 26032275443.5, 26071972489.0, 26097114623.0, 26098577835.5, 26111938714.5, 26151904940.0, 26178241106.5, 26191682762.0, 26205124417.5, 26231460584.0, 26257796750.5, 26271157629.5, 26283324476.0, 26284787688.5, 26310854675.0, 26335996809.0, 26337460021.5, 26338923234.0, 26351090080.5, 26364450959.5, 26390787126.0, 26417123292.5, 26430564948.0, 26444006603.5, 26470342770.0, 26496678936.5, 26506895625.0, 26518793291.5, 26522206662.0, 26523669874.5, 26550006041.0, 26574878995.0, 26576342207.5, 26577805420.0, 26589972266.5, 26603333145.5, 26629669312.0, 26656005478.5, 26693105478.0, 26717978432.0, 26735561122.5, 26745777811.0, 26757675477.5, 26761088848.0, 26762552060.5, 26764015273.0, 26788888227.0, 26813761181.0, 26815224393.5, 26816687606.0, 26825710262.0, 26837607928.5, 26868551498.0, 26894887664.5, 26908329320.0, 26921770975.5, 26956860618.0, 26974443308.5, 26984659997.0, 26998020876.0, 26999971034.0, 27001434246.5, 27002897459.0, 27011920115.0, 27036523889.0, 27052643367.0, 27055569792.0, 27076490114.5, 27107433684.0, 27147211506.0, 27160653161.5, 27197206016.5, 27213325494.5, 27223542183.0, 27236903062.0, 27238853220.0, 27240316432.5, 27241779645.0, 27275406075.0, 27292988765.5, 27294451978.0, 27316835513.0, 27346315870.0, 27386093692.0, 27407071857.0, 27415825333.0, 27436088202.5, 27452207680.5, 27462424369.0, 27475785248.0, 27477735406.0, 27480661831.0, 27489684487.0, 27516020653.5, 27531870951.5, 27533334164.0, 27542356820.0, 27555717699.0, 27579840054.5, 27604713008.5, 27611534222.5, 27624975878.0, 27638417533.5, 27645954043.0, 27654707519.0, 27674970388.5, 27691089866.5, 27701306555.0, 27714667434.0, 27716617592.0, 27728566673.0, 27738783361.5, 27754902839.5, 27770753137.5, 27772216350.0, 27781239006.0, 27794599885.0, 27818722240.5, 27845058407.0, 27863858064.0, 27877299719.5, 27884836229.0, 27892071328.5, 27893534541.0, 27895052917.5, 27913852574.5, 27929972052.5, 27950405429.5, 27953818800.0, 27956962990.5, 27958426203.0, 27967448859.0, 27993785025.5, 28009904503.5, 28020121192.0, 28034945283.5, 28044243547.5, 28063758500.0, 28078281181.5, 28079744394.0, 28085403805.5, 28102740250.0, 28116181905.5, 28123718415.0, 28130953514.5, 28132416727.0, 28133935103.5, 28152734760.5, 28179070927.0, 28189287615.5, 28192700986.0, 28196114356.5, 28204812668.5, 28213566144.5, 28227309210.0, 28241965475.5, 28248786689.5, 28259003378.0, 28273827469.5, 28283125733.5, 28302640686.0, 28317163367.5, 28318626580.0, 28324285991.5, 28343085648.5, 28362600601.0, 28369835700.5, 28371298913.0, 28372817289.5, 28387609151.0, 28403728629.0, 28417953113.0, 28428169801.5, 28433046384.5, 28443694854.5, 28452448330.5, 28466191396.0, 28482310874.0, 28508102252.5, 28512709655.5, 28522007919.5, 28529323795.5, 28530787008.0, 28542986084.5, 28556045553.5, 28557508766.0, 28576609833.0, 28601482787.0, 28608717886.5, 28610181099.0, 28611699475.5, 28626491337.0, 28651364291.0, 28667051987.5, 28671928570.5, 28682577040.5, 28691330516.5, 28694312105.5, 28705073582.0, 28715533648.5, 28722656272.5, 28736767750.0, 28746984438.5, 28757745915.0, 28768205981.5, 28769669194.0, 28781868270.5, 28794927739.5, 28796390952.0, 28802050363.5, 28820850020.5, 28840364973.0, 28847600072.5, 28849031055.0, 28849332465.0, 28850581661.5, 28853261840.5, 28865373523.0, 28890246477.0, 28905934173.5, 28910810756.5, 28921459226.5, 28930212702.5, 28933194291.5, 28943955768.0, 28954415834.5, 28955879047.0, 28961538458.5, 28975649936.0, 28985866624.5, 28996628101.0, 29007088167.5, 29008551380.0, 29017606266.0, 29033809925.5, 29035240908.0, 29035542318.0, 29086482258.5, 29087913241.0, 29088214651.0, 29089463847.5, 29092144026.5, 29104255709.0, 29130591875.5, 29156928042.0, 29169094888.5, 29172076477.5, 29193298020.5, 29194761233.0, 29203816119.0, 29217927596.5, 29224748810.5, 29235510287.0, 29245970353.5, 29247433566.0, 29256488452.0, 29272692111.5, 29274123094.0, 29274424504.0, 29325364444.5, 29326795427.0, 29327096837.0, 29328346033.5, 29331026212.5, 29343137895.0, 29369474061.5, 29395810228.0, 29407977074.5, 29410958663.5, 29421720140.0, 29432180206.5, 29433643419.0, 29442698305.0, 29456809782.5, 29474392473.0, 29484852539.5, 29486315752.0, 29495370638.0, 29505285916.5, 29506749129.0, 29508731517.0, 29511574297.5, 29513005280.0, 29513306690.0, 29538211874.0, 29565677613.0, 29565979023.0, 29568691432.0, 29582020081.0, 29608356247.5, 29634692414.0, 29646859260.5, 29660602326.0, 29671062392.5, 29672525605.0, 29681580491.0, 29691495769.5, 29692958982.0, 29697155181.0, 29702513182.5, 29713274659.0, 29723734725.5, 29725197938.0, 29734252824.0, 29744168102.5, 29745631315.0, 29747613703.0, 29750456483.5, 29751887466.0, 29752188876.0, 29777094060.0, 29803128816.5, 29804559799.0, 29804861209.0, 29815077897.5, 29828406546.5, 29857886903.5, 29887204659.0, 29899484512.0, 29911407791.0, 29920462677.0, 29937500579.5, 29952156845.0, 29962616911.5, 29964080124.0, 29983050288.5, 29984513501.0, 29986495889.0, 29989338669.5, 29990769652.0, 29991071062.0, 30026160704.5, 30042011002.5, 30043441985.0, 30043743395.0, 30053960083.5, 30067288732.5, 30096769089.5, 30138366698.0, 30150289977.0, 30185681029.5, 30201499097.5, 30202962310.0, 30221932474.5, 30223395687.0, 30225378075.0, 30228220855.5, 30229651838.0, 30229953248.0, 30240169936.5, 30266204693.0, 30282324171.0, 30282625581.0, 30292842269.5, 30306170918.5, 30335651275.5, 30374267295.0, 30396708672.5, 30405462148.5, 30424563215.5, 30440381283.5, 30441844496.0, 30460814660.5, 30462277873.0, 30465422063.5, 30468534024.0, 30468835434.0, 30479052122.5, 30505356059.0, 30521507767.0, 30531724455.5, 30545053104.5, 30569175460.0, 30595511626.5, 30613149481.0, 30628054349.0, 30635590858.5, 30644344334.5, 30664607204.0, 30680726682.0, 30699696846.5, 30701160059.0, 30704573429.5, 30717902078.5, 30744238245.0, 30770574411.5, 30783935290.5, 30808057646.0, 30847835468.0, 30866936535.0, 30874473044.5, 30883226520.5, 30903489390.0, 30919608868.0, 30938579032.5, 30940042245.0, 30943455615.5, 30983120431.0, 31009456597.5, 31030434762.5, 31054557118.0, 31086717654.0, 31105818721.0, 31113355230.5, 31122108706.5, 31142672986.0, 31167545940.0, 31177461218.5, 31178924431.0, 31182337801.5, 31195666450.5, 31216644615.5, 31232764093.5, 31248338783.5, 31269316948.5, 31293439304.0, 31325599840.0, 31344700907.0, 31352237416.5, 31363671071.5, 31390308648.0, 31417806617.0, 31421219987.5, 31434548636.5, 31455526801.5, 31481862968.0, 31508199134.5, 31532321490.0, 31564783436.0, 31591119602.5, 31599873078.5, 31602553257.5, 31604016470.0, 31629190834.0, 31656688803.0, 31660102173.5, 31673430822.5, 31694408987.5, 31720745154.0, 31747081320.5, 31771203676.0, 31839917067.0, 31842898656.0, 31869234822.5, 31895570989.0, 31909201048.0, 31933291173.5, 31959627340.0, 31985963506.5, 32006941671.5, 32020302550.5, 32049782907.5, 32078799253.0, 32081780842.0, 32108117008.5, 32134453175.0, 32148083234.0, 32172173359.5, 32193151524.5, 32208726214.5, 32224845692.5, 32259184736.5, 32317681439.0, 32320663028.0, 32360629253.5, 32411055545.5, 32432033710.5, 32458369877.0, 32498066922.5, 32527547279.5, 32596529850.5, 32649937731.5, 32670915896.5, 32697252063.0, 32723588229.5, 32736949108.5, 32766429465.5, 32846173513.0, 32909798082.5, 32936134249.0, 32962470415.5, 32975831294.5, 33148680268.5, 33175016435.0, 33201352601.5, 33214713480.5, 33297601718.5, 33387562454.5, 33413898621.0, 33440234787.5, 33533339714.0, 33626444640.5, 33652780807.0, 33679116973.5, 34140761867.5, 34260202960.5, 34379644053.5, 34499085146.5, 34618526239.5, 34857408425.5])
labels = array([11.0, 7.0, 0.0, 11.0, 7.0, 4.0, 17.0, 11.0, 10.0, 11.0, 4.0, 7.0, 17.0, 7.0, 4.0, 11.0, 7.0, 11.0, 4.0, 7.0, 4.0, 17.0, 7.0, 19.0, 4.0, 11.0, 4.0, 10.0, 4.0, 17.0, 15.0, 17.0, 11.0, 21.0, 4.0, 7.0, 11.0, 4.0, 11.0, 7.0, 21.0, 4.0, 19.0, 7.0, 17.0, 4.0, 17.0, 11.0, 17.0, 21.0, 4.0, 7.0, 13.0, 7.0, 11.0, 4.0, 21.0, 4.0, 19.0, 7.0, 17.0, 9.0, 11.0, 17.0, 21.0, 4.0, 7.0, 4.0, 13.0, 7.0, 11.0, 4.0, 7.0, 21.0, 19.0, 7.0, 18.0, 7.0, 13.0, 9.0, 7.0, 13.0, 4.0, 21.0, 4.0, 7.0, 13.0, 19.0, 7.0, 21.0, 19.0, 18.0, 7.0, 13.0, 9.0, 7.0, 13.0, 4.0, 21.0, 4.0, 19.0, 21.0, 7.0, 21.0, 19.0, 18.0, 1.0, 7.0, 13.0, 9.0, 7.0, 13.0, 21.0, 4.0, 7.0, 19.0, 21.0, 19.0, 7.0, 21.0, 1.0, 13.0, 19.0, 13.0, 18.0, 1.0, 7.0, 13.0, 9.0, 7.0, 13.0, 21.0, 7.0, 4.0, 19.0, 13.0, 19.0, 7.0, 21.0, 1.0, 19.0, 13.0, 18.0, 1.0, 17.0, 1.0, 7.0, 13.0, 19.0, 7.0, 21.0, 1.0, 21.0, 7.0, 13.0, 19.0, 7.0, 21.0, 3.0, 1.0, 10.0, 1.0, 21.0, 13.0, 18.0, 1.0, 11.0, 5.0, 7.0, 13.0, 19.0, 7.0, 19.0, 21.0, 10.0, 7.0, 15.0, 13.0, 19.0, 7.0, 21.0, 2.0, 1.0, 10.0, 1.0, 19.0, 21.0, 1.0, 11.0, 5.0, 7.0, 10.0, 9.0, 16.0, 7.0, 21.0, 4.0, 7.0, 4.0, 19.0, 7.0, 21.0, 2.0, 17.0, 2.0, 1.0, 10.0, 1.0, 18.0, 1.0, 11.0, 5.0, 7.0, 4.0, 19.0, 7.0, 21.0, 4.0, 21.0, 7.0, 1.0, 15.0, 13.0, 19.0, 7.0, 2.0, 17.0, 2.0, 1.0, 10.0, 1.0, 16.0, 13.0, 1.0, 11.0, 7.0, 1.0, 21.0, 7.0, 4.0, 13.0, 19.0, 18.0, 10.0, 21.0, 7.0, 1.0, 4.0, 15.0, 18.0, 19.0, 7.0, 2.0, 17.0, 2.0, 1.0, 10.0, 7.0, 1.0, 11.0, 4.0, 1.0, 13.0, 7.0, 4.0, 19.0, 10.0, 21.0, 7.0, 4.0, 15.0, 0.0, 13.0, 19.0, 2.0, 11.0, 2.0, 1.0, 10.0, 4.0, 1.0, 11.0, 4.0, 1.0, 13.0, 21.0, 4.0, 7.0, 10.0, 19.0, 21.0, 18.0, 17.0, 15.0, 0.0, 19.0, 2.0, 17.0, 2.0, 1.0, 17.0, 7.0, 1.0, 10.0, 4.0, 7.0, 13.0, 12.0, 1.0, 21.0, 10.0, 19.0, 10.0, 21.0, 18.0, 17.0, 15.0, 0.0, 17.0, 7.0, 13.0, 19.0, 2.0, 17.0, 2.0, 1.0, 7.0, 5.0, 13.0, 21.0, 4.0, 1.0, 10.0, 16.0, 12.0, 11.0, 7.0, 21.0, 10.0, 21.0, 18.0, 15.0, 7.0, 19.0, 2.0, 9.0, 2.0, 7.0, 5.0, 13.0, 10.0, 4.0, 7.0, 16.0, 13.0, 19.0, 1.0, 4.0, 21.0, 7.0, 18.0, 7.0, 9.0, 16.0, 2.0, 7.0, 2.0, 18.0, 13.0, 4.0, 10.0, 4.0, 16.0, 19.0, 9.0, 7.0, 21.0, 19.0, 18.0, 1.0, 7.0, 2.0, 13.0, 2.0, 7.0, 2.0, 13.0, 7.0, 21.0, 7.0, 19.0, 11.0, 7.0, 21.0, 15.0, 19.0, 13.0, 18.0, 1.0, 7.0, 13.0, 2.0, 7.0, 2.0, 21.0, 7.0, 19.0, 7.0, 13.0, 1.0, 13.0, 18.0, 1.0, 7.0, 9.0, 2.0, 7.0, 2.0, 7.0, 13.0, 7.0, 3.0, 1.0, 13.0, 18.0, 1.0, 7.0, 18.0, 2.0, 16.0, 2.0, 13.0, 7.0, 15.0, 13.0, 7.0, 12.0, 1.0, 13.0, 9.0, 1.0, 7.0, 13.0, 2.0, 9.0, 7.0, 15.0, 7.0, 12.0, 1.0, 18.0, 1.0, 13.0, 20.0, 7.0, 16.0, 2.0, 16.0, 7.0, 15.0, 7.0, 12.0, 1.0, 13.0, 7.0, 13.0, 2.0, 7.0, 15.0, 19.0, 12.0, 1.0, 10.0, 1.0, 11.0, 7.0, 13.0, 2.0, 21.0, 15.0, 0.0, 19.0, 12.0, 10.0, 11.0, 2.0, 14.0, 15.0, 7.0, 12.0, 10.0, 11.0, 21.0, 15.0, 7.0, 2.0, 12.0, 10.0, 16.0, 15.0, 7.0, 2.0, 12.0, 15.0, 7.0, 2.0, 12.0, 15.0, 18.0, 7.0, 13.0, 12.0, 20.0, 7.0, 13.0, 7.0, 2.0, 7.0, 2.0, 7.0, 16.0, 13.0])
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
        outputs[defaultindys] = 17.0
        return outputs
    return thresh_search(energys)

numthresholds = 522



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


