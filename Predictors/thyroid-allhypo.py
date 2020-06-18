#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/4533694/phpqJqmHb -o Predictors/thyroid-allhypo_QC.py -target Class -stopat 71.18 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:00:03.50. Finished on: May-21-2020 20:23:04.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        5-way classifier
Best-guess accuracy:                58.35%
Model accuracy:                     76.53% (2143/2800 correct)
Improvement over best guess:        18.18% (of possible 41.65%)
Model capacity (MEC):               971 bits
Generalization ratio:               2.20 bits/bit
Confusion Matrix:
 [6.54% 2.00% 1.18% 0.07% 0.04%]
 [2.57% 47.64% 6.93% 0.96% 0.18%]
 [1.04% 6.39% 19.64% 0.46% 0.00%]
 [0.11% 0.68% 0.29% 2.11% 0.07%]
 [0.00% 0.36% 0.14% 0.00% 0.61%]

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
TRAINFILE = "phpqJqmHb.csv"


#Number of attributes
num_attr = 26
n_classes = 5


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="Class"


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
    clean.mapping={'3': 0, '1': 1, '5': 2, '2': 3, '4': 4}

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
energy_thresholds = array([119.56, 122.405, 144.787483, 160.039966, 166.79500000000002, 168.37, 170.365, 171.395, 173.88, 177.79, 180.21748300000002, 180.437483, 182.005448, 184.47, 184.685, 186.718558, 189.118558, 193.54, 195.535, 198.055, 199.3475, 199.49, 200.8, 202.15044799999998, 205.32999999999998, 205.855, 206.248956, 206.457514, 206.975, 207.19248299999998, 207.862483, 208.565, 208.8425, 209.02, 212.592483, 213.12, 213.865, 214.71, 216.815, 217.38, 218.155, 219.19, 220.06, 220.491075, 221.325, 221.76, 221.87, 221.981075, 222.111075, 222.14999999999998, 222.26, 222.76748300000003, 223.38748300000003, 223.62, 223.64, 223.755, 224.07, 224.49, 226.17, 226.49, 226.715, 226.927483, 227.412483, 227.85, 228.165, 228.32999999999998, 228.41000000000003, 228.63500000000002, 229.26, 229.765, 229.95, 230.035, 230.215, 230.485, 230.578558, 231.282483, 231.42249999999999, 231.815, 231.91, 232.0, 232.035, 232.32, 233.315, 233.56, 233.60000000000002, 233.65855800000003, 233.718558, 233.925, 234.215, 234.51999999999998, 234.72, 235.17000000000002, 235.52, 235.56, 235.61, 235.803558, 235.93, 236.187483, 236.372483, 236.45749999999998, 236.888558, 237.14000000000001, 237.375, 237.498558, 238.265, 238.33999999999997, 238.48000000000002, 240.613558, 240.65, 241.15499999999997, 242.387483, 242.407483, 242.446041, 242.763558, 242.81, 242.875, 243.185, 243.802483, 243.86, 244.492948, 244.60500000000002, 244.71748300000002, 244.747483, 244.78, 244.925, 245.235, 245.285, 245.365, 246.672483, 246.794966, 246.952483, 247.14499999999998, 247.267483, 247.387483, 247.49, 247.598558, 247.688558, 247.745, 247.84, 247.935, 247.95499999999998, 247.995, 248.03000000000003, 248.13500000000002, 248.40500000000003, 248.70000000000002, 249.195, 249.52355799999998, 249.713558, 249.995, 250.16, 250.787483, 250.987483, 251.25248299999998, 251.69, 251.89855799999998, 252.015, 252.12748299999998, 252.32, 252.48294800000002, 252.5741315, 252.60411449999998, 252.84000000000003, 252.992948, 253.157948, 253.252483, 253.472483, 253.755, 253.87748299999998, 254.832948, 254.90248300000002, 254.99, 255.09, 255.185, 255.225, 255.26, 255.32999999999998, 255.41, 255.465, 255.60000000000002, 255.687483, 255.732483, 255.757483, 255.792948, 255.815, 255.83499999999998, 255.982483, 256.14, 256.275, 256.365, 256.475, 256.67065449999996, 256.895, 256.95000000000005, 257.293558, 257.62, 257.71500000000003, 257.91499999999996, 258.07, 258.117483, 258.212483, 258.432483, 258.675, 258.895, 258.96000000000004, 259.012483, 259.25, 259.3375, 259.42855799999995, 259.845, 259.935, 260.0325, 260.08544800000004, 260.087948, 260.165, 260.615, 260.65999999999997, 261.002948, 261.12294799999995, 261.365, 261.59000000000003, 261.62, 261.655, 261.672483, 261.685431, 261.73794799999996, 261.80499999999995, 261.845, 261.982948, 262.411075, 262.46999999999997, 262.622483, 262.765, 263.39500000000004, 263.52500000000003, 263.855, 263.885, 263.99, 264.095, 264.4977065, 264.6177065, 264.73, 264.78499999999997, 264.78999999999996, 264.855, 264.995, 265.17, 265.315, 265.435, 265.53, 265.64, 265.66999999999996, 265.727483, 265.967948, 266.185, 266.507948, 266.5427065, 266.6227065, 266.73, 266.86, 266.957483, 267.387948, 267.412948, 267.866075, 268.092483, 268.44000000000005, 268.56, 268.7, 269.105, 269.217948, 269.267948, 269.41499999999996, 269.5327065, 270.09000000000003, 270.262948, 270.6402235, 270.67499999999995, 270.98, 271.302483, 271.45, 271.5, 271.52, 271.5477065, 271.5827065, 271.612483, 271.95500000000004, 271.995, 272.21107500000005, 272.26, 272.37, 272.52, 272.725, 272.75, 272.877948, 272.997948, 273.142483, 273.17748300000005, 273.265, 273.4527065, 273.555413, 273.5951895, 273.67248300000006, 273.777483, 273.977483, 274.185, 274.322948, 274.521075, 274.695, 274.74, 274.78, 274.78499999999997, 274.83500000000004, 274.95, 275.095, 275.182948, 275.202948, 275.22, 275.275, 275.506075, 275.541075, 275.5527065, 275.66999999999996, 275.7152235, 275.7552235, 275.81, 275.875, 275.91499999999996, 276.06248300000004, 276.125, 276.155, 276.22, 276.80499999999995, 276.83000000000004, 276.84000000000003, 276.87, 276.92607499999997, 276.991075, 277.17, 277.222483, 277.472483, 277.56, 277.987483, 278.444966, 278.462483, 278.475, 278.49, 278.5277065, 278.777483, 279.03499999999997, 279.03999999999996, 279.041075, 279.15294800000004, 279.365, 279.638558, 279.94748300000003, 279.962483, 279.98, 280.002483, 280.047483, 280.115, 280.19, 280.262948, 280.625, 280.89, 281.23, 281.34000000000003, 281.39, 281.42999999999995, 281.5027065, 281.72, 281.81, 281.925, 282.045, 282.12, 282.18, 282.482483, 282.6, 282.76, 282.917483, 283.1591145, 283.207483, 283.34000000000003, 283.395, 283.435, 283.53, 283.63, 283.65999999999997, 284.027948, 284.102483, 284.19248300000004, 284.46770649999996, 284.7195795, 284.87294799999995, 284.91999999999996, 284.96000000000004, 284.97, 284.9766315, 285.0216315, 285.065, 285.11, 285.217948, 285.61, 285.725, 285.84499999999997, 285.97499999999997, 286.277483, 286.357948, 286.427948, 286.49, 286.52, 286.5427065, 286.818558, 286.865, 287.09000000000003, 287.19248300000004, 287.280431, 287.35294799999997, 287.43855800000006, 287.728558, 287.83, 287.96500000000003, 288.095, 288.185, 288.2, 288.23, 288.295, 288.345, 288.355, 288.51794800000005, 288.5527065, 288.825, 288.975, 289.115, 289.15999999999997, 289.33000000000004, 289.405, 289.46000000000004, 289.5227065, 289.677483, 289.81, 290.105, 290.255, 290.305, 290.412483, 290.51, 290.5427065, 290.64270650000003, 290.807948, 291.0, 291.102483, 291.5827065, 291.665, 291.845, 291.911075, 292.222483, 292.292483, 292.352483, 292.375, 292.385, 292.425, 292.485, 292.51, 292.87, 292.952948, 293.21, 293.565, 293.63, 293.705, 293.8452235, 293.95500000000004, 294.06, 294.125, 294.382948, 294.525, 294.57, 294.655, 294.90999999999997, 294.98, 295.0152235, 295.0502235, 295.2, 295.282948, 295.53499999999997, 295.882948, 296.22, 296.367948, 296.385896, 296.40294800000004, 296.43, 296.5227065, 296.81, 296.90500000000003, 297.015, 297.082483, 297.102483, 297.1202235, 297.15522350000003, 297.20000000000005, 297.225, 297.47, 297.753558, 297.775, 297.89, 298.01, 298.07, 298.257483, 298.412483, 298.5077065, 298.58770649999997, 298.655, 298.75, 298.85, 299.0325, 299.225, 299.375, 299.5377065, 299.63770650000004, 299.77, 299.82500000000005, 299.855, 299.895, 299.96, 300.06, 300.0725, 300.0875, 300.5067295, 300.5677065, 300.615, 300.74, 300.83, 300.932483, 301.117483, 301.2, 301.337483, 301.4477065, 301.5777065, 301.65, 301.71000000000004, 301.76, 301.998558, 302.02, 302.13, 302.32, 302.355, 302.61, 302.735, 302.8, 302.9, 303.187948, 303.3302235, 303.41022350000003, 303.615, 303.695, 303.78, 303.882483, 303.917483, 303.942948, 304.33500000000004, 304.385, 304.45500000000004, 304.51, 304.5327065, 304.5677065, 304.6052235, 304.777483, 304.925, 305.232483, 305.4177065, 306.072483, 306.185, 306.225, 306.2716315, 306.287948, 306.335, 306.385, 306.395, 306.65999999999997, 306.844983, 307.1581715, 307.187948, 307.41999999999996, 307.5527065, 307.905, 307.97, 308.29999999999995, 308.40248299999996, 308.64270650000003, 308.7302235, 308.7552235, 308.805431, 309.135, 309.16499999999996, 309.267483, 309.28294800000003, 309.322948, 309.485, 309.5777065, 309.735, 310.4275, 310.53999999999996, 310.5477065, 310.7325, 310.8566485, 311.014966, 311.17499999999995, 311.28499999999997, 311.375, 311.47270649999996, 311.632483, 311.64, 311.7991485, 312.072948, 312.19, 312.225, 312.442483, 312.5277065, 312.719338, 313.047, 313.117483, 313.227483, 313.59270649999996, 313.65, 313.815, 313.85294799999997, 314.06, 314.20500000000004, 314.282483, 314.52, 314.5427065, 314.68499999999995, 314.91499999999996, 314.94, 315.235, 315.28, 315.41499999999996, 315.813558, 316.08, 316.17499999999995, 316.2652235, 316.958558, 317.0252235, 317.0452235, 317.122483, 317.417483, 317.5527065, 317.885, 317.904983, 318.00498300000004, 318.14, 318.315, 318.501075, 318.57000000000005, 318.651075, 318.985, 319.0266315, 319.04411450000003, 319.06248300000004, 319.205, 319.79248300000006, 319.862483, 320.097483, 320.627483, 320.847483, 320.93, 321.025, 321.3275, 321.4975, 321.5652235, 321.905, 322.35, 322.4977065, 322.6227065, 322.93, 323.07000000000005, 323.175, 323.265, 323.29499999999996, 323.325, 323.45500000000004, 323.53999999999996, 323.545, 323.72, 323.917483, 324.13, 324.15748299999996, 324.32248300000003, 324.585, 324.84000000000003, 324.902483, 325.027483, 325.15794800000003, 325.387948, 325.63, 325.863558, 325.935, 325.975, 326.03499999999997, 326.10794799999996, 326.1302235, 326.552948, 326.755, 326.79499999999996, 326.81, 326.977483, 327.042483, 327.135, 327.235, 327.255, 327.70000000000005, 327.72, 327.735, 327.9166315, 328.242483, 328.355, 328.595, 328.795, 328.90500000000003, 329.3977065, 329.656075, 329.69000000000005, 329.74, 329.78499999999997, 330.12, 330.165, 330.35, 330.45248300000003, 330.575, 330.73, 330.865, 331.0, 331.275, 331.48, 331.515, 331.53499999999997, 331.71000000000004, 331.97, 332.07663149999996, 332.4, 332.57, 332.78499999999997, 333.0, 333.3, 333.47, 333.57, 333.7, 333.89, 334.04499999999996, 334.095, 334.262948, 334.35, 334.40999999999997, 334.45500000000004, 334.525, 334.69500000000005, 334.70750000000004, 334.739983, 334.867483, 335.0, 335.20248300000003, 335.315, 335.42, 335.68, 335.905, 336.170431, 336.262483, 336.28, 336.456075, 336.51770650000003, 336.5677065, 336.777483, 336.992483, 337.03, 337.2575, 337.482483, 337.592483, 337.814983, 338.3675, 338.5675, 338.78999999999996, 339.5, 339.66999999999996, 339.735, 340.095, 340.3075, 340.53499999999997, 340.765, 341.025, 341.265, 341.32, 341.39, 341.661075, 341.685, 341.78248299999996, 341.91748299999995, 341.97, 342.27748299999996, 342.325, 342.37, 342.475, 342.635, 342.875, 343.362483, 343.472483, 343.69748300000003, 343.84000000000003, 343.877483, 344.22, 344.372483, 344.437483, 344.485, 344.53499999999997, 344.781075, 345.001075, 345.087483, 345.207483, 345.775, 345.85294799999997, 346.347948, 347.23650599999996, 347.589983, 347.877466, 348.092483, 348.182483, 349.253558, 350.506075, 351.237483, 351.40999999999997, 351.495, 352.229983, 352.6425, 352.8275, 353.0475, 353.8, 353.86, 353.87, 353.94, 354.635, 354.969983, 355.27, 355.345, 355.5325, 355.605, 355.96498299999996, 356.095, 356.147483, 356.17248300000006, 356.492483, 356.85607500000003, 357.13, 357.265, 357.32748300000003, 357.572483, 358.02248299999997, 358.352483, 358.83500000000004, 358.88, 359.485, 359.766075, 359.861075, 359.932483, 360.64748299999997, 361.35, 362.13, 362.22, 362.3, 362.495, 363.015, 363.425, 363.46000000000004, 364.26, 364.616075, 365.22, 365.58500000000004, 366.04499999999996, 366.15, 367.54, 368.405, 368.78999999999996, 369.08, 369.21000000000004, 370.307483, 370.796075, 371.145, 373.74, 373.865, 374.045, 374.65999999999997, 374.82748300000003, 374.925, 374.997483, 375.172483, 375.42499999999995, 375.52, 375.82, 376.225, 376.88, 378.031075, 378.20248300000003, 380.595, 380.76748299999997, 381.362483, 381.635, 383.507483, 383.93499999999995, 384.26, 384.47, 384.76748299999997, 384.92499999999995, 386.122948, 386.88, 388.507483, 388.96, 391.128558, 391.57, 392.0, 392.915, 395.4575, 395.59355800000003, 396.62, 396.7625, 399.53499999999997, 399.957483, 401.48249999999996, 401.5625, 401.797483, 402.2025, 402.355, 402.65248299999996, 403.957483, 404.862483, 405.837483, 408.69748300000003, 409.40500000000003, 411.667483, 411.773558, 412.078558, 412.65999999999997, 413.07, 416.02, 417.03, 421.7475, 422.71000000000004, 424.219983, 424.527483, 424.857483, 426.612483, 434.272483, 436.742483, 448.4625, 452.305, 455.2575, 458.60249999999996, 459.409983, 460.5525, 462.20500000000004, 462.765, 478.5475, 486.3175, 504.04999999999995, 509.50249999999994, 513.115, 517.04, 521.335, 527.052483, 544.7710750000001, 547.06, 555.735, 564.0250000000001, 565.5, 688.4575, 803.385])
labels = array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 4.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 4.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 4.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
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
        outputs[defaultindys] = 1.0
        return outputs
    return thresh_search(energys)

numthresholds = 971



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


