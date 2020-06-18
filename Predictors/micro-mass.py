#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/1593706/phpSZJq5T -o Predictors/micro-mass_QC.py -target Class -stopat 96.94 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:01:13.40. Finished on: May-22-2020 21:23:18.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        10-way classifier
Best-guess accuracy:                10.00%
Model accuracy:                     79.44% (286/360 correct)
Improvement over best guess:        69.44% (of possible 90.0%)
Model capacity (MEC):               195 bits
Generalization ratio:               1.46 bits/bit
Confusion Matrix:
 [7.78% 0.28% 0.00% 0.00% 0.56% 0.28% 0.00% 0.28% 0.28% 0.56%]
 [0.28% 6.94% 0.00% 0.00% 0.56% 0.56% 0.56% 0.56% 0.28% 0.28%]
 [0.28% 0.00% 8.89% 0.56% 0.00% 0.00% 0.00% 0.00% 0.00% 0.28%]
 [0.00% 0.00% 0.00% 7.78% 1.11% 0.00% 0.28% 0.56% 0.00% 0.28%]
 [0.00% 0.00% 0.28% 0.28% 8.06% 0.28% 0.28% 0.28% 0.00% 0.56%]
 [0.56% 0.28% 0.00% 0.00% 0.00% 7.78% 0.00% 0.56% 0.56% 0.28%]
 [0.00% 0.00% 0.56% 0.28% 0.83% 0.00% 7.78% 0.00% 0.28% 0.28%]
 [0.28% 0.00% 0.00% 0.56% 0.28% 0.83% 0.00% 7.22% 0.28% 0.56%]
 [0.28% 0.00% 0.00% 0.28% 0.28% 0.00% 0.00% 0.28% 8.61% 0.28%]
 [0.28% 0.56% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.56% 8.61%]

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
TRAINFILE = "phpSZJq5T.csv"


#Number of attributes
num_attr = 1300
n_classes = 10


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
    clean.mapping={'1': 0, '3': 1, '4': 2, '5': 3, '6': 4, '2': 5, '8': 6, '9': 7, '10': 8, '7': 9}

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
energy_thresholds = array([16143568.296394499, 17080610.4952465, 18505321.218955003, 19031977.824179, 19728421.019536998, 21195754.8198385, 22562099.553482, 23004836.040959, 23435590.6084935, 24861313.760259, 25261244.412966996, 25991609.683512002, 26034506.510658503, 26262729.8993, 26685422.346078, 26955066.361365, 27018065.936988, 27292056.303322498, 27834965.9851145, 29158763.384713, 29449790.107488498, 29587995.3890425, 29964717.665692, 30236241.199502498, 30471315.4091235, 30518652.772527, 30691677.1843785, 30964010.892937, 31314889.624002002, 31538563.446174003, 31607874.7607805, 31901277.317878, 32176415.4905455, 32260803.3759215, 32317014.035409503, 32393468.2136655, 33049878.1478935, 33245295.756718002, 33515798.602970503, 33747185.8389965, 33790371.6801855, 33805758.36267801, 33942888.98987301, 34082282.84229, 34229927.667547, 34540909.499303505, 35280121.754187, 35955139.5857615, 36131760.073008, 36496010.709359504, 36992871.17147, 37141921.8880245, 37277162.8028635, 37623796.35462, 37855969.718085006, 37866921.2112495, 37955470.435829, 38079420.2944105, 38176545.7607525, 38290332.1149405, 38377252.619946, 38436907.6738395, 38468050.767767504, 38513303.866336495, 38665134.976568505, 38797903.408942506, 38994743.5891165, 39218367.7929135, 39394769.0863725, 39588189.195076, 39777484.40156, 40006922.868844494, 40143617.65015149, 40227135.776739, 40287418.912727, 40432596.456552, 40606764.789803, 40649018.174261, 40705310.4124795, 40779689.4396435, 40810680.972482, 40832295.680732004, 40871337.590462, 40983788.4178545, 41155570.673958, 41611929.688847505, 41691364.9498955, 41764694.52314, 41956513.488043, 42368592.501837, 42778163.903024, 43040898.3420495, 43442295.971869, 44194781.958655, 44818842.3422015, 45201509.4422675, 45488780.825903, 45585945.098427504, 45671533.9823085, 45861101.3882035, 46125294.858937, 46358297.7042915, 46457980.267553, 46547847.590736, 47433474.28959051, 48279224.131418005, 48394314.6631245, 48507907.1299495, 48633424.379889995, 48822602.74481849, 49108085.343815, 49234043.371832, 49303932.555856, 49353890.208138496, 49427253.306292996, 49649078.048057, 49807671.9755385, 50039109.177197, 50416690.742991, 50780547.212286994, 51144952.1749655, 51617471.523474, 51999369.594246, 53275215.46595301, 54615307.502698496, 55187260.24940249, 55388515.834967494, 56941687.034283, 57683809.23404701, 58167433.4784535, 59258732.631900996, 59673368.61987649, 60026971.4332955, 61430739.54455, 62496209.5754385, 63054512.738418, 63278114.979807004, 63340832.382576, 64165070.6624835, 66359380.806178994, 66565792.2824785, 66966141.21497001, 67922588.690313, 68868725.8964925, 69809214.344796, 70821769.1168285, 71263773.0451725, 72106675.9155355, 72897626.6794485, 73293365.83652501, 73858559.9556715, 74695134.4358605, 77058400.69693601, 77725309.2411885, 78529596.73737702, 78804499.9082115, 79566448.76386449, 81694010.0587875, 83320893.85132849, 83772834.7523285, 84118144.31382, 85866128.73999551, 86513118.5658635, 87594453.85544601, 89359765.355386, 90314227.984628, 90535411.07938349, 91651130.00940701, 92419349.33396599, 92540719.150706, 94010746.3402785, 97871693.4162045, 98657954.5499325, 99394712.11939101, 99926389.529883, 101858085.862157, 102249632.101416, 102518738.929972, 103335446.8317925, 105076997.7945345, 106501430.595741, 106834909.836125, 111308469.2386845, 111601713.538919, 113951408.91883, 116323958.730839, 119146744.9903655, 122243953.44348651, 124974659.3516885, 129266412.626348, 136433706.767706, 140056535.55567202, 145784811.82294402, 152984749.0958925, 169044015.7835755])
labels = array([8.0, 5.0, 8.0, 0.0, 5.0, 8.0, 7.0, 5.0, 0.0, 5.0, 7.0, 1.0, 7.0, 0.0, 5.0, 1.0, 7.0, 4.0, 5.0, 8.0, 1.0, 5.0, 7.0, 1.0, 5.0, 9.0, 5.0, 0.0, 8.0, 7.0, 0.0, 9.0, 5.0, 1.0, 7.0, 0.0, 1.0, 8.0, 6.0, 4.0, 7.0, 4.0, 5.0, 1.0, 4.0, 1.0, 5.0, 0.0, 7.0, 0.0, 1.0, 9.0, 5.0, 1.0, 5.0, 4.0, 1.0, 4.0, 8.0, 4.0, 7.0, 1.0, 5.0, 8.0, 0.0, 4.0, 0.0, 9.0, 1.0, 0.0, 9.0, 4.0, 0.0, 8.0, 5.0, 0.0, 1.0, 9.0, 0.0, 8.0, 1.0, 0.0, 8.0, 7.0, 9.0, 0.0, 4.0, 9.0, 4.0, 9.0, 4.0, 1.0, 5.0, 4.0, 5.0, 6.0, 3.0, 5.0, 4.0, 9.0, 1.0, 9.0, 1.0, 7.0, 4.0, 0.0, 2.0, 1.0, 6.0, 9.0, 6.0, 9.0, 7.0, 5.0, 6.0, 9.0, 1.0, 0.0, 9.0, 5.0, 9.0, 8.0, 4.0, 9.0, 6.0, 8.0, 3.0, 2.0, 8.0, 4.0, 1.0, 2.0, 9.0, 0.0, 4.0, 2.0, 8.0, 3.0, 6.0, 2.0, 8.0, 4.0, 7.0, 4.0, 3.0, 2.0, 0.0, 9.0, 3.0, 2.0, 9.0, 3.0, 2.0, 6.0, 2.0, 8.0, 6.0, 8.0, 6.0, 3.0, 2.0, 8.0, 3.0, 4.0, 2.0, 6.0, 3.0, 6.0, 2.0, 6.0, 2.0, 6.0, 3.0, 8.0, 2.0, 6.0, 2.0, 6.0, 8.0, 2.0, 0.0, 3.0, 6.0, 2.0, 3.0, 6.0, 2.0, 3.0, 8.0, 2.0, 3.0, 8.0, 2.0, 6.0, 8.0])
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
        outputs[defaultindys] = 7.0
        return outputs
    return thresh_search(energys)

numthresholds = 195



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


