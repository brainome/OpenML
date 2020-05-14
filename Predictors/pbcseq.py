#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53336/pbcseq.arff -o Predictors/pbcseq_QC.py -target binaryClass -stopat 84.37 -f QC -e 100 --yes
# Total compiler execution time: 0:00:10.80. Finished on: Apr-21-2020 15:14:34.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.02%
Model accuracy:                     85.39% (1661/1945 correct)
Improvement over best guess:        35.37% (of possible 49.98%)
Model capacity (MEC):               471 bits
Generalization ratio:               3.52 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     42.88% (834/1945)
True Positives:                     42.52% (827/1945)
False Negatives:                    7.51% (146/1945)
False Positives:                    7.10% (138/1945)
True Pos. Rate/Sensitivity/Recall:  0.85
True Neg. Rate/Specificity:         0.86
Precision:                          0.86
F-1 Measure:                        0.85
False Negative Rate/Miss Rate:      0.15
Critical Success Index:             0.74

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
TRAINFILE = "pbcseq.csv"


#Number of attributes
num_attr = 18
n_classes = 2


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

# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([4727576891.0, 4727577920.735001, 4727578978.28, 4727579242.450001, 4727579350.875, 4727580251.015, 4727580456.950001, 4727580546.955, 4727580723.344999, 4727581275.535, 4727581652.675, 4727581893.455, 4727582145.945, 4727582914.575, 4727583019.745, 4727583363.880001, 4727583830.130001, 4727583874.245001, 4727583902.014999, 4727584090.68, 4727584135.44, 4727584341.455, 4727584468.725, 4727584487.785, 4727584501.004999, 4727584761.905001, 4727584931.38, 4727584952.485, 4727585626.46, 4727585657.1, 4727585667.055, 4727585683.65, 4727585717.759999, 4727585759.9, 4727585781.724999, 4727585826.065001, 4727585872.184999, 4727586255.764999, 4727586325.135, 4727586355.025, 4727586764.165, 4727586929.684999, 4727587190.844999, 4727587265.684999, 4727587374.790001, 4727587513.975, 4727588044.08, 4727588175.24, 4727590985.035, 4727591646.48, 4727592381.6, 4727593526.01, 4820695467.57, 4913799009.820001, 4966460414.210001, 4966460557.965, 4966460759.65, 4966460947.775, 4966461097.78, 4966461179.285, 4966462641.23, 4966463127.65, 4966463879.75, 4966463952.395, 4966464091.43, 4966464192.745, 4966465129.295, 4966465209.639999, 4966465466.95, 4966465525.905, 4966466152.78, 4966466472.73, 4966467509.125, 4966467744.550001, 4966467849.780001, 4966467985.095, 4966468143.245, 4966468296.295, 4966469112.885, 4966469141.725, 4966469289.559999, 4966469464.475, 4966469808.78, 4966469989.004999, 4966470132.31, 4966470165.190001, 4966470262.365, 4966470520.824999, 4966471006.355, 4966471125.834999, 4966471482.985001, 4966471833.125, 4966472095.635, 4966472680.495, 4966473455.705, 5059575537.84, 5152676298.6449995, 5152676433.125, 5152679009.595, 5152679406.555, 5152679450.77, 5152679549.76, 5205344804.135, 5205345808.95, 5205346351.535, 5205346784.525, 5205352974.594999, 5205353732.91, 6411902307.895, 6411903905.17, 6411904464.835001, 6411905421.349999, 6411905896.120001, 6411907039.440001, 6411907216.58, 6411907257.184999, 6411907324.275, 6411907781.22, 6411907818.11, 6411907898.6, 6411908000.48, 6411908411.935, 6411908500.505, 6411908593.630001, 6411908605.27, 6411909221.860001, 6411909263.120001, 6411909510.094999, 6411909739.935, 6411910657.15, 6411910920.89, 6411911057.755001, 6411911367.175, 6411911503.87, 6411911809.494999, 6411912136.74, 6411912295.360001, 6411912608.51, 6411912672.84, 6411912806.49, 6411913146.184999, 6411913466.795, 6411913920.315001, 6411913987.295, 6411914167.309999, 6411914963.295, 6411916316.684999, 6466780993.934999, 6466781948.129999, 6466786343.635, 6466786501.715, 6466790722.2699995, 6466790969.904999, 6466792899.695, 6466793261.495001, 6598120623.799999, 6598120958.17, 6598123858.040001, 6650783914.115, 6650784728.21, 6650785191.32, 6650785398.97, 6650785532.085, 6650785689.889999, 6650785896.48, 6650786322.885, 6650787284.84, 6650787802.875, 6650787890.51, 6650787972.01, 6650788360.035, 6650788384.2699995, 6650788407.46, 6650788551.995, 6650788884.044999, 6650789206.594999, 6650789451.475, 6650790032.875, 6650790588.045, 6650790926.655001, 6650791274.575001, 6650791654.335, 6650791849.01, 6650792606.775, 6650792724.29, 6650792771.190001, 6650792990.290001, 6650793087.315001, 6650793154.4800005, 6650793223.055, 6650793258.225, 6650793350.04, 6650794382.57, 6650794457.83, 6650794599.934999, 6650795092.275, 6650796164.18, 6650796326.77, 6651897848.3, 6653001140.05, 6653005963.575, 6679336779.905, 6705664083.88, 6705664218.71, 6705664321.799999, 6705664552.719999, 6705665033.275, 6705666356.29, 6705666467.889999, 6705667137.825, 6705668016.21, 6705668737.035, 6705669782.690001, 6705670623.594999, 6705671087.91, 6705672759.73, 6705673011.025, 6705673358.919999, 6705673605.09, 6705673675.435, 6705673808.860001, 6705674543.07, 6705676586.605, 6705678644.42, 6771339056.82, 6836998099.959999, 6836999592.48, 6837000923.385, 6837002445.345, 6837004639.775, 6889670298.09, 6889671328.99, 6891879476.685, 6891882982.295, 7489646329.25, 7903411490.45, 7903411659.804999, 7903411977.93, 7903412101.365, 7903412225.924999, 7903412306.594999, 7903413639.13, 7903413771.33, 7903413872.76, 7903414086.54, 7903414209.084999, 7903415447.685001, 7903416521.195001, 7903416769.83, 7903417030.32, 7903417153.48, 7903417504.615, 7903417673.610001, 7903417795.985, 7903417838.629999, 7903418889.24, 7903419038.34, 7903419798.820001, 7903419827.695001, 7903420100.035, 7903420621.765, 7903420693.900001, 7903420840.825001, 7903420873.695, 7903421027.865, 7903421157.23, 7903421169.23, 7903421203.895, 7903421244.879999, 7903421268.119999, 7903421447.12, 7903421580.94, 7903422489.6, 7903422646.559999, 7903423741.785, 7903423995.794999, 7903424006.309999, 7903424192.215, 7903424686.275, 7903424777.48, 7903425211.695, 7903425271.085001, 7903425385.325, 7903425573.459999, 8089624355.09, 8089629391.715, 8092931799.945001, 8119263771.24, 8142293577.469999, 8142294887.799999, 8142296813.429999, 8142297058.07, 8142297293.67, 8142297486.99, 8142297574.76, 8142297664.865, 8142297731.155001, 8142297820.37, 8142297861.559999, 8142298512.379999, 8142298525.885, 8142298643.059999, 8142300586.75, 8142301691.325001, 8142301732.45, 8142301753.139999, 8142302036.205, 8142302319.235001, 8142302640.545, 8142302646.965, 8142302665.28, 8142303250.42, 8142303558.58, 8142303871.27, 8142303908.865, 8142303981.129999, 8142304221.535, 8142304364.055, 8142304410.495001, 8142304445.76, 8142304671.875, 8142304851.385, 8142304928.485, 8142305453.46, 8142306447.01, 8142306593.26, 8142306784.035, 8142307043.58, 8142307843.625, 8239806109.975, 8328504750.610001, 8328507376.375, 8328510065.849999, 8328511532.805, 8381181571.440001, 8381181972.68, 8381183068.595, 8381183859.014999, 8381186862.415, 8381188537.360001, 8385591956.96, 8389995199.914999, 8390000079.110001, 8567394524.075001, 8601438044.895, 8721980366.105, 9201411126.02, 9587734801.95, 9587735470.395, 9587737290.33, 9587737575.135002, 9587738251.325, 9587738316.235, 9587739446.465, 9587739673.055, 9587739951.455, 9587740008.33, 9587740761.855, 9587740898.515, 9587742298.405, 9587742572.09, 9587742601.16, 9587743143.0, 9587744081.735, 9587744137.39, 9587744608.485, 9587744670.564999, 9587744791.314999, 9587744923.98, 9587745592.630001, 9587745767.45, 9587746172.134998, 9587746535.02, 9587747121.119999, 9587748106.415, 9587748297.805, 9587748322.015, 9587748988.27, 9587749611.97, 9642617799.465, 9642618782.455, 9642621438.64, 9642621816.869999, 9707189745.345, 9771748306.259998, 9773952906.175, 9826618020.48, 9826618728.625, 9826619057.274998, 9826620422.185, 9826620528.119999, 9826620674.384998, 9826620759.619999, 9826620892.075, 9826621077.98, 9826621254.705, 9826621896.125, 9826622004.125, 9826622050.814999, 9826622106.58, 9826622196.54, 9826622938.255001, 9826623005.915, 9826623085.645, 9826623145.46, 9826623243.675, 9826624425.735, 9826624507.665, 9826625306.67, 9826625493.8, 9826626207.775, 9826626657.715, 9826626902.86, 9826626978.205, 9826627165.060001, 9826627272.945, 9826627293.82, 9826627304.585, 9826627318.725, 9826627358.050001, 9826627402.405, 9826627425.32, 9826627481.744999, 9826627685.285, 9826629166.005001, 9826629450.255001, 9826630071.009998, 9826630312.075, 9827729356.415, 9881497634.96, 9881497889.325, 9881500234.885, 9881502327.869999, 9881502535.83, 9881502665.455, 9881502826.919998, 9881504059.485, 9881504345.46, 9881504603.515, 9881505031.85, 9881505754.28, 9881506987.23, 9881511774.235, 9947171939.715, 10012829654.64, 10012834622.225, 10012835151.41, 10012835550.724998, 10012837766.954998, 10012839022.855, 10016138906.085001, 10042466659.335001, 10065505554.715, 10065507094.66, 10065510940.41, 10066613432.59, 10761894363.42, 11326942891.23, 11326946286.895, 11326951296.005001, 11355118024.315, 11355119553.675, 11355120095.61, 11355121513.2, 11565827845.86, 11565834591.045, 11685277803.5, 13167835551.150002, 14530956989.53, 14530959218.035, 14717172000.2])
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

numthresholds=471



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


