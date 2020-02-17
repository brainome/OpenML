#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 18:29:47
# Invocation: btc -v analcatdata-broadway-2.csv -o analcatdata-broadway-2.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                71.57%
Model accuracy:                     100.00% (95/95 correct)
Improvement over best guess:        28.43% (of possible 28.43%)
Model capacity (MEC):               23 bits
Generalization ratio:               4.13 bits/bit
Model efficiency:                   1.23%/parameter
System behavior
True Negatives:                     28.42% (27/95)
True Positives:                     71.58% (68/95)
False Negatives:                    0.00% (0/95)
False Positives:                    0.00% (0/95)
True Pos. Rate/Sensitivity/Recall:  1.00
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        1.00
False Negative Rate/Miss Rate:      0.00
Critical Success Index:             1.00
Model bias:                         1.00% higher chance to pick class 0
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="analcatdata-broadway-2.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 9

mappings = [{1448411008.0: 0, 348641025.0: 1, 1901793794.0: 2, 3865183620.0: 3, 170481158.0: 4, 3938574601.0: 5, 142605961.0: 6, 4238284810.0: 7, 3377415436.0: 8, 3889474832.0: 9, 778510995.0: 10, 444639253.0: 11, 1618659861.0: 12, 786222230.0: 13, 3756318614.0: 14, 20985369.0: 15, 1776639260.0: 16, 502395804.0: 17, 2927307294.0: 18, 1499881631.0: 19, 3786171676.0: 20, 1434509217.0: 21, 1279077794.0: 22, 1638930332.0: 23, 589765796.0: 24, 3292883617.0: 25, 1284101547.0: 26, 659273772.0: 27, 3595220141.0: 28, 587998382.0: 29, 4284001071.0: 30, 3296235700.0: 31, 2798143417.0: 32, 4081668921.0: 33, 4185523515.0: 34, 2211305148.0: 35, 4082829884.0: 36, 2622402367.0: 37, 255676610.0: 38, 3378105924.0: 39, 3207999173.0: 40, 1977039429.0: 41, 3257798342.0: 42, 874651339.0: 43, 2692692690.0: 44, 3438935763.0: 45, 3450666453.0: 46, 57705558.0: 47, 1406684887.0: 48, 1607631061.0: 49, 3065425497.0: 50, 417954907.0: 51, 2369380828.0: 52, 2122844895.0: 53, 782263392.0: 54, 1294303328.0: 55, 1663489510.0: 56, 1473365611.0: 57, 2834869484.0: 58, 672572912.0: 59, 282581360.0: 60, 1776.0: 61, 1771338492.0: 62, 4165923197.0: 63, 2615169150.0: 64, 3703800191.0: 65, 2737551465.0: 66, 2908512717.0: 67, 3796382369.0: 68, 2910062045.0: 69, 2404128539.0: 70, 1534065726.0: 71, 2358007975.0: 72, 2765190236.0: 73, 619243283.0: 74, 3486589747.0: 75, 2996628925.0: 76, 2774110390.0: 77, 2934779111.0: 78, 2340225153.0: 79, 1993779448.0: 80, 784430357.0: 81, 3356007431.0: 82, 1628501267.0: 83, 3713568086.0: 84, 2904430446.0: 85, 1033525424.0: 86, 3018237383.0: 87, 2167201544.0: 88, 4167852135.0: 89, 4083393966.0: 90, 3056583856.0: 91, 4156554627.0: 92, 1300701999.0: 93, 2632612306.0: 94}, {4273697156.0: 0, 2070504212.0: 1, 1768978383.0: 2}, {1303016265.0: 0, 4063104189.0: 1}, {1.66667: 0, 2.0: 1, 3.0: 2, 3.66667: 3, 5.0: 4, 4.33333: 5, 2.66667: 6, 1.33333: 7, 3.33333: 8, 4.66667: 9, 4.0: 10, 1.0: 11, 2.33333: 12, 1684325040.0: 13}, {1.33333: 0, 1.0: 1, 3.0: 2, 4.33333: 3, 5.0: 4, 3.66667: 5, 2.33333: 6, 2.0: 7, 4.0: 8, 2.66667: 9, 4.66667: 10, 3.33333: 11, 1684325040.0: 12, 1.66667: 13}, {34.74: 0, 40.66: 1, 42.33: 2, 46.92: 3, 1684325040.0: 4, 49.4: 5, 50.1: 6, 48.38: 7, 86.67: 8, 53.69: 9, 54.75: 10, 87.56: 11, 56.94: 12, 57.3: 13, 56.26: 14, 87.23: 15, 60.0: 16, 60.59: 17, 61.93: 18, 88.44: 19, 88.2: 20, 65.21: 21, 65.22: 22, 67.76: 23, 68.75: 24, 90.35: 25, 70.07: 26, 71.9: 27, 70.36: 28, 73.85: 29, 91.91: 30, 72.53: 31, 76.97: 32, 77.31: 33, 79.9: 34, 92.41: 35, 81.39: 36, 92.21: 37, 82.61: 38, 93.2: 39, 83.91: 40, 86.04: 41, 87.43: 42, 87.13: 43, 89.2: 44, 90.72: 45, 90.61: 46, 92.19: 47, 92.44: 48, 92.1: 49, 92.78: 50, 96.63: 51, 97.29: 52, 97.24: 53, 97.47: 54, 97.02: 55, 100.0: 56, 101.58: 57, 96.65: 58, 96.69: 59, 99.15: 60, 95.54: 61, 50.76: 62, 71.75: 63, 40.04: 64, 62.87: 65, 90.97: 66, 79.69: 67, 77.09: 68, 29.35: 69, 77.28: 70, 73.27: 71, 82.31: 72, 54.05: 73, 56.99: 74, 74.4: 75, 44.08: 76, 68.14: 77, 71.26: 78, 64.87: 79, 80.88: 80, 81.77: 81, 46.75: 82, 84.8: 83, 30.83: 84, 92.97: 85, 91.71: 86, 96.89: 87, 88.91: 88, 83.33: 89, 99.94: 90}, {2762405025.0: 0, 440313483.0: 1, 2045107345.0: 2, 1396881080.0: 3, 2244252154.0: 4}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 7, 8]

transform_true = False

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
def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist=[]
    clean.testfile=testfile
    
    def convert(cell):
        value=str(cell)
        try:
            result=int(value)
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                return result

    def convertclassid(cell):
        if (clean.testfile):
            return convert(cell)
        value=str(cell)
        if (value==''):
            raise ValueError("All cells in the target column need to contain a class label.")
        try:
            result=int(value)
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                if (not str(result) in clean.classlist):
                    clean.classlist=clean.classlist+[str(result)]
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                if (result in clean.classlist):
                    result=clean.classlist.index(result)
                else:
                    clean.classlist=clean.classlist+[result]
                    result=clean.classlist.index(result)
                return result
    rowcount=0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f=open(outfile,"w+")
        if (headerless==False):
            next(reader,None)
        outbuf=[]
        for row in reader:
            rowcount=rowcount+1
            rowlen=num_attr
            if (not testfile):
                rowlen=rowlen+1    
            if (not len(row)==rowlen):
                raise ValueError("Column count must match trained predictor. Row "+str(rowcount)+" differs.")
            i=0
            for elem in row:
                if(i+1<len(row)):
                    outbuf.append(str(convert(elem)))
                    outbuf.append(',')
                else:
                    classid=str(convertclassid(elem))
                    outbuf.append(classid)
                i=i+1
            if (len(outbuf)<IOBUF):
                outbuf.append("\n")
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)==2):
            raise ValueError("Number of classes must be 2.")


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    h_0 = max((((0.022535468 * float(x[0]))+ (-0.6332237 * float(x[1]))+ (-0.072306134 * float(x[2]))+ (-0.009636772 * float(x[3]))+ (0.2587576 * float(x[4]))+ (-1.2285391 * float(x[5]))+ (0.39285642 * float(x[6]))+ (0.032831606 * float(x[7]))+ (-7.7398314 * float(x[8]))) + 4.067052), 0)
    h_1 = max((((-1.9529221 * float(x[0]))+ (0.359758 * float(x[1]))+ (-0.47665387 * float(x[2]))+ (-1.5891769 * float(x[3]))+ (-3.3119912 * float(x[4]))+ (-0.54145575 * float(x[5]))+ (0.76008564 * float(x[6]))+ (-12.040704 * float(x[7]))+ (0.13690527 * float(x[8]))) + -0.538095), 0)
    o_0 = (1.459639 * h_0)+ (-0.4365398 * h_1) + -3.4435723

    if num_output_logits==1:
        return o_0>=0
    else:
        return argmax([eval('o'+str(i)) for i in range(num_output_logits)])

# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()

    if not args.validate: # Then predict
        if not args.cleanfile: # File is not preprocessed
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            test_tensor = np.loadtxt(cleanfile,delimiter=',',dtype='float64')
            os.remove(cleanfile)
        else: # File is already preprocessed
            test_tensor = np.loadtxt(args.File,delimiter = ',',dtype = 'float64')               
        test_tensor = Normalize(test_tensor)
        if transform_true:
            test_tensor = transform(test_tensor)
        with open(args.csvfile,'r') as csvinput:
            writer = csv.writer(sys.stdout, lineterminator='\n')
            reader = csv.reader(csvinput)
            if (not args.headerless):
                writer.writerow((next(reader, None)+['Prediction']))
            i=0
            for row in reader:
                if (classify(test_tensor[i])):
                    pred="1"
                else:
                    pred="0"
                row.append(pred)
                writer.writerow(row)
                i=i+1
    elif args.validate: # Then validate this predictor, always clean first.
        tempdir=tempfile.gettempdir()
        temp_name = next(tempfile._get_candidate_names())
        cleanfile=tempdir+os.sep+temp_name
        clean(args.csvfile,cleanfile, -1, args.headerless)
        val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
        os.remove(cleanfile)
        val_tensor = Normalize(val_tensor)
        if transform_true:
            trans = transform(val_tensor[:,:-1])
            val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
        count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0 = 0,0,0,0,0,0,0,0
        for i,row in enumerate(val_tensor):
            if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                correct_count+=1
                if int(float(row[-1]))==1:
                    num_class_1+=1
                    num_TP+=1
                else:
                    num_class_0+=1
                    num_TN+=1
            else:
                if int(float(row[-1]))==1:
                    num_class_1+=1
                    num_FN+=1
                else:
                    num_class_0+=1
                    num_FP+=1
            count+=1

        model_cap=23

        FN=float(num_FN)*100.0/float(count)
        FP=float(num_FP)*100.0/float(count)
        TN=float(num_TN)*100.0/float(count)
        TP=float(num_TP)*100.0/float(count)
        num_correct=correct_count

        if int(num_TP+num_FN)!=0:
            TPR=num_TP/(num_TP+num_FN) # Sensitivity, Recall
        if int(num_TN+num_FP)!=0:
            TNR=num_TN/(num_TN+num_FP) # Specificity, 
        if int(num_TP+num_FP)!=0:
            PPV=num_TP/(num_TP+num_FP) # Recall
        if int(num_FN+num_TP)!=0:
            FNR=num_FN/(num_FN+num_TP) # Miss rate
        if int(2*num_TP+num_FP+num_FN)!=0:
            FONE=2*num_TP/(2*num_TP+num_FP+num_FN) # F1 Score
        if int(num_TP+num_FN+num_FP)!=0:
            TS=num_TP/(num_TP+num_FN+num_FP) # Critical Success Index

        randguess=int(float(10000.0*max(num_class_1,num_class_0))/count)/100.0
        modelacc=int(float(num_correct*10000)/count)/100.0

        print("System Type:                        Binary classifier")
        print("Best-guess accuracy:                {:.2f}%".format(randguess))
        print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
        print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
        print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
        print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")
        print("Model efficiency:                   {:.2f}%/parameter".format(int(100*(modelacc-randguess)/model_cap)/100.0))
        print("System behavior")
        print("True Negatives:                     {:.2f}%".format(TN)+" ("+str(int(num_TN))+"/"+str(count)+")")
        print("True Positives:                     {:.2f}%".format(TP)+" ("+str(int(num_TP))+"/"+str(count)+")")
        print("False Negatives:                    {:.2f}%".format(FN)+" ("+str(int(num_FN))+"/"+str(count)+")")
        print("False Positives:                    {:.2f}%".format(FP)+" ("+str(int(num_FP))+"/"+str(count)+")")
        if int(num_TP+num_FN)!=0:
            print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
        if int(num_TN+num_FP)!=0:
            print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
        if int(num_TP+num_FP)!=0:
            print("Precision:                          {:.2f}".format(PPV))
        if int(2*num_TP+num_FP+num_FN)!=0:
            print("F-1 Measure:                        {:.2f}".format(FONE))
        if int(num_TP+num_FN)!=0:
            print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
        if int(num_TP+num_FN+num_FP)!=0:    
            print("Critical Success Index:             {:.2f}".format(TS))


