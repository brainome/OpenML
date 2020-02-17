#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 15:20:05
# Invocation: btc -v -v dermatology-1.csv -o dermatology-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                69.39%
Model accuracy:                     98.63% (361/366 correct)
Improvement over best guess:        29.24% (of possible 30.61%)
Model capacity (MEC):               73 bits
Generalization ratio:               4.94 bits/bit
Model efficiency:                   0.40%/parameter
System behavior
True Negatives:                     69.40% (254/366)
True Positives:                     29.23% (107/366)
False Negatives:                    1.37% (5/366)
False Positives:                    0.00% (0/366)
True Pos. Rate/Sensitivity/Recall:  0.96
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        0.98
False Negative Rate/Miss Rate:      0.04
Critical Success Index:             0.96
Model bias:                         100.00% higher chance to pick class 0
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
TRAINFILE="dermatology-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 34

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 15.0: 4, 16.0: 5, 17.0: 6, 18.0: 7, 19.0: 8, 20.0: 9, 22.0: 10, 23.0: 11, 24.0: 12, 25.0: 13, 27.0: 14, 28.0: 15, 29.0: 16, 30.0: 17, 31.0: 18, 33.0: 19, 34.0: 20, 35.0: 21, 36.0: 22, 37.0: 23, 38.0: 24, 40.0: 25, 41.0: 26, 42.0: 27, 43.0: 28, 44.0: 29, 45.0: 30, 46.0: 31, 47.0: 32, 48.0: 33, 1684325040.0: 34, 50.0: 35, 51.0: 36, 52.0: 37, 53.0: 38, 49.0: 39, 55.0: 40, 56.0: 41, 57.0: 42, 60.0: 43, 61.0: 44, 62.0: 45, 64.0: 46, 65.0: 47, 67.0: 48, 68.0: 49, 70.0: 50, 75.0: 51, 13.0: 52, 32.0: 53, 21.0: 54, 39.0: 55, 26.0: 56, 12.0: 57, 58.0: 58, 63.0: 59, 0.0: 60}]
list_of_cols_to_normalize = [33]

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
            if (not (result==0 or result==1)):
                raise ValueError("Integer class labels need to be 0 or 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                if (not (result==0 or result==1)):
                    raise ValueError("Numeric class labels need to be 0 or 1.")
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
            if (row==[]):  # Skip empty rows
                continue
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
    h_0 = max((((-1.4551826 * float(x[0]))+ (-1.6121548 * float(x[1]))+ (0.5386306 * float(x[2]))+ (-2.103922 * float(x[3]))+ (0.8602883 * float(x[4]))+ (-0.1248394 * float(x[5]))+ (-0.6606389 * float(x[6]))+ (0.92732185 * float(x[7]))+ (-0.34691337 * float(x[8]))+ (1.115183 * float(x[9]))+ (-0.010342947 * float(x[10]))+ (0.13607533 * float(x[11]))+ (0.010330368 * float(x[12]))+ (-1.186057 * float(x[13]))+ (-1.3666904 * float(x[14]))+ (-3.219143 * float(x[15]))+ (-1.5173396 * float(x[16]))+ (-0.17917092 * float(x[17]))+ (-0.5879987 * float(x[18]))+ (1.8930516 * float(x[19]))+ (1.2127498 * float(x[20]))+ (0.88842005 * float(x[21]))+ (1.0040551 * float(x[22]))+ (-0.2126817 * float(x[23]))+ (0.27982822 * float(x[24]))+ (-0.6026802 * float(x[25]))+ (0.8893292 * float(x[26]))+ (-3.2347326 * float(x[27]))+ (-0.17068811 * float(x[28]))+ (-1.3414567 * float(x[29]))+ (-0.43700907 * float(x[30]))+ (-2.551485 * float(x[31]))+ (0.13685408 * float(x[32]))+ (-26.760035 * float(x[33]))) + -1.1195498), 0)
    h_1 = max((((1.0413916 * float(x[0]))+ (-0.56553596 * float(x[1]))+ (-1.272079 * float(x[2]))+ (2.2463696 * float(x[3]))+ (-0.963663 * float(x[4]))+ (5.9920826 * float(x[5]))+ (6.164123 * float(x[6]))+ (5.408191 * float(x[7]))+ (-2.6634724 * float(x[8]))+ (-2.721571 * float(x[9]))+ (-3.0518444 * float(x[10]))+ (4.8802056 * float(x[11]))+ (5.4003687 * float(x[12]))+ (-2.8525105 * float(x[13]))+ (8.395753 * float(x[14]))+ (5.6758437 * float(x[15]))+ (-0.6677913 * float(x[16]))+ (-1.1295884 * float(x[17]))+ (-1.8388308 * float(x[18]))+ (-1.7665616 * float(x[19]))+ (-1.2020372 * float(x[20]))+ (-2.5353072 * float(x[21]))+ (-1.5374508 * float(x[22]))+ (-2.009358 * float(x[23]))+ (4.4208875 * float(x[24]))+ (-2.4828098 * float(x[25]))+ (5.875926 * float(x[26]))+ (7.2778816 * float(x[27]))+ (4.797557 * float(x[28]))+ (5.1142073 * float(x[29]))+ (6.5841665 * float(x[30]))+ (0.080528155 * float(x[31]))+ (6.3029184 * float(x[32]))+ (0.48191163 * float(x[33]))) + -0.06769251), 0)
    o_0 = (-0.264165 * h_0)+ (-0.30471048 * h_1) + 2.5570567

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

        model_cap=73

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


