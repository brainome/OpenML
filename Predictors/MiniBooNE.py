#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 13:57:02
# Invocation: btc -target signal -v -v MiniBooNE-1.csv -o MiniBooNE-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                71.93%
Model accuracy:                     85.12% (110720/130064 correct)
Improvement over best guess:        13.19% (of possible 28.07%)
Model capacity (MEC):               105 bits
Generalization ratio:               1054.47 bits/bit
Model efficiency:                   0.12%/parameter
System behavior
True Negatives:                     17.40% (22631/130064)
True Positives:                     67.73% (88089/130064)
False Negatives:                    4.21% (5476/130064)
False Positives:                    10.66% (13868/130064)
True Pos. Rate/Sensitivity/Recall:  0.94
True Neg. Rate/Specificity:         0.62
Precision:                          0.86
F-1 Measure:                        0.90
False Negative Rate/Miss Rate:      0.06
Critical Success Index:             0.82
Model bias:                         14.16% higher chance to pick class 1
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
TRAINFILE="MiniBooNE-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50

list_of_cols_to_normalize = [19]

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
    h_0 = max((((12.35848 * float(x[0]))+ (0.8435814 * float(x[1]))+ (1.2749399 * float(x[2]))+ (0.95930433 * float(x[3]))+ (0.3033341 * float(x[4]))+ (-2.4788318 * float(x[5]))+ (-8.715932 * float(x[6]))+ (-1.0906731 * float(x[7]))+ (-2.381257 * float(x[8]))+ (-0.4976771 * float(x[9]))+ (-3.9523458 * float(x[10]))+ (3.485699 * float(x[11]))+ (10.060928 * float(x[12]))+ (2.208644 * float(x[13]))+ (-9.45382 * float(x[14]))+ (-1.8891242 * float(x[15]))+ (-3.983124 * float(x[16]))+ (9.710874 * float(x[17]))+ (0.45397297 * float(x[18]))+ (-6.938863e-05 * float(x[19]))+ (-16.585608 * float(x[20]))+ (-0.35362792 * float(x[21]))+ (-8.019787 * float(x[22]))+ (-19.483047 * float(x[23]))+ (-1.0439008 * float(x[24]))+ (-15.995073 * float(x[25]))+ (7.693561 * float(x[26]))+ (2.3237417 * float(x[27]))+ (0.24880442 * float(x[28]))+ (-3.7408378 * float(x[29]))+ (15.383533 * float(x[30]))+ (-11.854345 * float(x[31]))+ (8.781429 * float(x[32]))+ (4.9055233 * float(x[33]))+ (-0.34695384 * float(x[34]))+ (-1.3572601 * float(x[35]))+ (-14.395614 * float(x[36]))+ (6.8951383 * float(x[37]))+ (0.9221306 * float(x[38]))+ (-0.043933302 * float(x[39]))+ (10.304209 * float(x[40]))+ (-9.060319 * float(x[41]))+ (-11.074369 * float(x[42]))+ (16.8058 * float(x[43]))+ (0.6610985 * float(x[44]))+ (0.14489396 * float(x[45]))+ (-7.949911 * float(x[46]))+ (13.604415 * float(x[47]))+ (20.808258 * float(x[48]))+ (-0.022647958 * float(x[49]))) + 0.3028723), 0)
    h_1 = max((((0.61210746 * float(x[0]))+ (0.6796819 * float(x[1]))+ (-1.3311695 * float(x[2]))+ (-0.9324053 * float(x[3]))+ (-0.57978106 * float(x[4]))+ (0.37645277 * float(x[5]))+ (-0.7122577 * float(x[6]))+ (0.6228087 * float(x[7]))+ (0.52121997 * float(x[8]))+ (-0.89564717 * float(x[9]))+ (0.37748656 * float(x[10]))+ (-12.892824 * float(x[11]))+ (0.8150249 * float(x[12]))+ (-0.48039272 * float(x[13]))+ (-0.71189487 * float(x[14]))+ (-174.41985 * float(x[15]))+ (-0.3031017 * float(x[16]))+ (0.10622135 * float(x[17]))+ (-0.56952953 * float(x[18]))+ (-33.20954 * float(x[19]))+ (-0.75254625 * float(x[20]))+ (-0.76861846 * float(x[21]))+ (-18.800064 * float(x[22]))+ (0.050303888 * float(x[23]))+ (-0.15917937 * float(x[24]))+ (0.26231614 * float(x[25]))+ (-2.4790478 * float(x[26]))+ (0.5563375 * float(x[27]))+ (-0.6931874 * float(x[28]))+ (0.106761135 * float(x[29]))+ (0.43915412 * float(x[30]))+ (0.47685876 * float(x[31]))+ (0.48019758 * float(x[32]))+ (-13.016411 * float(x[33]))+ (-0.86737704 * float(x[34]))+ (0.31706256 * float(x[35]))+ (0.25944912 * float(x[36]))+ (-0.33512157 * float(x[37]))+ (-0.72096133 * float(x[38]))+ (0.07142904 * float(x[39]))+ (-8.050932 * float(x[40]))+ (0.2724846 * float(x[41]))+ (0.5656455 * float(x[42]))+ (0.8234096 * float(x[43]))+ (-0.27532384 * float(x[44]))+ (0.488128 * float(x[45]))+ (0.41019678 * float(x[46]))+ (-0.49757686 * float(x[47]))+ (-0.12373826 * float(x[48]))+ (-0.42644396 * float(x[49]))) + -0.3559619), 0)
    o_0 = (0.0013444279 * h_0)+ (-0.21596572 * h_1) + -2.6948364

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

        model_cap=105

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

