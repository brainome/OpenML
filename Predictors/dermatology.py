#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 20:52:00
# Invocation: btc -server brain.brainome.ai Data/dermatology.csv -o Models/dermatology.py -v -v -v -stopat 100 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                69.39%
Model accuracy:                     100.00% (366/366 correct)
Improvement over best guess:        30.61% (of possible 30.61%)
Model capacity (MEC):               73 bits
Generalization ratio:               5.01 bits/bit
Model efficiency:                   0.41%/parameter
System behavior
True Negatives:                     69.40% (254/366)
True Positives:                     30.60% (112/366)
False Negatives:                    0.00% (0/366)
False Positives:                    0.00% (0/366)
True Pos. Rate/Sensitivity/Recall:  1.00
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        1.00
False Negative Rate/Miss Rate:      0.00
Critical Success Index:             1.00

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
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="dermatology.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 34
n_classes = 2

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 15.0: 4, 16.0: 5, 17.0: 6, 18.0: 7, 19.0: 8, 20.0: 9, 22.0: 10, 23.0: 11, 24.0: 12, 25.0: 13, 27.0: 14, 28.0: 15, 29.0: 16, 30.0: 17, 31.0: 18, 33.0: 19, 34.0: 20, 35.0: 21, 36.0: 22, 37.0: 23, 38.0: 24, 40.0: 25, 41.0: 26, 42.0: 27, 43.0: 28, 44.0: 29, 45.0: 30, 46.0: 31, 47.0: 32, 48.0: 33, 49.0: 34, 50.0: 35, 51.0: 36, 52.0: 37, 53.0: 38, 55.0: 39, 56.0: 40, 57.0: 41, 60.0: 42, 61.0: 43, 62.0: 44, 64.0: 45, 65.0: 46, 67.0: 47, 68.0: 48, 70.0: 49, 75.0: 50, 1684325040.0: 51, 13.0: 52, 32.0: 53, 21.0: 54, 39.0: 55, 26.0: 56, 12.0: 57, 58.0: 58, 63.0: 59, 0.0: 60}]
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
    clean.mapping={}
    

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
            raise ValueError("All cells in the target column must contain a class label.")

        if (not clean.mapping=={}):
            result=-1
            try:
                result=clean.mapping[cell]
            except:
                raise ValueError("Class label '"+value+"' encountered in input not defined in user-provided mapping.")
            if (not result==int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(int(result*100)/100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not result==int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result<0):
                raise ValueError("Integer class labels must be positive and contiguous.")

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
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)>=2):
            raise ValueError("Number of classes must be at least 2.")



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    o=[0]*num_output_logits
    h_0 = max((((-0.4108968 * float(x[0]))+ (-0.57406193 * float(x[1]))+ (-0.89487064 * float(x[2]))+ (-1.7755895 * float(x[3]))+ (0.30274567 * float(x[4]))+ (-0.124886505 * float(x[5]))+ (0.038475275 * float(x[6]))+ (0.9272745 * float(x[7]))+ (0.29140028 * float(x[8]))+ (1.8616401 * float(x[9]))+ (0.051979102 * float(x[10]))+ (0.13602853 * float(x[11]))+ (0.8323483 * float(x[12]))+ (-0.88368994 * float(x[13]))+ (-2.1825142 * float(x[14]))+ (-2.0685713 * float(x[15]))+ (-0.35648772 * float(x[16]))+ (0.35622254 * float(x[17]))+ (0.15422936 * float(x[18]))+ (2.3256376 * float(x[19]))+ (0.35003218 * float(x[20]))+ (0.9132538 * float(x[21]))+ (0.60954785 * float(x[22]))+ (0.54332304 * float(x[23]))+ (0.2675923 * float(x[24]))+ (0.0019306375 * float(x[25]))+ (0.8892767 * float(x[26]))+ (-1.4361957 * float(x[27]))+ (-0.17073572 * float(x[28]))+ (-1.0142204 * float(x[29]))+ (-0.10191923 * float(x[30]))+ (-0.7809618 * float(x[31]))+ (0.13679068 * float(x[32]))+ (-11.407605 * float(x[33]))) + -0.5317107), 0)
    h_1 = max((((0.7329329 * float(x[0]))+ (0.59102124 * float(x[1]))+ (-1.1596951 * float(x[2]))+ (1.2970501 * float(x[3]))+ (-0.25924253 * float(x[4]))+ (6.508207 * float(x[5]))+ (3.581946 * float(x[6]))+ (7.229176 * float(x[7]))+ (-0.31165 * float(x[8]))+ (-0.52317595 * float(x[9]))+ (-0.9497141 * float(x[10]))+ (6.5070925 * float(x[11]))+ (3.8903923 * float(x[12]))+ (-0.32296434 * float(x[13]))+ (10.867177 * float(x[14]))+ (2.4175885 * float(x[15]))+ (-0.6122598 * float(x[16]))+ (-0.8257383 * float(x[17]))+ (-0.5653084 * float(x[18]))+ (-0.9274679 * float(x[19]))+ (-0.14535421 * float(x[20]))+ (-2.045634 * float(x[21]))+ (-2.6741996 * float(x[22]))+ (-1.4828124 * float(x[23]))+ (7.800064 * float(x[24]))+ (-0.8874636 * float(x[25]))+ (7.810761 * float(x[26]))+ (10.611561 * float(x[27]))+ (9.799784 * float(x[28]))+ (9.702164 * float(x[29]))+ (9.010373 * float(x[30]))+ (0.09024958 * float(x[31]))+ (0.75693196 * float(x[32]))+ (0.123164624 * float(x[33]))) + 0.6354472), 0)
    o[0] = (-0.8284108 * h_0)+ (-0.4285477 * h_1) + 3.4201646

    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)

# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    
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
        if n_classes==2:
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
        else:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanvalfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanvalfile, -1, args.headerless)
            val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
            os.remove(cleanfile)
            val_tensor = Normalize(val_tensor)
            if transform_true:
                trans = transform(val_tensor[:,:-1])
                val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
            numeachclass={}
            count,correct_count = 0,0
            for i,row in enumerate(val_tensor):
                if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                    correct_count+=1
                    if int(float(val_tensor[i,-1])) in numeachclass.keys():
                        numeachclass[int(float(val_tensor[i,-1]))]+=1
                    else:
                        numeachclass[int(float(val_tensor[i,-1]))]=0
                count+=1

        model_cap=73

        if n_classes==2:

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
        else:
            num_correct=correct_count
            modelacc=int(float(num_correct*10000)/count)/100.0
            randguess=round(max(numeachclass.values())/sum(numeachclass.values())*100,2)
            print("System Type:                        "+str(n_classes)+"-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")






