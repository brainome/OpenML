#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 03:08:39
# Invocation: btc -v fri_c1_1000_25-8.csv -o fri_c1_1000_25-8.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                54.60%
Model accuracy:                     94.00% (940/1000 correct)
Improvement over best guess:        39.40% (of possible 45.4%)
Model capacity (MEC):               82 bits
Generalization ratio:               11.46 bits/bit
Model efficiency:                   0.48%/parameter
System behavior
True Negatives:                     51.10% (511/1000)
True Positives:                     42.90% (429/1000)
False Negatives:                    2.50% (25/1000)
False Positives:                    3.50% (35/1000)
True Pos. Rate/Sensitivity/Recall:  0.94
True Neg. Rate/Specificity:         0.94
Precision:                          0.92
F-1 Measure:                        0.93
False Negative Rate/Miss Rate:      0.06
Critical Success Index:             0.88
Model bias:                         0.16% higher chance to pick class 0
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="fri_c1_1000_25-8.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 25

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
    h_0 = max((((17.839548 * float(x[0]))+ (21.99961 * float(x[1]))+ (0.7264185 * float(x[2]))+ (-4.498801 * float(x[3]))+ (-3.4623263 * float(x[4]))+ (-0.31579512 * float(x[5]))+ (-0.64714164 * float(x[6]))+ (-0.3731275 * float(x[7]))+ (0.24735975 * float(x[8]))+ (0.275342 * float(x[9]))+ (1.806938 * float(x[10]))+ (1.1434736 * float(x[11]))+ (-0.5994948 * float(x[12]))+ (-1.0591596 * float(x[13]))+ (0.20462829 * float(x[14]))+ (-0.6416454 * float(x[15]))+ (-1.6716403 * float(x[16]))+ (0.011552736 * float(x[17]))+ (2.10306 * float(x[18]))+ (1.0011504 * float(x[19]))+ (0.40158013 * float(x[20]))+ (1.0406699 * float(x[21]))+ (-0.7683612 * float(x[22]))+ (0.5997306 * float(x[23]))+ (0.24586883 * float(x[24]))) + -2.0917509), 0)
    h_1 = max((((-8.411417 * float(x[0]))+ (-8.091652 * float(x[1]))+ (2.407469 * float(x[2]))+ (-5.346675 * float(x[3]))+ (-2.8280125 * float(x[4]))+ (-0.08645275 * float(x[5]))+ (-0.30017275 * float(x[6]))+ (0.10493745 * float(x[7]))+ (-0.2831013 * float(x[8]))+ (0.3740243 * float(x[9]))+ (0.4282568 * float(x[10]))+ (-0.5847707 * float(x[11]))+ (-0.5002589 * float(x[12]))+ (0.45457736 * float(x[13]))+ (-0.89812404 * float(x[14]))+ (0.56143564 * float(x[15]))+ (-0.76054305 * float(x[16]))+ (-0.8790645 * float(x[17]))+ (-0.985379 * float(x[18]))+ (-0.8174708 * float(x[19]))+ (-0.35695904 * float(x[20]))+ (-0.08826662 * float(x[21]))+ (-0.47321367 * float(x[22]))+ (-0.75852734 * float(x[23]))+ (1.4403079 * float(x[24]))) + -4.408735), 0)
    h_2 = max((((5.865413 * float(x[0]))+ (4.899499 * float(x[1]))+ (-0.13998874 * float(x[2]))+ (-0.21062124 * float(x[3]))+ (-0.10251727 * float(x[4]))+ (0.038290318 * float(x[5]))+ (-0.029360633 * float(x[6]))+ (0.004228974 * float(x[7]))+ (0.007191318 * float(x[8]))+ (0.05953521 * float(x[9]))+ (0.32386583 * float(x[10]))+ (0.122133926 * float(x[11]))+ (-0.018112399 * float(x[12]))+ (-0.05169494 * float(x[13]))+ (-0.16965877 * float(x[14]))+ (-0.017507264 * float(x[15]))+ (-0.35876983 * float(x[16]))+ (0.17948602 * float(x[17]))+ (0.29369032 * float(x[18]))+ (0.190336 * float(x[19]))+ (-0.0042212782 * float(x[20]))+ (0.22136585 * float(x[21]))+ (-0.12120176 * float(x[22]))+ (0.23792504 * float(x[23]))+ (-0.029428704 * float(x[24]))) + -7.762666), 0)
    o_0 = (1.2906728 * h_0)+ (0.4708851 * h_1)+ (-8.935065 * h_2) + -7.7809205
             
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
        if args.cleanfile:
            with open(args.csvfile,'r') as cleancsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                for cleanrow in cleancsvreader:
                    if len(cleanrow)==0:
                        continue
                print(str(','.join(str(j) for j in ([i for i in cleanrow])))+','+str(int(classify(cleanrow))))
        else:
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            with open(cleanfile,'r') as cleancsvfile, open(args.csvfile,'r') as dirtycsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                dirtycsvreader = csv.reader(dirtycsvfile)
                if (not args.headerless):
                        print(','.join(next(dirtycsvreader, None)+['Prediction']))
                for cleanrow,dirtyrow in zip(cleancsvreader,dirtycsvreader):
                    if len(cleanrow)==0:
                        continue
                    print(str(','.join(str(j) for j in ([i for i in dirtyrow])))+','+str(int(classify(cleanrow))))
            os.remove(cleanfile)
            
    else: # Then validate this predictor
        tempdir=tempfile.gettempdir()
        temp_name = next(tempfile._get_candidate_names())
        cleanvalfile=tempdir+os.sep+temp_name
        clean(args.csvfile,cleanvalfile, -1, args.headerless)
        with open(cleanvalfile,'r') as valcsvfile:
            count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0=0,0,0,0,0,0,0,0
            valcsvreader = csv.reader(valcsvfile)
            for valrow in valcsvreader:
                if len(valrow)==0:
                    continue
                if int(classify(valrow[:-1]))==int(float(valrow[-1])):
                    correct_count+=1
                    if int(float(valrow[-1]))==1:
                        num_class_1+=1
                        num_TP+=1
                    else:
                        num_class_0+=1
                        num_TN+=1
                else:
                    if int(float(valrow[-1]))==1:
                        num_class_1+=1
                        num_FN+=1
                    else:
                        num_class_0+=1
                        num_FP+=1
                count+=1

        model_cap=82

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


        os.remove(cleanvalfile)

