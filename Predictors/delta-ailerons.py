#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 16:30:33
# Invocation: btc -v -v delta-ailerons-1.csv -o delta-ailerons-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                53.06%
Model accuracy:                     93.95% (6698/7129 correct)
Improvement over best guess:        40.89% (of possible 46.94%)
Model capacity (MEC):               85 bits
Generalization ratio:               78.80 bits/bit
Model efficiency:                   0.48%/parameter
System behavior
True Negatives:                     50.60% (3607/7129)
True Positives:                     43.36% (3091/7129)
False Negatives:                    3.58% (255/7129)
False Positives:                    2.47% (176/7129)
True Pos. Rate/Sensitivity/Recall:  0.92
True Neg. Rate/Specificity:         0.95
Precision:                          0.95
F-1 Measure:                        0.93
False Negative Rate/Miss Rate:      0.08
Critical Success Index:             0.88
Model bias:                         27.70% higher chance to pick class 1
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
TRAINFILE="delta-ailerons-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 5

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
    h_0 = max((((-39.377052 * float(x[0]))+ (-3.2416925 * float(x[1]))+ (7.5063777 * float(x[2]))+ (-3.7448285 * float(x[3]))+ (-63.457676 * float(x[4]))) + 0.34961033), 0)
    h_1 = max((((-8.16066 * float(x[0]))+ (-2.362343 * float(x[1]))+ (2.0862477 * float(x[2]))+ (0.28220886 * float(x[3]))+ (63.72294 * float(x[4]))) + 0.12076394), 0)
    h_2 = max((((-3.2888873 * float(x[0]))+ (-0.78261006 * float(x[1]))+ (0.0056117885 * float(x[2]))+ (-0.5594206 * float(x[3]))+ (-0.41799626 * float(x[4]))) + -0.8727744), 0)
    h_3 = max((((-0.80513453 * float(x[0]))+ (-3.5430412 * float(x[1]))+ (1.0575624 * float(x[2]))+ (-1.8683865 * float(x[3]))+ (65.26782 * float(x[4]))) + 0.33900273), 0)
    h_4 = max((((-0.48063165 * float(x[0]))+ (-4.690692 * float(x[1]))+ (1.3533059 * float(x[2]))+ (-1.3665667 * float(x[3]))+ (63.85356 * float(x[4]))) + 0.4850274), 0)
    h_5 = max((((0.23759905 * float(x[0]))+ (-0.5588662 * float(x[1]))+ (0.48213536 * float(x[2]))+ (-0.08262567 * float(x[3]))+ (0.30501384 * float(x[4]))) + -0.50381047), 0)
    h_6 = max((((0.7099656 * float(x[0]))+ (-4.4504776 * float(x[1]))+ (0.7013665 * float(x[2]))+ (0.15187341 * float(x[3]))+ (65.99897 * float(x[4]))) + 0.05627337), 0)
    h_7 = max((((-0.66386664 * float(x[0]))+ (-0.93014354 * float(x[1]))+ (-0.09788967 * float(x[2]))+ (0.4895955 * float(x[3]))+ (0.82536656 * float(x[4]))) + -0.028018216), 0)
    h_8 = max((((0.6374258 * float(x[0]))+ (-0.6134041 * float(x[1]))+ (-0.8541331 * float(x[2]))+ (0.68128204 * float(x[3]))+ (-0.31177363 * float(x[4]))) + -0.528994), 0)
    h_9 = max((((-0.3795257 * float(x[0]))+ (-0.7963462 * float(x[1]))+ (0.8484117 * float(x[2]))+ (-0.3330968 * float(x[3]))+ (0.4189266 * float(x[4]))) + -0.5882624), 0)
    h_10 = max((((0.65923357 * float(x[0]))+ (-0.42505994 * float(x[1]))+ (-0.5901593 * float(x[2]))+ (-0.8523875 * float(x[3]))+ (0.7133502 * float(x[4]))) + -0.14715633), 0)
    h_11 = max((((-0.13031933 * float(x[0]))+ (-4.8185444 * float(x[1]))+ (0.8896481 * float(x[2]))+ (0.39940697 * float(x[3]))+ (64.659424 * float(x[4]))) + 0.061582845), 0)
    o_0 = (15.71127 * h_0)+ (-8.855221 * h_1)+ (-0.1270078 * h_2)+ (-2.5795355 * h_3)+ (-3.0169668 * h_4)+ (-0.8620202 * h_5)+ (-13.175443 * h_6)+ (-0.9998391 * h_7)+ (0.7727489 * h_8)+ (0.14537169 * h_9)+ (0.90823287 * h_10)+ (-11.575091 * h_11) + -1.1399409
             
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

        model_cap=85

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

