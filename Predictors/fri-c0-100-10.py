#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 15:21:50
# Invocation: btc -v -v fri-c0-100-10-1.csv -o fri-c0-100-10-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.00%
Model accuracy:                     86.00% (86/100 correct)
Improvement over best guess:        31.00% (of possible 45.0%)
Model capacity (MEC):               85 bits
Generalization ratio:               1.01 bits/bit
Model efficiency:                   0.36%/parameter
System behavior
True Negatives:                     42.00% (42/100)
True Positives:                     44.00% (44/100)
False Negatives:                    1.00% (1/100)
False Positives:                    13.00% (13/100)
True Pos. Rate/Sensitivity/Recall:  0.98
True Neg. Rate/Specificity:         0.76
Precision:                          0.77
F-1 Measure:                        0.86
False Negative Rate/Miss Rate:      0.02
Critical Success Index:             0.76
Model bias:                         19.20% higher chance to pick class 1
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
TRAINFILE="fri-c0-100-10-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 10

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
    h_0 = max((((2.2610703 * float(x[0]))+ (3.3021326 * float(x[1]))+ (0.14781792 * float(x[2]))+ (2.071076 * float(x[3]))+ (1.5987109 * float(x[4]))+ (0.41992614 * float(x[5]))+ (1.3836042 * float(x[6]))+ (-2.9085848 * float(x[7]))+ (-1.7530409 * float(x[8]))+ (-0.044226985 * float(x[9]))) + 1.3973914), 0)
    h_1 = max((((1.8923414 * float(x[0]))+ (-0.0140409 * float(x[1]))+ (-0.8043631 * float(x[2]))+ (-0.34558004 * float(x[3]))+ (0.50456667 * float(x[4]))+ (0.5173708 * float(x[5]))+ (-0.09695044 * float(x[6]))+ (-0.20071858 * float(x[7]))+ (0.2689046 * float(x[8]))+ (1.5268058 * float(x[9]))) + 0.27805737), 0)
    h_2 = max((((-0.54912 * float(x[0]))+ (1.7809606 * float(x[1]))+ (0.25372693 * float(x[2]))+ (-0.046645623 * float(x[3]))+ (-0.09026204 * float(x[4]))+ (-0.8360094 * float(x[5]))+ (-0.14564331 * float(x[6]))+ (0.46074915 * float(x[7]))+ (0.24404189 * float(x[8]))+ (-0.57481956 * float(x[9]))) + -1.4404017), 0)
    h_3 = max((((0.02368934 * float(x[0]))+ (0.6835528 * float(x[1]))+ (0.35449263 * float(x[2]))+ (-1.2401501 * float(x[3]))+ (-1.688844 * float(x[4]))+ (0.030481663 * float(x[5]))+ (0.22847593 * float(x[6]))+ (0.46477214 * float(x[7]))+ (-1.6542019 * float(x[8]))+ (0.43629882 * float(x[9]))) + 0.028705735), 0)
    h_4 = max((((0.37395045 * float(x[0]))+ (0.66683304 * float(x[1]))+ (0.24084777 * float(x[2]))+ (0.1796152 * float(x[3]))+ (0.40724418 * float(x[4]))+ (0.24646622 * float(x[5]))+ (-0.38561863 * float(x[6]))+ (1.0414169 * float(x[7]))+ (-1.5356653 * float(x[8]))+ (-0.47777334 * float(x[9]))) + -0.28446984), 0)
    h_5 = max((((-0.88227135 * float(x[0]))+ (1.9452127 * float(x[1]))+ (1.5270913 * float(x[2]))+ (0.08441679 * float(x[3]))+ (0.14693426 * float(x[4]))+ (-0.5770893 * float(x[5]))+ (1.022058 * float(x[6]))+ (0.021093752 * float(x[7]))+ (0.7062069 * float(x[8]))+ (-0.31754708 * float(x[9]))) + -0.5490327), 0)
    h_6 = max((((0.9822394 * float(x[0]))+ (0.879167 * float(x[1]))+ (-0.42739984 * float(x[2]))+ (-0.16383786 * float(x[3]))+ (-0.03920708 * float(x[4]))+ (-0.9729101 * float(x[5]))+ (-1.187626 * float(x[6]))+ (-0.1328748 * float(x[7]))+ (-0.66669554 * float(x[8]))+ (-0.26520303 * float(x[9]))) + -0.36444983), 0)
    o_0 = (2.0054963 * h_0)+ (1.1040665 * h_1)+ (3.0621324 * h_2)+ (-3.324361 * h_3)+ (0.7954232 * h_4)+ (1.4411949 * h_5)+ (3.3659985 * h_6) + -3.6911259
             
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

