#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 21:17:16
# Invocation: btc -v fri-c0-100-10-8.csv -o fri-c0-100-10-8.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.00%
Model accuracy:                     94.00% (94/100 correct)
Improvement over best guess:        39.00% (of possible 45.0%)
Model capacity (MEC):               49 bits
Generalization ratio:               1.91 bits/bit
Model efficiency:                   0.79%/parameter
System behavior
True Negatives:                     41.00% (41/100)
True Positives:                     53.00% (53/100)
False Negatives:                    2.00% (2/100)
False Positives:                    4.00% (4/100)
True Pos. Rate/Sensitivity/Recall:  0.96
True Neg. Rate/Specificity:         0.91
Precision:                          0.93
F-1 Measure:                        0.95
False Negative Rate/Miss Rate:      0.04
Critical Success Index:             0.90
Model bias:                         0.12% higher chance to pick class 1
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
TRAINFILE="fri-c0-100-10-8.csv"


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
    h_0 = max((((-2.176085 * float(x[0]))+ (-3.7919815 * float(x[1]))+ (0.4535201 * float(x[2]))+ (-4.3783183 * float(x[3]))+ (-4.035052 * float(x[4]))+ (0.5470011 * float(x[5]))+ (0.2374514 * float(x[6]))+ (1.3187447 * float(x[7]))+ (-1.0670769 * float(x[8]))+ (-0.6066007 * float(x[9]))) + 0.73204494), 0)
    h_1 = max((((-0.59574544 * float(x[0]))+ (-0.0882753 * float(x[1]))+ (-0.15940899 * float(x[2]))+ (0.09969818 * float(x[3]))+ (1.3910056 * float(x[4]))+ (1.9961951 * float(x[5]))+ (-1.3968427 * float(x[6]))+ (-1.6052783 * float(x[7]))+ (0.9035176 * float(x[8]))+ (-0.6340258 * float(x[9]))) + 1.3000638), 0)
    h_2 = max((((-0.5191406 * float(x[0]))+ (-0.94628465 * float(x[1]))+ (0.12105479 * float(x[2]))+ (-0.37950957 * float(x[3]))+ (0.8375766 * float(x[4]))+ (-1.5893023 * float(x[5]))+ (-0.031069405 * float(x[6]))+ (0.023218978 * float(x[7]))+ (0.5954172 * float(x[8]))+ (-0.30293584 * float(x[9]))) + -1.3462274), 0)
    h_3 = max((((0.40043655 * float(x[0]))+ (-0.3464236 * float(x[1]))+ (0.25297326 * float(x[2]))+ (1.6167215 * float(x[3]))+ (-0.5196437 * float(x[4]))+ (1.993656 * float(x[5]))+ (0.6050244 * float(x[6]))+ (-0.76145554 * float(x[7]))+ (-1.0914054 * float(x[8]))+ (-0.68311286 * float(x[9]))) + 1.0284423), 0)
    o_0 = (3.236666 * h_0)+ (0.41651562 * h_1)+ (1.8296646 * h_2)+ (-2.3115366 * h_3) + -1.7782092
             
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

        model_cap=49

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

