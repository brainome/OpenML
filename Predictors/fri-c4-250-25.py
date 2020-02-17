#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 12:47:06
# Invocation: btc -v -v fri-c4-250-25-7.csv -o fri-c4-250-25-7.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                54.40%
Model accuracy:                     86.80% (217/250 correct)
Improvement over best guess:        32.40% (of possible 45.6%)
Model capacity (MEC):               81 bits
Generalization ratio:               2.67 bits/bit
Model efficiency:                   0.40%/parameter
System behavior
True Negatives:                     48.80% (122/250)
True Positives:                     38.00% (95/250)
False Negatives:                    7.60% (19/250)
False Positives:                    5.60% (14/250)
True Pos. Rate/Sensitivity/Recall:  0.83
True Neg. Rate/Specificity:         0.90
Precision:                          0.87
F-1 Measure:                        0.85
False Negative Rate/Miss Rate:      0.17
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


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="fri-c4-250-25-7.csv"


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


# Calculate equilibrium energy ($_i=1)
def eqenergy(row):
    result=0
    for elem in row:
        result = result + float(elem)
    return result

# Classifier 
def classify(row):
    energy=eqenergy(row)
    if (energy>16.313727999999998):
        return 1.0
    if (energy>13.382867999999998):
        return 0.0
    if (energy>13.118165999999999):
        return 1.0
    if (energy>11.051533500000001):
        return 0.0
    if (energy>8.480433499999998):
        return 1.0
    if (energy>8.126865000000002):
        return 0.0
    if (energy>7.3778205):
        return 1.0
    if (energy>6.985045500000001):
        return 0.0
    if (energy>6.697804):
        return 1.0
    if (energy>6.473174499999999):
        return 0.0
    if (energy>6.3702784999999995):
        return 1.0
    if (energy>6.315920499999999):
        return 0.0
    if (energy>6.003993499999997):
        return 1.0
    if (energy>5.686703499999998):
        return 0.0
    if (energy>5.6020895):
        return 1.0
    if (energy>5.529653999999999):
        return 0.0
    if (energy>4.8702214999999995):
        return 1.0
    if (energy>4.7167819999999985):
        return 0.0
    if (energy>4.320938):
        return 1.0
    if (energy>4.118845):
        return 0.0
    if (energy>4.1018875):
        return 1.0
    if (energy>3.906273):
        return 0.0
    if (energy>3.5429284999999995):
        return 1.0
    if (energy>3.408617):
        return 0.0
    if (energy>3.058600499999999):
        return 1.0
    if (energy>2.5214744999999996):
        return 0.0
    if (energy>2.351644000000001):
        return 1.0
    if (energy>2.209975):
        return 0.0
    if (energy>2.1213875):
        return 1.0
    if (energy>2.0018395000000004):
        return 0.0
    if (energy>1.9809440000000005):
        return 1.0
    if (energy>1.8859330000000003):
        return 0.0
    if (energy>1.6074624999999998):
        return 1.0
    if (energy>1.5021480000000005):
        return 0.0
    if (energy>1.403527):
        return 1.0
    if (energy>1.369948):
        return 0.0
    if (energy>1.3602595000000002):
        return 1.0
    if (energy>0.9041515):
        return 0.0
    if (energy>0.7075780000000007):
        return 1.0
    if (energy>0.22184449999999922):
        return 0.0
    if (energy>-0.07538449999999956):
        return 1.0
    if (energy>-0.6425124999999992):
        return 0.0
    if (energy>-0.8077229999999997):
        return 1.0
    if (energy>-0.9211669999999997):
        return 0.0
    if (energy>-0.9485669999999999):
        return 1.0
    if (energy>-1.3096349999999994):
        return 0.0
    if (energy>-1.3911895000000003):
        return 1.0
    if (energy>-1.7976759999999996):
        return 0.0
    if (energy>-1.9372149999999997):
        return 1.0
    if (energy>-2.1534145000000002):
        return 0.0
    if (energy>-2.2532385):
        return 1.0
    if (energy>-2.6643134999999996):
        return 0.0
    if (energy>-3.1481759999999994):
        return 1.0
    if (energy>-3.222790500000001):
        return 0.0
    if (energy>-3.280103):
        return 1.0
    if (energy>-3.3301990000000004):
        return 0.0
    if (energy>-3.3796275000000007):
        return 1.0
    if (energy>-3.679954500000001):
        return 0.0
    if (energy>-3.6932080000000003):
        return 1.0
    if (energy>-3.7462385000000005):
        return 0.0
    if (energy>-3.9739820000000003):
        return 1.0
    if (energy>-4.0097380000000005):
        return 0.0
    if (energy>-4.325012000000001):
        return 1.0
    if (energy>-5.4136345):
        return 0.0
    if (energy>-5.918354999999998):
        return 1.0
    if (energy>-6.8112785):
        return 0.0
    if (energy>-7.173560500000001):
        return 1.0
    if (energy>-7.338121499999998):
        return 0.0
    if (energy>-7.631463999999999):
        return 1.0
    if (energy>-7.834178):
        return 0.0
    if (energy>-7.972703500000001):
        return 1.0
    if (energy>-8.1310155):
        return 0.0
    if (energy>-8.396878000000001):
        return 1.0
    if (energy>-8.8629405):
        return 0.0
    if (energy>-8.915368):
        return 1.0
    if (energy>-8.9659925):
        return 0.0
    if (energy>-10.005333):
        return 1.0
    if (energy>-10.8099965):
        return 0.0
    if (energy>-10.882395999999998):
        return 1.0
    if (energy>-13.043650499999998):
        return 0.0
    if (energy>-13.9435255):
        return 1.0
    return 0.0

numthresholds=81


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

        model_cap=numthresholds

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
