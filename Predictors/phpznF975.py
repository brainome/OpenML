#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:50:33
# Invocation: btc -v -v phpznF975-1.csv -o phpznF975-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                98.34%
Model accuracy:                     98.56% (3633/3686 correct)
Improvement over best guess:        0.22% (of possible 1.66%)
Model capacity (MEC):               86 bits
Generalization ratio:               42.24 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     1.22% (45/3686)
True Positives:                     97.34% (3588/3686)
False Negatives:                    1.00% (37/3686)
False Positives:                    0.43% (16/3686)
True Pos. Rate/Sensitivity/Recall:  0.99
True Neg. Rate/Specificity:         0.74
Precision:                          1.00
F-1 Measure:                        0.99
False Negative Rate/Miss Rate:      0.01
Critical Success Index:             0.99
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
TRAINFILE="phpznF975-1.csv"


#Number of attributes
num_attr = 400

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
    if (energy>37.169760999999994):
        return 1.0
    if (energy>36.9532305):
        return 0.0
    if (energy>33.84175649999999):
        return 1.0
    if (energy>33.566961500000005):
        return 0.0
    if (energy>30.9230095):
        return 1.0
    if (energy>30.785349500000002):
        return 0.0
    if (energy>28.9164655):
        return 1.0
    if (energy>28.906354500000003):
        return 0.0
    if (energy>27.660476000000003):
        return 1.0
    if (energy>27.58012250000001):
        return 0.0
    if (energy>25.494120000000002):
        return 1.0
    if (energy>25.458852):
        return 0.0
    if (energy>23.952602499999998):
        return 1.0
    if (energy>23.912715000000002):
        return 0.0
    if (energy>20.680532999999997):
        return 1.0
    if (energy>20.654991):
        return 0.0
    if (energy>16.661463499999982):
        return 1.0
    if (energy>16.658269499999985):
        return 0.0
    if (energy>16.270258499999997):
        return 1.0
    if (energy>16.2400795):
        return 0.0
    if (energy>14.792396499999988):
        return 1.0
    if (energy>14.731901999999995):
        return 0.0
    if (energy>11.768537):
        return 1.0
    if (energy>11.757545000000002):
        return 0.0
    if (energy>11.710781000000011):
        return 1.0
    if (energy>11.6703695):
        return 0.0
    if (energy>11.482000499999995):
        return 1.0
    if (energy>11.421610000000006):
        return 0.0
    if (energy>11.017724500000003):
        return 1.0
    if (energy>10.972044):
        return 0.0
    if (energy>10.652384999999988):
        return 1.0
    if (energy>10.646669499999994):
        return 0.0
    if (energy>9.575596499999996):
        return 1.0
    if (energy>9.562593999999988):
        return 0.0
    if (energy>8.773476500000001):
        return 1.0
    if (energy>8.758350999999994):
        return 0.0
    if (energy>8.39194299999999):
        return 1.0
    if (energy>8.3849545):
        return 0.0
    if (energy>8.091774999999998):
        return 1.0
    if (energy>8.084576499999997):
        return 0.0
    if (energy>5.317317500000006):
        return 1.0
    if (energy>5.292965500000008):
        return 0.0
    if (energy>2.1269259999999957):
        return 1.0
    if (energy>2.1231489999999944):
        return 0.0
    if (energy>1.2387529999999938):
        return 1.0
    if (energy>1.234676499999999):
        return 0.0
    if (energy>-1.8026735000000036):
        return 1.0
    if (energy>-1.8087395000000102):
        return 0.0
    if (energy>-2.6703040000000104):
        return 1.0
    if (energy>-2.67876850000001):
        return 0.0
    if (energy>-2.7133089999999895):
        return 1.0
    if (energy>-2.751768499999989):
        return 0.0
    if (energy>-3.001092499999994):
        return 1.0
    if (energy>-3.0121120000000055):
        return 0.0
    if (energy>-3.272665000000006):
        return 1.0
    if (energy>-3.2803060000000075):
        return 0.0
    if (energy>-3.8524705):
        return 1.0
    if (energy>-3.898829000000001):
        return 0.0
    if (energy>-4.098914499999999):
        return 1.0
    if (energy>-4.102590500000002):
        return 0.0
    if (energy>-4.204608500000013):
        return 1.0
    if (energy>-4.227267500000016):
        return 0.0
    if (energy>-4.749151499999998):
        return 1.0
    if (energy>-4.783907999999998):
        return 0.0
    if (energy>-5.4431940000000045):
        return 1.0
    if (energy>-5.464630499999998):
        return 0.0
    if (energy>-7.558727999999993):
        return 1.0
    if (energy>-7.578945000000003):
        return 0.0
    if (energy>-8.27633299999999):
        return 1.0
    if (energy>-8.325330499999993):
        return 0.0
    if (energy>-8.509471000000001):
        return 1.0
    if (energy>-8.551641500000002):
        return 0.0
    if (energy>-9.010931000000019):
        return 1.0
    if (energy>-9.047180000000012):
        return 0.0
    if (energy>-10.224875999999984):
        return 1.0
    if (energy>-10.333794999999995):
        return 0.0
    if (energy>-10.90887050000001):
        return 1.0
    if (energy>-10.985946500000004):
        return 0.0
    if (energy>-11.069934500000002):
        return 1.0
    if (energy>-11.080353499999989):
        return 0.0
    if (energy>-18.092124):
        return 1.0
    if (energy>-18.1704455):
        return 0.0
    if (energy>-25.11570850000001):
        return 1.0
    if (energy>-25.13590050000002):
        return 0.0
    if (energy>-40.35260400000001):
        return 1.0
    if (energy>-41.4641925):
        return 0.0
    return 1.0

numthresholds=86


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

