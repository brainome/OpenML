#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 16:31:10
# Invocation: btc -v -v fri-c2-500-50-1.csv -o fri-c2-500-50-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                59.00%
Model accuracy:                     81.40% (407/500 correct)
Improvement over best guess:        22.40% (of possible 41.0%)
Model capacity (MEC):               157 bits
Generalization ratio:               2.59 bits/bit
Model efficiency:                   0.14%/parameter
System behavior
True Negatives:                     32.20% (161/500)
True Positives:                     49.20% (246/500)
False Negatives:                    9.80% (49/500)
False Positives:                    8.80% (44/500)
True Pos. Rate/Sensitivity/Recall:  0.83
True Neg. Rate/Specificity:         0.79
Precision:                          0.85
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.17
Critical Success Index:             0.73
Model bias:                         18.14% higher chance to pick class 0
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
TRAINFILE="fri-c2-500-50-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50

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
    h_0 = max((((-12.9444475 * float(x[0]))+ (-12.585749 * float(x[1]))+ (-8.731382 * float(x[2]))+ (6.3707757 * float(x[3]))+ (4.252172 * float(x[4]))+ (-2.0011518 * float(x[5]))+ (0.933576 * float(x[6]))+ (3.030106 * float(x[7]))+ (0.38229764 * float(x[8]))+ (-1.6178037 * float(x[9]))+ (-0.0065252176 * float(x[10]))+ (-0.6213671 * float(x[11]))+ (-1.9321741 * float(x[12]))+ (1.6537921 * float(x[13]))+ (-0.72690094 * float(x[14]))+ (1.621101 * float(x[15]))+ (2.308794 * float(x[16]))+ (-1.5495021 * float(x[17]))+ (-0.9885341 * float(x[18]))+ (0.20735125 * float(x[19]))+ (-1.00924 * float(x[20]))+ (2.392969 * float(x[21]))+ (-0.9627709 * float(x[22]))+ (0.765544 * float(x[23]))+ (-2.2237482 * float(x[24]))+ (1.0184287 * float(x[25]))+ (1.1161863 * float(x[26]))+ (-2.4730911 * float(x[27]))+ (5.001869 * float(x[28]))+ (1.3451413 * float(x[29]))+ (-0.806403 * float(x[30]))+ (-0.02498153 * float(x[31]))+ (-1.6654695 * float(x[32]))+ (-0.089848205 * float(x[33]))+ (-1.9888047 * float(x[34]))+ (-0.24376151 * float(x[35]))+ (0.9965877 * float(x[36]))+ (-4.7274218 * float(x[37]))+ (-2.114971 * float(x[38]))+ (1.0038296 * float(x[39]))+ (1.0733827 * float(x[40]))+ (1.8950976 * float(x[41]))+ (0.41826916 * float(x[42]))+ (0.52198285 * float(x[43]))+ (0.16135006 * float(x[44]))+ (-2.1076362 * float(x[45]))+ (2.0971863 * float(x[46]))+ (-2.5723007 * float(x[47]))+ (-1.774842 * float(x[48]))+ (-0.3066472 * float(x[49]))) + 3.6560113), 0)
    h_1 = max((((3.1549616 * float(x[0]))+ (-0.5068376 * float(x[1]))+ (0.28012186 * float(x[2]))+ (-0.19186868 * float(x[3]))+ (1.9566647 * float(x[4]))+ (-0.1403909 * float(x[5]))+ (0.32625556 * float(x[6]))+ (0.1461164 * float(x[7]))+ (0.65629435 * float(x[8]))+ (1.1855813 * float(x[9]))+ (0.43354946 * float(x[10]))+ (-1.327836 * float(x[11]))+ (3.6615436 * float(x[12]))+ (-0.1503914 * float(x[13]))+ (1.9716632 * float(x[14]))+ (-0.98480445 * float(x[15]))+ (0.29642797 * float(x[16]))+ (1.8330541 * float(x[17]))+ (-0.73309964 * float(x[18]))+ (-0.33604017 * float(x[19]))+ (-0.6529736 * float(x[20]))+ (-0.68900174 * float(x[21]))+ (-0.55196685 * float(x[22]))+ (-1.1523674 * float(x[23]))+ (1.872569 * float(x[24]))+ (1.1744235 * float(x[25]))+ (-2.14346 * float(x[26]))+ (-1.6721553 * float(x[27]))+ (1.2372096 * float(x[28]))+ (0.37884128 * float(x[29]))+ (-2.2150085 * float(x[30]))+ (2.041178 * float(x[31]))+ (0.7496648 * float(x[32]))+ (0.27286005 * float(x[33]))+ (0.44278196 * float(x[34]))+ (0.20458657 * float(x[35]))+ (0.9728737 * float(x[36]))+ (0.9617713 * float(x[37]))+ (-1.1995546 * float(x[38]))+ (-1.7442219 * float(x[39]))+ (-2.4024901 * float(x[40]))+ (0.04001059 * float(x[41]))+ (-2.7325618 * float(x[42]))+ (1.7573541 * float(x[43]))+ (-2.6633618 * float(x[44]))+ (-0.6320162 * float(x[45]))+ (2.2455566 * float(x[46]))+ (1.1668354 * float(x[47]))+ (1.1614921 * float(x[48]))+ (-0.6540007 * float(x[49]))) + -0.32102767), 0)
    h_2 = max((((-0.50147384 * float(x[0]))+ (-1.3442894 * float(x[1]))+ (2.2202706 * float(x[2]))+ (-1.6471016 * float(x[3]))+ (-0.18002878 * float(x[4]))+ (-0.5364598 * float(x[5]))+ (1.1100365 * float(x[6]))+ (-0.7321719 * float(x[7]))+ (-0.071466625 * float(x[8]))+ (1.1840425 * float(x[9]))+ (-0.38759148 * float(x[10]))+ (-0.60810876 * float(x[11]))+ (3.3598099 * float(x[12]))+ (0.29096597 * float(x[13]))+ (1.8860886 * float(x[14]))+ (-0.52303886 * float(x[15]))+ (1.7196442 * float(x[16]))+ (0.5769351 * float(x[17]))+ (-0.43556526 * float(x[18]))+ (-1.1533928 * float(x[19]))+ (-0.42642832 * float(x[20]))+ (-0.19736794 * float(x[21]))+ (-1.4619479 * float(x[22]))+ (0.91361415 * float(x[23]))+ (0.06581761 * float(x[24]))+ (0.7767601 * float(x[25]))+ (-1.8333578 * float(x[26]))+ (-1.5512022 * float(x[27]))+ (0.17094764 * float(x[28]))+ (1.6132007 * float(x[29]))+ (-1.6863173 * float(x[30]))+ (0.04925138 * float(x[31]))+ (-0.09552627 * float(x[32]))+ (-0.4953021 * float(x[33]))+ (0.8086716 * float(x[34]))+ (-0.8021955 * float(x[35]))+ (0.04853568 * float(x[36]))+ (-0.47165933 * float(x[37]))+ (-1.1729543 * float(x[38]))+ (-0.6451023 * float(x[39]))+ (0.6946317 * float(x[40]))+ (0.39260504 * float(x[41]))+ (-2.0545607 * float(x[42]))+ (0.5746213 * float(x[43]))+ (0.24350715 * float(x[44]))+ (0.1536649 * float(x[45]))+ (1.3327612 * float(x[46]))+ (-0.2910467 * float(x[47]))+ (1.0852039 * float(x[48]))+ (0.4045966 * float(x[49]))) + 0.59893984), 0)
    o_0 = (0.6890004 * h_0)+ (1.9473534 * h_1)+ (-2.9356751 * h_2) + -3.0731645
             
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

        model_cap=157

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

