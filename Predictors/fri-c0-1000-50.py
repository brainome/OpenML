#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 15:23:40
# Invocation: btc -v -v fri-c0-1000-50-1.csv -o fri-c0-1000-50-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                51.00%
Model accuracy:                     84.60% (846/1000 correct)
Improvement over best guess:        33.60% (of possible 49.0%)
Model capacity (MEC):               209 bits
Generalization ratio:               4.04 bits/bit
Model efficiency:                   0.16%/parameter
System behavior
True Negatives:                     44.30% (443/1000)
True Positives:                     40.30% (403/1000)
False Negatives:                    8.70% (87/1000)
False Positives:                    6.70% (67/1000)
True Pos. Rate/Sensitivity/Recall:  0.82
True Neg. Rate/Specificity:         0.87
Precision:                          0.86
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.18
Critical Success Index:             0.72
Model bias:                         11.32% higher chance to pick class 0
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
TRAINFILE="fri-c0-1000-50-1.csv"


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
    h_0 = max((((18.796587 * float(x[0]))+ (17.261898 * float(x[1]))+ (-1.3339926 * float(x[2]))+ (16.83456 * float(x[3]))+ (6.6176953 * float(x[4]))+ (-0.93750614 * float(x[5]))+ (-0.46690494 * float(x[6]))+ (4.1725826 * float(x[7]))+ (2.275122 * float(x[8]))+ (2.1071832 * float(x[9]))+ (0.1437398 * float(x[10]))+ (1.1448265 * float(x[11]))+ (4.2984667 * float(x[12]))+ (0.26867124 * float(x[13]))+ (2.982767 * float(x[14]))+ (-0.57854736 * float(x[15]))+ (-1.1373829 * float(x[16]))+ (2.187686 * float(x[17]))+ (-2.3739092 * float(x[18]))+ (4.2862387 * float(x[19]))+ (-1.6022234 * float(x[20]))+ (-6.052945 * float(x[21]))+ (1.5427463 * float(x[22]))+ (0.61123985 * float(x[23]))+ (1.7460976 * float(x[24]))+ (-2.9839177 * float(x[25]))+ (1.9854554 * float(x[26]))+ (-2.5694408 * float(x[27]))+ (-2.1454964 * float(x[28]))+ (5.269807 * float(x[29]))+ (2.0245955 * float(x[30]))+ (0.10699375 * float(x[31]))+ (-3.0255363 * float(x[32]))+ (5.294641 * float(x[33]))+ (0.81269765 * float(x[34]))+ (0.41463244 * float(x[35]))+ (0.5186705 * float(x[36]))+ (1.1866897 * float(x[37]))+ (6.996833 * float(x[38]))+ (2.3196228 * float(x[39]))+ (-3.5354397 * float(x[40]))+ (-0.8836489 * float(x[41]))+ (3.1219935 * float(x[42]))+ (-2.4118252 * float(x[43]))+ (-3.5010262 * float(x[44]))+ (3.2788348 * float(x[45]))+ (-1.6531657 * float(x[46]))+ (1.8642884 * float(x[47]))+ (0.64265376 * float(x[48]))+ (-0.13626145 * float(x[49]))) + 2.4941003), 0)
    h_1 = max((((5.2281227 * float(x[0]))+ (0.50934744 * float(x[1]))+ (0.75285125 * float(x[2]))+ (7.929284 * float(x[3]))+ (3.4709895 * float(x[4]))+ (0.8087675 * float(x[5]))+ (-1.9158969 * float(x[6]))+ (2.7013855 * float(x[7]))+ (-0.33596012 * float(x[8]))+ (0.42915395 * float(x[9]))+ (-1.363117 * float(x[10]))+ (-4.412455 * float(x[11]))+ (-0.88675886 * float(x[12]))+ (2.2239554 * float(x[13]))+ (-1.0419259 * float(x[14]))+ (-0.17526881 * float(x[15]))+ (-0.25530535 * float(x[16]))+ (-0.118724585 * float(x[17]))+ (-1.3529041 * float(x[18]))+ (-4.2472215 * float(x[19]))+ (0.78230244 * float(x[20]))+ (0.985408 * float(x[21]))+ (-1.7970729 * float(x[22]))+ (-1.3454142 * float(x[23]))+ (1.2101732 * float(x[24]))+ (-0.61012894 * float(x[25]))+ (2.5012887 * float(x[26]))+ (3.9247553 * float(x[27]))+ (-1.3569546 * float(x[28]))+ (-0.30058149 * float(x[29]))+ (-2.3519914 * float(x[30]))+ (1.2126461 * float(x[31]))+ (2.2449317 * float(x[32]))+ (1.4930135 * float(x[33]))+ (-3.1748765 * float(x[34]))+ (3.2549357 * float(x[35]))+ (1.6104674 * float(x[36]))+ (-1.996334 * float(x[37]))+ (-0.17658257 * float(x[38]))+ (2.5697074 * float(x[39]))+ (-0.3617937 * float(x[40]))+ (3.135822 * float(x[41]))+ (-4.744754 * float(x[42]))+ (-4.9769807 * float(x[43]))+ (-0.46674976 * float(x[44]))+ (-5.7031507 * float(x[45]))+ (0.23881632 * float(x[46]))+ (2.89391 * float(x[47]))+ (1.8317981 * float(x[48]))+ (3.5244796 * float(x[49]))) + -4.284873), 0)
    h_2 = max((((-0.52791095 * float(x[0]))+ (-4.5949507 * float(x[1]))+ (-0.22543041 * float(x[2]))+ (-1.9370604 * float(x[3]))+ (-1.1690967 * float(x[4]))+ (1.7095208 * float(x[5]))+ (-2.6715138 * float(x[6]))+ (0.55688804 * float(x[7]))+ (0.33295572 * float(x[8]))+ (1.1310593 * float(x[9]))+ (-1.7677352 * float(x[10]))+ (1.9429792 * float(x[11]))+ (-1.0691004 * float(x[12]))+ (0.09725836 * float(x[13]))+ (0.2861919 * float(x[14]))+ (-0.10267864 * float(x[15]))+ (-2.5842054 * float(x[16]))+ (-2.3434515 * float(x[17]))+ (-0.2679299 * float(x[18]))+ (1.7331523 * float(x[19]))+ (-0.044255324 * float(x[20]))+ (2.3699734 * float(x[21]))+ (-0.7959868 * float(x[22]))+ (1.1989473 * float(x[23]))+ (-0.78297895 * float(x[24]))+ (1.6244382 * float(x[25]))+ (0.9358575 * float(x[26]))+ (2.3226695 * float(x[27]))+ (-0.33497286 * float(x[28]))+ (0.45681977 * float(x[29]))+ (-0.36849073 * float(x[30]))+ (-0.55575097 * float(x[31]))+ (-2.3110168 * float(x[32]))+ (0.23711628 * float(x[33]))+ (-3.3287077 * float(x[34]))+ (1.5057355 * float(x[35]))+ (1.2710408 * float(x[36]))+ (-2.393611 * float(x[37]))+ (0.35472977 * float(x[38]))+ (1.5286204 * float(x[39]))+ (0.7607732 * float(x[40]))+ (0.13673754 * float(x[41]))+ (-1.035984 * float(x[42]))+ (0.20223492 * float(x[43]))+ (-0.7628264 * float(x[44]))+ (1.828229 * float(x[45]))+ (0.23781754 * float(x[46]))+ (1.7719308 * float(x[47]))+ (2.1240475 * float(x[48]))+ (0.6287167 * float(x[49]))) + -5.5692163), 0)
    h_3 = max((((5.1860814 * float(x[0]))+ (3.8365138 * float(x[1]))+ (-0.34242398 * float(x[2]))+ (2.5289896 * float(x[3]))+ (1.6998585 * float(x[4]))+ (-0.43077973 * float(x[5]))+ (-0.12741101 * float(x[6]))+ (1.8368548 * float(x[7]))+ (-1.0233476 * float(x[8]))+ (0.7138444 * float(x[9]))+ (0.7809671 * float(x[10]))+ (-0.70289594 * float(x[11]))+ (1.1008594 * float(x[12]))+ (1.168666 * float(x[13]))+ (0.22740163 * float(x[14]))+ (-0.19498414 * float(x[15]))+ (0.8399628 * float(x[16]))+ (0.1516216 * float(x[17]))+ (0.016214779 * float(x[18]))+ (-0.13316113 * float(x[19]))+ (0.38403618 * float(x[20]))+ (-0.76525223 * float(x[21]))+ (2.6341448 * float(x[22]))+ (-0.8468362 * float(x[23]))+ (0.920462 * float(x[24]))+ (-1.575028 * float(x[25]))+ (0.095312536 * float(x[26]))+ (-0.20649014 * float(x[27]))+ (-1.11107 * float(x[28]))+ (1.2716846 * float(x[29]))+ (0.17889893 * float(x[30]))+ (1.0039049 * float(x[31]))+ (-0.1950419 * float(x[32]))+ (0.7335629 * float(x[33]))+ (1.207277 * float(x[34]))+ (0.6499841 * float(x[35]))+ (-0.77487826 * float(x[36]))+ (-0.8854319 * float(x[37]))+ (2.0363193 * float(x[38]))+ (0.4892061 * float(x[39]))+ (-1.0810527 * float(x[40]))+ (0.18529308 * float(x[41]))+ (0.09376861 * float(x[42]))+ (-1.5037227 * float(x[43]))+ (0.30067158 * float(x[44]))+ (0.054403044 * float(x[45]))+ (0.3252678 * float(x[46]))+ (1.6689562 * float(x[47]))+ (-1.4002987 * float(x[48]))+ (-0.15253112 * float(x[49]))) + -2.9753318), 0)
    o_0 = (0.6918435 * h_0)+ (1.0289356 * h_1)+ (-2.4384048 * h_2)+ (-2.7221317 * h_3) + -3.1956418
             
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

        model_cap=209

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

