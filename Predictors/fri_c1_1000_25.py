#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Feb-28-2020 17:45:39
# Invocation: btc Data/fri_c1_1000_25.csv -o Models/fri_c1_1000_25.py -v -v -v -stopat 90.90 -port 8090 -e 9
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                54.60%
Model accuracy:                     91.20% (912/1000 correct)
Improvement over best guess:        36.60% (of possible 45.4%)
Model capacity (MEC):               136 bits
Generalization ratio:               6.70 bits/bit
Model efficiency:                   0.26%/parameter
System behavior
True Negatives:                     51.90% (519/1000)
True Positives:                     39.30% (393/1000)
False Negatives:                    6.10% (61/1000)
False Positives:                    2.70% (27/1000)
True Pos. Rate/Sensitivity/Recall:  0.87
True Neg. Rate/Specificity:         0.95
Precision:                          0.94
F-1 Measure:                        0.90
False Negative Rate/Miss Rate:      0.13
Critical Success Index:             0.82
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


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="fri_c1_1000_25.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 25

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
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mapped to 0 and 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(result)

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mappable to 0 and 1.")
        finally:
            if (result<0 or result>1):
                raise ValueError("Alpha version restriction: Integer class labels can only be 0 or 1.")
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
    h_0 = max((((13.961276 * float(x[0]))+ (16.439154 * float(x[1]))+ (0.6667891 * float(x[2]))+ (-2.2299004 * float(x[3]))+ (-5.1778336 * float(x[4]))+ (-2.4077528 * float(x[5]))+ (-1.9031345 * float(x[6]))+ (1.2029068 * float(x[7]))+ (-0.55668885 * float(x[8]))+ (-0.53191257 * float(x[9]))+ (-2.6590846 * float(x[10]))+ (1.3396621 * float(x[11]))+ (4.021308 * float(x[12]))+ (-0.9614548 * float(x[13]))+ (-4.1174183 * float(x[14]))+ (0.23625414 * float(x[15]))+ (-3.83114 * float(x[16]))+ (-2.7410376 * float(x[17]))+ (2.1649086 * float(x[18]))+ (-1.0090101 * float(x[19]))+ (-0.7952016 * float(x[20]))+ (0.9650505 * float(x[21]))+ (1.2155364 * float(x[22]))+ (-0.38392124 * float(x[23]))+ (0.31520203 * float(x[24]))) + -1.6233498), 0)
    h_1 = max((((1.2101684 * float(x[0]))+ (2.845428 * float(x[1]))+ (-3.835337 * float(x[2]))+ (-3.8064919 * float(x[3]))+ (-2.901878 * float(x[4]))+ (-0.018410174 * float(x[5]))+ (1.7393419 * float(x[6]))+ (-2.4331706 * float(x[7]))+ (1.3668545 * float(x[8]))+ (-3.4473016 * float(x[9]))+ (-6.5648723 * float(x[10]))+ (-1.8367547 * float(x[11]))+ (-3.2481084 * float(x[12]))+ (0.07000772 * float(x[13]))+ (0.95634675 * float(x[14]))+ (-1.5481317 * float(x[15]))+ (-5.7587805 * float(x[16]))+ (1.8632295 * float(x[17]))+ (-5.377724 * float(x[18]))+ (-1.8327594 * float(x[19]))+ (2.5648725 * float(x[20]))+ (2.0851347 * float(x[21]))+ (-2.648424 * float(x[22]))+ (5.9502015 * float(x[23]))+ (-0.543259 * float(x[24]))) + -2.2981064), 0)
    h_2 = max((((5.486807 * float(x[0]))+ (2.2390256 * float(x[1]))+ (1.3443152 * float(x[2]))+ (1.5919584 * float(x[3]))+ (-1.4974047 * float(x[4]))+ (-1.8948876 * float(x[5]))+ (-2.5021746 * float(x[6]))+ (1.9587157 * float(x[7]))+ (0.8642875 * float(x[8]))+ (-1.5934839 * float(x[9]))+ (-1.2245244 * float(x[10]))+ (0.16716504 * float(x[11]))+ (2.5709295 * float(x[12]))+ (-0.77740705 * float(x[13]))+ (-1.055121 * float(x[14]))+ (0.56376445 * float(x[15]))+ (-3.0412102 * float(x[16]))+ (-0.28527564 * float(x[17]))+ (2.2568853 * float(x[18]))+ (-0.42208073 * float(x[19]))+ (-0.7562377 * float(x[20]))+ (0.74424505 * float(x[21]))+ (1.3476964 * float(x[22]))+ (-1.0209455 * float(x[23]))+ (0.04134449 * float(x[24]))) + -1.3110342), 0)
    h_3 = max((((-0.3529334 * float(x[0]))+ (-2.9695582 * float(x[1]))+ (0.07619308 * float(x[2]))+ (-1.1468333 * float(x[3]))+ (-0.65841436 * float(x[4]))+ (0.012948475 * float(x[5]))+ (-0.27327833 * float(x[6]))+ (0.4155945 * float(x[7]))+ (-0.23402926 * float(x[8]))+ (0.28433952 * float(x[9]))+ (0.16360107 * float(x[10]))+ (0.04885191 * float(x[11]))+ (-0.075521804 * float(x[12]))+ (0.06187956 * float(x[13]))+ (-0.15502003 * float(x[14]))+ (0.04609195 * float(x[15]))+ (-0.42825988 * float(x[16]))+ (-0.13821161 * float(x[17]))+ (-0.18263516 * float(x[18]))+ (-0.32090122 * float(x[19]))+ (-0.044598978 * float(x[20]))+ (0.15680104 * float(x[21]))+ (-0.098724 * float(x[22]))+ (-0.345739 * float(x[23]))+ (0.2069055 * float(x[24]))) + -4.4264393), 0)
    h_4 = max((((4.164854 * float(x[0]))+ (4.084371 * float(x[1]))+ (-0.5962068 * float(x[2]))+ (-0.90242517 * float(x[3]))+ (-0.5036156 * float(x[4]))+ (0.07208523 * float(x[5]))+ (0.4701857 * float(x[6]))+ (-0.5545408 * float(x[7]))+ (-0.504188 * float(x[8]))+ (0.4059723 * float(x[9]))+ (-0.14714547 * float(x[10]))+ (0.32858458 * float(x[11]))+ (0.05578417 * float(x[12]))+ (0.15210299 * float(x[13]))+ (-0.7241123 * float(x[14]))+ (0.00088853907 * float(x[15]))+ (0.07810517 * float(x[16]))+ (-0.35732353 * float(x[17]))+ (-0.34449992 * float(x[18]))+ (-0.16334267 * float(x[19]))+ (0.113811456 * float(x[20]))+ (-0.14406705 * float(x[21]))+ (-0.101629 * float(x[22]))+ (0.46392605 * float(x[23]))+ (-0.044872425 * float(x[24]))) + -5.4845195), 0)
    o_0 = (2.0512507 * h_0)+ (0.16135351 * h_1)+ (-3.3165047 * h_2)+ (10.678124 * h_3)+ (-8.606493 * h_4) + -3.6371121
             
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
    faulthandler.enable()
    
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

        model_cap=136

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

