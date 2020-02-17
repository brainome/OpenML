#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 20:05:34
# Invocation: btc -v GAMETES-Epistasis-2-Way-20atts-0-8.csv -o GAMETES-Epistasis-2-Way-20atts-0-8.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     81.18% (1299/1600 correct)
Improvement over best guess:        31.18% (of possible 50.0%)
Model capacity (MEC):               133 bits
Generalization ratio:               9.76 bits/bit
Model efficiency:                   0.23%/parameter
System behavior
True Negatives:                     43.50% (696/1600)
True Positives:                     37.69% (603/1600)
False Negatives:                    12.31% (197/1600)
False Positives:                    6.50% (104/1600)
True Pos. Rate/Sensitivity/Recall:  0.75
True Neg. Rate/Specificity:         0.87
Precision:                          0.85
F-1 Measure:                        0.80
False Negative Rate/Miss Rate:      0.25
Critical Success Index:             0.67
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
TRAINFILE="GAMETES-Epistasis-2-Way-20atts-0-8.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 20

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
    h_0 = max((((-12.961178 * float(x[0]))+ (4.2846184 * float(x[1]))+ (-0.3942944 * float(x[2]))+ (-0.50948715 * float(x[3]))+ (-4.9714537 * float(x[4]))+ (-1.3916112 * float(x[5]))+ (2.9840539 * float(x[6]))+ (-0.4165755 * float(x[7]))+ (0.27734712 * float(x[8]))+ (-0.93703634 * float(x[9]))+ (-7.7635756 * float(x[10]))+ (-1.6001923 * float(x[11]))+ (1.5189028 * float(x[12]))+ (6.0091963 * float(x[13]))+ (-1.0378987 * float(x[14]))+ (0.195921 * float(x[15]))+ (-2.3620024 * float(x[16]))+ (1.3039249 * float(x[17]))+ (14.420905 * float(x[18]))+ (-13.671978 * float(x[19]))) + -13.437748), 0)
    h_1 = max((((-9.6958065 * float(x[0]))+ (3.6377585 * float(x[1]))+ (-19.228558 * float(x[2]))+ (-10.172835 * float(x[3]))+ (7.7631807 * float(x[4]))+ (4.096089 * float(x[5]))+ (3.7723687 * float(x[6]))+ (-5.555501 * float(x[7]))+ (-2.0565681 * float(x[8]))+ (-1.916117 * float(x[9]))+ (10.849383 * float(x[10]))+ (-1.7128363 * float(x[11]))+ (-2.100155 * float(x[12]))+ (-0.029743822 * float(x[13]))+ (-2.841985 * float(x[14]))+ (-0.8789125 * float(x[15]))+ (-10.278059 * float(x[16]))+ (-1.3573859 * float(x[17]))+ (-10.762082 * float(x[18]))+ (4.172532 * float(x[19]))) + -1.5453408), 0)
    h_2 = max((((-5.6609707 * float(x[0]))+ (2.6082647 * float(x[1]))+ (-0.2922284 * float(x[2]))+ (-0.067895 * float(x[3]))+ (-1.9037861 * float(x[4]))+ (-0.45618263 * float(x[5]))+ (0.21454489 * float(x[6]))+ (-0.8599425 * float(x[7]))+ (0.8277171 * float(x[8]))+ (-0.4369876 * float(x[9]))+ (-14.817996 * float(x[10]))+ (-0.7251418 * float(x[11]))+ (1.8680452 * float(x[12]))+ (3.4618337 * float(x[13]))+ (0.60813564 * float(x[14]))+ (-1.2040828 * float(x[15]))+ (-0.6109987 * float(x[16]))+ (1.4028167 * float(x[17]))+ (-5.353083 * float(x[18]))+ (7.5831532 * float(x[19]))) + 1.4814743), 0)
    h_3 = max((((1.3209136 * float(x[0]))+ (2.969896 * float(x[1]))+ (0.095008016 * float(x[2]))+ (1.6604354 * float(x[3]))+ (-3.392497 * float(x[4]))+ (-1.6783364 * float(x[5]))+ (-0.3033805 * float(x[6]))+ (-0.612246 * float(x[7]))+ (0.91546804 * float(x[8]))+ (0.07789552 * float(x[9]))+ (-1.2512617 * float(x[10]))+ (-2.33028 * float(x[11]))+ (-0.8152693 * float(x[12]))+ (0.5134215 * float(x[13]))+ (1.1490326 * float(x[14]))+ (-0.18941739 * float(x[15]))+ (-1.1170712 * float(x[16]))+ (-0.20590843 * float(x[17]))+ (-1.1345961 * float(x[18]))+ (-0.43628347 * float(x[19]))) + -1.9279395), 0)
    h_4 = max((((1.1432066 * float(x[0]))+ (-0.02148755 * float(x[1]))+ (0.04509419 * float(x[2]))+ (-0.009171479 * float(x[3]))+ (0.120249465 * float(x[4]))+ (0.2783845 * float(x[5]))+ (0.009792284 * float(x[6]))+ (0.11230817 * float(x[7]))+ (0.015724175 * float(x[8]))+ (0.030411182 * float(x[9]))+ (0.057185717 * float(x[10]))+ (0.049148783 * float(x[11]))+ (-0.3859785 * float(x[12]))+ (-0.5793617 * float(x[13]))+ (-0.29748166 * float(x[14]))+ (0.337647 * float(x[15]))+ (0.09210192 * float(x[16]))+ (-0.32538474 * float(x[17]))+ (4.502453 * float(x[18]))+ (-3.410295 * float(x[19]))) + 2.0135517), 0)
    h_5 = max((((0.53734493 * float(x[0]))+ (0.2968206 * float(x[1]))+ (-0.09233513 * float(x[2]))+ (-0.0356068 * float(x[3]))+ (-0.07262253 * float(x[4]))+ (0.13455889 * float(x[5]))+ (0.08940059 * float(x[6]))+ (0.058412142 * float(x[7]))+ (0.10578198 * float(x[8]))+ (-0.014523223 * float(x[9]))+ (-0.943013 * float(x[10]))+ (0.039823785 * float(x[11]))+ (-0.107326016 * float(x[12]))+ (-0.2423962 * float(x[13]))+ (-0.28526112 * float(x[14]))+ (0.26528358 * float(x[15]))+ (-0.0008152996 * float(x[16]))+ (-0.23444225 * float(x[17]))+ (4.321616 * float(x[18]))+ (-4.0164356 * float(x[19]))) + -0.05872076), 0)
    o_0 = (-1.0351359 * h_0)+ (17.841541 * h_1)+ (-0.48593086 * h_2)+ (36.19094 * h_3)+ (-3.4232545 * h_4)+ (5.2226524 * h_5) + 6.6150546
             
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

        model_cap=133

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

