#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-11-2020 20:25:00
# Invocation: btc -v ailerons-2.csv -o ailerons-2.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                57.61%
Model accuracy:                     88.04% (12106/13750 correct)
Improvement over best guess:        30.43% (of possible 42.39%)
Model capacity (MEC):               127 bits
Generalization ratio:               95.32 bits/bit
Model efficiency:                   0.23%/parameter
System behavior
True Negatives:                     52.17% (7174/13750)
True Positives:                     35.87% (4932/13750)
False Negatives:                    6.52% (896/13750)
False Positives:                    5.44% (748/13750)
True Pos. Rate/Sensitivity/Recall:  0.85
True Neg. Rate/Specificity:         0.91
Precision:                          0.87
F-1 Measure:                        0.86
False Negative Rate/Miss Rate:      0.15
Critical Success Index:             0.75
Model bias:                         0.39% higher chance to pick class 1
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
TRAINFILE="ailerons-2.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 40

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
    h_0 = max((((-49.753178 * float(x[0]))+ (28.025558 * float(x[1]))+ (-0.43977544 * float(x[2]))+ (-6.951976 * float(x[3]))+ (-9.628641 * float(x[4]))+ (-8.427624 * float(x[5]))+ (11.431514 * float(x[6]))+ (17.44399 * float(x[7]))+ (3.1937325 * float(x[8]))+ (-8.950385 * float(x[9]))+ (-6.3395495 * float(x[10]))+ (-6.2755275 * float(x[11]))+ (-5.560392 * float(x[12]))+ (-7.2788873 * float(x[13]))+ (-7.24583 * float(x[14]))+ (-7.3781834 * float(x[15]))+ (-5.753316 * float(x[16]))+ (-5.8629484 * float(x[17]))+ (-5.6792364 * float(x[18]))+ (-5.4734025 * float(x[19]))+ (-5.8324776 * float(x[20]))+ (-6.523835 * float(x[21]))+ (-5.8857164 * float(x[22]))+ (-7.219304 * float(x[23]))+ (1.4144952 * float(x[24]))+ (-1.2090471 * float(x[25]))+ (1.8902428 * float(x[26]))+ (0.02355304 * float(x[27]))+ (0.75705886 * float(x[28]))+ (0.055182654 * float(x[29]))+ (1.5193567 * float(x[30]))+ (0.4121847 * float(x[31]))+ (1.2458943 * float(x[32]))+ (-0.33105192 * float(x[33]))+ (1.412374 * float(x[34]))+ (0.8528937 * float(x[35]))+ (1.3172042 * float(x[36]))+ (1.5123618 * float(x[37]))+ (-9.232527 * float(x[38]))+ (-6.7367773 * float(x[39]))) + -9.946835), 0)
    h_1 = max((((15.488707 * float(x[0]))+ (3.7425647 * float(x[1]))+ (-1.9548649 * float(x[2]))+ (-8.704389 * float(x[3]))+ (-10.857045 * float(x[4]))+ (-2.0249848 * float(x[5]))+ (15.684796 * float(x[6]))+ (2.5182028 * float(x[7]))+ (-1.9159639 * float(x[8]))+ (8.875578 * float(x[9]))+ (-6.802044 * float(x[10]))+ (-6.2354746 * float(x[11]))+ (-7.077637 * float(x[12]))+ (-7.359243 * float(x[13]))+ (-6.8331947 * float(x[14]))+ (-7.1341434 * float(x[15]))+ (-7.612375 * float(x[16]))+ (-7.626416 * float(x[17]))+ (-5.9349957 * float(x[18]))+ (-7.6137238 * float(x[19]))+ (-5.7702646 * float(x[20]))+ (-6.182348 * float(x[21]))+ (-7.2642035 * float(x[22]))+ (-6.743738 * float(x[23]))+ (-1.0027673 * float(x[24]))+ (0.78240585 * float(x[25]))+ (-1.5315127 * float(x[26]))+ (1.2073804 * float(x[27]))+ (-0.57039285 * float(x[28]))+ (-0.27842578 * float(x[29]))+ (-0.9995836 * float(x[30]))+ (0.20986037 * float(x[31]))+ (-0.45950964 * float(x[32]))+ (-1.3765992 * float(x[33]))+ (-1.522367 * float(x[34]))+ (0.4944414 * float(x[35]))+ (-0.1755551 * float(x[36]))+ (0.24730085 * float(x[37]))+ (-12.314129 * float(x[38]))+ (-5.7129545 * float(x[39]))) + -12.061615), 0)
    h_2 = max((((0.0017264355 * float(x[0]))+ (-0.00025881626 * float(x[1]))+ (2.7932415 * float(x[2]))+ (-0.92315394 * float(x[3]))+ (1.4975961 * float(x[4]))+ (0.58376557 * float(x[5]))+ (-0.40896443 * float(x[6]))+ (0.0044434713 * float(x[7]))+ (-12.197025 * float(x[8]))+ (-0.074730486 * float(x[9]))+ (2.477225 * float(x[10]))+ (3.5148923 * float(x[11]))+ (2.9503884 * float(x[12]))+ (3.7872162 * float(x[13]))+ (3.0905511 * float(x[14]))+ (3.9244907 * float(x[15]))+ (2.5663025 * float(x[16]))+ (2.5451155 * float(x[17]))+ (2.8006716 * float(x[18]))+ (2.1389248 * float(x[19]))+ (3.194488 * float(x[20]))+ (3.3964915 * float(x[21]))+ (1.5388135 * float(x[22]))+ (2.1409872 * float(x[23]))+ (12.193052 * float(x[24]))+ (12.631878 * float(x[25]))+ (-14.475179 * float(x[26]))+ (19.296896 * float(x[27]))+ (19.557589 * float(x[28]))+ (-15.880605 * float(x[29]))+ (28.619534 * float(x[30]))+ (3.304849 * float(x[31]))+ (7.1974216 * float(x[32]))+ (-10.069243 * float(x[33]))+ (7.110037 * float(x[34]))+ (16.0151 * float(x[35]))+ (20.48316 * float(x[36]))+ (-23.410057 * float(x[37]))+ (1.243647 * float(x[38]))+ (2.714175 * float(x[39]))) + -4.3929663), 0)
    o_0 = (-3.7293858e-06 * h_0)+ (-3.764843e-05 * h_1)+ (1.7537535 * h_2) + -5.2275653
             
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

        model_cap=127

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

