#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 21:20:53
# Invocation: btc -v fri-c0-1000-50-2.csv -o fri-c0-1000-50-2.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                51.00%
Model accuracy:                     87.60% (876/1000 correct)
Improvement over best guess:        36.60% (of possible 49.0%)
Model capacity (MEC):               261 bits
Generalization ratio:               3.35 bits/bit
Model efficiency:                   0.14%/parameter
System behavior
True Negatives:                     44.80% (448/1000)
True Positives:                     42.80% (428/1000)
False Negatives:                    8.20% (82/1000)
False Positives:                    4.20% (42/1000)
True Pos. Rate/Sensitivity/Recall:  0.84
True Neg. Rate/Specificity:         0.91
Precision:                          0.91
F-1 Measure:                        0.87
False Negative Rate/Miss Rate:      0.16
Critical Success Index:             0.78
Model bias:                         0.08% higher chance to pick class 0
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
TRAINFILE="fri-c0-1000-50-2.csv"


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
    h_0 = max((((-14.704047 * float(x[0]))+ (-17.063549 * float(x[1]))+ (0.5811973 * float(x[2]))+ (-14.861977 * float(x[3]))+ (-6.7004213 * float(x[4]))+ (4.9528623 * float(x[5]))+ (1.2858659 * float(x[6]))+ (-4.127553 * float(x[7]))+ (3.0583901 * float(x[8]))+ (-2.7580929 * float(x[9]))+ (-0.75155705 * float(x[10]))+ (2.9507923 * float(x[11]))+ (0.2516685 * float(x[12]))+ (4.1519465 * float(x[13]))+ (-1.5724759 * float(x[14]))+ (2.5478442 * float(x[15]))+ (-2.1817486 * float(x[16]))+ (-5.4657927 * float(x[17]))+ (2.783593 * float(x[18]))+ (0.8290835 * float(x[19]))+ (0.97508955 * float(x[20]))+ (-0.7485593 * float(x[21]))+ (-1.593737 * float(x[22]))+ (1.5891067 * float(x[23]))+ (1.4736632 * float(x[24]))+ (0.768788 * float(x[25]))+ (-0.809025 * float(x[26]))+ (0.63095695 * float(x[27]))+ (-0.2635968 * float(x[28]))+ (-3.7235718 * float(x[29]))+ (-2.175235 * float(x[30]))+ (-2.8952074 * float(x[31]))+ (1.3851048 * float(x[32]))+ (-0.4858132 * float(x[33]))+ (-0.46309552 * float(x[34]))+ (-3.5379071 * float(x[35]))+ (-0.94757265 * float(x[36]))+ (-3.97417 * float(x[37]))+ (2.7601924 * float(x[38]))+ (-3.3238099 * float(x[39]))+ (-0.43894312 * float(x[40]))+ (-1.3281348 * float(x[41]))+ (-1.670275 * float(x[42]))+ (2.148776 * float(x[43]))+ (-1.7335427 * float(x[44]))+ (1.8041401 * float(x[45]))+ (-1.8036143 * float(x[46]))+ (-4.075812 * float(x[47]))+ (0.1134478 * float(x[48]))+ (-4.9806857 * float(x[49]))) + 4.1976514), 0)
    h_1 = max((((1.1350342 * float(x[0]))+ (-5.211216 * float(x[1]))+ (1.3961707 * float(x[2]))+ (-4.7919226 * float(x[3]))+ (-1.1774564 * float(x[4]))+ (-0.105869725 * float(x[5]))+ (-1.210886 * float(x[6]))+ (4.616806 * float(x[7]))+ (-3.330838 * float(x[8]))+ (2.6352055 * float(x[9]))+ (0.3204872 * float(x[10]))+ (-0.48122215 * float(x[11]))+ (2.3046706 * float(x[12]))+ (-8.616646 * float(x[13]))+ (0.784064 * float(x[14]))+ (-3.7921011 * float(x[15]))+ (-1.5944285 * float(x[16]))+ (5.826689 * float(x[17]))+ (2.3151736 * float(x[18]))+ (-4.346155 * float(x[19]))+ (0.14188762 * float(x[20]))+ (1.7911717 * float(x[21]))+ (-3.0611367 * float(x[22]))+ (0.5952679 * float(x[23]))+ (-0.19891138 * float(x[24]))+ (2.5550275 * float(x[25]))+ (-1.2921922 * float(x[26]))+ (0.41631272 * float(x[27]))+ (-2.9510396 * float(x[28]))+ (3.2862976 * float(x[29]))+ (-4.386745 * float(x[30]))+ (0.41534036 * float(x[31]))+ (-1.6335505 * float(x[32]))+ (4.502106 * float(x[33]))+ (-3.0483816 * float(x[34]))+ (3.0559735 * float(x[35]))+ (-3.0257695 * float(x[36]))+ (3.2401705 * float(x[37]))+ (-2.887686 * float(x[38]))+ (-1.3204864 * float(x[39]))+ (-2.602059 * float(x[40]))+ (1.0887815 * float(x[41]))+ (2.5099883 * float(x[42]))+ (1.6133875 * float(x[43]))+ (0.57417935 * float(x[44]))+ (1.7660882 * float(x[45]))+ (2.5344849 * float(x[46]))+ (-3.3511724 * float(x[47]))+ (0.3091479 * float(x[48]))+ (-1.2069336 * float(x[49]))) + -4.0006013), 0)
    h_2 = max((((-1.6560206 * float(x[0]))+ (-0.7678217 * float(x[1]))+ (-1.057445 * float(x[2]))+ (-1.809737 * float(x[3]))+ (0.9367521 * float(x[4]))+ (1.2336909 * float(x[5]))+ (2.7820926 * float(x[6]))+ (-0.6850854 * float(x[7]))+ (0.039669424 * float(x[8]))+ (-0.54995817 * float(x[9]))+ (-1.9318771 * float(x[10]))+ (1.3606148 * float(x[11]))+ (1.000823 * float(x[12]))+ (-0.39237192 * float(x[13]))+ (1.0014415 * float(x[14]))+ (1.8591113 * float(x[15]))+ (-2.3546462 * float(x[16]))+ (-1.872162 * float(x[17]))+ (2.1221776 * float(x[18]))+ (1.506628 * float(x[19]))+ (1.5593716 * float(x[20]))+ (0.8917619 * float(x[21]))+ (-1.8399763 * float(x[22]))+ (2.5091746 * float(x[23]))+ (0.46330878 * float(x[24]))+ (-1.0507827 * float(x[25]))+ (-0.5905841 * float(x[26]))+ (-0.12714091 * float(x[27]))+ (0.17459263 * float(x[28]))+ (-2.6541111 * float(x[29]))+ (0.0015172617 * float(x[30]))+ (1.2990408 * float(x[31]))+ (1.3141377 * float(x[32]))+ (1.4335431 * float(x[33]))+ (-0.43736 * float(x[34]))+ (0.06486444 * float(x[35]))+ (2.185131 * float(x[36]))+ (2.2805288 * float(x[37]))+ (-0.12723196 * float(x[38]))+ (-0.7132703 * float(x[39]))+ (0.479248 * float(x[40]))+ (0.6621785 * float(x[41]))+ (-0.61655796 * float(x[42]))+ (2.8788855 * float(x[43]))+ (0.9448494 * float(x[44]))+ (1.2266204 * float(x[45]))+ (1.1105914 * float(x[46]))+ (-3.641982 * float(x[47]))+ (-1.4533399 * float(x[48]))+ (-0.4037788 * float(x[49]))) + 0.12819245), 0)
    h_3 = max((((-0.48632875 * float(x[0]))+ (0.0024197167 * float(x[1]))+ (0.2444517 * float(x[2]))+ (0.46689504 * float(x[3]))+ (0.27123758 * float(x[4]))+ (0.8134332 * float(x[5]))+ (-2.9794357 * float(x[6]))+ (0.45022005 * float(x[7]))+ (1.2338965 * float(x[8]))+ (-0.6762505 * float(x[9]))+ (-0.8614166 * float(x[10]))+ (1.1631459 * float(x[11]))+ (0.8329679 * float(x[12]))+ (-0.39767137 * float(x[13]))+ (0.681964 * float(x[14]))+ (-0.004398255 * float(x[15]))+ (0.20513864 * float(x[16]))+ (-0.4120825 * float(x[17]))+ (0.84609085 * float(x[18]))+ (-1.1040666 * float(x[19]))+ (-2.376104 * float(x[20]))+ (1.1925739 * float(x[21]))+ (0.33629215 * float(x[22]))+ (2.9473517 * float(x[23]))+ (-1.4771843 * float(x[24]))+ (-0.93662214 * float(x[25]))+ (-0.35626575 * float(x[26]))+ (0.5633833 * float(x[27]))+ (-2.7561023 * float(x[28]))+ (2.5682755 * float(x[29]))+ (-2.3051188 * float(x[30]))+ (1.9992524 * float(x[31]))+ (1.7531029 * float(x[32]))+ (0.3272919 * float(x[33]))+ (-0.7235859 * float(x[34]))+ (-1.2438247 * float(x[35]))+ (0.2527063 * float(x[36]))+ (0.2667378 * float(x[37]))+ (-0.68247014 * float(x[38]))+ (-0.26653403 * float(x[39]))+ (-0.032700084 * float(x[40]))+ (0.3095816 * float(x[41]))+ (0.65541977 * float(x[42]))+ (-0.4749504 * float(x[43]))+ (-0.63447374 * float(x[44]))+ (-1.2967526 * float(x[45]))+ (0.8600763 * float(x[46]))+ (-1.5897979 * float(x[47]))+ (-0.15594779 * float(x[48]))+ (-1.0900111 * float(x[49]))) + -1.878835), 0)
    h_4 = max((((1.6959705 * float(x[0]))+ (-2.466809 * float(x[1]))+ (1.5687249 * float(x[2]))+ (0.8015758 * float(x[3]))+ (-0.39997414 * float(x[4]))+ (2.5985627 * float(x[5]))+ (0.7431882 * float(x[6]))+ (-1.2712735 * float(x[7]))+ (-0.3133365 * float(x[8]))+ (-2.3102698 * float(x[9]))+ (0.6996774 * float(x[10]))+ (1.9238803 * float(x[11]))+ (1.6160692 * float(x[12]))+ (-0.6285925 * float(x[13]))+ (-0.8236074 * float(x[14]))+ (-1.0985917 * float(x[15]))+ (0.018410051 * float(x[16]))+ (0.9976614 * float(x[17]))+ (-1.0292166 * float(x[18]))+ (-0.9706752 * float(x[19]))+ (0.92666155 * float(x[20]))+ (-3.6064944 * float(x[21]))+ (-1.1933918 * float(x[22]))+ (-3.2218573 * float(x[23]))+ (0.87766004 * float(x[24]))+ (2.0896895 * float(x[25]))+ (-0.9468936 * float(x[26]))+ (0.42043844 * float(x[27]))+ (0.6423225 * float(x[28]))+ (-0.46113062 * float(x[29]))+ (0.91924435 * float(x[30]))+ (-1.0522742 * float(x[31]))+ (-2.557537 * float(x[32]))+ (0.68757015 * float(x[33]))+ (0.45994338 * float(x[34]))+ (0.29380065 * float(x[35]))+ (-0.98709184 * float(x[36]))+ (-2.4832735 * float(x[37]))+ (1.2188058 * float(x[38]))+ (-3.1151922 * float(x[39]))+ (-2.5977468 * float(x[40]))+ (-1.3476076 * float(x[41]))+ (-0.07139706 * float(x[42]))+ (0.04796263 * float(x[43]))+ (0.50936323 * float(x[44]))+ (-0.99597543 * float(x[45]))+ (-2.7248747 * float(x[46]))+ (0.7937724 * float(x[47]))+ (3.6400096 * float(x[48]))+ (-3.1286156 * float(x[49]))) + -0.35800177), 0)
    o_0 = (2.0271394 * h_0)+ (1.5404849 * h_1)+ (-2.6478696 * h_2)+ (-4.0111523 * h_3)+ (-3.2130592 * h_4) + -0.16790348
             
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

        model_cap=261

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

