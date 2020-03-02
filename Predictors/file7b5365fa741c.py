#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Mar-02-2020 11:23:34
# Invocation: btc Data/file7b5365fa741c.csv -o Models/file7b5365fa741c.py -v -v -v -stopat 100 -port 8090 -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                97.00%
Model accuracy:                     99.95% (4145/4147 correct)
Improvement over best guess:        2.95% (of possible 3.0%)
Model capacity (MEC):               101 bits
Generalization ratio:               41.03 bits/bit
Model efficiency:                   0.02%/parameter
System behavior
True Negatives:                     97.01% (4023/4147)
True Positives:                     2.94% (122/4147)
False Negatives:                    0.05% (2/4147)
False Positives:                    0.00% (0/4147)
True Pos. Rate/Sensitivity/Recall:  0.98
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        0.99
False Negative Rate/Miss Rate:      0.02
Critical Success Index:             0.98
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
TRAINFILE="file7b5365fa741c.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 48

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
    h_0 = max((((0.43037874 * float(x[0]))+ (0.20552675 * float(x[1]))+ (0.08976637 * float(x[2]))+ (-0.1526904 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.124825574 * float(x[5]))+ (0.78354603 * float(x[6]))+ (0.92732555 * float(x[7]))+ (-0.23311697 * float(x[8]))+ (0.5834501 * float(x[9]))+ (0.05778984 * float(x[10]))+ (0.13608912 * float(x[11]))+ (0.85119325 * float(x[12]))+ (-0.85792786 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (-0.9595632 * float(x[15]))+ (0.6652397 * float(x[16]))+ (0.5563135 * float(x[17]))+ (0.74002427 * float(x[18]))+ (0.9572367 * float(x[19]))+ (0.59831715 * float(x[20]))+ (-0.077041276 * float(x[21]))+ (0.56105834 * float(x[22]))+ (-0.76345116 * float(x[23]))+ (0.27984205 * float(x[24]))+ (-0.71329343 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-0.17067613 * float(x[28]))+ (-0.47088876 * float(x[29]))+ (0.5484674 * float(x[30]))+ (-0.08769934 * float(x[31]))+ (0.1368679 * float(x[32]))+ (-0.9624204 * float(x[33]))+ (0.23527099 * float(x[34]))+ (0.22419144 * float(x[35]))+ (0.23386799 * float(x[36]))+ (0.8874962 * float(x[37]))+ (0.3636406 * float(x[38]))+ (-0.2809842 * float(x[39]))+ (-0.12593609 * float(x[40]))+ (0.3952624 * float(x[41]))+ (-0.8795491 * float(x[42]))+ (0.33353344 * float(x[43]))+ (0.34127575 * float(x[44]))+ (-0.5792349 * float(x[45]))+ (-0.7421474 * float(x[46]))+ (-0.3691433 * float(x[47]))) + 0.09762701), 0)
    h_1 = max((((-1.2766416 * float(x[0]))+ (7.151253 * float(x[1]))+ (1.6468716 * float(x[2]))+ (1.1872027 * float(x[3]))+ (0.00010404019 * float(x[4]))+ (-4.00249 * float(x[5]))+ (10.062829 * float(x[6]))+ (-0.33689252 * float(x[7]))+ (1.213254 * float(x[8]))+ (0.5040696 * float(x[9]))+ (0.0006204225 * float(x[10]))+ (2.8708127 * float(x[11]))+ (2.5326574 * float(x[12]))+ (1.2363575 * float(x[13]))+ (-0.04729414 * float(x[14]))+ (0.0020586206 * float(x[15]))+ (1.7874929 * float(x[16]))+ (0.5210574 * float(x[17]))+ (6.75222 * float(x[18]))+ (1.8210881 * float(x[19]))+ (-2.766535 * float(x[20]))+ (0.6382733 * float(x[21]))+ (0.47834006 * float(x[22]))+ (1.9322327 * float(x[23]))+ (1.9495969 * float(x[24]))+ (-0.0074737654 * float(x[25]))+ (1.4334636 * float(x[26]))+ (8.357424 * float(x[27]))+ (1.842268 * float(x[28]))+ (-0.76924205 * float(x[29]))+ (9.62379 * float(x[30]))+ (6.100806 * float(x[31]))+ (-0.0016395594 * float(x[32]))+ (1.2012069 * float(x[33]))+ (-0.046690837 * float(x[34]))+ (1.9674354 * float(x[35]))+ (2.1743088 * float(x[36]))+ (9.532004 * float(x[37]))+ (6.9921584 * float(x[38]))+ (2.2955968 * float(x[39]))+ (6.796374e-05 * float(x[40]))+ (-3.2051604 * float(x[41]))+ (0.42043906 * float(x[42]))+ (2.8717022 * float(x[43]))+ (7.636509 * float(x[44]))+ (1.40229 * float(x[45]))+ (0.33883995 * float(x[46]))+ (2.484216 * float(x[47]))) + 1.7744597), 0)
    o_0 = (-0.41971418 * h_0)+ (-0.7630783 * h_1) + 2.2355666
             
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

        model_cap=101

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

