#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 00:53:13
# Invocation: btc -server brain.brainome.ai Data/letter.csv -o Models/letter.py -v -v -v -stopat 99.86 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                95.93%
Model accuracy:                     97.81% (19563/20000 correct)
Improvement over best guess:        1.88% (of possible 4.07%)
Model capacity (MEC):               37 bits
Generalization ratio:               528.72 bits/bit
Model efficiency:                   0.05%/parameter
System behavior
True Negatives:                     95.57% (19114/20000)
True Positives:                     2.25% (449/20000)
False Negatives:                    1.82% (364/20000)
False Positives:                    0.36% (73/20000)
True Pos. Rate/Sensitivity/Recall:  0.55
True Neg. Rate/Specificity:         1.00
Precision:                          0.86
F-1 Measure:                        0.67
False Negative Rate/Miss Rate:      0.45
Critical Success Index:             0.51

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
TRAINFILE="letter.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 16
n_classes = 2


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
            if (not result==int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(int(result*100)/100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not result==int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result<0):
                raise ValueError("Integer class labels must be positive and contiguous.")

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
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf=[]
        print(''.join(outbuf),end="", file=f)
        f.close()

        if (testfile==False and not len(clean.classlist)>=2):
            raise ValueError("Number of classes must be at least 2.")



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    o=[0]*num_output_logits
    h_0 = max((((-0.82599854 * float(x[0]))+ (-1.0508505 * float(x[1]))+ (-1.7947996 * float(x[2]))+ (-1.0949733 * float(x[3]))+ (-0.6504947 * float(x[4]))+ (-2.9516745 * float(x[5]))+ (-1.4151142 * float(x[6]))+ (-0.014957427 * float(x[7]))+ (-2.117683 * float(x[8]))+ (-2.557493 * float(x[9]))+ (-0.8844931 * float(x[10]))+ (-2.062571 * float(x[11]))+ (-0.09108967 * float(x[12]))+ (-3.0565882 * float(x[13]))+ (-1.7680243 * float(x[14]))+ (-3.786412 * float(x[15]))) + -0.2164673), 0)
    h_1 = max((((-0.47930768 * float(x[0]))+ (-0.17741828 * float(x[1]))+ (0.60632366 * float(x[2]))+ (0.82455033 * float(x[3]))+ (-0.8156877 * float(x[4]))+ (-0.0066552865 * float(x[5]))+ (0.45364517 * float(x[6]))+ (-0.33854505 * float(x[7]))+ (-0.85604876 * float(x[8]))+ (0.28181446 * float(x[9]))+ (-0.6602931 * float(x[10]))+ (-0.24125835 * float(x[11]))+ (-0.1877326 * float(x[12]))+ (-0.15901342 * float(x[13]))+ (0.7169279 * float(x[14]))+ (0.5777312 * float(x[15]))) + 8.897655), 0)
    o[0] = (-0.08495252 * h_0)+ (-1.2445985 * h_1) + 4.144024

    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)

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
        if n_classes==2:
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
        else:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanvalfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanvalfile, -1, args.headerless)
            with open(cleanvalfile,'r') as valcsvfile:
                count,correct_count=0,0
                valcsvreader = csv.reader(valcsvfile)
                numeachclass={}
                for i,valrow in enumerate(valcsvreader):
                    if len(valrow)==0:
                        continue
                    if int(classify(valrow[:-1]))==int(float(valrow[-1])):
                        correct_count+=1
                    if int(float(valrow[-1])) in numeachclass.keys():
                        numeachclass[int(float(valrow[-1]))]+=1
                    else:
                        numeachclass[int(float(valrow[-1]))]=0
                    count+=1

        model_cap=37

        if n_classes==2:

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
        else:
            num_correct=correct_count
            modelacc=int(float(num_correct*10000)/count)/100.0
            randguess=round(max(numeachclass.values())/sum(numeachclass.values())*100,2)
            print("System Type:                        "+str(n_classes)+"-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc)+" ("+str(int(num_correct))+"/"+str(count)+" correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc-randguess)+" (of possible "+str(round(100-randguess,2))+"%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct*100)/model_cap)/100.0)+" bits/bit")






        os.remove(cleanvalfile)

