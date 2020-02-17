#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:24:55
# Invocation: btc -v -v phpSNaed2-1.csv -o phpSNaed2-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                70.75%
Model accuracy:                     83.03% (230/277 correct)
Improvement over best guess:        12.28% (of possible 29.25%)
Model capacity (MEC):               65 bits
Generalization ratio:               3.53 bits/bit
Model efficiency:                   0.18%/parameter
System behavior
True Negatives:                     19.86% (55/277)
True Positives:                     63.18% (175/277)
False Negatives:                    7.58% (21/277)
False Positives:                    9.39% (26/277)
True Pos. Rate/Sensitivity/Recall:  0.89
True Neg. Rate/Specificity:         0.68
Precision:                          0.87
F-1 Measure:                        0.88
False Negative Rate/Miss Rate:      0.11
Critical Success Index:             0.79
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
TRAINFILE="phpSNaed2-1.csv"


#Number of attributes
num_attr = 9

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
    if (energy>21633748140.0):
        return 0.0
    if (energy>20907199528.0):
        return 1.0
    if (energy>20515676530.5):
        return 0.0
    if (energy>20201773825.0):
        return 1.0
    if (energy>20029808740.5):
        return 0.0
    if (energy>19893806722.5):
        return 1.0
    if (energy>19699102993.5):
        return 0.0
    if (energy>19633114168.5):
        return 1.0
    if (energy>19480041927.5):
        return 0.0
    if (energy>19015949158.5):
        return 1.0
    if (energy>18941410403.0):
        return 0.0
    if (energy>18864800881.5):
        return 1.0
    if (energy>18808272469.5):
        return 0.0
    if (energy>18692333544.5):
        return 1.0
    if (energy>18663483149.5):
        return 0.0
    if (energy>18637130168.0):
        return 1.0
    if (energy>18630148078.0):
        return 0.0
    if (energy>18551644720.0):
        return 1.0
    if (energy>18524570049.5):
        return 0.0
    if (energy>18473477601.0):
        return 1.0
    if (energy>18442760256.5):
        return 0.0
    if (energy>18407433583.5):
        return 1.0
    if (energy>18374673813.0):
        return 0.0
    if (energy>18211772645.5):
        return 1.0
    if (energy>18179111412.0):
        return 0.0
    if (energy>18169422904.0):
        return 1.0
    if (energy>18143915776.0):
        return 0.0
    if (energy>17983594286.5):
        return 1.0
    if (energy>17970074916.0):
        return 0.0
    if (energy>17776437198.0):
        return 1.0
    if (energy>17758650445.5):
        return 0.0
    if (energy>17468990104.5):
        return 1.0
    if (energy>17451452121.0):
        return 0.0
    if (energy>17237652330.0):
        return 1.0
    if (energy>17194871104.0):
        return 0.0
    if (energy>17136302617.5):
        return 1.0
    if (energy>17103056463.5):
        return 0.0
    if (energy>17060012828.0):
        return 1.0
    if (energy>17035124137.0):
        return 0.0
    if (energy>16765599430.0):
        return 1.0
    if (energy>16732499327.5):
        return 0.0
    if (energy>16687297925.5):
        return 1.0
    if (energy>16639165159.0):
        return 0.0
    if (energy>16577078251.5):
        return 1.0
    if (energy>16505940831.5):
        return 0.0
    if (energy>16447291065.0):
        return 1.0
    if (energy>16412176556.0):
        return 0.0
    if (energy>16220807802.0):
        return 1.0
    if (energy>16189711168.5):
        return 0.0
    if (energy>16113747432.5):
        return 1.0
    if (energy>16070538874.0):
        return 0.0
    if (energy>15796228677.0):
        return 1.0
    if (energy>15776853037.0):
        return 0.0
    if (energy>15657799381.0):
        return 1.0
    if (energy>15614807306.0):
        return 0.0
    if (energy>15580728429.0):
        return 1.0
    if (energy>15572267064.0):
        return 0.0
    if (energy>15462837779.5):
        return 1.0
    if (energy>15446172394.5):
        return 0.0
    if (energy>14926356641.5):
        return 1.0
    if (energy>14619633947.5):
        return 0.0
    if (energy>14557423563.0):
        return 1.0
    if (energy>14439029287.0):
        return 0.0
    if (energy>13361459426.0):
        return 1.0
    if (energy>13359147649.0):
        return 0.0
    return 1.0

numthresholds=65


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

