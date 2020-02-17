#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:25:25
# Invocation: btc -v -v analcatdata_vineyard-10.csv -o analcatdata_vineyard-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.55%
Model accuracy:                     57.90% (271/468 correct)
Improvement over best guess:        2.35% (of possible 44.45%)
Model capacity (MEC):               110 bits
Generalization ratio:               2.46 bits/bit
Model efficiency:                   0.02%/parameter
System behavior
True Negatives:                     35.90% (168/468)
True Positives:                     22.01% (103/468)
False Negatives:                    22.44% (105/468)
False Positives:                    19.66% (92/468)
True Pos. Rate/Sensitivity/Recall:  0.50
True Neg. Rate/Specificity:         0.65
Precision:                          0.53
F-1 Measure:                        0.51
False Negative Rate/Miss Rate:      0.50
Critical Success Index:             0.34
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
TRAINFILE="analcatdata_vineyard-10.csv"


#Number of attributes
num_attr = 3

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
    if (energy>2051.0):
        return 0.0
    if (energy>2050.5):
        return 1.0
    if (energy>2047.0):
        return 0.0
    if (energy>2047.0):
        return 1.0
    if (energy>2045.0):
        return 0.0
    if (energy>2045.0):
        return 1.0
    if (energy>2044.0):
        return 0.0
    if (energy>2044.0):
        return 1.0
    if (energy>2044.0):
        return 0.0
    if (energy>2043.0):
        return 1.0
    if (energy>2043.0):
        return 0.0
    if (energy>2042.0):
        return 1.0
    if (energy>2041.5):
        return 0.0
    if (energy>2041.0):
        return 1.0
    if (energy>2040.5):
        return 0.0
    if (energy>2039.5):
        return 1.0
    if (energy>2039.0):
        return 0.0
    if (energy>2039.0):
        return 1.0
    if (energy>2038.0):
        return 0.0
    if (energy>2036.0):
        return 1.0
    if (energy>2036.0):
        return 0.0
    if (energy>2035.0):
        return 1.0
    if (energy>2035.0):
        return 0.0
    if (energy>2034.5):
        return 1.0
    if (energy>2033.5):
        return 0.0
    if (energy>2033.0):
        return 1.0
    if (energy>2033.0):
        return 0.0
    if (energy>2031.0):
        return 1.0
    if (energy>2031.0):
        return 0.0
    if (energy>2031.0):
        return 1.0
    if (energy>2030.0):
        return 0.0
    if (energy>2030.0):
        return 1.0
    if (energy>2029.5):
        return 0.0
    if (energy>2029.0):
        return 1.0
    if (energy>2028.5):
        return 0.0
    if (energy>2027.0):
        return 1.0
    if (energy>2027.0):
        return 0.0
    if (energy>2026.5):
        return 1.0
    if (energy>2025.5):
        return 0.0
    if (energy>2025.0):
        return 1.0
    if (energy>2024.5):
        return 0.0
    if (energy>2024.0):
        return 1.0
    if (energy>2023.5):
        return 0.0
    if (energy>2023.0):
        return 1.0
    if (energy>2022.5):
        return 0.0
    if (energy>2022.0):
        return 1.0
    if (energy>2021.5):
        return 0.0
    if (energy>2021.0):
        return 1.0
    if (energy>2021.0):
        return 0.0
    if (energy>2020.0):
        return 1.0
    if (energy>2020.0):
        return 0.0
    if (energy>2019.5):
        return 1.0
    if (energy>2019.0):
        return 0.0
    if (energy>2018.0):
        return 1.0
    if (energy>2018.0):
        return 0.0
    if (energy>2017.5):
        return 1.0
    if (energy>2017.0):
        return 0.0
    if (energy>2017.0):
        return 1.0
    if (energy>2016.5):
        return 0.0
    if (energy>2016.0):
        return 1.0
    if (energy>2016.0):
        return 0.0
    if (energy>2015.0):
        return 1.0
    if (energy>2014.5):
        return 0.0
    if (energy>2014.0):
        return 1.0
    if (energy>2014.0):
        return 0.0
    if (energy>2013.5):
        return 1.0
    if (energy>2013.0):
        return 0.0
    if (energy>2013.0):
        return 1.0
    if (energy>2013.0):
        return 0.0
    if (energy>2012.0):
        return 1.0
    if (energy>2011.0):
        return 0.0
    if (energy>2010.5):
        return 1.0
    if (energy>2010.0):
        return 0.0
    if (energy>2009.0):
        return 1.0
    if (energy>2006.0):
        return 0.0
    if (energy>2006.0):
        return 1.0
    if (energy>2005.0):
        return 0.0
    if (energy>2005.0):
        return 1.0
    if (energy>2004.0):
        return 0.0
    if (energy>2004.0):
        return 1.0
    if (energy>2004.0):
        return 0.0
    if (energy>2004.0):
        return 1.0
    if (energy>2003.0):
        return 0.0
    if (energy>2003.0):
        return 1.0
    if (energy>2002.0):
        return 0.0
    if (energy>2001.5):
        return 1.0
    if (energy>2001.0):
        return 0.0
    if (energy>2000.0):
        return 1.0
    if (energy>1999.5):
        return 0.0
    if (energy>1999.0):
        return 1.0
    if (energy>1999.0):
        return 0.0
    if (energy>1999.0):
        return 1.0
    if (energy>1998.0):
        return 0.0
    if (energy>1997.5):
        return 1.0
    if (energy>1995.0):
        return 0.0
    if (energy>1994.5):
        return 1.0
    if (energy>1994.0):
        return 0.0
    if (energy>1993.5):
        return 1.0
    if (energy>1993.0):
        return 0.0
    if (energy>1993.0):
        return 1.0
    if (energy>1993.0):
        return 0.0
    if (energy>1992.5):
        return 1.0
    if (energy>1992.0):
        return 0.0
    if (energy>1992.0):
        return 1.0
    if (energy>1991.5):
        return 0.0
    if (energy>1991.0):
        return 1.0
    if (energy>1990.0):
        return 0.0
    if (energy>1990.0):
        return 1.0
    if (energy>1988.5):
        return 0.0
    if (energy>1988.0):
        return 1.0
    return 0.0

numthresholds=110


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

