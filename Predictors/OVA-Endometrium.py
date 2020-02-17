#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 12:59:57
# Invocation: btc -v -v OVA-Endometrium-1.csv -o OVA-Endometrium-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                96.05%
Model accuracy:                     97.73% (1510/1545 correct)
Improvement over best guess:        1.68% (of possible 3.95%)
Model capacity (MEC):               78 bits
Generalization ratio:               19.35 bits/bit
Model efficiency:                   0.02%/parameter
System behavior
True Negatives:                     95.02% (1468/1545)
True Positives:                     2.72% (42/1545)
False Negatives:                    1.23% (19/1545)
False Positives:                    1.04% (16/1545)
True Pos. Rate/Sensitivity/Recall:  0.69
True Neg. Rate/Specificity:         0.99
Precision:                          0.72
F-1 Measure:                        0.71
False Negative Rate/Miss Rate:      0.31
Critical Success Index:             0.55
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
TRAINFILE="OVA-Endometrium-1.csv"


#Number of attributes
num_attr = 10936

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
    if (energy>47313582.39999983):
        return 0.0
    if (energy>46770766.85000001):
        return 1.0
    if (energy>39865111.10000001):
        return 0.0
    if (energy>39859564.40000005):
        return 1.0
    if (energy>39779066.94999996):
        return 0.0
    if (energy>39765237.85000007):
        return 1.0
    if (energy>39273738.10000004):
        return 0.0
    if (energy>39269177.34999993):
        return 1.0
    if (energy>39120902.20000005):
        return 0.0
    if (energy>39118643.04999994):
        return 1.0
    if (energy>39055208.90000002):
        return 0.0
    if (energy>39038934.70000004):
        return 1.0
    if (energy>38723089.449999936):
        return 0.0
    if (energy>38715378.39999992):
        return 1.0
    if (energy>38711192.25000002):
        return 0.0
    if (energy>38710755.3000001):
        return 1.0
    if (energy>38348154.499999955):
        return 0.0
    if (energy>38339184.34999993):
        return 1.0
    if (energy>38328023.3999999):
        return 0.0
    if (energy>38325438.19999991):
        return 1.0
    if (energy>38088184.550000064):
        return 0.0
    if (energy>38086954.85000004):
        return 1.0
    if (energy>37988729.34999992):
        return 0.0
    if (energy>37982037.349999905):
        return 1.0
    if (energy>37912276.39999993):
        return 0.0
    if (energy>37907589.849999875):
        return 1.0
    if (energy>37799159.25000003):
        return 0.0
    if (energy>37795797.14999991):
        return 1.0
    if (energy>37641716.05000008):
        return 0.0
    if (energy>37633741.30000001):
        return 1.0
    if (energy>37505558.150000095):
        return 0.0
    if (energy>37497610.999999925):
        return 1.0
    if (energy>37477790.75000006):
        return 0.0
    if (energy>37468227.750000075):
        return 1.0
    if (energy>37077605.20000008):
        return 0.0
    if (energy>37076209.750000104):
        return 1.0
    if (energy>37033218.34999998):
        return 0.0
    if (energy>37026453.0500001):
        return 1.0
    if (energy>36982035.999999955):
        return 0.0
    if (energy>36978178.25):
        return 1.0
    if (energy>36969895.85000008):
        return 0.0
    if (energy>36949117.750000045):
        return 1.0
    if (energy>36567461.649999954):
        return 0.0
    if (energy>36566135.94999992):
        return 1.0
    if (energy>36218603.85000001):
        return 0.0
    if (energy>36212696.94999986):
        return 1.0
    if (energy>36200289.30000006):
        return 0.0
    if (energy>36195906.150000066):
        return 1.0
    if (energy>35504633.85000006):
        return 0.0
    if (energy>35493725.90000003):
        return 1.0
    if (energy>35358026.35000008):
        return 0.0
    if (energy>35354855.3500001):
        return 1.0
    if (energy>34462339.500000015):
        return 0.0
    if (energy>34453552.4999999):
        return 1.0
    if (energy>34402476.049999885):
        return 0.0
    if (energy>34387852.59999995):
        return 1.0
    if (energy>34158544.50000004):
        return 0.0
    if (energy>34150795.59999989):
        return 1.0
    if (energy>33577444.35000007):
        return 0.0
    if (energy>33551213.75000009):
        return 1.0
    if (energy>33394755.8000002):
        return 0.0
    if (energy>33387886.60000015):
        return 1.0
    if (energy>33320187.90000007):
        return 0.0
    if (energy>33306001.700000033):
        return 1.0
    if (energy>33074634.750000037):
        return 0.0
    if (energy>33060205.900000095):
        return 1.0
    if (energy>32886549.700000007):
        return 0.0
    if (energy>32824087.40000008):
        return 1.0
    if (energy>32816452.449999988):
        return 0.0
    if (energy>32771425.849999957):
        return 1.0
    if (energy>32725754.750000022):
        return 0.0
    if (energy>32717641.900000077):
        return 1.0
    if (energy>32437260.749999836):
        return 0.0
    if (energy>32369763.9499999):
        return 1.0
    if (energy>31591422.000000108):
        return 0.0
    if (energy>31567778.35000007):
        return 1.0
    if (energy>30986215.799999956):
        return 0.0
    if (energy>30931681.49999993):
        return 1.0
    return 0.0

numthresholds=78


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

