#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 12:07:00
# Invocation: btc -v -v LATER-OVA-Lung-10.csv -o LATER-OVA-Lung-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                91.84%
Model accuracy:                     92.23% (1425/1545 correct)
Improvement over best guess:        0.39% (of possible 8.16%)
Model capacity (MEC):               112 bits
Generalization ratio:               12.72 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     88.16% (1362/1545)
True Positives:                     4.08% (63/1545)
False Negatives:                    4.08% (63/1545)
False Positives:                    3.69% (57/1545)
True Pos. Rate/Sensitivity/Recall:  0.50
True Neg. Rate/Specificity:         0.96
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
TRAINFILE="LATER-OVA-Lung-10.csv"


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
    if (energy>43457464.49999994):
        return 0.0
    if (energy>43337070.249999866):
        return 1.0
    if (energy>42946217.04999983):
        return 0.0
    if (energy>42772973.94999982):
        return 1.0
    if (energy>42364580.79999985):
        return 0.0
    if (energy>42338299.299999945):
        return 1.0
    if (energy>41593713.40000013):
        return 0.0
    if (energy>41573280.95000006):
        return 1.0
    if (energy>40403543.65000002):
        return 0.0
    if (energy>40400468.99999994):
        return 1.0
    if (energy>40197021.100000024):
        return 0.0
    if (energy>40180892.449999966):
        return 1.0
    if (energy>39905815.050000206):
        return 0.0
    if (energy>39884113.700000055):
        return 1.0
    if (energy>39440106.350000046):
        return 0.0
    if (energy>39402501.29999999):
        return 1.0
    if (energy>39319578.150000006):
        return 0.0
    if (energy>39300633.54999997):
        return 1.0
    if (energy>39193496.49999995):
        return 0.0
    if (energy>39179243.95000003):
        return 1.0
    if (energy>39084118.79999995):
        return 0.0
    if (energy>39078588.09999996):
        return 1.0
    if (energy>38996398.199999884):
        return 0.0
    if (energy>38977979.95):
        return 1.0
    if (energy>38932784.20000001):
        return 0.0
    if (energy>38920325.34999999):
        return 1.0
    if (energy>38838644.69999997):
        return 0.0
    if (energy>38821561.89999993):
        return 1.0
    if (energy>38653437.34999995):
        return 0.0
    if (energy>38630067.20000009):
        return 1.0
    if (energy>38573175.45000006):
        return 0.0
    if (energy>38532279.59999995):
        return 1.0
    if (energy>38516649.7):
        return 0.0
    if (energy>38514368.20000009):
        return 1.0
    if (energy>38350941.35000004):
        return 0.0
    if (energy>38347124.49999994):
        return 1.0
    if (energy>38125457.44999994):
        return 0.0
    if (energy>38119233.75000002):
        return 1.0
    if (energy>38086954.85000004):
        return 0.0
    if (energy>38085821.649999924):
        return 1.0
    if (energy>38058630.899999976):
        return 0.0
    if (energy>38051628.40000004):
        return 1.0
    if (energy>37825365.949999996):
        return 0.0
    if (energy>37821274.10000005):
        return 1.0
    if (energy>37795475.29999985):
        return 0.0
    if (energy>37787541.60000008):
        return 1.0
    if (energy>37709660.84999986):
        return 0.0
    if (energy>37697632.249999866):
        return 1.0
    if (energy>37624523.05000007):
        return 0.0
    if (energy>37605766.199999996):
        return 1.0
    if (energy>37593949.599999905):
        return 0.0
    if (energy>37590582.69999984):
        return 1.0
    if (energy>37580768.00000006):
        return 0.0
    if (energy>37568365.00000003):
        return 1.0
    if (energy>37468227.750000075):
        return 0.0
    if (energy>37461883.05000009):
        return 1.0
    if (energy>37418120.29999994):
        return 0.0
    if (energy>37410009.09999995):
        return 1.0
    if (energy>37375225.00000008):
        return 0.0
    if (energy>37374714.750000045):
        return 1.0
    if (energy>36942432.100000024):
        return 0.0
    if (energy>36939042.80000009):
        return 1.0
    if (energy>36883150.35000012):
        return 0.0
    if (energy>36877184.20000001):
        return 1.0
    if (energy>36848730.55):
        return 0.0
    if (energy>36841491.3):
        return 1.0
    if (energy>36680059.89999993):
        return 0.0
    if (energy>36672936.349999964):
        return 1.0
    if (energy>36588640.95000002):
        return 0.0
    if (energy>36587496.3000001):
        return 1.0
    if (energy>36548668.04999998):
        return 0.0
    if (energy>36538877.599999994):
        return 1.0
    if (energy>36238841.45000006):
        return 0.0
    if (energy>36237592.00000006):
        return 1.0
    if (energy>36157049.999999955):
        return 0.0
    if (energy>36156292.449999996):
        return 1.0
    if (energy>36092021.90000008):
        return 0.0
    if (energy>36086781.100000024):
        return 1.0
    if (energy>36034163.04999979):
        return 0.0
    if (energy>36032586.29999988):
        return 1.0
    if (energy>35960561.849999905):
        return 0.0
    if (energy>35958856.74999997):
        return 1.0
    if (energy>35775764.75):
        return 0.0
    if (energy>35768247.94999994):
        return 1.0
    if (energy>35747488.900000095):
        return 0.0
    if (energy>35732596.65000014):
        return 1.0
    if (energy>35704213.59999995):
        return 0.0
    if (energy>35685890.00000002):
        return 1.0
    if (energy>35539052.85000001):
        return 0.0
    if (energy>35527411.10000007):
        return 1.0
    if (energy>35519722.75000006):
        return 0.0
    if (energy>35507063.000000045):
        return 1.0
    if (energy>35172439.29999994):
        return 0.0
    if (energy>35165838.349999964):
        return 1.0
    if (energy>35069197.99999995):
        return 0.0
    if (energy>35034864.19999994):
        return 1.0
    if (energy>34713592.05000004):
        return 0.0
    if (energy>34701755.64999996):
        return 1.0
    if (energy>33242870.05000012):
        return 0.0
    if (energy>33207929.900000073):
        return 1.0
    if (energy>33166059.199999977):
        return 0.0
    if (energy>33162973.550000053):
        return 1.0
    if (energy>32777591.549999975):
        return 0.0
    if (energy>32752127.050000034):
        return 1.0
    if (energy>32324510.799999904):
        return 0.0
    if (energy>32284682.299999952):
        return 1.0
    if (energy>32253118.950000044):
        return 0.0
    if (energy>32235462.650000036):
        return 1.0
    if (energy>31746999.599999953):
        return 0.0
    if (energy>31648127.049999982):
        return 1.0
    if (energy>25712546.549999967):
        return 0.0
    if (energy>24815055.94999986):
        return 1.0
    return 0.0

numthresholds=112


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

