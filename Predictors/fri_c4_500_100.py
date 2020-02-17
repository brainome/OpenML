#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:49:30
# Invocation: btc -v -v fri_c4_500_100-10.csv -o fri_c4_500_100-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                56.60%
Model accuracy:                     77.20% (386/500 correct)
Improvement over best guess:        20.60% (of possible 43.4%)
Model capacity (MEC):               139 bits
Generalization ratio:               2.77 bits/bit
Model efficiency:                   0.14%/parameter
System behavior
True Negatives:                     31.40% (157/500)
True Positives:                     45.80% (229/500)
False Negatives:                    10.80% (54/500)
False Positives:                    12.00% (60/500)
True Pos. Rate/Sensitivity/Recall:  0.81
True Neg. Rate/Specificity:         0.72
Precision:                          0.79
F-1 Measure:                        0.80
False Negative Rate/Miss Rate:      0.19
Critical Success Index:             0.67
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
TRAINFILE="fri_c4_500_100-10.csv"


#Number of attributes
num_attr = 100

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
    if (energy>27.421442999999996):
        return 0.0
    if (energy>21.41433999999999):
        return 1.0
    if (energy>20.479589499999992):
        return 0.0
    if (energy>19.1920825):
        return 1.0
    if (energy>18.313495000000003):
        return 0.0
    if (energy>15.886556999999996):
        return 1.0
    if (energy>14.4090165):
        return 0.0
    if (energy>13.989792499999998):
        return 1.0
    if (energy>13.1580175):
        return 0.0
    if (energy>12.727869500000002):
        return 1.0
    if (energy>12.574728000000004):
        return 0.0
    if (energy>12.3610955):
        return 1.0
    if (energy>12.309855000000002):
        return 0.0
    if (energy>12.261478499999999):
        return 1.0
    if (energy>12.209674):
        return 0.0
    if (energy>12.141264):
        return 1.0
    if (energy>12.091614999999997):
        return 0.0
    if (energy>11.735486000000003):
        return 1.0
    if (energy>11.415659999999999):
        return 0.0
    if (energy>11.253065999999997):
        return 1.0
    if (energy>10.893995500000003):
        return 0.0
    if (energy>10.280854500000004):
        return 1.0
    if (energy>9.3310385):
        return 0.0
    if (energy>9.144859499999999):
        return 1.0
    if (energy>8.678477999999998):
        return 0.0
    if (energy>7.980446999999997):
        return 1.0
    if (energy>7.925404499999994):
        return 0.0
    if (energy>7.702613000000001):
        return 1.0
    if (energy>7.4963265):
        return 0.0
    if (energy>7.3019905000000005):
        return 1.0
    if (energy>7.129150500000001):
        return 0.0
    if (energy>6.82546):
        return 1.0
    if (energy>6.647455499999999):
        return 0.0
    if (energy>6.379255000000002):
        return 1.0
    if (energy>6.04212):
        return 0.0
    if (energy>5.995688000000005):
        return 1.0
    if (energy>5.635725000000004):
        return 0.0
    if (energy>5.2972294999999985):
        return 1.0
    if (energy>5.185599500000001):
        return 0.0
    if (energy>5.139009500000001):
        return 1.0
    if (energy>4.990805499999999):
        return 0.0
    if (energy>4.774932000000003):
        return 1.0
    if (energy>4.602770500000005):
        return 0.0
    if (energy>4.394228000000002):
        return 1.0
    if (energy>4.2117215):
        return 0.0
    if (energy>4.022532):
        return 1.0
    if (energy>3.9958040000000006):
        return 0.0
    if (energy>3.868771500000002):
        return 1.0
    if (energy>3.8059355000000004):
        return 0.0
    if (energy>3.7465235000000012):
        return 1.0
    if (energy>3.718415500000003):
        return 0.0
    if (energy>3.432660000000001):
        return 1.0
    if (energy>3.146629000000001):
        return 0.0
    if (energy>3.0327000000000015):
        return 1.0
    if (energy>2.583610000000001):
        return 0.0
    if (energy>2.4389934999999996):
        return 1.0
    if (energy>2.1481464999999993):
        return 0.0
    if (energy>1.9679019999999994):
        return 1.0
    if (energy>1.861191999999999):
        return 0.0
    if (energy>1.7512535000000025):
        return 1.0
    if (energy>1.6485820000000024):
        return 0.0
    if (energy>1.5395735000000008):
        return 1.0
    if (energy>1.4557410000000008):
        return 0.0
    if (energy>1.3986525000000016):
        return 1.0
    if (energy>1.3457490000000019):
        return 0.0
    if (energy>1.2860815000000012):
        return 1.0
    if (energy>1.0319479999999985):
        return 0.0
    if (energy>0.9717394999999991):
        return 1.0
    if (energy>0.9266355000000018):
        return 0.0
    if (energy>0.905637):
        return 1.0
    if (energy>0.3443154999999998):
        return 0.0
    if (energy>-0.6206304999999983):
        return 1.0
    if (energy>-0.6802889999999996):
        return 0.0
    if (energy>-1.8034914999999971):
        return 1.0
    if (energy>-1.933808499999996):
        return 0.0
    if (energy>-2.0545215000000008):
        return 1.0
    if (energy>-2.1925605000000017):
        return 0.0
    if (energy>-2.379392499999997):
        return 1.0
    if (energy>-2.4117179999999983):
        return 0.0
    if (energy>-2.605914499999999):
        return 1.0
    if (energy>-2.9338059999999997):
        return 0.0
    if (energy>-3.3329950000000004):
        return 1.0
    if (energy>-3.4385699999999986):
        return 0.0
    if (energy>-3.7242185):
        return 1.0
    if (energy>-4.116388999999997):
        return 0.0
    if (energy>-4.2755565):
        return 1.0
    if (energy>-4.535616000000001):
        return 0.0
    if (energy>-4.927851499999999):
        return 1.0
    if (energy>-5.5398539999999965):
        return 0.0
    if (energy>-5.896897):
        return 1.0
    if (energy>-6.0260795):
        return 0.0
    if (energy>-6.369896500000001):
        return 1.0
    if (energy>-6.395655499999999):
        return 0.0
    if (energy>-6.499234499999998):
        return 1.0
    if (energy>-6.522529499999998):
        return 0.0
    if (energy>-6.839083499999999):
        return 1.0
    if (energy>-6.868319):
        return 0.0
    if (energy>-6.957471500000002):
        return 1.0
    if (energy>-7.047761000000001):
        return 0.0
    if (energy>-7.091263499999998):
        return 1.0
    if (energy>-7.129685500000001):
        return 0.0
    if (energy>-7.530187):
        return 1.0
    if (energy>-7.789009):
        return 0.0
    if (energy>-7.856516000000001):
        return 1.0
    if (energy>-7.926108000000001):
        return 0.0
    if (energy>-8.147354):
        return 1.0
    if (energy>-8.2023375):
        return 0.0
    if (energy>-8.310109000000006):
        return 1.0
    if (energy>-8.415090000000003):
        return 0.0
    if (energy>-8.519105499999998):
        return 1.0
    if (energy>-8.533049):
        return 0.0
    if (energy>-9.4513775):
        return 1.0
    if (energy>-9.751837000000002):
        return 0.0
    if (energy>-9.903300499999999):
        return 1.0
    if (energy>-10.073445):
        return 0.0
    if (energy>-10.388948500000001):
        return 1.0
    if (energy>-10.474674499999995):
        return 0.0
    if (energy>-10.874848499999995):
        return 1.0
    if (energy>-11.246008499999995):
        return 0.0
    if (energy>-11.322209499999996):
        return 1.0
    if (energy>-11.374203999999999):
        return 0.0
    if (energy>-12.354175999999999):
        return 1.0
    if (energy>-12.699098499999998):
        return 0.0
    if (energy>-13.133241999999996):
        return 1.0
    if (energy>-13.5492265):
        return 0.0
    if (energy>-13.590279499999998):
        return 1.0
    if (energy>-13.660247):
        return 0.0
    if (energy>-14.461052000000002):
        return 1.0
    if (energy>-15.677516500000001):
        return 0.0
    if (energy>-16.050456499999996):
        return 1.0
    if (energy>-16.391870999999995):
        return 0.0
    if (energy>-16.592699999999997):
        return 1.0
    if (energy>-17.484852000000004):
        return 0.0
    if (energy>-18.1425165):
        return 1.0
    if (energy>-18.706832499999997):
        return 0.0
    if (energy>-19.256786000000012):
        return 1.0
    if (energy>-20.819586999999995):
        return 0.0
    if (energy>-21.701785000000008):
        return 1.0
    if (energy>-25.6520875):
        return 0.0
    return 1.0

numthresholds=139


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

