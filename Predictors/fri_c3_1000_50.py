#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:49:05
# Invocation: btc -v -v fri_c3_1000_50-10.csv -o fri_c3_1000_50-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.50%
Model accuracy:                     74.80% (748/1000 correct)
Improvement over best guess:        19.30% (of possible 44.5%)
Model capacity (MEC):               253 bits
Generalization ratio:               2.95 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     42.50% (425/1000)
True Positives:                     32.30% (323/1000)
False Negatives:                    12.20% (122/1000)
False Positives:                    13.00% (130/1000)
True Pos. Rate/Sensitivity/Recall:  0.73
True Neg. Rate/Specificity:         0.77
Precision:                          0.71
F-1 Measure:                        0.72
False Negative Rate/Miss Rate:      0.27
Critical Success Index:             0.56
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
TRAINFILE="fri_c3_1000_50-10.csv"


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
    if (energy>22.145087):
        return 0.0
    if (energy>21.026632999999997):
        return 1.0
    if (energy>18.80277):
        return 0.0
    if (energy>17.444909999999997):
        return 1.0
    if (energy>15.728345000000001):
        return 0.0
    if (energy>15.213627000000002):
        return 1.0
    if (energy>14.481980500000006):
        return 0.0
    if (energy>14.422898):
        return 1.0
    if (energy>14.144010999999999):
        return 0.0
    if (energy>13.397507000000001):
        return 1.0
    if (energy>13.228818):
        return 0.0
    if (energy>13.070425499999999):
        return 1.0
    if (energy>12.982962500000001):
        return 0.0
    if (energy>12.939172):
        return 1.0
    if (energy>12.886447):
        return 0.0
    if (energy>12.2226555):
        return 1.0
    if (energy>12.200267):
        return 0.0
    if (energy>12.070606999999999):
        return 1.0
    if (energy>11.6737805):
        return 0.0
    if (energy>11.475929999999998):
        return 1.0
    if (energy>11.394464500000002):
        return 0.0
    if (energy>11.2931315):
        return 1.0
    if (energy>10.4317665):
        return 0.0
    if (energy>10.281973500000001):
        return 1.0
    if (energy>9.945066000000004):
        return 0.0
    if (energy>9.768292):
        return 1.0
    if (energy>9.7241115):
        return 0.0
    if (energy>9.662680000000002):
        return 1.0
    if (energy>9.432390000000002):
        return 0.0
    if (energy>9.3348315):
        return 1.0
    if (energy>9.194333999999998):
        return 0.0
    if (energy>9.06082):
        return 1.0
    if (energy>8.9952385):
        return 0.0
    if (energy>8.888593499999999):
        return 1.0
    if (energy>8.7851475):
        return 0.0
    if (energy>8.7518095):
        return 1.0
    if (energy>8.698359):
        return 0.0
    if (energy>8.540606000000002):
        return 1.0
    if (energy>8.419183):
        return 0.0
    if (energy>8.346296500000001):
        return 1.0
    if (energy>8.245966999999998):
        return 0.0
    if (energy>8.212004):
        return 1.0
    if (energy>8.0438645):
        return 0.0
    if (energy>7.9011865000000014):
        return 1.0
    if (energy>7.8438965):
        return 0.0
    if (energy>7.769972499999999):
        return 1.0
    if (energy>7.7580839999999975):
        return 0.0
    if (energy>7.707182499999998):
        return 1.0
    if (energy>7.6627425):
        return 0.0
    if (energy>7.5982720000000015):
        return 1.0
    if (energy>7.449922000000003):
        return 0.0
    if (energy>7.368753500000002):
        return 1.0
    if (energy>7.141232499999998):
        return 0.0
    if (energy>7.102835499999998):
        return 1.0
    if (energy>7.057918999999998):
        return 0.0
    if (energy>6.7882035):
        return 1.0
    if (energy>6.744133999999999):
        return 0.0
    if (energy>6.734039499999998):
        return 1.0
    if (energy>6.726456999999999):
        return 0.0
    if (energy>6.4750905):
        return 1.0
    if (energy>6.4321625000000004):
        return 0.0
    if (energy>6.399043000000001):
        return 1.0
    if (energy>6.162811499999998):
        return 0.0
    if (energy>6.0153490000000005):
        return 1.0
    if (energy>6.003934000000001):
        return 0.0
    if (energy>5.961275499999999):
        return 1.0
    if (energy>5.72086):
        return 0.0
    if (energy>5.629348499999999):
        return 1.0
    if (energy>5.6224620000000005):
        return 0.0
    if (energy>5.612596000000002):
        return 1.0
    if (energy>5.581528499999999):
        return 0.0
    if (energy>5.557391000000001):
        return 1.0
    if (energy>5.523631500000002):
        return 0.0
    if (energy>5.472149999999999):
        return 1.0
    if (energy>5.342146499999998):
        return 0.0
    if (energy>5.270887499999997):
        return 1.0
    if (energy>5.056646500000001):
        return 0.0
    if (energy>5.028987500000002):
        return 1.0
    if (energy>4.836802499999999):
        return 0.0
    if (energy>4.730771000000001):
        return 1.0
    if (energy>4.6229305):
        return 0.0
    if (energy>4.525710999999999):
        return 1.0
    if (energy>4.435502):
        return 0.0
    if (energy>4.205097):
        return 1.0
    if (energy>4.044340999999999):
        return 0.0
    if (energy>3.8542335000000003):
        return 1.0
    if (energy>3.8295224999999995):
        return 0.0
    if (energy>3.7727615000000005):
        return 1.0
    if (energy>3.707905499999999):
        return 0.0
    if (energy>3.6643389999999982):
        return 1.0
    if (energy>3.6528254999999987):
        return 0.0
    if (energy>3.633229500000001):
        return 1.0
    if (energy>3.3744249999999996):
        return 0.0
    if (energy>3.3440655):
        return 1.0
    if (energy>3.3374815):
        return 0.0
    if (energy>3.2904764999999996):
        return 1.0
    if (energy>3.242946):
        return 0.0
    if (energy>3.218302):
        return 1.0
    if (energy>3.1849505):
        return 0.0
    if (energy>3.0979089999999996):
        return 1.0
    if (energy>3.050576499999999):
        return 0.0
    if (energy>2.9051574999999983):
        return 1.0
    if (energy>2.7177649999999995):
        return 0.0
    if (energy>2.7014465000000003):
        return 1.0
    if (energy>2.690769500000001):
        return 0.0
    if (energy>2.6014920000000012):
        return 1.0
    if (energy>2.555547999999999):
        return 0.0
    if (energy>2.4718915):
        return 1.0
    if (energy>2.4313515000000003):
        return 0.0
    if (energy>2.334628500000001):
        return 1.0
    if (energy>2.299255):
        return 0.0
    if (energy>2.2660885):
        return 1.0
    if (energy>2.1075314999999994):
        return 0.0
    if (energy>1.9876965000000004):
        return 1.0
    if (energy>1.934080999999999):
        return 0.0
    if (energy>1.7286185):
        return 1.0
    if (energy>1.6802294999999998):
        return 0.0
    if (energy>1.6381044999999994):
        return 1.0
    if (energy>1.6069604999999996):
        return 0.0
    if (energy>1.5299509999999994):
        return 1.0
    if (energy>1.4947645000000005):
        return 0.0
    if (energy>1.4751920000000012):
        return 1.0
    if (energy>1.3643210000000003):
        return 0.0
    if (energy>1.2571415):
        return 1.0
    if (energy>1.220570499999999):
        return 0.0
    if (energy>1.2132299999999994):
        return 1.0
    if (energy>1.1090419999999983):
        return 0.0
    if (energy>1.0371115):
        return 1.0
    if (energy>0.9071879999999993):
        return 0.0
    if (energy>0.8190579999999996):
        return 1.0
    if (energy>0.7406825000000004):
        return 0.0
    if (energy>0.6800795000000002):
        return 1.0
    if (energy>0.5523889999999999):
        return 0.0
    if (energy>0.44115350000000003):
        return 1.0
    if (energy>0.3944050000000005):
        return 0.0
    if (energy>0.3685250000000001):
        return 1.0
    if (energy>0.34467099999999984):
        return 0.0
    if (energy>0.27142449999999996):
        return 1.0
    if (energy>0.19217199999999945):
        return 0.0
    if (energy>0.08718199999999969):
        return 1.0
    if (energy>-0.0029935000000006484):
        return 0.0
    if (energy>-0.13118000000000118):
        return 1.0
    if (energy>-0.2538135000000008):
        return 0.0
    if (energy>-0.28517199999999976):
        return 1.0
    if (energy>-0.2884104999999999):
        return 0.0
    if (energy>-0.3839270000000004):
        return 1.0
    if (energy>-0.49016249999999995):
        return 0.0
    if (energy>-0.7354345):
        return 1.0
    if (energy>-0.78673):
        return 0.0
    if (energy>-0.9597949999999995):
        return 1.0
    if (energy>-1.0592765):
        return 0.0
    if (energy>-1.1237729999999995):
        return 1.0
    if (energy>-1.2793270000000003):
        return 0.0
    if (energy>-1.3043644999999988):
        return 1.0
    if (energy>-1.346804999999999):
        return 0.0
    if (energy>-1.4365875000000006):
        return 1.0
    if (energy>-1.5082315):
        return 0.0
    if (energy>-1.6836995000000008):
        return 1.0
    if (energy>-1.852788000000001):
        return 0.0
    if (energy>-1.910900499999999):
        return 1.0
    if (energy>-1.9510034999999988):
        return 0.0
    if (energy>-2.004108499999999):
        return 1.0
    if (energy>-2.0627115000000003):
        return 0.0
    if (energy>-2.1161465):
        return 1.0
    if (energy>-2.1737760000000006):
        return 0.0
    if (energy>-2.243229):
        return 1.0
    if (energy>-2.371947):
        return 0.0
    if (energy>-2.3892439999999997):
        return 1.0
    if (energy>-2.4619530000000003):
        return 0.0
    if (energy>-2.5889374999999983):
        return 1.0
    if (energy>-2.6921015000000006):
        return 0.0
    if (energy>-2.785924999999998):
        return 1.0
    if (energy>-2.8811215000000026):
        return 0.0
    if (energy>-3.0887185000000015):
        return 1.0
    if (energy>-3.2866494999999984):
        return 0.0
    if (energy>-3.4024825000000014):
        return 1.0
    if (energy>-3.4974885000000007):
        return 0.0
    if (energy>-3.5320044999999993):
        return 1.0
    if (energy>-3.6331999999999978):
        return 0.0
    if (energy>-3.89039):
        return 1.0
    if (energy>-4.079305999999999):
        return 0.0
    if (energy>-4.1229365):
        return 1.0
    if (energy>-4.2171639999999995):
        return 0.0
    if (energy>-4.309239000000002):
        return 1.0
    if (energy>-4.348838499999999):
        return 0.0
    if (energy>-4.3707329999999995):
        return 1.0
    if (energy>-4.5020545):
        return 0.0
    if (energy>-4.553553):
        return 1.0
    if (energy>-4.601082):
        return 0.0
    if (energy>-4.645503):
        return 1.0
    if (energy>-4.755496500000001):
        return 0.0
    if (energy>-5.137076499999999):
        return 1.0
    if (energy>-5.585310999999999):
        return 0.0
    if (energy>-5.799469500000001):
        return 1.0
    if (energy>-5.904997000000001):
        return 0.0
    if (energy>-5.948639000000002):
        return 1.0
    if (energy>-5.985446500000002):
        return 0.0
    if (energy>-5.9906785):
        return 1.0
    if (energy>-6.124340499999995):
        return 0.0
    if (energy>-6.174422):
        return 1.0
    if (energy>-6.234847):
        return 0.0
    if (energy>-6.2800839999999996):
        return 1.0
    if (energy>-6.334292999999999):
        return 0.0
    if (energy>-6.3632445):
        return 1.0
    if (energy>-6.403375499999999):
        return 0.0
    if (energy>-6.5714369999999995):
        return 1.0
    if (energy>-6.661443499999998):
        return 0.0
    if (energy>-6.687215999999998):
        return 1.0
    if (energy>-6.717120499999999):
        return 0.0
    if (energy>-6.909047499999998):
        return 1.0
    if (energy>-7.188889):
        return 0.0
    if (energy>-7.242206999999999):
        return 1.0
    if (energy>-7.398581000000002):
        return 0.0
    if (energy>-7.568177999999999):
        return 1.0
    if (energy>-7.8079719999999995):
        return 0.0
    if (energy>-7.886845499999999):
        return 1.0
    if (energy>-7.9630295):
        return 0.0
    if (energy>-8.076548):
        return 1.0
    if (energy>-8.203857000000003):
        return 0.0
    if (energy>-8.284451500000001):
        return 1.0
    if (energy>-8.323747999999998):
        return 0.0
    if (energy>-8.338371500000001):
        return 1.0
    if (energy>-8.402348000000003):
        return 0.0
    if (energy>-8.470405500000002):
        return 1.0
    if (energy>-8.493807):
        return 0.0
    if (energy>-8.864765000000002):
        return 1.0
    if (energy>-9.1451475):
        return 0.0
    if (energy>-9.2981825):
        return 1.0
    if (energy>-9.412316):
        return 0.0
    if (energy>-9.579191999999999):
        return 1.0
    if (energy>-9.78412):
        return 0.0
    if (energy>-9.977367999999998):
        return 1.0
    if (energy>-10.337733):
        return 0.0
    if (energy>-10.464984999999999):
        return 1.0
    if (energy>-11.090346):
        return 0.0
    if (energy>-11.1566405):
        return 1.0
    if (energy>-11.375117499999998):
        return 0.0
    if (energy>-11.433171):
        return 1.0
    if (energy>-12.0670915):
        return 0.0
    if (energy>-12.379515999999999):
        return 1.0
    if (energy>-12.851678000000001):
        return 0.0
    if (energy>-13.0517575):
        return 1.0
    if (energy>-13.149899999999999):
        return 0.0
    if (energy>-13.321137999999998):
        return 1.0
    if (energy>-13.560066499999998):
        return 0.0
    if (energy>-13.967589999999998):
        return 1.0
    if (energy>-14.165797999999995):
        return 0.0
    if (energy>-14.567266499999999):
        return 1.0
    if (energy>-14.893391000000001):
        return 0.0
    if (energy>-15.460626):
        return 1.0
    if (energy>-16.196212000000003):
        return 0.0
    if (energy>-17.010109):
        return 1.0
    if (energy>-17.3070715):
        return 0.0
    return 1.0

numthresholds=253


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

