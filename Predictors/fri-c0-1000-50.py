#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 00:11:44
# Invocation: btc -server brain.brainome.ai Data/fri-c0-1000-50.csv -o Models/fri-c0-1000-50.py -v -v -v -stopat 88.6 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                51.00%
Model accuracy:                     91.50% (915/1000 correct)
Improvement over best guess:        40.50% (of possible 49.0%)
Model capacity (MEC):               261 bits
Generalization ratio:               3.50 bits/bit
Model efficiency:                   0.15%/parameter
System behavior
True Negatives:                     47.60% (476/1000)
True Positives:                     43.90% (439/1000)
False Negatives:                    5.10% (51/1000)
False Positives:                    3.40% (34/1000)
True Pos. Rate/Sensitivity/Recall:  0.90
True Neg. Rate/Specificity:         0.93
Precision:                          0.93
F-1 Measure:                        0.91
False Negative Rate/Miss Rate:      0.10
Critical Success Index:             0.84

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
TRAINFILE="fri-c0-1000-50.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50
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
    h_0 = max((((12.413478 * float(x[0]))+ (18.434917 * float(x[1]))+ (-1.3052704 * float(x[2]))+ (19.225065 * float(x[3]))+ (10.421562 * float(x[4]))+ (-3.305833 * float(x[5]))+ (-0.46844703 * float(x[6]))+ (0.34696844 * float(x[7]))+ (-1.922143 * float(x[8]))+ (4.0670815 * float(x[9]))+ (-1.5255688 * float(x[10]))+ (1.1688634 * float(x[11]))+ (-6.0278163 * float(x[12]))+ (-3.3737202 * float(x[13]))+ (-2.0499136 * float(x[14]))+ (-8.408323 * float(x[15]))+ (2.3812406 * float(x[16]))+ (14.472981 * float(x[17]))+ (-4.517437 * float(x[18]))+ (3.210278 * float(x[19]))+ (1.6144748 * float(x[20]))+ (-4.9191217 * float(x[21]))+ (2.9136684 * float(x[22]))+ (-1.3603978 * float(x[23]))+ (-0.93726945 * float(x[24]))+ (1.1137825 * float(x[25]))+ (-7.7184057 * float(x[26]))+ (0.9899204 * float(x[27]))+ (-1.4380851 * float(x[28]))+ (-3.8123806 * float(x[29]))+ (7.6809998 * float(x[30]))+ (-0.5155487 * float(x[31]))+ (-1.6753129 * float(x[32]))+ (-6.196287 * float(x[33]))+ (1.7251222 * float(x[34]))+ (-0.78271914 * float(x[35]))+ (-0.24662054 * float(x[36]))+ (-1.264216 * float(x[37]))+ (6.6937656 * float(x[38]))+ (-3.4334044 * float(x[39]))+ (-3.505115 * float(x[40]))+ (0.26837602 * float(x[41]))+ (-3.4310603 * float(x[42]))+ (4.3645673 * float(x[43]))+ (-5.100739 * float(x[44]))+ (-6.8427653 * float(x[45]))+ (1.8462536 * float(x[46]))+ (2.0809083 * float(x[47]))+ (12.82514 * float(x[48]))+ (-0.09047291 * float(x[49]))) + 0.8609844), 0)
    h_1 = max((((6.0119443 * float(x[0]))+ (2.602209 * float(x[1]))+ (0.304653 * float(x[2]))+ (1.7998731 * float(x[3]))+ (4.3302603 * float(x[4]))+ (3.1251676 * float(x[5]))+ (-6.1316886 * float(x[6]))+ (-4.443861 * float(x[7]))+ (-1.9420502 * float(x[8]))+ (0.4743815 * float(x[9]))+ (0.5865724 * float(x[10]))+ (6.3338284 * float(x[11]))+ (7.4843535 * float(x[12]))+ (2.2338223 * float(x[13]))+ (3.2931645 * float(x[14]))+ (0.1876716 * float(x[15]))+ (4.19493 * float(x[16]))+ (-4.4231443 * float(x[17]))+ (-0.8473953 * float(x[18]))+ (3.5892107 * float(x[19]))+ (-1.5169247 * float(x[20]))+ (-2.1127703 * float(x[21]))+ (-7.861736 * float(x[22]))+ (3.0425286 * float(x[23]))+ (1.1102983 * float(x[24]))+ (-0.04554664 * float(x[25]))+ (2.268067 * float(x[26]))+ (-3.7001736 * float(x[27]))+ (-3.3658886 * float(x[28]))+ (-2.8462534 * float(x[29]))+ (-9.107874 * float(x[30]))+ (-0.5727426 * float(x[31]))+ (1.3126245 * float(x[32]))+ (0.5033793 * float(x[33]))+ (-5.301594 * float(x[34]))+ (-4.9167495 * float(x[35]))+ (-3.6084452 * float(x[36]))+ (2.342179 * float(x[37]))+ (-2.5000112 * float(x[38]))+ (0.2499517 * float(x[39]))+ (-0.188033 * float(x[40]))+ (3.1493008 * float(x[41]))+ (-1.6587527 * float(x[42]))+ (-2.8583179 * float(x[43]))+ (0.70369494 * float(x[44]))+ (1.8912796 * float(x[45]))+ (1.368705 * float(x[46]))+ (1.2854759 * float(x[47]))+ (-4.7467804 * float(x[48]))+ (-3.1757615 * float(x[49]))) + -6.24578), 0)
    h_2 = max((((-3.1872396 * float(x[0]))+ (-3.7532985 * float(x[1]))+ (-1.9617659 * float(x[2]))+ (-3.927776 * float(x[3]))+ (-1.559746 * float(x[4]))+ (-0.31852773 * float(x[5]))+ (1.9913176 * float(x[6]))+ (0.20984605 * float(x[7]))+ (1.478968 * float(x[8]))+ (-0.7753377 * float(x[9]))+ (-2.0402954 * float(x[10]))+ (1.4084058 * float(x[11]))+ (-0.36313772 * float(x[12]))+ (0.15222533 * float(x[13]))+ (0.26971355 * float(x[14]))+ (0.44304186 * float(x[15]))+ (0.6980625 * float(x[16]))+ (-2.2150078 * float(x[17]))+ (2.6584432 * float(x[18]))+ (0.30757216 * float(x[19]))+ (-2.2640128 * float(x[20]))+ (-2.7155533 * float(x[21]))+ (-1.2696141 * float(x[22]))+ (-1.2770987 * float(x[23]))+ (-0.092985116 * float(x[24]))+ (2.277831 * float(x[25]))+ (2.6923034 * float(x[26]))+ (-2.5443654 * float(x[27]))+ (-3.3232164 * float(x[28]))+ (-1.7972307 * float(x[29]))+ (1.1710546 * float(x[30]))+ (-0.2871829 * float(x[31]))+ (-1.2615228 * float(x[32]))+ (1.3256731 * float(x[33]))+ (-3.0535533 * float(x[34]))+ (-1.7752968 * float(x[35]))+ (-1.9481088 * float(x[36]))+ (4.646945 * float(x[37]))+ (0.11898778 * float(x[38]))+ (-1.6979212 * float(x[39]))+ (-0.08683137 * float(x[40]))+ (-0.012196904 * float(x[41]))+ (-1.22057 * float(x[42]))+ (-3.0266638 * float(x[43]))+ (0.7365868 * float(x[44]))+ (4.442033 * float(x[45]))+ (-1.2602072 * float(x[46]))+ (-2.9245403 * float(x[47]))+ (-1.9833665 * float(x[48]))+ (-1.7953618 * float(x[49]))) + 2.5953443), 0)
    h_3 = max((((-1.8912096 * float(x[0]))+ (-0.46952233 * float(x[1]))+ (1.7319685 * float(x[2]))+ (-4.416244 * float(x[3]))+ (-0.9302696 * float(x[4]))+ (-0.4716524 * float(x[5]))+ (-3.0891583 * float(x[6]))+ (0.7840342 * float(x[7]))+ (-3.394195 * float(x[8]))+ (-0.25550392 * float(x[9]))+ (-1.6524005 * float(x[10]))+ (0.15925413 * float(x[11]))+ (-1.1859804 * float(x[12]))+ (-0.9327988 * float(x[13]))+ (1.6084687 * float(x[14]))+ (0.34117675 * float(x[15]))+ (0.34857902 * float(x[16]))+ (0.5751118 * float(x[17]))+ (-1.1416676 * float(x[18]))+ (1.1763598 * float(x[19]))+ (2.454991 * float(x[20]))+ (-2.9657335 * float(x[21]))+ (0.54289734 * float(x[22]))+ (-0.5088553 * float(x[23]))+ (0.60419023 * float(x[24]))+ (-1.7867725 * float(x[25]))+ (1.9268401 * float(x[26]))+ (1.7734127 * float(x[27]))+ (0.10406123 * float(x[28]))+ (1.4049081 * float(x[29]))+ (-4.2972994 * float(x[30]))+ (-1.1547453 * float(x[31]))+ (0.6634733 * float(x[32]))+ (-1.8015147 * float(x[33]))+ (0.46410084 * float(x[34]))+ (-4.823288 * float(x[35]))+ (-2.2947223 * float(x[36]))+ (1.5966997 * float(x[37]))+ (-0.9174091 * float(x[38]))+ (-1.282853 * float(x[39]))+ (-0.08043454 * float(x[40]))+ (-1.4528183 * float(x[41]))+ (-1.5416796 * float(x[42]))+ (-0.3160644 * float(x[43]))+ (0.85225636 * float(x[44]))+ (0.89311165 * float(x[45]))+ (1.4970311 * float(x[46]))+ (2.280243 * float(x[47]))+ (0.79694587 * float(x[48]))+ (1.8005059 * float(x[49]))) + -1.0432323), 0)
    h_4 = max((((-0.46545032 * float(x[0]))+ (2.2681808 * float(x[1]))+ (-0.570648 * float(x[2]))+ (2.0629482 * float(x[3]))+ (-0.91528916 * float(x[4]))+ (-0.87429196 * float(x[5]))+ (0.849708 * float(x[6]))+ (2.2439835 * float(x[7]))+ (0.4305256 * float(x[8]))+ (-1.2055955 * float(x[9]))+ (-1.393132 * float(x[10]))+ (-0.20798036 * float(x[11]))+ (-3.786148 * float(x[12]))+ (-0.8263099 * float(x[13]))+ (-1.454406 * float(x[14]))+ (1.5604596 * float(x[15]))+ (-0.785488 * float(x[16]))+ (-0.7734967 * float(x[17]))+ (1.4673178 * float(x[18]))+ (0.1471698 * float(x[19]))+ (0.71627796 * float(x[20]))+ (-3.3319714 * float(x[21]))+ (1.1521488 * float(x[22]))+ (-0.585254 * float(x[23]))+ (-1.4496692 * float(x[24]))+ (-0.38588402 * float(x[25]))+ (5.023993 * float(x[26]))+ (0.98838884 * float(x[27]))+ (-1.1284429 * float(x[28]))+ (3.680108 * float(x[29]))+ (-0.55555505 * float(x[30]))+ (-0.99011797 * float(x[31]))+ (-0.86751145 * float(x[32]))+ (-0.4401562 * float(x[33]))+ (0.14681461 * float(x[34]))+ (-0.89462906 * float(x[35]))+ (-1.0087801 * float(x[36]))+ (2.820386 * float(x[37]))+ (-1.1984018 * float(x[38]))+ (-1.3354099 * float(x[39]))+ (0.7362194 * float(x[40]))+ (-2.0037992 * float(x[41]))+ (-1.4485322 * float(x[42]))+ (-1.0777484 * float(x[43]))+ (1.6397152 * float(x[44]))+ (2.9870203 * float(x[45]))+ (0.007878895 * float(x[46]))+ (-1.9758474 * float(x[47]))+ (-0.45319754 * float(x[48]))+ (0.8523447 * float(x[49]))) + -2.049653), 0)
    o[0] = (0.8545722 * h_0)+ (3.9127655 * h_1)+ (-5.9989 * h_2)+ (-8.002925 * h_3)+ (9.304098 * h_4) + -2.281166

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

        model_cap=261

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

