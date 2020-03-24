#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:43:20
# Invocation: btc -server brain.brainome.ai Data/GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1.csv -o Models/GAMETES-Epistasis-2-Way-20atts-0.py -v -v -v -stopat 80.56 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     79.12% (1266/1600 correct)
Improvement over best guess:        29.12% (of possible 50.0%)
Model capacity (MEC):               133 bits
Generalization ratio:               9.51 bits/bit
Model efficiency:                   0.21%/parameter
System behavior
True Negatives:                     42.00% (672/1600)
True Positives:                     37.12% (594/1600)
False Negatives:                    12.88% (206/1600)
False Positives:                    8.00% (128/1600)
True Pos. Rate/Sensitivity/Recall:  0.74
True Neg. Rate/Specificity:         0.84
Precision:                          0.82
F-1 Measure:                        0.78
False Negative Rate/Miss Rate:      0.26
Critical Success Index:             0.64

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
TRAINFILE="GAMETES-Epistasis-2-Way-20atts-0.4H-EDM-1-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 20
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
    h_0 = max((((-7.0474677 * float(x[0]))+ (-3.5505972 * float(x[1]))+ (0.43797553 * float(x[2]))+ (4.15163 * float(x[3]))+ (0.97181875 * float(x[4]))+ (8.649084 * float(x[5]))+ (-6.8879414 * float(x[6]))+ (1.9041913 * float(x[7]))+ (7.1790657 * float(x[8]))+ (-1.0655086 * float(x[9]))+ (2.0693107 * float(x[10]))+ (3.8443727 * float(x[11]))+ (2.8242881 * float(x[12]))+ (0.37165418 * float(x[13]))+ (-7.0951667 * float(x[14]))+ (-5.0436783 * float(x[15]))+ (3.4691637 * float(x[16]))+ (-2.0228143 * float(x[17]))+ (5.6076403 * float(x[18]))+ (-16.535831 * float(x[19]))) + -0.1913017), 0)
    h_1 = max((((-3.7936046 * float(x[0]))+ (1.5704575 * float(x[1]))+ (-0.59015733 * float(x[2]))+ (-1.2703856 * float(x[3]))+ (-0.7122313 * float(x[4]))+ (-2.1821783 * float(x[5]))+ (0.9182762 * float(x[6]))+ (-0.7476713 * float(x[7]))+ (-0.2973934 * float(x[8]))+ (0.7732612 * float(x[9]))+ (-3.4785402 * float(x[10]))+ (-0.6225471 * float(x[11]))+ (1.058385 * float(x[12]))+ (1.0661192 * float(x[13]))+ (0.5393977 * float(x[14]))+ (-0.3546651 * float(x[15]))+ (-1.0658293 * float(x[16]))+ (0.7839337 * float(x[17]))+ (14.989924 * float(x[18]))+ (18.897804 * float(x[19]))) + -7.3489156), 0)
    h_2 = max((((1.8642207 * float(x[0]))+ (1.7517371 * float(x[1]))+ (-1.9467968 * float(x[2]))+ (0.9802621 * float(x[3]))+ (-1.6038662 * float(x[4]))+ (-2.2593157 * float(x[5]))+ (0.14894898 * float(x[6]))+ (-3.2964528 * float(x[7]))+ (-1.3779888 * float(x[8]))+ (1.3478073 * float(x[9]))+ (-6.84435 * float(x[10]))+ (-3.777602 * float(x[11]))+ (1.2177187 * float(x[12]))+ (-2.1244218 * float(x[13]))+ (0.20762695 * float(x[14]))+ (1.540041 * float(x[15]))+ (5.21826 * float(x[16]))+ (-1.0461551 * float(x[17]))+ (3.610927 * float(x[18]))+ (-4.948853 * float(x[19]))) + -6.000956), 0)
    h_3 = max((((2.7701843 * float(x[0]))+ (-2.881213 * float(x[1]))+ (0.4261981 * float(x[2]))+ (1.4068121 * float(x[3]))+ (2.5504193 * float(x[4]))+ (0.03411617 * float(x[5]))+ (3.4484148 * float(x[6]))+ (0.88374716 * float(x[7]))+ (0.53007454 * float(x[8]))+ (-0.79405576 * float(x[9]))+ (-0.67915046 * float(x[10]))+ (0.8482733 * float(x[11]))+ (-0.9464136 * float(x[12]))+ (0.76891756 * float(x[13]))+ (-0.23776296 * float(x[14]))+ (0.5710961 * float(x[15]))+ (1.1261307 * float(x[16]))+ (-0.9479352 * float(x[17]))+ (4.365609 * float(x[18]))+ (-0.34932578 * float(x[19]))) + 0.44587028), 0)
    h_4 = max((((-0.8383119 * float(x[0]))+ (-0.15094714 * float(x[1]))+ (-0.5400352 * float(x[2]))+ (1.61969 * float(x[3]))+ (0.9669995 * float(x[4]))+ (1.3270289 * float(x[5]))+ (-1.4177194 * float(x[6]))+ (-1.1498194 * float(x[7]))+ (0.89017385 * float(x[8]))+ (0.87218213 * float(x[9]))+ (-1.3778864 * float(x[10]))+ (0.4479628 * float(x[11]))+ (-0.6631368 * float(x[12]))+ (-1.1061498 * float(x[13]))+ (1.5326748 * float(x[14]))+ (-0.7656506 * float(x[15]))+ (-0.7387354 * float(x[16]))+ (-0.23997971 * float(x[17]))+ (2.030151 * float(x[18]))+ (5.7416577 * float(x[19]))) + 2.0959222), 0)
    h_5 = max((((0.7327226 * float(x[0]))+ (-1.7021817 * float(x[1]))+ (-0.1269249 * float(x[2]))+ (2.1664119 * float(x[3]))+ (2.0019126 * float(x[4]))+ (1.5783913 * float(x[5]))+ (0.09332887 * float(x[6]))+ (-0.38907588 * float(x[7]))+ (1.3363678 * float(x[8]))+ (0.26264155 * float(x[9]))+ (-1.2558628 * float(x[10]))+ (0.87807757 * float(x[11]))+ (-0.8453874 * float(x[12]))+ (-0.5146694 * float(x[13]))+ (0.79718626 * float(x[14]))+ (-0.49822676 * float(x[15]))+ (0.20794316 * float(x[16]))+ (-0.77908707 * float(x[17]))+ (1.8099785 * float(x[18]))+ (1.1667821 * float(x[19]))) + 0.017075699), 0)
    o[0] = (0.17684676 * h_0)+ (-0.555189 * h_1)+ (11.193332 * h_2)+ (1.262757 * h_3)+ (2.1791143 * h_4)+ (-2.8548398 * h_5) + -5.2796836

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

        model_cap=133

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

