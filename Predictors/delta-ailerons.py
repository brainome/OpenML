#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 23:11:01
# Invocation: btc -server brain.brainome.ai Data/delta-ailerons.csv -o Models/delta-ailerons.py -v -v -v -stopat 94.54 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                53.06%
Model accuracy:                     94.16% (6713/7129 correct)
Improvement over best guess:        41.10% (of possible 46.94%)
Model capacity (MEC):               106 bits
Generalization ratio:               63.33 bits/bit
Model efficiency:                   0.38%/parameter
System behavior
True Negatives:                     50.57% (3605/7129)
True Positives:                     43.60% (3108/7129)
False Negatives:                    3.34% (238/7129)
False Positives:                    2.50% (178/7129)
True Pos. Rate/Sensitivity/Recall:  0.93
True Neg. Rate/Specificity:         0.95
Precision:                          0.95
F-1 Measure:                        0.94
False Negative Rate/Miss Rate:      0.07
Critical Success Index:             0.88

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
TRAINFILE="delta-ailerons.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 5
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
    h_0 = max((((-20.00128 * float(x[0]))+ (-1.5029258 * float(x[1]))+ (-4.766201 * float(x[2]))+ (-14.511092 * float(x[3]))+ (-44.4326 * float(x[4]))) + 0.12826708), 0)
    h_1 = max((((-13.716565 * float(x[0]))+ (1.8003263 * float(x[1]))+ (3.7682083 * float(x[2]))+ (0.96580297 * float(x[3]))+ (-58.532383 * float(x[4]))) + 0.36655882), 0)
    h_2 = max((((0.14960697 * float(x[0]))+ (-1.0765419 * float(x[1]))+ (-1.0850483 * float(x[2]))+ (-0.534004 * float(x[3]))+ (57.55556 * float(x[4]))) + 0.67852676), 0)
    h_3 = max((((3.407349 * float(x[0]))+ (-1.5863405 * float(x[1]))+ (-0.62726253 * float(x[2]))+ (-0.22240284 * float(x[3]))+ (57.98095 * float(x[4]))) + 0.6195218), 0)
    h_4 = max((((3.94394 * float(x[0]))+ (-2.2951324 * float(x[1]))+ (-1.3936075 * float(x[2]))+ (0.09043444 * float(x[3]))+ (58.53735 * float(x[4]))) + 0.5096005), 0)
    h_5 = max((((5.212556 * float(x[0]))+ (-2.5913844 * float(x[1]))+ (-0.6152082 * float(x[2]))+ (0.6958245 * float(x[3]))+ (57.604446 * float(x[4]))) + 0.1559294), 0)
    h_6 = max((((-0.8305348 * float(x[0]))+ (-0.03676723 * float(x[1]))+ (0.12564233 * float(x[2]))+ (0.61046183 * float(x[3]))+ (0.5541959 * float(x[4]))) + -0.20488328), 0)
    h_7 = max((((4.5954347 * float(x[0]))+ (-1.3032968 * float(x[1]))+ (-0.74940354 * float(x[2]))+ (0.571704 * float(x[3]))+ (58.66574 * float(x[4]))) + 0.07948294), 0)
    h_8 = max((((-0.90614945 * float(x[0]))+ (0.70803684 * float(x[1]))+ (-0.8996024 * float(x[2]))+ (-0.02770905 * float(x[3]))+ (-0.038805183 * float(x[4]))) + -0.6200248), 0)
    h_9 = max((((4.0581584 * float(x[0]))+ (-1.2198398 * float(x[1]))+ (-2.4118915 * float(x[2]))+ (-1.3940105 * float(x[3]))+ (57.104168 * float(x[4]))) + 0.36854511), 0)
    h_10 = max((((-5.623228 * float(x[0]))+ (2.8679144 * float(x[1]))+ (1.2227288 * float(x[2]))+ (-0.64829373 * float(x[3]))+ (-58.5854 * float(x[4]))) + 0.29247254), 0)
    h_11 = max((((4.365792 * float(x[0]))+ (-2.1361077 * float(x[1]))+ (-0.7050314 * float(x[2]))+ (-1.3001633 * float(x[3]))+ (56.871555 * float(x[4]))) + 0.59042746), 0)
    h_12 = max((((-8.170787 * float(x[0]))+ (-4.00453 * float(x[1]))+ (-2.5573173 * float(x[2]))+ (-13.114713 * float(x[3]))+ (30.059679 * float(x[4]))) + -0.11011992), 0)
    h_13 = max((((0.40235886 * float(x[0]))+ (0.9117747 * float(x[1]))+ (0.06467843 * float(x[2]))+ (-0.1084549 * float(x[3]))+ (-0.32338578 * float(x[4]))) + -0.059243273), 0)
    h_14 = max((((4.5934668 * float(x[0]))+ (-1.2968311 * float(x[1]))+ (-2.4233649 * float(x[2]))+ (-0.5925729 * float(x[3]))+ (58.717194 * float(x[4]))) + 0.4680739), 0)
    o[0] = (15.204011 * h_0)+ (15.721026 * h_1)+ (-2.1729155 * h_2)+ (-2.2141218 * h_3)+ (-2.7159398 * h_4)+ (-5.801537 * h_5)+ (-0.22122584 * h_6)+ (-12.558394 * h_7)+ (-0.16689488 * h_8)+ (-4.2007203 * h_9)+ (16.058386 * h_10)+ (-2.506171 * h_11)+ (-18.286753 * h_12)+ (0.9853318 * h_13)+ (-3.0077446 * h_14) + -2.5201223

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

        model_cap=106

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

