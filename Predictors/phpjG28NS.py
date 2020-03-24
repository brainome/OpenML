#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:22:24
# Invocation: btc -server brain.brainome.ai Data/phpjG28NS.csv -o Models/phpjG28NS.py -v -v -v -stopat 90.51 -port 8100 -f QC -e 100
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                89.32%
Model accuracy:                     100.00% (253/253 correct)
Improvement over best guess:        10.68% (of possible 10.68%)
Model capacity (MEC):               46 bits
Generalization ratio:               5.50 bits/bit
Model efficiency:                   0.23%/parameter
System behavior
True Negatives:                     89.33% (226/253)
True Positives:                     10.67% (27/253)
False Negatives:                    0.00% (0/253)
False Positives:                    0.00% (0/253)
True Pos. Rate/Sensitivity/Recall:  1.00
True Neg. Rate/Specificity:         1.00
Precision:                          1.00
F-1 Measure:                        1.00
False Negative Rate/Miss Rate:      0.00
Critical Success Index:             1.00

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

from bisect import bisect_left

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="phpjG28NS.csv"


#Number of attributes
num_attr = 37
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



# Calculate energy

import numpy as np
energy_thresholds=np.array([1401.82, 1444.875, 1501.975, 1556.2250000000001, 2288.785, 2339.205, 3335.7850000000003, 3424.8050000000003, 3486.1850000000004, 3544.02, 4124.67, 4129.549999999999, 4981.795, 4990.360000000001, 6109.8150000000005, 6186.95, 7323.210000000001, 7461.14, 12362.925, 12603.895, 12781.24, 13971.36, 16167.650000000001, 16483.915, 17193.434999999998, 17856.235, 19655.82, 20540.525, 25702.31, 25999.655, 26382.155, 27108.225000000006, 29176.49, 31109.840000000004, 33113.119999999995, 33254.06, 36335.425, 36817.700000000004, 39456.275, 40309.795, 41803.15, 42132.485, 58120.465, 65517.14, 117495.895, 159210.16500000004])
def eqenergy(rows):
    return np.sum(rows,axis=1)
def classify(rows):
    energys=eqenergy(rows)
    start_label=1
    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys=np.argwhere(np.logical_and(numers<len(energy_thresholds),numers>=0)).reshape(-1)
        defaultindys=np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds),numers>=0))).reshape(-1)
        outputs=np.zeros(input_energys.shape[0])
        outputs[indys]=(numers[indys]+start_label)%2
        outputs[defaultindys]=0
        return outputs
    return thresh_search(energys)

numthresholds=46


# Main method
model_cap=numthresholds
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    if numthresholds<10:
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
                    for i,valrow in enumerate(valcsvreader):
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








    else:
        if not args.validate: # Then predict
            if args.cleanfile:
                cleanarr=np.loadtxt(args.csvfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr)
                for k,o in enumerate(outputs):

                    print(str(','.join(str(j) for j in ([i for i in cleanarr[k]])))+','+str(int(o)))
            else:
                tempdir=tempfile.gettempdir()
                cleanfile=tempdir+os.sep+"clean.csv"
                clean(args.csvfile,cleanfile, -1, args.headerless, True)
                with open(args.csvfile,'r') as dirtycsvfile:
                    dirtycsvreader = csv.reader(dirtycsvfile)
                    if (not args.headerless):
                            print(','.join(next(dirtycsvreader, None)+['Prediction']))
                    cleanarr=np.loadtxt(cleanfile,delimiter=',',dtype='float64')
                    outputs=classify(cleanarr)
                    for k,dirtyrow in enumerate(dirtycsvreader):

                        print(str(','.join(str(j) for j in ([i for i in dirtyrow])))+','+str(int(outputs[k])))
                os.remove(cleanfile)
                
        else: # Then validate this predictor
            if n_classes==2:
                tempdir=tempfile.gettempdir()
                temp_name = next(tempfile._get_candidate_names())
                cleanvalfile=tempdir+os.sep+temp_name

                clean(args.csvfile,cleanvalfile, -1, args.headerless)
                cleanarr=np.loadtxt(cleanvalfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr[:,:-1])
                count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0=0,0,0,0,0,0,0,0
                correct_count=int(np.sum(outputs.reshape(-1)==cleanarr[:,-1].reshape(-1)))
                count=outputs.shape[0]
                num_TP=int(np.sum(np.logical_and(outputs.reshape(-1)==1,cleanarr[:,-1].reshape(-1)==1)))
                num_TN=int(np.sum(np.logical_and(outputs.reshape(-1)==0,cleanarr[:,-1].reshape(-1)==0)))
                num_FN=int(np.sum(np.logical_and(outputs.reshape(-1)==0,cleanarr[:,-1].reshape(-1)==1)))
                num_FP=int(np.sum(np.logical_and(outputs.reshape(-1)==1,cleanarr[:,-1].reshape(-1)==0)))
                num_class_0=int(np.sum(cleanarr[:,-1].reshape(-1)==0))
                num_class_1=int(np.sum(cleanarr[:,-1].reshape(-1)==1))
            else:
                tempdir=tempfile.gettempdir()
                temp_name = next(tempfile._get_candidate_names())
                cleanvalfile=tempdir+os.sep+temp_name

                clean(args.csvfile,cleanvalfile, -1, args.headerless)
                cleanarr=np.loadtxt(cleanvalfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr[:,:-1])
                count,correct_count=0,0
                numeachclass={}
                for k,o in enumerate(outputs):
                    if int(o)==int(float(cleanarr[k,-1])):
                        correct_count+=1
                    if int(float(cleanarr[k,-1])) in numeachclass.keys():
                        numeachclass[int(float(cleanarr[k,-1]))]+=1
                    else:
                        numeachclass[int(float(cleanarr[k,-1]))]=0
                    count+=1


    

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
    

