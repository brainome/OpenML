#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:58:23
# Invocation: btc -server brain.brainome.ai Data/eucalyptus.csv -o Models/eucalyptus.py -v -v -v -stopat 78.26 -port 8100 -f QC -e 100
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                70.92%
Model accuracy:                     91.03% (670/736 correct)
Improvement over best guess:        20.11% (of possible 29.08%)
Model capacity (MEC):               169 bits
Generalization ratio:               3.96 bits/bit
Model efficiency:                   0.11%/parameter
System behavior
True Negatives:                     24.59% (181/736)
True Positives:                     66.44% (489/736)
False Negatives:                    4.48% (33/736)
False Positives:                    4.48% (33/736)
True Pos. Rate/Sensitivity/Recall:  0.94
True Neg. Rate/Specificity:         0.85
Precision:                          0.94
F-1 Measure:                        0.94
False Negative Rate/Miss Rate:      0.06
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

from bisect import bisect_left

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="eucalyptus.csv"


#Number of attributes
num_attr = 19
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
energy_thresholds=np.array([5189394638.35, 5308799295.91, 5428203826.195, 5428204311.695, 5493849731.094999, 5622978967.945, 5631443613.44, 5861787083.995001, 5861787582.504999, 5861788077.815001, 5927433471.469999, 6073492810.76, 6094363842.235001, 6410669119.075, 6458850346.985, 6527003146.12, 6546974725.495001, 6694946968.665001, 6785784843.700001, 6817323403.145, 6830787585.065001, 6844251747.215, 6844252227.325001, 6844252857.665, 6885965624.605, 6997488818.485001, 6997489620.905001, 7213511293.75, 7232208597.664999, 7250905914.535, 7250907007.385, 7250907141.790001, 7263879935.405, 7584942658.414999, 7584942697.66, 7664024403.1, 7695833230.465, 7768249092.93, 7769904168.27, 7818085396.3949995, 7907131686.08, 8136413545.344999, 8136413949.23, 8136414361.280001, 8147747743.370001, 8174903907.485, 8222901024.705, 8434784160.299999, 8434784405.65, 8496060860.1050005, 8512022436.975, 8619641852.35, 8632645970.220001, 8679259409.035, 8712868757.82, 8841438796.15, 8841439072.5, 8841439419.3, 8856045056.6, 8907173323.52, 8907173838.654999, 8907174404.245, 8924848835.05, 8942523236.544998, 8942523256.649998, 9025697309.635, 9025697336.02, 9215069229.395, 9268827910.52, 9313828680.71, 9313828710.195, 9374668054.97, 9439107328.685, 9516661632.745, 9516661671.850002, 9587035811.240002, 9657410909.82, 9657410936.454998, 9719366640.675, 9815810252.65, 9818187079.099998, 9818187108.849998, 9846227549.145, 9883544653.08, 9892819815.84, 9935021093.27, 10005459912.115, 10052073416.95, 10070449283.980001, 10208339413.91, 10270484317.565, 10270484914.25, 10302155659.140001, 10476902523.82, 10534691661.095001, 10552365139.465, 10566885930.09, 10581406687.349998, 10581407173.035, 10581407659.074999, 10581407731.175, 10581407737.05, 10581428256.61, 10643757858.349998, 10649672146.885002, 10762739633.394999, 10897945938.915, 10899793046.85, 10899793553.865002, 10899794072.64, 10972156669.1, 11127374658.67, 11173987328.885, 11323125123.27, 11350307933.404999, 11350307955.619999, 11399260041.18, 11491757801.38, 11491758051.84, 11521737960.786, 11551717588.961, 11551717602.795, 11569918963.285, 11720331481.68, 11720331967.675, 11720332432.55, 11725055848.099998, 11729779397.38, 11898413031.835001, 11898413053.505001, 11968113692.545, 12098656962.785, 12098656969.89, 12121066925.119999, 12121067447.955, 12138485484.329998, 12155903000.98, 12177155249.575, 12182490201.865, 12187826171.43, 12225556040.125, 12314508874.435001, 12359596045.22, 12466202713.214998, 12527722228.255001, 12527722268.849998, 12609615473.244999, 13090671183.785, 13295341553.470001, 14304915963.36, 14330040676.864998, 14330040692.025, 14472520831.06, 14727999519.365, 14934225021.060001, 14934225048.560001, 15573041840.2, 17433131954.15, 17583379727.71, 17839786632.449997, 17839786951.4, 18098572839.055, 18118529170.8, 18789849207.63, 18921912992.0, 19599900952.58, 19642104277.05, 19642105061.1, 19931116138.25])
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

numthresholds=169


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
    

