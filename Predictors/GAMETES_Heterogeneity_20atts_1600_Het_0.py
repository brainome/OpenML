#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:58:18
# Invocation: btc -server brain.brainome.ai Data/GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001.csv -o Models/GAMETES_Heterogeneity_20atts_1600_Het_0.py -v -v -v -stopat 100 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     74.06% (1185/1600 correct)
Improvement over best guess:        24.06% (of possible 50.0%)
Model capacity (MEC):               155 bits
Generalization ratio:               7.64 bits/bit
Model efficiency:                   0.15%/parameter
System behavior
True Negatives:                     38.81% (621/1600)
True Positives:                     35.25% (564/1600)
False Negatives:                    14.75% (236/1600)
False Positives:                    11.19% (179/1600)
True Pos. Rate/Sensitivity/Recall:  0.70
True Neg. Rate/Specificity:         0.78
Precision:                          0.76
F-1 Measure:                        0.73
False Negative Rate/Miss Rate:      0.29
Critical Success Index:             0.58

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
TRAINFILE="GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001.csv"


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
    h_0 = max((((7.410396 * float(x[0]))+ (5.391074 * float(x[1]))+ (1.621635 * float(x[2]))+ (5.6149073 * float(x[3]))+ (-3.5928385 * float(x[4]))+ (2.3615603 * float(x[5]))+ (1.1621435 * float(x[6]))+ (-3.659586 * float(x[7]))+ (0.36147663 * float(x[8]))+ (3.189219 * float(x[9]))+ (2.4676363 * float(x[10]))+ (0.9314559 * float(x[11]))+ (3.4548273 * float(x[12]))+ (1.1020579 * float(x[13]))+ (5.262968 * float(x[14]))+ (2.913715 * float(x[15]))+ (4.8742023 * float(x[16]))+ (14.704549 * float(x[17]))+ (1.6511562 * float(x[18]))+ (7.0940614 * float(x[19]))) + 0.7445114), 0)
    h_1 = max((((-3.8905425 * float(x[0]))+ (-3.501639 * float(x[1]))+ (-1.395354 * float(x[2]))+ (-2.901663 * float(x[3]))+ (1.7235742 * float(x[4]))+ (-1.0483408 * float(x[5]))+ (1.1462824 * float(x[6]))+ (1.6326624 * float(x[7]))+ (2.5608842 * float(x[8]))+ (-3.7232318 * float(x[9]))+ (-2.1028333 * float(x[10]))+ (-0.66608226 * float(x[11]))+ (1.399593 * float(x[12]))+ (-1.7855114 * float(x[13]))+ (1.1656625 * float(x[14]))+ (3.3754578 * float(x[15]))+ (-2.078038 * float(x[16]))+ (1.5636479 * float(x[17]))+ (-3.5529714 * float(x[18]))+ (0.55694115 * float(x[19]))) + -4.695469), 0)
    h_2 = max((((-0.10783833 * float(x[0]))+ (0.32615992 * float(x[1]))+ (0.25247675 * float(x[2]))+ (-0.025905235 * float(x[3]))+ (0.2463627 * float(x[4]))+ (-0.16707347 * float(x[5]))+ (0.23693907 * float(x[6]))+ (0.06960886 * float(x[7]))+ (-0.17177959 * float(x[8]))+ (-0.045212477 * float(x[9]))+ (-0.16559966 * float(x[10]))+ (1.2425021 * float(x[11]))+ (-0.22339025 * float(x[12]))+ (0.38520637 * float(x[13]))+ (-0.4979398 * float(x[14]))+ (0.16262771 * float(x[15]))+ (0.75144917 * float(x[16]))+ (0.41550156 * float(x[17]))+ (3.1740055 * float(x[18]))+ (-3.1903234 * float(x[19]))) + -0.43412688), 0)
    h_3 = max((((0.11824959 * float(x[0]))+ (0.13109493 * float(x[1]))+ (1.084261 * float(x[2]))+ (-0.21115327 * float(x[3]))+ (-0.10977665 * float(x[4]))+ (0.105554104 * float(x[5]))+ (-0.21391498 * float(x[6]))+ (-0.35807672 * float(x[7]))+ (-0.335133 * float(x[8]))+ (-0.00021056451 * float(x[9]))+ (-0.4481917 * float(x[10]))+ (0.31693512 * float(x[11]))+ (-0.5731499 * float(x[12]))+ (0.27937338 * float(x[13]))+ (-0.006245706 * float(x[14]))+ (0.35543105 * float(x[15]))+ (1.6076818 * float(x[16]))+ (3.1216545 * float(x[17]))+ (0.98419577 * float(x[18]))+ (-0.032882217 * float(x[19]))) + -2.0383856), 0)
    h_4 = max((((-0.60179156 * float(x[0]))+ (0.18220143 * float(x[1]))+ (-0.98510087 * float(x[2]))+ (-0.1976711 * float(x[3]))+ (-1.1035419 * float(x[4]))+ (1.1896305 * float(x[5]))+ (0.060823407 * float(x[6]))+ (-0.769834 * float(x[7]))+ (-1.5189893 * float(x[8]))+ (-1.7849399 * float(x[9]))+ (-2.8310323 * float(x[10]))+ (2.4635057 * float(x[11]))+ (-2.481387 * float(x[12]))+ (-2.634379 * float(x[13]))+ (0.049737692 * float(x[14]))+ (0.43810967 * float(x[15]))+ (-0.42002586 * float(x[16]))+ (1.6373781 * float(x[17]))+ (-0.1534766 * float(x[18]))+ (-0.00831633 * float(x[19]))) + -1.4989003), 0)
    h_5 = max((((1.701544 * float(x[0]))+ (1.2197205 * float(x[1]))+ (0.5630502 * float(x[2]))+ (1.3552272 * float(x[3]))+ (-0.5490741 * float(x[4]))+ (0.56953055 * float(x[5]))+ (0.5481211 * float(x[6]))+ (-0.69988775 * float(x[7]))+ (0.2227317 * float(x[8]))+ (0.5417211 * float(x[9]))+ (0.5117233 * float(x[10]))+ (0.48294672 * float(x[11]))+ (0.91586053 * float(x[12]))+ (0.35486117 * float(x[13]))+ (1.1647948 * float(x[14]))+ (0.76219356 * float(x[15]))+ (0.62815547 * float(x[16]))+ (2.4726493 * float(x[17]))+ (1.4333354 * float(x[18]))+ (0.95300364 * float(x[19]))) + 0.706207), 0)
    h_6 = max((((0.52095455 * float(x[0]))+ (-1.0346925 * float(x[1]))+ (0.5967578 * float(x[2]))+ (-0.7801168 * float(x[3]))+ (0.034120593 * float(x[4]))+ (0.06396517 * float(x[5]))+ (0.14624584 * float(x[6]))+ (-0.42143393 * float(x[7]))+ (0.45007876 * float(x[8]))+ (0.010257918 * float(x[9]))+ (-0.44614935 * float(x[10]))+ (-1.0360204 * float(x[11]))+ (0.040649515 * float(x[12]))+ (-0.05084459 * float(x[13]))+ (0.4224872 * float(x[14]))+ (0.12646237 * float(x[15]))+ (-2.0248644 * float(x[16]))+ (1.6167593 * float(x[17]))+ (0.5427337 * float(x[18]))+ (-0.76568 * float(x[19]))) + -0.8496692), 0)
    o[0] = (0.53428197 * h_0)+ (30.559582 * h_1)+ (1.4815677 * h_2)+ (-2.000755 * h_3)+ (28.492815 * h_4)+ (-2.16731 * h_5)+ (2.8594573 * h_6) + -0.3950811

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

        model_cap=155

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

