#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:28:21
# Invocation: btc -server brain.brainome.ai Data/GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1.csv -o Models/GAMETES_Epistasis_3-Way_20atts_0.py -v -v -v -stopat 62.82 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     65.87% (1054/1600 correct)
Improvement over best guess:        15.87% (of possible 50.0%)
Model capacity (MEC):               221 bits
Generalization ratio:               4.76 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     33.31% (533/1600)
True Positives:                     32.56% (521/1600)
False Negatives:                    17.44% (279/1600)
False Positives:                    16.69% (267/1600)
True Pos. Rate/Sensitivity/Recall:  0.65
True Neg. Rate/Specificity:         0.67
Precision:                          0.66
F-1 Measure:                        0.66
False Negative Rate/Miss Rate:      0.35
Critical Success Index:             0.49

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
TRAINFILE="GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1.csv"


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
    h_0 = max((((-9.911499 * float(x[0]))+ (-4.7127194 * float(x[1]))+ (-0.10587121 * float(x[2]))+ (-7.0064473 * float(x[3]))+ (-5.5973444 * float(x[4]))+ (-10.540284 * float(x[5]))+ (3.9694622 * float(x[6]))+ (-2.1739874 * float(x[7]))+ (1.8443558 * float(x[8]))+ (-6.7328086 * float(x[9]))+ (3.6144936 * float(x[10]))+ (1.2285774 * float(x[11]))+ (8.876645 * float(x[12]))+ (5.113495 * float(x[13]))+ (-3.539466 * float(x[14]))+ (-12.47622 * float(x[15]))+ (-4.412203 * float(x[16]))+ (-7.9666333 * float(x[17]))+ (2.873606 * float(x[18]))+ (6.435442 * float(x[19]))) + -1.3557813), 0)
    h_1 = max((((-0.9924945 * float(x[0]))+ (2.210009 * float(x[1]))+ (2.2014847 * float(x[2]))+ (4.313686 * float(x[3]))+ (0.41754228 * float(x[4]))+ (4.702004 * float(x[5]))+ (-3.8847237 * float(x[6]))+ (-2.4550176 * float(x[7]))+ (-0.57088196 * float(x[8]))+ (1.4765278 * float(x[9]))+ (3.2157216 * float(x[10]))+ (-5.775164 * float(x[11]))+ (-8.675211 * float(x[12]))+ (8.02412 * float(x[13]))+ (1.9934225 * float(x[14]))+ (8.441815 * float(x[15]))+ (6.0865917 * float(x[16]))+ (4.774707 * float(x[17]))+ (-0.94333494 * float(x[18]))+ (-8.6420355 * float(x[19]))) + 2.0794852), 0)
    h_2 = max((((-2.5340562 * float(x[0]))+ (-6.5297856 * float(x[1]))+ (-0.24409607 * float(x[2]))+ (-0.7653031 * float(x[3]))+ (-2.5953822 * float(x[4]))+ (2.4414513 * float(x[5]))+ (1.3049253 * float(x[6]))+ (-0.48710135 * float(x[7]))+ (-0.05104115 * float(x[8]))+ (-0.7230508 * float(x[9]))+ (0.6434597 * float(x[10]))+ (-1.4448953 * float(x[11]))+ (0.52407116 * float(x[12]))+ (2.4320955 * float(x[13]))+ (-0.3889224 * float(x[14]))+ (-6.969663 * float(x[15]))+ (-0.91138756 * float(x[16]))+ (-3.250953 * float(x[17]))+ (0.21214624 * float(x[18]))+ (3.1153517 * float(x[19]))) + 0.68778026), 0)
    h_3 = max((((-0.036870573 * float(x[0]))+ (-1.505902 * float(x[1]))+ (-0.031308576 * float(x[2]))+ (-0.28077948 * float(x[3]))+ (-0.2614757 * float(x[4]))+ (-1.295576 * float(x[5]))+ (-1.0949991 * float(x[6]))+ (2.337879 * float(x[7]))+ (-0.45406938 * float(x[8]))+ (-3.2661746 * float(x[9]))+ (0.29455677 * float(x[10]))+ (-1.6209635 * float(x[11]))+ (-0.98085237 * float(x[12]))+ (-1.4509115 * float(x[13]))+ (0.85447836 * float(x[14]))+ (-2.0447755 * float(x[15]))+ (-0.7640935 * float(x[16]))+ (0.26011392 * float(x[17]))+ (0.8669786 * float(x[18]))+ (0.15368672 * float(x[19]))) + -1.8979254), 0)
    h_4 = max((((-0.45336127 * float(x[0]))+ (-0.20796928 * float(x[1]))+ (-0.043106876 * float(x[2]))+ (0.43929327 * float(x[3]))+ (0.2762834 * float(x[4]))+ (1.6216297 * float(x[5]))+ (-1.251016 * float(x[6]))+ (0.20344551 * float(x[7]))+ (0.27293962 * float(x[8]))+ (-1.0142964 * float(x[9]))+ (0.8147842 * float(x[10]))+ (-1.1231325 * float(x[11]))+ (-1.4886963 * float(x[12]))+ (1.0741097 * float(x[13]))+ (1.3561833 * float(x[14]))+ (1.679858 * float(x[15]))+ (3.2331545 * float(x[16]))+ (-0.6606788 * float(x[17]))+ (1.8238473 * float(x[18]))+ (1.4837906 * float(x[19]))) + -0.71600384), 0)
    h_5 = max((((0.034305 * float(x[0]))+ (-0.7902941 * float(x[1]))+ (-0.32049036 * float(x[2]))+ (-0.1039964 * float(x[3]))+ (-0.19184393 * float(x[4]))+ (0.1893636 * float(x[5]))+ (-0.2837581 * float(x[6]))+ (1.1226617 * float(x[7]))+ (0.30735406 * float(x[8]))+ (-1.5939015 * float(x[9]))+ (0.07064114 * float(x[10]))+ (-0.84600085 * float(x[11]))+ (0.4689486 * float(x[12]))+ (-0.5371091 * float(x[13]))+ (0.18082233 * float(x[14]))+ (-0.34933162 * float(x[15]))+ (-0.9856993 * float(x[16]))+ (-0.12568992 * float(x[17]))+ (0.64208466 * float(x[18]))+ (-0.29805765 * float(x[19]))) + -0.23766968), 0)
    h_6 = max((((0.93345785 * float(x[0]))+ (-1.2000484 * float(x[1]))+ (-0.5643016 * float(x[2]))+ (0.50674194 * float(x[3]))+ (-1.1451093 * float(x[4]))+ (-1.4741998 * float(x[5]))+ (-3.139816 * float(x[6]))+ (-1.8539147 * float(x[7]))+ (-2.6398766 * float(x[8]))+ (-0.027756866 * float(x[9]))+ (0.9724859 * float(x[10]))+ (-1.337919 * float(x[11]))+ (-0.85003775 * float(x[12]))+ (-1.2186466 * float(x[13]))+ (-0.6319833 * float(x[14]))+ (1.6488557 * float(x[15]))+ (-3.5995936 * float(x[16]))+ (-1.236224 * float(x[17]))+ (-3.1771722 * float(x[18]))+ (1.3147981 * float(x[19]))) + -0.095494665), 0)
    h_7 = max((((-1.7944663 * float(x[0]))+ (0.5754763 * float(x[1]))+ (-0.06359727 * float(x[2]))+ (-4.17073 * float(x[3]))+ (0.6541439 * float(x[4]))+ (0.28809902 * float(x[5]))+ (-0.54535353 * float(x[6]))+ (-0.22470109 * float(x[7]))+ (-0.10869139 * float(x[8]))+ (-3.9014015 * float(x[9]))+ (-0.2737456 * float(x[10]))+ (-2.2237132 * float(x[11]))+ (-0.013315317 * float(x[12]))+ (-1.6533498 * float(x[13]))+ (-0.8755836 * float(x[14]))+ (0.4862726 * float(x[15]))+ (-0.8218417 * float(x[16]))+ (0.51978284 * float(x[17]))+ (0.1741822 * float(x[18]))+ (1.0991055 * float(x[19]))) + 0.1584652), 0)
    h_8 = max((((-0.47944018 * float(x[0]))+ (-0.26018655 * float(x[1]))+ (0.059900317 * float(x[2]))+ (-0.39603648 * float(x[3]))+ (0.4575002 * float(x[4]))+ (-0.78333384 * float(x[5]))+ (-0.43041852 * float(x[6]))+ (0.8052494 * float(x[7]))+ (-1.0994234 * float(x[8]))+ (-0.094332695 * float(x[9]))+ (-0.5418472 * float(x[10]))+ (0.29825893 * float(x[11]))+ (-1.0069824 * float(x[12]))+ (0.14952834 * float(x[13]))+ (0.7209619 * float(x[14]))+ (-0.24496025 * float(x[15]))+ (-0.29192072 * float(x[16]))+ (0.2052077 * float(x[17]))+ (-0.23036355 * float(x[18]))+ (0.51432216 * float(x[19]))) + 0.63643336), 0)
    h_9 = max((((-0.14610712 * float(x[0]))+ (-0.18853594 * float(x[1]))+ (-0.27295935 * float(x[2]))+ (-0.034058228 * float(x[3]))+ (0.090920985 * float(x[4]))+ (0.790283 * float(x[5]))+ (-0.6027831 * float(x[6]))+ (0.067548156 * float(x[7]))+ (0.2752981 * float(x[8]))+ (-0.7620426 * float(x[9]))+ (0.46473342 * float(x[10]))+ (-0.48192373 * float(x[11]))+ (-0.29626125 * float(x[12]))+ (-0.038173713 * float(x[13]))+ (0.57179815 * float(x[14]))+ (0.65122426 * float(x[15]))+ (2.0087984 * float(x[16]))+ (-0.76221776 * float(x[17]))+ (1.4316527 * float(x[18]))+ (1.496234 * float(x[19]))) + -0.37294346), 0)
    o[0] = (25.59334 * h_0)+ (-0.31858036 * h_1)+ (-19.255882 * h_2)+ (4.220061 * h_3)+ (2.7063112 * h_4)+ (-4.444643 * h_5)+ (-53.025208 * h_6)+ (3.719667 * h_7)+ (-1.649567 * h_8)+ (-3.8364625 * h_9) + 2.7854517

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

        model_cap=221

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

