#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Feb-28-2020 17:18:05
# Invocation: btc Data/fri-c2-500-50.csv -o Models/fri-c2-500-50.py -v -v -v -stopat 92 -port 8090 -e 9
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                59.00%
Model accuracy:                     89.80% (449/500 correct)
Improvement over best guess:        30.80% (of possible 41.0%)
Model capacity (MEC):               209 bits
Generalization ratio:               2.14 bits/bit
Model efficiency:                   0.14%/parameter
System behavior
True Negatives:                     36.20% (181/500)
True Positives:                     53.60% (268/500)
False Negatives:                    5.40% (27/500)
False Positives:                    4.80% (24/500)
True Pos. Rate/Sensitivity/Recall:  0.91
True Neg. Rate/Specificity:         0.88
Precision:                          0.92
F-1 Measure:                        0.91
False Negative Rate/Miss Rate:      0.09
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
TRAINFILE="fri-c2-500-50.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50

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
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mapped to 0 and 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(result)

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mappable to 0 and 1.")
        finally:
            if (result<0 or result>1):
                raise ValueError("Alpha version restriction: Integer class labels can only be 0 or 1.")
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


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    h_0 = max((((-11.067242 * float(x[0]))+ (-10.796282 * float(x[1]))+ (-10.108956 * float(x[2]))+ (3.8093026 * float(x[3]))+ (1.2276869 * float(x[4]))+ (-0.10320607 * float(x[5]))+ (1.2363977 * float(x[6]))+ (4.0305233 * float(x[7]))+ (4.3789434 * float(x[8]))+ (-5.24719 * float(x[9]))+ (-2.9863586 * float(x[10]))+ (-0.29489625 * float(x[11]))+ (-2.9812253 * float(x[12]))+ (-5.031249 * float(x[13]))+ (-0.6920952 * float(x[14]))+ (2.5214968 * float(x[15]))+ (0.43477276 * float(x[16]))+ (-2.8145835 * float(x[17]))+ (-0.8100101 * float(x[18]))+ (4.4942946 * float(x[19]))+ (-0.4969352 * float(x[20]))+ (0.22643857 * float(x[21]))+ (-2.3381398 * float(x[22]))+ (-1.7620031 * float(x[23]))+ (-2.3606136 * float(x[24]))+ (-0.580503 * float(x[25]))+ (-0.94234234 * float(x[26]))+ (-5.21614 * float(x[27]))+ (3.7940247 * float(x[28]))+ (-3.5939724 * float(x[29]))+ (2.9612923 * float(x[30]))+ (1.8173177 * float(x[31]))+ (-1.3672489 * float(x[32]))+ (1.8477049 * float(x[33]))+ (1.1321819 * float(x[34]))+ (2.4768736 * float(x[35]))+ (3.9742534 * float(x[36]))+ (2.0124807 * float(x[37]))+ (-0.6958869 * float(x[38]))+ (2.3983192 * float(x[39]))+ (-0.7985042 * float(x[40]))+ (-3.6158729 * float(x[41]))+ (-3.1023128 * float(x[42]))+ (1.3454262 * float(x[43]))+ (-0.5294517 * float(x[44]))+ (-2.892064 * float(x[45]))+ (8.784397 * float(x[46]))+ (5.826574 * float(x[47]))+ (1.2282118 * float(x[48]))+ (-4.857531 * float(x[49]))) + 12.2620125), 0)
    h_1 = max((((2.6686358 * float(x[0]))+ (-2.534332 * float(x[1]))+ (0.03121443 * float(x[2]))+ (0.15637831 * float(x[3]))+ (2.7876592 * float(x[4]))+ (-1.0506883 * float(x[5]))+ (-2.0161314 * float(x[6]))+ (-0.40295652 * float(x[7]))+ (1.4797583 * float(x[8]))+ (1.6488419 * float(x[9]))+ (1.516777 * float(x[10]))+ (-5.3233833 * float(x[11]))+ (3.630146 * float(x[12]))+ (6.7467074 * float(x[13]))+ (-2.295288 * float(x[14]))+ (0.9079676 * float(x[15]))+ (-3.566939 * float(x[16]))+ (-1.9614645 * float(x[17]))+ (4.539278 * float(x[18]))+ (-3.1557908 * float(x[19]))+ (0.116483375 * float(x[20]))+ (0.9969969 * float(x[21]))+ (2.664635 * float(x[22]))+ (-1.3401481 * float(x[23]))+ (-1.0300472 * float(x[24]))+ (-0.10034897 * float(x[25]))+ (-0.9619921 * float(x[26]))+ (-1.6106889 * float(x[27]))+ (1.6325114 * float(x[28]))+ (2.6030655 * float(x[29]))+ (-3.5444496 * float(x[30]))+ (1.8766775 * float(x[31]))+ (-3.0400345 * float(x[32]))+ (-2.4443347 * float(x[33]))+ (-1.8095495 * float(x[34]))+ (0.13398346 * float(x[35]))+ (-2.734211 * float(x[36]))+ (-2.5228424 * float(x[37]))+ (-0.56043476 * float(x[38]))+ (0.81389743 * float(x[39]))+ (-1.8078852 * float(x[40]))+ (0.14916742 * float(x[41]))+ (-2.0178237 * float(x[42]))+ (-0.5394818 * float(x[43]))+ (1.0471454 * float(x[44]))+ (1.2105623 * float(x[45]))+ (-4.807819 * float(x[46]))+ (-3.490685 * float(x[47]))+ (2.3141587 * float(x[48]))+ (2.287566 * float(x[49]))) + -1.908694), 0)
    h_2 = max((((-1.3773441 * float(x[0]))+ (3.182933 * float(x[1]))+ (0.2426704 * float(x[2]))+ (-3.3647077 * float(x[3]))+ (3.7727437 * float(x[4]))+ (1.3807002 * float(x[5]))+ (-2.3852546 * float(x[6]))+ (-0.17351659 * float(x[7]))+ (0.46593004 * float(x[8]))+ (0.055025965 * float(x[9]))+ (-0.60657483 * float(x[10]))+ (-2.4021842 * float(x[11]))+ (1.4151562 * float(x[12]))+ (2.4449975 * float(x[13]))+ (-0.29382315 * float(x[14]))+ (2.0659804 * float(x[15]))+ (-1.300347 * float(x[16]))+ (0.8451779 * float(x[17]))+ (2.8971806 * float(x[18]))+ (-3.2551138 * float(x[19]))+ (-2.7433605 * float(x[20]))+ (1.9609054 * float(x[21]))+ (0.21031487 * float(x[22]))+ (-2.2952394 * float(x[23]))+ (-2.6851473 * float(x[24]))+ (-1.3100609 * float(x[25]))+ (-1.5391563 * float(x[26]))+ (-1.2652596 * float(x[27]))+ (-1.4148276 * float(x[28]))+ (-1.9801996 * float(x[29]))+ (0.538653 * float(x[30]))+ (1.6834596 * float(x[31]))+ (-0.23451522 * float(x[32]))+ (-1.7263361 * float(x[33]))+ (-0.049256857 * float(x[34]))+ (-0.8559466 * float(x[35]))+ (1.0708596 * float(x[36]))+ (-1.1229287 * float(x[37]))+ (0.98366207 * float(x[38]))+ (0.049879085 * float(x[39]))+ (-1.4907193 * float(x[40]))+ (-3.0357752 * float(x[41]))+ (2.7653966 * float(x[42]))+ (-2.94861 * float(x[43]))+ (-1.3996105 * float(x[44]))+ (1.0322565 * float(x[45]))+ (-4.191792 * float(x[46]))+ (-3.1384068 * float(x[47]))+ (2.5677762 * float(x[48]))+ (0.6256018 * float(x[49]))) + -1.305864), 0)
    h_3 = max((((-3.7161205 * float(x[0]))+ (-2.607068 * float(x[1]))+ (1.9287311 * float(x[2]))+ (-2.3033903 * float(x[3]))+ (-1.1832845 * float(x[4]))+ (0.37298802 * float(x[5]))+ (0.27445456 * float(x[6]))+ (0.5527911 * float(x[7]))+ (0.58421075 * float(x[8]))+ (-0.880784 * float(x[9]))+ (-0.9606599 * float(x[10]))+ (-0.61477464 * float(x[11]))+ (-0.60186285 * float(x[12]))+ (-1.6046778 * float(x[13]))+ (0.028362278 * float(x[14]))+ (-0.055890817 * float(x[15]))+ (-1.1072533 * float(x[16]))+ (-0.98258436 * float(x[17]))+ (-0.30266926 * float(x[18]))+ (0.95494884 * float(x[19]))+ (1.9867265 * float(x[20]))+ (-0.20939608 * float(x[21]))+ (-0.2572592 * float(x[22]))+ (0.23349087 * float(x[23]))+ (0.3917536 * float(x[24]))+ (0.5080143 * float(x[25]))+ (-0.0994785 * float(x[26]))+ (-2.3142114 * float(x[27]))+ (1.8108009 * float(x[28]))+ (1.160146 * float(x[29]))+ (-0.5678782 * float(x[30]))+ (1.3894011 * float(x[31]))+ (-0.4274701 * float(x[32]))+ (-0.8256566 * float(x[33]))+ (-0.43371055 * float(x[34]))+ (0.39923105 * float(x[35]))+ (0.057115056 * float(x[36]))+ (2.331099 * float(x[37]))+ (-1.363965 * float(x[38]))+ (1.6640007 * float(x[39]))+ (-0.35033783 * float(x[40]))+ (-0.30951425 * float(x[41]))+ (-2.8911595 * float(x[42]))+ (1.902472 * float(x[43]))+ (0.4719591 * float(x[44]))+ (1.2035663 * float(x[45]))+ (2.272378 * float(x[46]))+ (2.5980234 * float(x[47]))+ (0.7964984 * float(x[48]))+ (-0.56741434 * float(x[49]))) + -2.8594365), 0)
    o_0 = (0.9699589 * h_0)+ (1.0138273 * h_1)+ (-1.3953642 * h_2)+ (-4.0533056 * h_3) + -2.2701843
             
    if num_output_logits==1:
        return o_0>=0
    else:
        return argmax([eval('o'+str(i)) for i in range(num_output_logits)])

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

        model_cap=209

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

