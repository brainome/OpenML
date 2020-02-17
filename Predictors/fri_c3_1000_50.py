#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 03:23:22
# Invocation: btc -v fri_c3_1000_50-8.csv -o fri_c3_1000_50-8.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.50%
Model accuracy:                     91.80% (918/1000 correct)
Improvement over best guess:        36.30% (of possible 44.5%)
Model capacity (MEC):               261 bits
Generalization ratio:               3.51 bits/bit
Model efficiency:                   0.13%/parameter
System behavior
True Negatives:                     52.10% (521/1000)
True Positives:                     39.70% (397/1000)
False Negatives:                    4.80% (48/1000)
False Positives:                    3.40% (34/1000)
True Pos. Rate/Sensitivity/Recall:  0.89
True Neg. Rate/Specificity:         0.94
Precision:                          0.92
F-1 Measure:                        0.91
False Negative Rate/Miss Rate:      0.11
Critical Success Index:             0.83
Model bias:                         0.08% higher chance to pick class 0
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
TRAINFILE="fri_c3_1000_50-8.csv"


#Number of output logits
num_output_logits = 1

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
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
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
    h_0 = max((((18.517277 * float(x[0]))+ (22.470158 * float(x[1]))+ (4.710969 * float(x[2]))+ (-10.593464 * float(x[3]))+ (6.1901717 * float(x[4]))+ (-3.2417164 * float(x[5]))+ (-1.6067001 * float(x[6]))+ (0.5836885 * float(x[7]))+ (-2.5954247 * float(x[8]))+ (0.09930889 * float(x[9]))+ (-0.9810349 * float(x[10]))+ (1.8604385 * float(x[11]))+ (0.18126103 * float(x[12]))+ (2.1558354 * float(x[13]))+ (-1.0251226 * float(x[14]))+ (0.35233465 * float(x[15]))+ (-0.19741653 * float(x[16]))+ (2.028754 * float(x[17]))+ (-0.40511826 * float(x[18]))+ (-1.7552575 * float(x[19]))+ (-1.847514 * float(x[20]))+ (-3.3058941 * float(x[21]))+ (-1.1108369 * float(x[22]))+ (2.286316 * float(x[23]))+ (-0.20395242 * float(x[24]))+ (2.7377903 * float(x[25]))+ (0.37585482 * float(x[26]))+ (3.1364892 * float(x[27]))+ (0.8942351 * float(x[28]))+ (-0.78304124 * float(x[29]))+ (1.0757996 * float(x[30]))+ (1.4623432 * float(x[31]))+ (-0.52370507 * float(x[32]))+ (-0.26008907 * float(x[33]))+ (1.3704618 * float(x[34]))+ (2.1475098 * float(x[35]))+ (-2.579253 * float(x[36]))+ (-1.6176671 * float(x[37]))+ (-4.3533745 * float(x[38]))+ (1.8493842 * float(x[39]))+ (-1.5775343 * float(x[40]))+ (0.37746757 * float(x[41]))+ (-0.66024226 * float(x[42]))+ (-1.3763009 * float(x[43]))+ (-0.7260845 * float(x[44]))+ (2.6712525 * float(x[45]))+ (2.8579304 * float(x[46]))+ (-1.4845594 * float(x[47]))+ (-2.2306263 * float(x[48]))+ (0.011990182 * float(x[49]))) + 8.346374), 0)
    h_1 = max((((-7.0649 * float(x[0]))+ (-6.4208097 * float(x[1]))+ (3.3302448 * float(x[2]))+ (-6.742551 * float(x[3]))+ (-2.5188086 * float(x[4]))+ (-0.0067419666 * float(x[5]))+ (-0.23171136 * float(x[6]))+ (-2.6607504 * float(x[7]))+ (-2.2382004 * float(x[8]))+ (-0.4552356 * float(x[9]))+ (-0.6978651 * float(x[10]))+ (0.40025508 * float(x[11]))+ (-0.62594837 * float(x[12]))+ (-0.2574496 * float(x[13]))+ (0.5094954 * float(x[14]))+ (-0.33805758 * float(x[15]))+ (-0.4878048 * float(x[16]))+ (0.1225376 * float(x[17]))+ (-0.6504973 * float(x[18]))+ (-0.17203029 * float(x[19]))+ (0.66562444 * float(x[20]))+ (1.074933 * float(x[21]))+ (0.022136228 * float(x[22]))+ (-0.41187605 * float(x[23]))+ (-0.31531328 * float(x[24]))+ (-0.2577456 * float(x[25]))+ (1.8485994 * float(x[26]))+ (0.2857046 * float(x[27]))+ (-1.3803807 * float(x[28]))+ (0.35037643 * float(x[29]))+ (-1.4075184 * float(x[30]))+ (0.67538977 * float(x[31]))+ (-0.8319171 * float(x[32]))+ (0.83786386 * float(x[33]))+ (-0.40066475 * float(x[34]))+ (2.1516516 * float(x[35]))+ (-0.5427667 * float(x[36]))+ (0.6676206 * float(x[37]))+ (1.1980765 * float(x[38]))+ (-0.13167483 * float(x[39]))+ (-0.34054086 * float(x[40]))+ (1.3010911 * float(x[41]))+ (0.85904115 * float(x[42]))+ (0.8204455 * float(x[43]))+ (0.059932724 * float(x[44]))+ (0.54427135 * float(x[45]))+ (0.15326865 * float(x[46]))+ (-0.7148091 * float(x[47]))+ (-1.228887 * float(x[48]))+ (1.0993799 * float(x[49]))) + -18.219658), 0)
    h_2 = max((((3.0638475 * float(x[0]))+ (6.057858 * float(x[1]))+ (2.3476782 * float(x[2]))+ (1.6394264 * float(x[3]))+ (2.2625403 * float(x[4]))+ (-2.1210055 * float(x[5]))+ (-0.6124007 * float(x[6]))+ (-1.0730312 * float(x[7]))+ (-1.209169 * float(x[8]))+ (2.584494 * float(x[9]))+ (-0.8769131 * float(x[10]))+ (-0.13776025 * float(x[11]))+ (-1.6235607 * float(x[12]))+ (1.7230753 * float(x[13]))+ (0.31550398 * float(x[14]))+ (-0.1036678 * float(x[15]))+ (2.4638941 * float(x[16]))+ (2.074462 * float(x[17]))+ (0.19890772 * float(x[18]))+ (0.69707716 * float(x[19]))+ (-0.7648179 * float(x[20]))+ (0.47549093 * float(x[21]))+ (-0.26673838 * float(x[22]))+ (-0.3344493 * float(x[23]))+ (0.73872507 * float(x[24]))+ (-1.385014 * float(x[25]))+ (1.5300813 * float(x[26]))+ (-0.406512 * float(x[27]))+ (-0.30168074 * float(x[28]))+ (-0.84066767 * float(x[29]))+ (-1.7224268 * float(x[30]))+ (-1.4072239 * float(x[31]))+ (-2.277493 * float(x[32]))+ (0.58013266 * float(x[33]))+ (-2.5802326 * float(x[34]))+ (-1.4700068 * float(x[35]))+ (-0.18138497 * float(x[36]))+ (0.44303724 * float(x[37]))+ (-2.4223762 * float(x[38]))+ (1.0085088 * float(x[39]))+ (-0.20772882 * float(x[40]))+ (0.750844 * float(x[41]))+ (-0.8628692 * float(x[42]))+ (-3.2938976 * float(x[43]))+ (0.22132966 * float(x[44]))+ (-0.30478463 * float(x[45]))+ (0.04453976 * float(x[46]))+ (-0.65940607 * float(x[47]))+ (-2.7761066 * float(x[48]))+ (1.9326613 * float(x[49]))) + -3.338433), 0)
    h_3 = max((((6.471045 * float(x[0]))+ (5.7962894 * float(x[1]))+ (0.61451423 * float(x[2]))+ (-6.0316043 * float(x[3]))+ (1.7213398 * float(x[4]))+ (2.0946062 * float(x[5]))+ (-2.4010763 * float(x[6]))+ (0.64387953 * float(x[7]))+ (0.020938054 * float(x[8]))+ (1.5501426 * float(x[9]))+ (-0.16550438 * float(x[10]))+ (0.45596153 * float(x[11]))+ (0.37946087 * float(x[12]))+ (0.6374136 * float(x[13]))+ (0.8274058 * float(x[14]))+ (1.4862627 * float(x[15]))+ (-2.106944 * float(x[16]))+ (-0.72543085 * float(x[17]))+ (-0.21992639 * float(x[18]))+ (-0.8111946 * float(x[19]))+ (0.5388984 * float(x[20]))+ (-0.84478205 * float(x[21]))+ (0.42208454 * float(x[22]))+ (0.52098656 * float(x[23]))+ (1.8155032 * float(x[24]))+ (1.9081113 * float(x[25]))+ (-0.99896836 * float(x[26]))+ (-0.94989765 * float(x[27]))+ (-1.5286037 * float(x[28]))+ (0.12217593 * float(x[29]))+ (1.1614538 * float(x[30]))+ (-0.13236834 * float(x[31]))+ (0.24635918 * float(x[32]))+ (0.240327 * float(x[33]))+ (0.9669296 * float(x[34]))+ (-0.7566612 * float(x[35]))+ (-1.4053171 * float(x[36]))+ (-1.2214402 * float(x[37]))+ (-0.99545795 * float(x[38]))+ (1.8758644 * float(x[39]))+ (-0.7778817 * float(x[40]))+ (1.3704565 * float(x[41]))+ (-0.40892756 * float(x[42]))+ (-2.284951 * float(x[43]))+ (0.09591732 * float(x[44]))+ (-0.89233476 * float(x[45]))+ (1.2744902 * float(x[46]))+ (-0.11160751 * float(x[47]))+ (-0.19880885 * float(x[48]))+ (-2.0898845 * float(x[49]))) + -3.0904953), 0)
    h_4 = max((((5.7864738 * float(x[0]))+ (6.627446 * float(x[1]))+ (1.8264645 * float(x[2]))+ (-2.297446 * float(x[3]))+ (2.019468 * float(x[4]))+ (-2.2008574 * float(x[5]))+ (0.87085617 * float(x[6]))+ (0.6326213 * float(x[7]))+ (-0.7223168 * float(x[8]))+ (-2.2408051 * float(x[9]))+ (-0.19548295 * float(x[10]))+ (0.866619 * float(x[11]))+ (0.7683763 * float(x[12]))+ (-0.28004497 * float(x[13]))+ (-0.97001916 * float(x[14]))+ (-1.1398858 * float(x[15]))+ (0.31536344 * float(x[16]))+ (0.7287265 * float(x[17]))+ (-0.0376568 * float(x[18]))+ (-0.8289105 * float(x[19]))+ (-0.83367044 * float(x[20]))+ (-1.083566 * float(x[21]))+ (-0.8711725 * float(x[22]))+ (1.113582 * float(x[23]))+ (-1.9660622 * float(x[24]))+ (0.7015052 * float(x[25]))+ (0.005047319 * float(x[26]))+ (2.819485 * float(x[27]))+ (1.4067385 * float(x[28]))+ (-0.7367944 * float(x[29]))+ (-0.21726649 * float(x[30]))+ (1.5372366 * float(x[31]))+ (0.22695962 * float(x[32]))+ (-1.1325266 * float(x[33]))+ (1.442023 * float(x[34]))+ (2.3444602 * float(x[35]))+ (0.26598558 * float(x[36]))+ (-0.2197739 * float(x[37]))+ (-0.68884534 * float(x[38]))+ (-0.3403709 * float(x[39]))+ (0.0038378267 * float(x[40]))+ (-1.8330005 * float(x[41]))+ (0.20379585 * float(x[42]))+ (2.9098659 * float(x[43]))+ (-1.2117912 * float(x[44]))+ (1.9998965 * float(x[45]))+ (-0.0063759503 * float(x[46]))+ (-0.44701484 * float(x[47]))+ (0.9075479 * float(x[48]))+ (1.3643913 * float(x[49]))) + -3.819168), 0)
    o_0 = (2.9856534 * h_0)+ (7.318558 * h_1)+ (-3.3590467 * h_2)+ (-4.826068 * h_3)+ (-5.8862066 * h_4) + -4.021504
             
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

        model_cap=261

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

