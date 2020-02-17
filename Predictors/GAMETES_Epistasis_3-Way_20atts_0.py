#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-11-2020 19:09:47
# Invocation: btc -v GAMETES_Epistasis_3-Way_20atts_0-1.csv -o GAMETES_Epistasis_3-Way_20atts_0-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     63.37% (1014/1600 correct)
Improvement over best guess:        13.37% (of possible 50.0%)
Model capacity (MEC):               221 bits
Generalization ratio:               4.58 bits/bit
Model efficiency:                   0.06%/parameter
System behavior
True Negatives:                     35.94% (575/1600)
True Positives:                     27.44% (439/1600)
False Negatives:                    22.56% (361/1600)
False Positives:                    14.06% (225/1600)
True Pos. Rate/Sensitivity/Recall:  0.55
True Neg. Rate/Specificity:         0.72
Precision:                          0.66
F-1 Measure:                        0.60
False Negative Rate/Miss Rate:      0.45
Critical Success Index:             0.43
Model bias:                         0.82% higher chance to pick class 1
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
TRAINFILE="GAMETES_Epistasis_3-Way_20atts_0-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 20

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
    h_0 = max((((-6.1355414 * float(x[0]))+ (-4.0926766 * float(x[1]))+ (0.4152905 * float(x[2]))+ (-8.269476 * float(x[3]))+ (-5.7026067 * float(x[4]))+ (-10.55503 * float(x[5]))+ (7.6253166 * float(x[6]))+ (-1.4141212 * float(x[7]))+ (0.17362964 * float(x[8]))+ (-7.3667827 * float(x[9]))+ (1.1207811 * float(x[10]))+ (2.6075633 * float(x[11]))+ (9.529397 * float(x[12]))+ (4.60173 * float(x[13]))+ (-4.9220667 * float(x[14]))+ (-12.589013 * float(x[15]))+ (-3.8404737 * float(x[16]))+ (-8.9948435 * float(x[17]))+ (2.404628 * float(x[18]))+ (7.346297 * float(x[19]))) + -1.51123), 0)
    h_1 = max((((0.7750705 * float(x[0]))+ (5.79859 * float(x[1]))+ (1.2211245 * float(x[2]))+ (2.670842 * float(x[3]))+ (1.1025409 * float(x[4]))+ (3.859125 * float(x[5]))+ (-1.5556787 * float(x[6]))+ (-2.836515 * float(x[7]))+ (0.7526868 * float(x[8]))+ (3.5552306 * float(x[9]))+ (2.114656 * float(x[10]))+ (-0.74987894 * float(x[11]))+ (-1.5916387 * float(x[12]))+ (-1.9444497 * float(x[13]))+ (0.6318053 * float(x[14]))+ (8.628838 * float(x[15]))+ (3.8886163 * float(x[16]))+ (5.2889175 * float(x[17]))+ (-6.5977645 * float(x[18]))+ (-8.003813 * float(x[19]))) + -0.019375373), 0)
    h_2 = max((((-0.22278537 * float(x[0]))+ (-1.0890049 * float(x[1]))+ (1.1715717 * float(x[2]))+ (-2.2441206 * float(x[3]))+ (3.0449526 * float(x[4]))+ (0.6074746 * float(x[5]))+ (1.9532415 * float(x[6]))+ (3.2023077 * float(x[7]))+ (2.3998878 * float(x[8]))+ (2.8252423 * float(x[9]))+ (0.17437238 * float(x[10]))+ (0.5119929 * float(x[11]))+ (3.6120634 * float(x[12]))+ (-0.3786903 * float(x[13]))+ (2.703959 * float(x[14]))+ (-2.311137 * float(x[15]))+ (-3.026312 * float(x[16]))+ (1.5878217 * float(x[17]))+ (1.4212921 * float(x[18]))+ (1.4510195 * float(x[19]))) + 2.9924288), 0)
    h_3 = max((((0.18949956 * float(x[0]))+ (-0.053194147 * float(x[1]))+ (0.5272765 * float(x[2]))+ (-1.2136347 * float(x[3]))+ (0.9114579 * float(x[4]))+ (-0.31595242 * float(x[5]))+ (0.0912323 * float(x[6]))+ (2.609997 * float(x[7]))+ (0.54266465 * float(x[8]))+ (0.22084336 * float(x[9]))+ (0.84699786 * float(x[10]))+ (-0.8036175 * float(x[11]))+ (-0.86393636 * float(x[12]))+ (0.546127 * float(x[13]))+ (2.244066 * float(x[14]))+ (0.09754197 * float(x[15]))+ (-0.5697397 * float(x[16]))+ (0.71952534 * float(x[17]))+ (1.0246145 * float(x[18]))+ (1.6275654 * float(x[19]))) + 0.005040768), 0)
    h_4 = max((((-0.38502413 * float(x[0]))+ (-0.26545215 * float(x[1]))+ (-0.90519553 * float(x[2]))+ (1.7876489 * float(x[3]))+ (-1.4251031 * float(x[4]))+ (-0.38278687 * float(x[5]))+ (0.3667881 * float(x[6]))+ (2.8591552 * float(x[7]))+ (-0.7709949 * float(x[8]))+ (-1.1638155 * float(x[9]))+ (1.8093936 * float(x[10]))+ (-1.8519285 * float(x[11]))+ (-0.15880786 * float(x[12]))+ (1.2706618 * float(x[13]))+ (2.008066 * float(x[14]))+ (0.8784658 * float(x[15]))+ (1.338602 * float(x[16]))+ (0.30171773 * float(x[17]))+ (-0.19393751 * float(x[18]))+ (3.0851445 * float(x[19]))) + 2.0862696), 0)
    h_5 = max((((0.34738246 * float(x[0]))+ (1.6964903 * float(x[1]))+ (0.17037912 * float(x[2]))+ (0.78452945 * float(x[3]))+ (-0.4673096 * float(x[4]))+ (1.2385358 * float(x[5]))+ (0.010931686 * float(x[6]))+ (-0.76789397 * float(x[7]))+ (0.48139668 * float(x[8]))+ (1.8773563 * float(x[9]))+ (0.706929 * float(x[10]))+ (-0.014118313 * float(x[11]))+ (0.4751124 * float(x[12]))+ (-1.9244711 * float(x[13]))+ (-0.21765596 * float(x[14]))+ (1.7632338 * float(x[15]))+ (1.2675571 * float(x[16]))+ (2.26254 * float(x[17]))+ (-3.0475717 * float(x[18]))+ (-2.9664252 * float(x[19]))) + 1.1497488), 0)
    h_6 = max((((0.32752705 * float(x[0]))+ (1.060268 * float(x[1]))+ (0.24092677 * float(x[2]))+ (0.8142517 * float(x[3]))+ (0.4545262 * float(x[4]))+ (1.2630304 * float(x[5]))+ (-0.06015875 * float(x[6]))+ (-1.0528835 * float(x[7]))+ (0.037238333 * float(x[8]))+ (1.4610084 * float(x[9]))+ (0.5375512 * float(x[10]))+ (0.021418266 * float(x[11]))+ (1.066276 * float(x[12]))+ (-0.2300749 * float(x[13]))+ (-0.053756326 * float(x[14]))+ (1.8683004 * float(x[15]))+ (0.39807245 * float(x[16]))+ (1.29469 * float(x[17]))+ (-1.4639139 * float(x[18]))+ (-1.4685547 * float(x[19]))) + 0.03543383), 0)
    h_7 = max((((-0.76808625 * float(x[0]))+ (1.5500871 * float(x[1]))+ (0.16045435 * float(x[2]))+ (1.8675932 * float(x[3]))+ (0.54572725 * float(x[4]))+ (0.95895314 * float(x[5]))+ (-0.38056204 * float(x[6]))+ (-0.56635445 * float(x[7]))+ (0.26464957 * float(x[8]))+ (-0.5308393 * float(x[9]))+ (0.35559386 * float(x[10]))+ (-0.42187548 * float(x[11]))+ (-0.83867407 * float(x[12]))+ (0.62595063 * float(x[13]))+ (0.5343039 * float(x[14]))+ (2.860888 * float(x[15]))+ (1.6590619 * float(x[16]))+ (0.47216132 * float(x[17]))+ (-1.4407669 * float(x[18]))+ (-1.6683655 * float(x[19]))) + 0.043578982), 0)
    h_8 = max((((-0.9539652 * float(x[0]))+ (-0.56421924 * float(x[1]))+ (-1.6950773 * float(x[2]))+ (-1.4817194 * float(x[3]))+ (0.9506217 * float(x[4]))+ (-1.8162813 * float(x[5]))+ (-0.019966347 * float(x[6]))+ (-0.8372673 * float(x[7]))+ (1.6256078 * float(x[8]))+ (-1.5148264 * float(x[9]))+ (-2.6746378 * float(x[10]))+ (-1.214006 * float(x[11]))+ (1.2890712 * float(x[12]))+ (-1.7259371 * float(x[13]))+ (0.09836853 * float(x[14]))+ (-2.9208593 * float(x[15]))+ (-0.58998716 * float(x[16]))+ (2.2330756 * float(x[17]))+ (-1.25766 * float(x[18]))+ (0.7484232 * float(x[19]))) + -1.1717758), 0)
    h_9 = max((((0.40682456 * float(x[0]))+ (-0.1863004 * float(x[1]))+ (0.020933084 * float(x[2]))+ (-0.8102755 * float(x[3]))+ (0.07408586 * float(x[4]))+ (-0.98222846 * float(x[5]))+ (-1.6831613 * float(x[6]))+ (0.3700982 * float(x[7]))+ (-0.6580265 * float(x[8]))+ (0.1021673 * float(x[9]))+ (0.076008305 * float(x[10]))+ (-0.42235053 * float(x[11]))+ (0.2169842 * float(x[12]))+ (1.0217205 * float(x[13]))+ (0.20337245 * float(x[14]))+ (0.48640925 * float(x[15]))+ (-0.71465933 * float(x[16]))+ (-0.08978544 * float(x[17]))+ (0.67354393 * float(x[18]))+ (1.0324717 * float(x[19]))) + -1.6466583), 0)
    o_0 = (20.609558 * h_0)+ (-2.5706542 * h_1)+ (-1.3457086 * h_2)+ (3.4809988 * h_3)+ (-1.3830978 * h_4)+ (3.1160948 * h_5)+ (4.076068 * h_6)+ (2.5155861 * h_7)+ (32.83575 * h_8)+ (-3.406321 * h_9) + 2.1288474
             
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

        model_cap=221

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

