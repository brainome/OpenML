#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 13:44:04
# Invocation: btc -v -v ailerons-1.csv -o ailerons-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                57.61%
Model accuracy:                     86.08% (11837/13750 correct)
Improvement over best guess:        28.47% (of possible 42.39%)
Model capacity (MEC):               211 bits
Generalization ratio:               56.09 bits/bit
Model efficiency:                   0.13%/parameter
System behavior
True Negatives:                     35.35% (4861/13750)
True Positives:                     50.73% (6976/13750)
False Negatives:                    6.88% (946/13750)
False Positives:                    7.03% (967/13750)
True Pos. Rate/Sensitivity/Recall:  0.88
True Neg. Rate/Specificity:         0.83
Precision:                          0.88
F-1 Measure:                        0.88
False Negative Rate/Miss Rate:      0.12
Critical Success Index:             0.78
Model bias:                         28.58% higher chance to pick class 0
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
TRAINFILE="ailerons-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 40

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
            if (not (result==0 or result==1)):
                raise ValueError("Integer class labels need to be 0 or 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                if (not (result==0 or result==1)):
                    raise ValueError("Numeric class labels need to be 0 or 1.")
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
    h_0 = max((((55.72388 * float(x[0]))+ (37.96717 * float(x[1]))+ (-45.987835 * float(x[2]))+ (-27.938118 * float(x[3]))+ (-8.197888 * float(x[4]))+ (-39.238403 * float(x[5]))+ (13.835618 * float(x[6]))+ (18.490894 * float(x[7]))+ (28.521255 * float(x[8]))+ (2.8442183 * float(x[9]))+ (-10.18738 * float(x[10]))+ (-10.258325 * float(x[11]))+ (-9.54322 * float(x[12]))+ (-11.291569 * float(x[13]))+ (-11.259387 * float(x[14]))+ (-11.340325 * float(x[15]))+ (-9.715518 * float(x[16]))+ (-9.591064 * float(x[17]))+ (-9.407362 * float(x[18]))+ (-9.036736 * float(x[19]))+ (-9.401337 * float(x[20]))+ (-9.962444 * float(x[21]))+ (-9.318549 * float(x[22]))+ (-10.491631 * float(x[23]))+ (-9.377176 * float(x[24]))+ (-0.71329343 * float(x[25]))+ (-2.5953357 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-7.879924 * float(x[28]))+ (-0.47088876 * float(x[29]))+ (-17.350864 * float(x[30]))+ (-0.08769934 * float(x[31]))+ (-14.771955 * float(x[32]))+ (-1.1777568 * float(x[33]))+ (-17.203001 * float(x[34]))+ (1.7961125 * float(x[35]))+ (-15.402266 * float(x[36]))+ (-0.6828385 * float(x[37]))+ (-8.940525 * float(x[38]))+ (-10.009162 * float(x[39]))) + 12.325568), 0)
    h_1 = max((((-62.60086 * float(x[0]))+ (3.0051787 * float(x[1]))+ (0.94544804 * float(x[2]))+ (-0.8118101 * float(x[3]))+ (5.005731 * float(x[4]))+ (0.07583081 * float(x[5]))+ (-3.3076982 * float(x[6]))+ (1.8882321 * float(x[7]))+ (-21.216108 * float(x[8]))+ (-0.54719067 * float(x[9]))+ (7.0368185 * float(x[10]))+ (7.396734 * float(x[11]))+ (6.554627 * float(x[12]))+ (6.1970882 * float(x[13]))+ (6.72363 * float(x[14]))+ (6.1816573 * float(x[15]))+ (5.7054415 * float(x[16]))+ (5.529656 * float(x[17]))+ (7.221102 * float(x[18]))+ (5.1985755 * float(x[19]))+ (7.0430965 * float(x[20]))+ (6.3050137 * float(x[21]))+ (5.2294726 * float(x[22]))+ (5.5873733 * float(x[23]))+ (18.069561 * float(x[24]))+ (-1.145685 * float(x[25]))+ (5.334344 * float(x[26]))+ (-3.5157375 * float(x[27]))+ (29.445354 * float(x[28]))+ (-7.6153345 * float(x[29]))+ (16.856573 * float(x[30]))+ (-0.491852 * float(x[31]))+ (24.456434 * float(x[32]))+ (-5.7100415 * float(x[33]))+ (28.952559 * float(x[34]))+ (-13.749125 * float(x[35]))+ (19.907608 * float(x[36]))+ (-11.851766 * float(x[37]))+ (5.0602107 * float(x[38]))+ (6.62188 * float(x[39]))) + -3.5373995), 0)
    h_2 = max((((9.478838 * float(x[0]))+ (5.9199505 * float(x[1]))+ (47.37968 * float(x[2]))+ (29.808083 * float(x[3]))+ (21.32273 * float(x[4]))+ (35.591595 * float(x[5]))+ (-21.300562 * float(x[6]))+ (2.716908 * float(x[7]))+ (-34.86304 * float(x[8]))+ (-4.9277906 * float(x[9]))+ (16.781273 * float(x[10]))+ (16.456617 * float(x[11]))+ (17.308943 * float(x[12]))+ (16.105656 * float(x[13]))+ (17.509659 * float(x[14]))+ (17.375479 * float(x[15]))+ (17.382162 * float(x[16]))+ (16.722116 * float(x[17]))+ (17.37675 * float(x[18]))+ (16.987453 * float(x[19]))+ (16.107958 * float(x[20]))+ (16.213444 * float(x[21]))+ (15.758349 * float(x[22]))+ (16.643787 * float(x[23]))+ (19.176882 * float(x[24]))+ (-0.76722586 * float(x[25]))+ (9.466601 * float(x[26]))+ (-0.19904077 * float(x[27]))+ (21.499866 * float(x[28]))+ (0.38160232 * float(x[29]))+ (33.252964 * float(x[30]))+ (0.6654611 * float(x[31]))+ (26.75719 * float(x[32]))+ (-1.7328032 * float(x[33]))+ (27.56048 * float(x[34]))+ (1.0295906 * float(x[35]))+ (29.773989 * float(x[36]))+ (0.40540364 * float(x[37]))+ (15.504226 * float(x[38]))+ (15.620199 * float(x[39]))) + -11.219873), 0)
    h_3 = max((((-2.0434456 * float(x[0]))+ (0.10922369 * float(x[1]))+ (-6.160288 * float(x[2]))+ (3.152999 * float(x[3]))+ (-2.605883 * float(x[4]))+ (-1.1332309 * float(x[5]))+ (0.7062382 * float(x[6]))+ (0.06404251 * float(x[7]))+ (21.062746 * float(x[8]))+ (0.12508665 * float(x[9]))+ (-3.9170063 * float(x[10]))+ (-4.2556825 * float(x[11]))+ (-3.633788 * float(x[12]))+ (-3.7355633 * float(x[13]))+ (-3.1442447 * float(x[14]))+ (-3.8520052 * float(x[15]))+ (-2.7786107 * float(x[16]))+ (-2.7653186 * float(x[17]))+ (-3.8667266 * float(x[18]))+ (-3.3508348 * float(x[19]))+ (-3.0015154 * float(x[20]))+ (-2.3087351 * float(x[21]))+ (-3.664368 * float(x[22]))+ (-3.123972 * float(x[23]))+ (-22.472595 * float(x[24]))+ (11.980357 * float(x[25]))+ (-9.837193 * float(x[26]))+ (24.993351 * float(x[27]))+ (-33.638416 * float(x[28]))+ (29.566576 * float(x[29]))+ (-20.868763 * float(x[30]))+ (15.847634 * float(x[31]))+ (-28.284842 * float(x[32]))+ (27.859337 * float(x[33]))+ (-33.498943 * float(x[34]))+ (37.36169 * float(x[35]))+ (-23.156301 * float(x[36]))+ (31.266663 * float(x[37]))+ (-3.2607446 * float(x[38]))+ (-2.092594 * float(x[39]))) + 8.457528), 0)
    h_4 = max((((0.0064890976 * float(x[0]))+ (0.051300082 * float(x[1]))+ (-0.31798127 * float(x[2]))+ (-0.22425511 * float(x[3]))+ (-0.16616671 * float(x[4]))+ (-0.056841753 * float(x[5]))+ (1.0345035 * float(x[6]))+ (0.14355296 * float(x[7]))+ (0.5836807 * float(x[8]))+ (0.7382789 * float(x[9]))+ (-0.8556103 * float(x[10]))+ (-0.13112138 * float(x[11]))+ (-1.2180704 * float(x[12]))+ (0.07033205 * float(x[13]))+ (-0.08925243 * float(x[14]))+ (-0.22339301 * float(x[15]))+ (-1.216287 * float(x[16]))+ (0.1463574 * float(x[17]))+ (-1.3032929 * float(x[18]))+ (-1.4232694 * float(x[19]))+ (0.051537074 * float(x[20]))+ (0.08373169 * float(x[21]))+ (-0.11134191 * float(x[22]))+ (0.44643706 * float(x[23]))+ (0.08941037 * float(x[24]))+ (0.23445912 * float(x[25]))+ (-0.3402604 * float(x[26]))+ (0.10476287 * float(x[27]))+ (-0.25697792 * float(x[28]))+ (0.7945974 * float(x[29]))+ (-0.6903355 * float(x[30]))+ (0.31814724 * float(x[31]))+ (0.41425127 * float(x[32]))+ (-0.17195491 * float(x[33]))+ (-1.1983169 * float(x[34]))+ (-0.46755967 * float(x[35]))+ (-0.303565 * float(x[36]))+ (0.8509379 * float(x[37]))+ (-1.0564033 * float(x[38]))+ (-0.7427274 * float(x[39]))) + -0.09974395), 0)
    o_0 = (0.0033878896 * h_0)+ (-0.030145228 * h_1)+ (-0.019947072 * h_2)+ (0.9244508 * h_3)+ (-1.2156088 * h_4) + 5.7971396
             
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

        model_cap=211

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

