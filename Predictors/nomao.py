#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 16:35:55
# Invocation: btc -v -v nomao-1.csv -o nomao-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                71.43%
Model accuracy:                     94.92% (32715/34465 correct)
Improvement over best guess:        23.49% (of possible 28.57%)
Model capacity (MEC):               241 bits
Generalization ratio:               135.74 bits/bit
Model efficiency:                   0.09%/parameter
System behavior
True Negatives:                     68.93% (23756/34465)
True Positives:                     25.99% (8959/34465)
False Negatives:                    2.57% (885/34465)
False Positives:                    2.51% (865/34465)
True Pos. Rate/Sensitivity/Recall:  0.91
True Neg. Rate/Specificity:         0.96
Precision:                          0.91
F-1 Measure:                        0.91
False Negative Rate/Miss Rate:      0.09
Critical Success Index:             0.84
Model bias:                         40.68% higher chance to pick class 0
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
TRAINFILE="nomao-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 118

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
    h_0 = max((((-0.37987915 * float(x[0]))+ (0.52424663 * float(x[1]))+ (0.35029426 * float(x[2]))+ (-1.4613814 * float(x[3]))+ (0.5606927 * float(x[4]))+ (-1.6809903 * float(x[5]))+ (-0.33450818 * float(x[6]))+ (0.6382741 * float(x[7]))+ (-0.2347059 * float(x[8]))+ (1.2353122 * float(x[9]))+ (-0.2617 * float(x[10]))+ (-0.66710013 * float(x[11]))+ (-0.05877537 * float(x[12]))+ (-0.3081739 * float(x[13]))+ (-0.46538386 * float(x[14]))+ (0.5436225 * float(x[15]))+ (-1.339748 * float(x[16]))+ (-0.38497362 * float(x[17]))+ (0.44845945 * float(x[18]))+ (0.7862815 * float(x[19]))+ (0.2556759 * float(x[20]))+ (0.6060658 * float(x[21]))+ (0.13305648 * float(x[22]))+ (-0.045894478 * float(x[23]))+ (-0.42361504 * float(x[24]))+ (-0.20768265 * float(x[25]))+ (1.2015107 * float(x[26]))+ (0.37724316 * float(x[27]))+ (-0.04280521 * float(x[28]))+ (-1.2352142 * float(x[29]))+ (0.03255397 * float(x[30]))+ (-0.057231378 * float(x[31]))+ (-0.4117796 * float(x[32]))+ (0.421671 * float(x[33]))+ (1.0535408 * float(x[34]))+ (0.8233607 * float(x[35]))+ (0.15915117 * float(x[36]))+ (-1.6130636 * float(x[37]))+ (-0.4242852 * float(x[38]))+ (0.12840334 * float(x[39]))+ (0.2121086 * float(x[40]))+ (0.7333053 * float(x[41]))+ (-0.69895273 * float(x[42]))+ (0.3341502 * float(x[43]))+ (0.30375233 * float(x[44]))+ (-0.56208473 * float(x[45]))+ (-0.25631413 * float(x[46]))+ (0.116693325 * float(x[47]))+ (0.89017755 * float(x[48]))+ (-0.9911113 * float(x[49])))+ ((1.0666019 * float(x[50]))+ (1.0491465 * float(x[51]))+ (-0.9099677 * float(x[52]))+ (-1.0873501 * float(x[53]))+ (0.2630289 * float(x[54]))+ (0.04909956 * float(x[55]))+ (0.3309498 * float(x[56]))+ (1.5168397 * float(x[57]))+ (-1.0081365 * float(x[58]))+ (-2.701512 * float(x[59]))+ (2.5889843 * float(x[60]))+ (-1.3404262 * float(x[61]))+ (0.058753632 * float(x[62]))+ (0.41810167 * float(x[63]))+ (-0.635992 * float(x[64]))+ (0.21541631 * float(x[65]))+ (1.3655415 * float(x[66]))+ (-0.37731987 * float(x[67]))+ (1.7155502 * float(x[68]))+ (-2.0173116 * float(x[69]))+ (0.17933184 * float(x[70]))+ (-0.36897045 * float(x[71]))+ (-0.10462623 * float(x[72]))+ (1.8154702 * float(x[73]))+ (1.7189902 * float(x[74]))+ (-0.8332752 * float(x[75]))+ (-1.3851132 * float(x[76]))+ (-1.2213947 * float(x[77]))+ (-0.87081516 * float(x[78]))+ (0.72499734 * float(x[79]))+ (0.39267984 * float(x[80]))+ (-0.30755195 * float(x[81]))+ (0.9490833 * float(x[82]))+ (0.6973517 * float(x[83]))+ (0.09492732 * float(x[84]))+ (0.61064374 * float(x[85]))+ (-0.4780277 * float(x[86]))+ (0.4859829 * float(x[87]))+ (-0.7048877 * float(x[88]))+ (-0.2111559 * float(x[89]))+ (-0.89307106 * float(x[90]))+ (-0.4485226 * float(x[91]))+ (0.0814684 * float(x[92]))+ (-0.12478364 * float(x[93]))+ (-1.1329134 * float(x[94]))+ (0.41549942 * float(x[95]))+ (-0.6332652 * float(x[96]))+ (-1.702541 * float(x[97]))+ (1.1834639 * float(x[98]))+ (-0.7272671 * float(x[99])))+ ((1.2781695 * float(x[100]))+ (0.07400371 * float(x[101]))+ (-1.156966 * float(x[102]))+ (0.5838892 * float(x[103]))+ (0.14828584 * float(x[104]))+ (-0.21504724 * float(x[105]))+ (-0.42191854 * float(x[106]))+ (0.7079524 * float(x[107]))+ (-0.655022 * float(x[108]))+ (-0.8233363 * float(x[109]))+ (0.51112866 * float(x[110]))+ (-0.3917811 * float(x[111]))+ (0.30764595 * float(x[112]))+ (-0.68984777 * float(x[113]))+ (0.22842255 * float(x[114]))+ (0.3628458 * float(x[115]))+ (-1.0461807 * float(x[116]))+ (0.53708655 * float(x[117]))) + 2.2822342), 0)
    h_1 = max((((-0.5498634 * float(x[0]))+ (-0.4446335 * float(x[1]))+ (0.46072435 * float(x[2]))+ (-0.6831014 * float(x[3]))+ (0.6980394 * float(x[4]))+ (-0.18465264 * float(x[5]))+ (-0.19340691 * float(x[6]))+ (-1.019854 * float(x[7]))+ (-1.1177644 * float(x[8]))+ (-0.18901213 * float(x[9]))+ (-0.6952877 * float(x[10]))+ (0.8230339 * float(x[11]))+ (0.13876444 * float(x[12]))+ (-0.4833742 * float(x[13]))+ (0.31240875 * float(x[14]))+ (-1.0792685 * float(x[15]))+ (-0.8399744 * float(x[16]))+ (0.10512753 * float(x[17]))+ (-0.54967237 * float(x[18]))+ (-0.58287925 * float(x[19]))+ (-0.16412805 * float(x[20]))+ (0.27884486 * float(x[21]))+ (0.7898826 * float(x[22]))+ (0.03009147 * float(x[23]))+ (0.82131463 * float(x[24]))+ (-0.21954402 * float(x[25]))+ (-0.626488 * float(x[26]))+ (0.3339169 * float(x[27]))+ (-0.8604129 * float(x[28]))+ (-0.7121102 * float(x[29]))+ (0.44977108 * float(x[30]))+ (0.46690258 * float(x[31]))+ (-0.5539368 * float(x[32]))+ (-0.72800344 * float(x[33]))+ (0.037725333 * float(x[34]))+ (0.8583759 * float(x[35]))+ (-0.9792889 * float(x[36]))+ (0.011814592 * float(x[37]))+ (0.5099778 * float(x[38]))+ (-0.8574807 * float(x[39]))+ (-0.3742802 * float(x[40]))+ (0.26717117 * float(x[41]))+ (0.67276543 * float(x[42]))+ (-0.5090652 * float(x[43]))+ (-0.020799499 * float(x[44]))+ (0.7711587 * float(x[45]))+ (-1.1454159 * float(x[46]))+ (0.010909535 * float(x[47]))+ (-0.6043604 * float(x[48]))+ (-0.3160706 * float(x[49])))+ ((-0.32235977 * float(x[50]))+ (-0.4165524 * float(x[51]))+ (0.7957955 * float(x[52]))+ (-0.20396227 * float(x[53]))+ (-1.520607 * float(x[54]))+ (0.30548444 * float(x[55]))+ (-0.02254105 * float(x[56]))+ (0.5966153 * float(x[57]))+ (-0.05162904 * float(x[58]))+ (0.43953857 * float(x[59]))+ (-0.5028887 * float(x[60]))+ (-0.68370396 * float(x[61]))+ (-0.29974258 * float(x[62]))+ (-0.001660407 * float(x[63]))+ (-0.32163227 * float(x[64]))+ (0.6330574 * float(x[65]))+ (-0.9755938 * float(x[66]))+ (-0.61419344 * float(x[67]))+ (-0.009700873 * float(x[68]))+ (0.8603135 * float(x[69]))+ (-0.4525168 * float(x[70]))+ (0.48605785 * float(x[71]))+ (0.31711867 * float(x[72]))+ (-1.1012619 * float(x[73]))+ (-0.9586829 * float(x[74]))+ (0.69292545 * float(x[75]))+ (-0.7410149 * float(x[76]))+ (-1.0670918 * float(x[77]))+ (0.12699486 * float(x[78]))+ (-1.0981557 * float(x[79]))+ (0.2382512 * float(x[80]))+ (0.68102163 * float(x[81]))+ (-0.621201 * float(x[82]))+ (-0.15033577 * float(x[83]))+ (0.23068503 * float(x[84]))+ (-0.7111092 * float(x[85]))+ (-0.35066047 * float(x[86]))+ (0.07661867 * float(x[87]))+ (0.3136391 * float(x[88]))+ (-0.040976636 * float(x[89]))+ (-0.033688623 * float(x[90]))+ (-0.38633204 * float(x[91]))+ (-0.83614916 * float(x[92]))+ (0.2122315 * float(x[93]))+ (-0.33703062 * float(x[94]))+ (-0.7668441 * float(x[95]))+ (-0.584719 * float(x[96]))+ (-0.14880295 * float(x[97]))+ (-0.7889511 * float(x[98]))+ (0.3997898 * float(x[99])))+ ((0.12034268 * float(x[100]))+ (0.59362507 * float(x[101]))+ (0.85726434 * float(x[102]))+ (-0.5628389 * float(x[103]))+ (-0.12178711 * float(x[104]))+ (0.019057972 * float(x[105]))+ (-0.7636365 * float(x[106]))+ (-1.318541 * float(x[107]))+ (-0.91128796 * float(x[108]))+ (-0.8569101 * float(x[109]))+ (-0.1636196 * float(x[110]))+ (0.44526082 * float(x[111]))+ (0.41542217 * float(x[112]))+ (0.2028698 * float(x[113]))+ (-0.10646242 * float(x[114]))+ (-0.64101565 * float(x[115]))+ (0.22891292 * float(x[116]))+ (0.25634333 * float(x[117]))) + -1.0985968), 0)
    o_0 = (2.805571 * h_0)+ (0.019720105 * h_1) + -8.1274805
             
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

        model_cap=241

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

