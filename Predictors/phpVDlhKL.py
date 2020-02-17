#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-12-2020 03:56:13
# Invocation: btc -v phpVDlhKL-1.csv -o phpVDlhKL-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                54.68%
Model accuracy:                     54.68% (35/64 correct)
Improvement over best guess:        0.00% (of possible 45.32%)
Model capacity (MEC):               230 bits
Generalization ratio:               0.15 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     0.00% (0/64)
True Positives:                     54.69% (35/64)
False Negatives:                    0.00% (0/64)
False Positives:                    45.31% (29/64)
True Pos. Rate/Sensitivity/Recall:  1.00
True Neg. Rate/Specificity:         0.00
Precision:                          0.55
F-1 Measure:                        0.71
False Negative Rate/Miss Rate:      0.00
Critical Success Index:             0.55
Model bias:                         1.00% higher chance to pick class 1
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
TRAINFILE="phpVDlhKL-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 229

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
    h_0 = ((((0.43037874 * float(x[0]))+ (0.20552675 * float(x[1]))+ (0.08976637 * float(x[2]))+ (-0.1526904 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.124825574 * float(x[5]))+ (0.78354603 * float(x[6]))+ (0.92732555 * float(x[7]))+ (-0.23311697 * float(x[8]))+ (0.5834501 * float(x[9]))+ (0.05778984 * float(x[10]))+ (0.13608912 * float(x[11]))+ (0.85119325 * float(x[12]))+ (-0.85792786 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (-0.9595632 * float(x[15]))+ (0.6652397 * float(x[16]))+ (1.4490677 * float(x[17]))+ (0.74002427 * float(x[18]))+ (0.9572367 * float(x[19]))+ (0.59831715 * float(x[20]))+ (-0.077041276 * float(x[21]))+ (0.56105834 * float(x[22]))+ (-0.76345116 * float(x[23]))+ (0.27984205 * float(x[24]))+ (-0.71329343 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-0.17067613 * float(x[28]))+ (0.42186546 * float(x[29]))+ (0.5484674 * float(x[30]))+ (0.8050549 * float(x[31]))+ (0.1368679 * float(x[32]))+ (-0.024832893 * float(x[33]))+ (0.23527099 * float(x[34]))+ (0.22419144 * float(x[35]))+ (0.23386799 * float(x[36]))+ (0.8874962 * float(x[37]))+ (0.3636406 * float(x[38]))+ (-0.2809842 * float(x[39]))+ (-0.12593609 * float(x[40]))+ (0.3952624 * float(x[41]))+ (-0.8795491 * float(x[42]))+ (0.33353344 * float(x[43]))+ (0.34127575 * float(x[44]))+ (-0.5792349 * float(x[45]))+ (-0.7421474 * float(x[46]))+ (-0.3691433 * float(x[47]))+ (-0.27257845 * float(x[48]))+ (0.14039354 * float(x[49])))+ ((-0.122796975 * float(x[50]))+ (0.9767477 * float(x[51]))+ (0.09684387 * float(x[52]))+ (-0.5822465 * float(x[53]))+ (-0.677381 * float(x[54]))+ (0.30621666 * float(x[55]))+ (-0.4934168 * float(x[56]))+ (-0.067378454 * float(x[57]))+ (-0.5111488 * float(x[58]))+ (-0.68206084 * float(x[59]))+ (-0.7792497 * float(x[60]))+ (0.31265917 * float(x[61]))+ (-0.7236341 * float(x[62]))+ (-0.6068353 * float(x[63]))+ (-0.26254967 * float(x[64]))+ (0.6419865 * float(x[65]))+ (-0.80579746 * float(x[66]))+ (0.6758898 * float(x[67]))+ (-0.8078032 * float(x[68]))+ (0.95291895 * float(x[69]))+ (-0.0626976 * float(x[70]))+ (0.9535222 * float(x[71]))+ (0.20969103 * float(x[72]))+ (0.47852716 * float(x[73]))+ (-0.9216244 * float(x[74]))+ (-0.43438607 * float(x[75]))+ (-0.7596069 * float(x[76]))+ (-0.4077196 * float(x[77]))+ (-0.7625446 * float(x[78]))+ (-0.36403364 * float(x[79]))+ (-0.17147401 * float(x[80]))+ (-0.871705 * float(x[81]))+ (0.38494423 * float(x[82]))+ (0.13320291 * float(x[83]))+ (-0.46922103 * float(x[84]))+ (0.04649611 * float(x[85]))+ (-0.812119 * float(x[86]))+ (0.15189299 * float(x[87]))+ (0.8585924 * float(x[88]))+ (-0.36286208 * float(x[89]))+ (0.33482075 * float(x[90]))+ (-0.7364043 * float(x[91]))+ (0.4326544 * float(x[92]))+ (-0.42118782 * float(x[93]))+ (-0.6336173 * float(x[94]))+ (0.17302588 * float(x[95]))+ (-0.9597849 * float(x[96]))+ (0.65788007 * float(x[97]))+ (-0.99060905 * float(x[98]))+ (0.35563308 * float(x[99])))+ ((-0.45998406 * float(x[100]))+ (0.47038805 * float(x[101]))+ (0.9243771 * float(x[102]))+ (-0.50249374 * float(x[103]))+ (0.15231466 * float(x[104]))+ (0.18408386 * float(x[105]))+ (0.14450382 * float(x[106]))+ (-0.55383676 * float(x[107]))+ (0.905498 * float(x[108]))+ (-0.10574924 * float(x[109]))+ (0.69281733 * float(x[110]))+ (0.39895856 * float(x[111]))+ (-0.4051261 * float(x[112]))+ (0.62759566 * float(x[113]))+ (-0.20698851 * float(x[114]))+ (0.7622064 * float(x[115]))+ (0.16254574 * float(x[116]))+ (0.7634707 * float(x[117]))+ (0.38506317 * float(x[118]))+ (0.45050856 * float(x[119]))+ (0.0026487638 * float(x[120]))+ (0.91216725 * float(x[121]))+ (0.2879804 * float(x[122]))+ (-0.1522899 * float(x[123]))+ (0.21278642 * float(x[124]))+ (-0.9616136 * float(x[125]))+ (-0.39685038 * float(x[126]))+ (0.32034707 * float(x[127]))+ (-0.41984478 * float(x[128]))+ (0.23603086 * float(x[129]))+ (-0.1424626 * float(x[130]))+ (-0.7290519 * float(x[131]))+ (-0.40343535 * float(x[132]))+ (0.13992982 * float(x[133]))+ (0.18174553 * float(x[134]))+ (0.1486505 * float(x[135]))+ (0.30640164 * float(x[136]))+ (0.30420655 * float(x[137]))+ (-0.13716313 * float(x[138]))+ (0.7930932 * float(x[139]))+ (-0.26487625 * float(x[140]))+ (-0.12827015 * float(x[141]))+ (0.78384674 * float(x[142]))+ (0.61238796 * float(x[143]))+ (0.40777716 * float(x[144]))+ (0.09320802 * float(x[145]))+ (0.83896524 * float(x[146]))+ (0.4284826 * float(x[147]))+ (0.997694 * float(x[148]))+ (-0.7011034 * float(x[149])))+ ((0.7362521 * float(x[150]))+ (-0.67501414 * float(x[151]))+ (0.23111913 * float(x[152]))+ (-0.75236005 * float(x[153]))+ (0.69601643 * float(x[154]))+ (1.5522255 * float(x[155]))+ (0.13820148 * float(x[156]))+ (-0.1856334 * float(x[157]))+ (-0.861666 * float(x[158]))+ (0.39485756 * float(x[159]))+ (0.7998396 * float(x[160]))+ (1.3368654 * float(x[161]))+ (0.73276466 * float(x[162]))+ (1.8886305 * float(x[163]))+ (0.7116067 * float(x[164]))+ (-0.97657186 * float(x[165]))+ (-0.28004387 * float(x[166]))+ (0.4599811 * float(x[167]))+ (-0.65674067 * float(x[168]))+ (0.042073213 * float(x[169]))+ (-0.89132404 * float(x[170]))+ (-0.60000694 * float(x[171]))+ (-0.9629564 * float(x[172]))+ (0.5873954 * float(x[173]))+ (-0.5521506 * float(x[174]))+ (-0.30929664 * float(x[175]))+ (0.8561626 * float(x[176]))+ (0.4088288 * float(x[177]))+ (-0.93632215 * float(x[178]))+ (-0.6706117 * float(x[179]))+ (0.2429568 * float(x[180]))+ (0.15445718 * float(x[181]))+ (-0.5242144 * float(x[182]))+ (0.868428 * float(x[183]))+ (0.22793192 * float(x[184]))+ (0.07126561 * float(x[185]))+ (0.17981996 * float(x[186]))+ (0.46024406 * float(x[187]))+ (-0.37611002 * float(x[188]))+ (-0.20355788 * float(x[189]))+ (0.357275 * float(x[190]))+ (-0.62761396 * float(x[191]))+ (0.8887448 * float(x[192]))+ (0.4791016 * float(x[193]))+ (-0.019082382 * float(x[194]))+ (-0.5451707 * float(x[195]))+ (-0.49128702 * float(x[196]))+ (-0.88394165 * float(x[197]))+ (-0.13116676 * float(x[198]))+ (-0.37640825 * float(x[199])))+ ((0.39268696 * float(x[200]))+ (0.6930912 * float(x[201]))+ (-0.64079267 * float(x[202]))+ (-0.9506425 * float(x[203]))+ (0.07208677 * float(x[204]))+ (0.35878554 * float(x[205]))+ (-0.09260631 * float(x[206]))+ (0.96591264 * float(x[207]))+ (0.7933426 * float(x[208]))+ (0.9806779 * float(x[209]))+ (-0.56620604 * float(x[210]))+ (0.3261564 * float(x[211]))+ (-0.47335523 * float(x[212]))+ (-0.958698 * float(x[213]))+ (0.5167573 * float(x[214]))+ (-0.3599657 * float(x[215]))+ (-0.2330722 * float(x[216]))+ (0.17663422 * float(x[217]))+ (0.6620969 * float(x[218]))+ (0.2579637 * float(x[219]))+ (0.7453013 * float(x[220]))+ (-0.45291594 * float(x[221]))+ (0.59609365 * float(x[222]))+ (-0.6287281 * float(x[223]))+ (1.7983376 * float(x[224]))+ (0.37497655 * float(x[225]))+ (-0.5689846 * float(x[226]))+ (0.8947412 * float(x[227]))+ (0.46171162 * float(x[228]))) + 1.9279687))
    o_0=h_0
             
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

        model_cap=230

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

