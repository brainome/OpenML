#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 11:00:01
# Invocation: btc -v -v eucalyptus-1.csv -o eucalyptus-1.py -f ME
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


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="eucalyptus-1.csv"


#Number of attributes
num_attr = 19

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


# Calculate equilibrium energy ($_i=1)
def eqenergy(row):
    result=0
    for elem in row:
        result = result + float(elem)
    return result

# Classifier 
def classify(row):
    energy=eqenergy(row)
    if (energy>19931116138.25):
        return 1.0
    if (energy>19642105061.1):
        return 0.0
    if (energy>19642104277.05):
        return 1.0
    if (energy>19599900952.58):
        return 0.0
    if (energy>18921912992.0):
        return 1.0
    if (energy>18789849207.629997):
        return 0.0
    if (energy>18118529170.799995):
        return 1.0
    if (energy>18098572839.055):
        return 0.0
    if (energy>17839786951.4):
        return 1.0
    if (energy>17839786632.449997):
        return 0.0
    if (energy>17583379727.71):
        return 1.0
    if (energy>17433131954.15):
        return 0.0
    if (energy>15573041840.2):
        return 1.0
    if (energy>14934225048.560001):
        return 0.0
    if (energy>14934225021.06):
        return 1.0
    if (energy>14727999519.364998):
        return 0.0
    if (energy>14472520831.06):
        return 1.0
    if (energy>14330040692.024998):
        return 0.0
    if (energy>14330040676.864998):
        return 1.0
    if (energy>14304915963.36):
        return 0.0
    if (energy>13295341553.470001):
        return 1.0
    if (energy>13090671183.785):
        return 0.0
    if (energy>12609615473.244999):
        return 1.0
    if (energy>12527722268.849998):
        return 0.0
    if (energy>12527722228.255001):
        return 1.0
    if (energy>12466202713.215):
        return 0.0
    if (energy>12359596045.220001):
        return 1.0
    if (energy>12314508874.435001):
        return 0.0
    if (energy>12225556040.125):
        return 1.0
    if (energy>12187826171.43):
        return 0.0
    if (energy>12182490201.865):
        return 1.0
    if (energy>12177155249.575):
        return 0.0
    if (energy>12155903000.98):
        return 1.0
    if (energy>12138485484.329998):
        return 0.0
    if (energy>12121067447.955):
        return 1.0
    if (energy>12121066925.119999):
        return 0.0
    if (energy>12098656969.89):
        return 1.0
    if (energy>12098656962.785):
        return 0.0
    if (energy>11968113692.545):
        return 1.0
    if (energy>11898413053.505001):
        return 0.0
    if (energy>11898413031.835):
        return 1.0
    if (energy>11729779397.380001):
        return 0.0
    if (energy>11725055848.099998):
        return 1.0
    if (energy>11720332432.55):
        return 0.0
    if (energy>11720331967.675):
        return 1.0
    if (energy>11720331481.68):
        return 0.0
    if (energy>11569918963.285):
        return 1.0
    if (energy>11551717602.794998):
        return 0.0
    if (energy>11551717588.960999):
        return 1.0
    if (energy>11521737960.786):
        return 0.0
    if (energy>11491758051.84):
        return 1.0
    if (energy>11491757801.380001):
        return 0.0
    if (energy>11399260041.18):
        return 1.0
    if (energy>11350307955.619999):
        return 0.0
    if (energy>11350307933.404999):
        return 1.0
    if (energy>11323125123.27):
        return 0.0
    if (energy>11173987328.884998):
        return 1.0
    if (energy>11127374658.669998):
        return 0.0
    if (energy>10972156669.100002):
        return 1.0
    if (energy>10899794072.64):
        return 0.0
    if (energy>10899793553.865):
        return 1.0
    if (energy>10899793046.849998):
        return 0.0
    if (energy>10897945938.915):
        return 1.0
    if (energy>10762739633.395):
        return 0.0
    if (energy>10649672146.885002):
        return 1.0
    if (energy>10643757858.349998):
        return 0.0
    if (energy>10581428256.61):
        return 1.0
    if (energy>10581407737.05):
        return 0.0
    if (energy>10581407731.175):
        return 1.0
    if (energy>10581407659.075):
        return 0.0
    if (energy>10581407173.035):
        return 1.0
    if (energy>10581406687.349998):
        return 0.0
    if (energy>10566885930.09):
        return 1.0
    if (energy>10552365139.465):
        return 0.0
    if (energy>10534691661.095001):
        return 1.0
    if (energy>10476902523.82):
        return 0.0
    if (energy>10302155659.14):
        return 1.0
    if (energy>10270484914.25):
        return 0.0
    if (energy>10270484317.564999):
        return 1.0
    if (energy>10208339413.91):
        return 0.0
    if (energy>10070449283.980003):
        return 1.0
    if (energy>10052073416.95):
        return 0.0
    if (energy>10005459912.115):
        return 1.0
    if (energy>9935021093.27):
        return 0.0
    if (energy>9892819815.839998):
        return 1.0
    if (energy>9883544653.079998):
        return 0.0
    if (energy>9846227549.145):
        return 1.0
    if (energy>9818187108.849998):
        return 0.0
    if (energy>9818187079.1):
        return 1.0
    if (energy>9815810252.650002):
        return 0.0
    if (energy>9719366640.675):
        return 1.0
    if (energy>9657410936.454998):
        return 0.0
    if (energy>9657410909.82):
        return 1.0
    if (energy>9587035811.24):
        return 0.0
    if (energy>9516661671.849998):
        return 1.0
    if (energy>9516661632.744999):
        return 0.0
    if (energy>9439107328.685):
        return 1.0
    if (energy>9374668054.97):
        return 0.0
    if (energy>9313828710.195):
        return 1.0
    if (energy>9313828680.710001):
        return 0.0
    if (energy>9268827910.52):
        return 1.0
    if (energy>9215069229.395):
        return 0.0
    if (energy>9025697336.02):
        return 1.0
    if (energy>9025697309.635002):
        return 0.0
    if (energy>8942523256.65):
        return 1.0
    if (energy>8942523236.544998):
        return 0.0
    if (energy>8924848835.05):
        return 1.0
    if (energy>8907174404.244999):
        return 0.0
    if (energy>8907173838.655):
        return 1.0
    if (energy>8907173323.52):
        return 0.0
    if (energy>8856045056.6):
        return 1.0
    if (energy>8841439419.3):
        return 0.0
    if (energy>8841439072.5):
        return 1.0
    if (energy>8841438796.15):
        return 0.0
    if (energy>8712868757.82):
        return 1.0
    if (energy>8679259409.035):
        return 0.0
    if (energy>8632645970.22):
        return 1.0
    if (energy>8619641852.35):
        return 0.0
    if (energy>8512022436.975):
        return 1.0
    if (energy>8496060860.105):
        return 0.0
    if (energy>8434784405.650001):
        return 1.0
    if (energy>8434784160.3):
        return 0.0
    if (energy>8222901024.705):
        return 1.0
    if (energy>8174903907.484999):
        return 0.0
    if (energy>8147747743.37):
        return 1.0
    if (energy>8136414361.280001):
        return 0.0
    if (energy>8136413949.23):
        return 1.0
    if (energy>8136413545.344999):
        return 0.0
    if (energy>7907131686.08):
        return 1.0
    if (energy>7818085396.395):
        return 0.0
    if (energy>7769904168.27):
        return 1.0
    if (energy>7768249092.93):
        return 0.0
    if (energy>7695833230.465):
        return 1.0
    if (energy>7664024403.1):
        return 0.0
    if (energy>7584942697.66):
        return 1.0
    if (energy>7584942658.414999):
        return 0.0
    if (energy>7263879935.405):
        return 1.0
    if (energy>7250907141.790001):
        return 0.0
    if (energy>7250907007.385):
        return 1.0
    if (energy>7250905914.535):
        return 0.0
    if (energy>7232208597.665):
        return 1.0
    if (energy>7213511293.75):
        return 0.0
    if (energy>6997489620.905001):
        return 1.0
    if (energy>6997488818.485001):
        return 0.0
    if (energy>6885965624.6050005):
        return 1.0
    if (energy>6844252857.665):
        return 0.0
    if (energy>6844252227.325001):
        return 1.0
    if (energy>6844251747.215):
        return 0.0
    if (energy>6830787585.065):
        return 1.0
    if (energy>6817323403.145):
        return 0.0
    if (energy>6785784843.7):
        return 1.0
    if (energy>6694946968.664999):
        return 0.0
    if (energy>6546974725.495001):
        return 1.0
    if (energy>6527003146.12):
        return 0.0
    if (energy>6458850346.985001):
        return 1.0
    if (energy>6410669119.075001):
        return 0.0
    if (energy>6094363842.235001):
        return 1.0
    if (energy>6073492810.76):
        return 0.0
    if (energy>5927433471.469999):
        return 1.0
    if (energy>5861788077.815001):
        return 0.0
    if (energy>5861787582.505):
        return 1.0
    if (energy>5861787083.995001):
        return 0.0
    if (energy>5631443613.44):
        return 1.0
    if (energy>5622978967.945):
        return 0.0
    if (energy>5493849731.094999):
        return 1.0
    if (energy>5428204311.695):
        return 0.0
    if (energy>5428203826.195):
        return 1.0
    if (energy>5308799295.91):
        return 0.0
    if (energy>5189394638.35):
        return 1.0
    return 0.0

numthresholds=169


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

        model_cap=numthresholds

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

