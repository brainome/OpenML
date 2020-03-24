#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 01:17:50
# Invocation: btc -server brain.brainome.ai Data/ailerons.csv -o Models/ailerons.py -v -v -v -stopat 89.03 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                57.61%
Model accuracy:                     88.13% (12119/13750 correct)
Improvement over best guess:        30.52% (of possible 42.39%)
Model capacity (MEC):               211 bits
Generalization ratio:               57.43 bits/bit
Model efficiency:                   0.14%/parameter
System behavior
True Negatives:                     36.36% (4999/13750)
True Positives:                     51.78% (7120/13750)
False Negatives:                    5.83% (802/13750)
False Positives:                    6.03% (829/13750)
True Pos. Rate/Sensitivity/Recall:  0.90
True Neg. Rate/Specificity:         0.86
Precision:                          0.90
F-1 Measure:                        0.90
False Negative Rate/Miss Rate:      0.10
Critical Success Index:             0.81

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
TRAINFILE="ailerons.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 40
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
    h_0 = max((((-44.161884 * float(x[0]))+ (-17.825975 * float(x[1]))+ (3.478715 * float(x[2]))+ (2.2158983 * float(x[3]))+ (-1.0429996 * float(x[4]))+ (-1.6242687 * float(x[5]))+ (13.304675 * float(x[6]))+ (4.5357165 * float(x[7]))+ (11.668574 * float(x[8]))+ (-1.2177633 * float(x[9]))+ (-1.4703037 * float(x[10]))+ (-1.4127471 * float(x[11]))+ (-0.6965644 * float(x[12]))+ (-2.4052904 * float(x[13]))+ (-2.3561254 * float(x[14]))+ (-2.2097101 * float(x[15]))+ (-0.5848886 * float(x[16]))+ (-0.67118365 * float(x[17]))+ (-0.4874734 * float(x[18]))+ (-0.05638308 * float(x[19]))+ (-0.42026782 * float(x[20]))+ (-0.98634076 * float(x[21]))+ (-0.34977588 * float(x[22]))+ (-1.6086626 * float(x[23]))+ (-5.024499 * float(x[24]))+ (-2.4226995 * float(x[25]))+ (14.756116 * float(x[26]))+ (-6.8063636 * float(x[27]))+ (-24.751055 * float(x[28]))+ (-0.47088876 * float(x[29]))+ (-2.995327 * float(x[30]))+ (-0.08769934 * float(x[31]))+ (-13.428753 * float(x[32]))+ (4.150397 * float(x[33]))+ (-3.981912 * float(x[34]))+ (2.1601257 * float(x[35]))+ (-8.125086 * float(x[36]))+ (1.8851538 * float(x[37]))+ (-0.8092037 * float(x[38]))+ (-1.1258099 * float(x[39]))) + 0.47930923), 0)
    h_1 = max((((12.588574 * float(x[0]))+ (2.3770406 * float(x[1]))+ (-0.069079675 * float(x[2]))+ (0.3335522 * float(x[3]))+ (-1.0999002 * float(x[4]))+ (-0.74938565 * float(x[5]))+ (-1.26282 * float(x[6]))+ (-0.8438783 * float(x[7]))+ (13.667264 * float(x[8]))+ (-0.092841156 * float(x[9]))+ (-2.4836483 * float(x[10]))+ (-2.1038742 * float(x[11]))+ (-2.945994 * float(x[12]))+ (-3.4454923 * float(x[13]))+ (-2.9194524 * float(x[14]))+ (-3.1807444 * float(x[15]))+ (-3.6589942 * float(x[16]))+ (-3.2912443 * float(x[17]))+ (-1.5998496 * float(x[18]))+ (-2.7987573 * float(x[19]))+ (-0.95519143 * float(x[20]))+ (-1.1615146 * float(x[21]))+ (-2.2434824 * float(x[22]))+ (-1.7105851 * float(x[23]))+ (-23.914455 * float(x[24]))+ (0.78240585 * float(x[25]))+ (11.494019 * float(x[26]))+ (4.728804 * float(x[27]))+ (-0.5168782 * float(x[28]))+ (-0.9077941 * float(x[29]))+ (-42.163124 * float(x[30]))+ (0.8392429 * float(x[31]))+ (-43.02963 * float(x[32]))+ (-24.718681 * float(x[33]))+ (-21.04421 * float(x[34]))+ (-23.118366 * float(x[35]))+ (-23.48825 * float(x[36]))+ (23.596762 * float(x[37]))+ (-1.8953243 * float(x[38]))+ (-0.69094634 * float(x[39]))) + 2.7555225), 0)
    h_2 = max((((2.041427 * float(x[0]))+ (0.38400757 * float(x[1]))+ (2.2002175 * float(x[2]))+ (1.5529628 * float(x[3]))+ (1.0836626 * float(x[4]))+ (0.34845752 * float(x[5]))+ (-0.5394634 * float(x[6]))+ (-0.11124358 * float(x[7]))+ (-12.933824 * float(x[8]))+ (0.009890538 * float(x[9]))+ (2.8038306 * float(x[10]))+ (1.656284 * float(x[11]))+ (3.219833 * float(x[12]))+ (2.4179108 * float(x[13]))+ (3.330634 * float(x[14]))+ (1.8382185 * float(x[15]))+ (1.6719785 * float(x[16]))+ (1.4029186 * float(x[17]))+ (1.1618778 * float(x[18]))+ (1.9143888 * float(x[19]))+ (2.099177 * float(x[20]))+ (1.2567842 * float(x[21]))+ (1.6468432 * float(x[22]))+ (0.71691674 * float(x[23]))+ (24.640099 * float(x[24]))+ (0.31553623 * float(x[25]))+ (-12.227904 * float(x[26]))+ (-11.894577 * float(x[27]))+ (1.9081781 * float(x[28]))+ (-0.6977015 * float(x[29]))+ (44.200447 * float(x[30]))+ (0.24884856 * float(x[31]))+ (45.42552 * float(x[32]))+ (33.061707 * float(x[33]))+ (21.718002 * float(x[34]))+ (31.907873 * float(x[35]))+ (25.74723 * float(x[36]))+ (-31.533257 * float(x[37]))+ (0.8741438 * float(x[38]))+ (1.1642153 * float(x[39]))) + -3.9744773), 0)
    h_3 = max((((-1.4609828 * float(x[0]))+ (-0.34875128 * float(x[1]))+ (1.4650027 * float(x[2]))+ (-2.6788657 * float(x[3]))+ (0.22527769 * float(x[4]))+ (2.0135114 * float(x[5]))+ (-1.0659719 * float(x[6]))+ (0.66426677 * float(x[7]))+ (-8.432639 * float(x[8]))+ (1.3334421 * float(x[9]))+ (1.0836959 * float(x[10]))+ (0.49131814 * float(x[11]))+ (0.023263827 * float(x[12]))+ (1.5430248 * float(x[13]))+ (0.4211552 * float(x[14]))+ (-0.12491057 * float(x[15]))+ (1.0660081 * float(x[16]))+ (0.6855845 * float(x[17]))+ (0.3178688 * float(x[18]))+ (0.87354577 * float(x[19]))+ (0.040978063 * float(x[20]))+ (1.0497825 * float(x[21]))+ (0.90152997 * float(x[22]))+ (-0.16311446 * float(x[23]))+ (9.460477 * float(x[24]))+ (18.832676 * float(x[25]))+ (-16.306541 * float(x[26]))+ (32.120625 * float(x[27]))+ (33.3153 * float(x[28]))+ (-0.34312123 * float(x[29]))+ (12.172052 * float(x[30]))+ (-0.011871377 * float(x[31]))+ (21.03758 * float(x[32]))+ (-26.882023 * float(x[33]))+ (6.500564 * float(x[34]))+ (-15.56119 * float(x[35]))+ (10.572911 * float(x[36]))+ (-0.9393055 * float(x[37]))+ (-0.6397545 * float(x[38]))+ (-0.22109547 * float(x[39]))) + -3.3056195), 0)
    h_4 = max((((-0.45579797 * float(x[0]))+ (-0.042421483 * float(x[1]))+ (-2.682005 * float(x[2]))+ (0.40201503 * float(x[3]))+ (-1.8091309 * float(x[4]))+ (0.52989256 * float(x[5]))+ (-0.2749527 * float(x[6]))+ (0.34823984 * float(x[7]))+ (6.110465 * float(x[8]))+ (0.9029833 * float(x[9]))+ (-3.2198925 * float(x[10]))+ (-2.268326 * float(x[11]))+ (-2.9596682 * float(x[12]))+ (-2.195323 * float(x[13]))+ (-2.3447886 * float(x[14]))+ (-1.8874402 * float(x[15]))+ (-1.912675 * float(x[16]))+ (-2.7178705 * float(x[17]))+ (-2.502801 * float(x[18]))+ (-2.03327 * float(x[19]))+ (-1.3579437 * float(x[20]))+ (-1.5522112 * float(x[21]))+ (-0.85224366 * float(x[22]))+ (-1.524288 * float(x[23]))+ (-12.2081585 * float(x[24]))+ (-20.376026 * float(x[25]))+ (15.790273 * float(x[26]))+ (-29.882492 * float(x[27]))+ (-35.809612 * float(x[28]))+ (0.5647695 * float(x[29]))+ (-13.980952 * float(x[30]))+ (-0.5538659 * float(x[31]))+ (-24.24904 * float(x[32]))+ (27.983562 * float(x[33]))+ (-7.341149 * float(x[34]))+ (16.382765 * float(x[35]))+ (-13.021332 * float(x[36]))+ (-0.324147 * float(x[37]))+ (-1.6622385 * float(x[38]))+ (-2.643511 * float(x[39]))) + 3.3786685), 0)
    o[0] = (0.013147134 * h_0)+ (0.368793 * h_1)+ (-2.2758954 * h_2)+ (-0.8542865 * h_3)+ (1.4710544 * h_4) + 4.191083

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

        model_cap=211

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

