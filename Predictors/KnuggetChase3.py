#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Mar-02-2020 11:11:41
# Invocation: btc Data/KnuggetChase3.csv -o Models/KnuggetChase3.py -v -v -v -stopat 100 -port 8090 -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                81.44%
Model accuracy:                     86.08% (167/194 correct)
Improvement over best guess:        4.64% (of possible 18.56%)
Model capacity (MEC):               124 bits
Generalization ratio:               1.34 bits/bit
Model efficiency:                   0.03%/parameter
System behavior
True Negatives:                     5.67% (11/194)
True Positives:                     80.41% (156/194)
False Negatives:                    1.03% (2/194)
False Positives:                    12.89% (25/194)
True Pos. Rate/Sensitivity/Recall:  0.99
True Neg. Rate/Specificity:         0.31
Precision:                          0.86
F-1 Measure:                        0.92
False Negative Rate/Miss Rate:      0.01
Critical Success Index:             0.85
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

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="KnuggetChase3.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 39

mappings = [{238.4: 0, 355.35: 1, 781.73: 2, 822.39: 3, 1092.32: 4, 1189.32: 5, 1294.27: 6, 1355.23: 7, 1404.72: 8, 1517.0: 9, 1593.02: 10, 1648.21: 11, 1800.5: 12, 1943.91: 13, 2372.72: 14, 2383.85: 15, 2569.87: 16, 2664.41: 17, 2752.14: 18, 2819.37: 19, 3208.05: 20, 3243.25: 21, 3268.21: 22, 3349.04: 23, 3548.61: 24, 4118.96: 25, 4777.91: 26, 4832.73: 27, 4980.42: 28, 5014.84: 29, 5352.73: 30, 5396.06: 31, 5496.65: 32, 5590.85: 33, 5735.89: 34, 6017.65: 35, 6090.84: 36, 6152.97: 37, 6172.93: 38, 6361.1: 39, 6560.0: 40, 6589.95: 41, 6656.28: 42, 6882.97: 43, 6896.24: 44, 7876.2: 45, 8321.46: 46, 8678.36: 47, 8841.71: 48, 8983.13: 49, 9703.67: 50, 9816.39: 51, 9914.72: 52, 9961.7: 53, 10297.93: 54, 10577.6: 55, 11411.18: 56, 13243.53: 57, 13490.6: 58, 13948.3: 59, 14395.0: 60, 15290.22: 61, 15944.27: 62, 16352.99: 63, 16702.0: 64, 19748.29: 65, 19777.28: 66, 22864.72: 67, 26838.69: 68, 27226.57: 69, 29970.57: 70, 30243.28: 71, 31877.94: 72, 32631.05: 73, 33320.87: 74, 37151.02: 75, 37755.59: 76, 39892.13: 77, 45051.12: 78, 45791.28: 79, 46024.85: 80, 46844.09: 81, 47999.33: 82, 50170.62: 83, 51383.14: 84, 59492.48: 85, 63297.51: 86, 70666.9: 87, 71508.8: 88, 115062.64: 89, 120439.45: 90, 141550.33: 91, 154013.92: 92, 193686.35: 93, 252155.78: 94, 582830.35: 95, 6778.53: 96, 34291.94: 97, 3988.71: 98, 169489.04: 99, 15056.5: 100, 14702.26: 101, 8310.37: 102, 1240.55: 103, 9151.3: 104, 3114.36: 105, 95809.13: 106, 2898.63: 107, 5883.89: 108, 6086.79: 109, 1679.07: 110, 13563.07: 111, 18875.0: 112, 2976.5: 113, 13244.88: 114, 24374.21: 115, 48459.37: 116, 169799.06: 117, 1066.0: 118, 6006.03: 119, 3685.14: 120, 4016.74: 121, 1609.1: 122, 726.48: 123, 8150.47: 124, 107406.67: 125, 9878.21: 126, 4689.17: 127, 10037.72: 128, 5332.5: 129, 2320.9: 130, 23842.81: 131, 45226.86: 132, 1110.86: 133, 502.31: 134, 114410.43: 135, 3010.96: 136, 1685.67: 137, 6765.0: 138, 8034.06: 139, 10710.87: 140, 68305.52: 141, 2782.69: 142, 719.39: 143, 19687.59: 144, 29080.61: 145, 8112.64: 146, 76313.12: 147, 3645.43: 148, 51033.62: 149, 26283.38: 150, 3814.64: 151, 148686.82: 152, 9065.86: 153, 11268.26: 154, 2229.05: 155, 52445.24: 156, 25188.5: 157, 22872.08: 158, 72932.91: 159, 3218.64: 160, 2092.83: 161, 568.57: 162, 2125.46: 163, 116852.98: 164, 48468.15: 165, 41048.67: 166, 871.59: 167, 2266.4: 168, 30048.37: 169, 64177.21: 170, 29456.04: 171, 13404.49: 172, 1681.86: 173, 103097.32: 174, 223888.39: 175, 9917.4: 176, 3029.38: 177, 8806.58: 178, 1796.86: 179, 9144.94: 180, 395438.84: 181, 3150.52: 182, 6175.54: 183, 21840.0: 184, 178710.34: 185, 33861.93: 186, 3264.45: 187, 1557.81: 188}]
list_of_cols_to_normalize = [21]

transform_true = False

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values()))+1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize,mappings):
            if i>=data_arr.shape[1]:
                break
            col = data_arr[:,i]
            normcol = column_norm(col,mapping)
            data_arr[:,i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([])
        components = np.array([])
        whiten = None
        explained_variance = np.array([])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

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
    h_0 = max((((1.1529176 * float(x[0]))+ (1.0107721 * float(x[1]))+ (-0.69849503 * float(x[2]))+ (-5.814841 * float(x[3]))+ (1.6914281 * float(x[4]))+ (0.63577354 * float(x[5]))+ (1.2146859 * float(x[6]))+ (-0.71573335 * float(x[7]))+ (-0.08154327 * float(x[8]))+ (0.6545143 * float(x[9]))+ (-0.6756764 * float(x[10]))+ (-0.98899335 * float(x[11]))+ (2.885963 * float(x[12]))+ (1.0678719 * float(x[13]))+ (-6.8732853 * float(x[14]))+ (0.9181652 * float(x[15]))+ (-0.69753045 * float(x[16]))+ (-0.27760842 * float(x[17]))+ (-0.4955553 * float(x[18]))+ (3.1616628 * float(x[19]))+ (0.8942471 * float(x[20]))+ (4.3415294 * float(x[21]))+ (-0.38330877 * float(x[22]))+ (7.4473825 * float(x[23]))+ (5.1752872 * float(x[24]))+ (45.848915 * float(x[25]))+ (43.6981 * float(x[26]))+ (-0.22624673 * float(x[27]))+ (-1.1626126 * float(x[28]))+ (-0.6546247 * float(x[29]))+ (2.1156769 * float(x[30]))+ (-1.8746257 * float(x[31]))+ (2.7735813 * float(x[32]))+ (4.0703073 * float(x[33]))+ (1.390234 * float(x[34]))+ (-0.1227935 * float(x[35]))+ (3.0341682 * float(x[36]))+ (0.85201246 * float(x[37]))+ (2.1619198 * float(x[38]))) + 0.27643976), 0)
    h_1 = max((((0.96071887 * float(x[0]))+ (0.0631827 * float(x[1]))+ (1.7682852 * float(x[2]))+ (8.469665 * float(x[3]))+ (-0.19487754 * float(x[4]))+ (2.0659244 * float(x[5]))+ (0.9159342 * float(x[6]))+ (1.3019636 * float(x[7]))+ (0.88475513 * float(x[8]))+ (-0.114173084 * float(x[9]))+ (1.5740663 * float(x[10]))+ (-0.21156473 * float(x[11]))+ (1.894 * float(x[12]))+ (-1.9234931 * float(x[13]))+ (6.398052 * float(x[14]))+ (1.2030313 * float(x[15]))+ (1.6188103 * float(x[16]))+ (1.5983651 * float(x[17]))+ (1.7545216 * float(x[18]))+ (1.1630927 * float(x[19]))+ (1.8113314 * float(x[20]))+ (0.88878757 * float(x[21]))+ (1.6212857 * float(x[22]))+ (1.3132334 * float(x[23]))+ (-5.7575965 * float(x[24]))+ (1.641126 * float(x[25]))+ (1.7715394 * float(x[26]))+ (-0.43978104 * float(x[27]))+ (0.99672127 * float(x[28]))+ (1.7597 * float(x[29]))+ (0.7068788 * float(x[30]))+ (0.19776984 * float(x[31]))+ (0.28027567 * float(x[32]))+ (0.98882186 * float(x[33]))+ (0.6133866 * float(x[34]))+ (0.93825155 * float(x[35]))+ (1.4948043 * float(x[36]))+ (1.9317062 * float(x[37]))+ (1.7312542 * float(x[38]))) + -0.60912013), 0)
    h_2 = max((((0.43468356 * float(x[0]))+ (2.002375 * float(x[1]))+ (0.2852889 * float(x[2]))+ (-8.257154 * float(x[3]))+ (3.8937669 * float(x[4]))+ (0.19014186 * float(x[5]))+ (1.8131412 * float(x[6]))+ (0.8856275 * float(x[7]))+ (1.1420071 * float(x[8]))+ (2.9609573 * float(x[9]))+ (0.7077648 * float(x[10]))+ (1.8126547 * float(x[11]))+ (1.4232764 * float(x[12]))+ (4.197063 * float(x[13]))+ (-3.5446303 * float(x[14]))+ (0.8183587 * float(x[15]))+ (0.1613756 * float(x[16]))+ (1.3135847 * float(x[17]))+ (1.3622446 * float(x[18]))+ (0.9578432 * float(x[19]))+ (1.3628818 * float(x[20]))+ (1.1737535 * float(x[21]))+ (-0.36169556 * float(x[22]))+ (0.21497315 * float(x[23]))+ (7.4243426 * float(x[24]))+ (-0.07594001 * float(x[25]))+ (0.24974656 * float(x[26]))+ (1.823512 * float(x[27]))+ (0.8142565 * float(x[28]))+ (1.3614908 * float(x[29]))+ (1.3085228 * float(x[30]))+ (1.8373394 * float(x[31]))+ (0.027661838 * float(x[32]))+ (0.88846666 * float(x[33]))+ (-0.23986869 * float(x[34]))+ (0.85357994 * float(x[35]))+ (0.45164013 * float(x[36]))+ (0.92303437 * float(x[37]))+ (0.15739027 * float(x[38]))) + 2.7605836), 0)
    o_0 = (0.0637905 * h_0)+ (-1.7004726 * h_1)+ (2.0394666 * h_2) + 1.586668

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
        if not args.cleanfile: # File is not preprocessed
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            test_tensor = np.loadtxt(cleanfile,delimiter=',',dtype='float64')
            os.remove(cleanfile)
        else: # File is already preprocessed
            test_tensor = np.loadtxt(args.File,delimiter = ',',dtype = 'float64')               
        test_tensor = Normalize(test_tensor)
        if transform_true:
            test_tensor = transform(test_tensor)
        with open(args.csvfile,'r') as csvinput:
            writer = csv.writer(sys.stdout, lineterminator='\n')
            reader = csv.reader(csvinput)
            if (not args.headerless):
                writer.writerow((next(reader, None)+['Prediction']))
            i=0
            for row in reader:
                if (classify(test_tensor[i])):
                    pred="1"
                else:
                    pred="0"
                row.append(pred)
                writer.writerow(row)
                i=i+1
    elif args.validate: # Then validate this predictor, always clean first.
        tempdir=tempfile.gettempdir()
        temp_name = next(tempfile._get_candidate_names())
        cleanfile=tempdir+os.sep+temp_name
        clean(args.csvfile,cleanfile, -1, args.headerless)
        val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
        os.remove(cleanfile)
        val_tensor = Normalize(val_tensor)
        if transform_true:
            trans = transform(val_tensor[:,:-1])
            val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
        count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0 = 0,0,0,0,0,0,0,0
        for i,row in enumerate(val_tensor):
            if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                correct_count+=1
                if int(float(row[-1]))==1:
                    num_class_1+=1
                    num_TP+=1
                else:
                    num_class_0+=1
                    num_TN+=1
            else:
                if int(float(row[-1]))==1:
                    num_class_1+=1
                    num_FN+=1
                else:
                    num_class_0+=1
                    num_FP+=1
            count+=1

        model_cap=124

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


