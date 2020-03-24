#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 00:26:27
# Invocation: btc -server brain.brainome.ai Data/fri_c3_1000_50.csv -o Models/fri_c3_1000_50.py -v -v -v -stopat 91.30 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                55.50%
Model accuracy:                     88.70% (887/1000 correct)
Improvement over best guess:        33.20% (of possible 44.5%)
Model capacity (MEC):               365 bits
Generalization ratio:               2.43 bits/bit
Model efficiency:                   0.09%/parameter
System behavior
True Negatives:                     49.40% (494/1000)
True Positives:                     39.30% (393/1000)
False Negatives:                    5.20% (52/1000)
False Positives:                    6.10% (61/1000)
True Pos. Rate/Sensitivity/Recall:  0.88
True Neg. Rate/Specificity:         0.89
Precision:                          0.87
F-1 Measure:                        0.87
False Negative Rate/Miss Rate:      0.12
Critical Success Index:             0.78

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
TRAINFILE="fri_c3_1000_50.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 50
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
    h_0 = max((((12.878301 * float(x[0]))+ (19.321014 * float(x[1]))+ (4.469814 * float(x[2]))+ (-11.132458 * float(x[3]))+ (-7.386926 * float(x[4]))+ (2.2964623 * float(x[5]))+ (-10.450652 * float(x[6]))+ (1.9702438 * float(x[7]))+ (2.7849576 * float(x[8]))+ (7.102367 * float(x[9]))+ (1.9219533 * float(x[10]))+ (9.355566 * float(x[11]))+ (2.7013886 * float(x[12]))+ (3.9992974 * float(x[13]))+ (-3.5789392 * float(x[14]))+ (3.1348474 * float(x[15]))+ (-1.4234332 * float(x[16]))+ (2.7372081 * float(x[17]))+ (-4.509921 * float(x[18]))+ (0.78502977 * float(x[19]))+ (0.053250592 * float(x[20]))+ (1.4455069 * float(x[21]))+ (5.135475 * float(x[22]))+ (-3.9548059 * float(x[23]))+ (-1.4108441 * float(x[24]))+ (-6.5998216 * float(x[25]))+ (-4.144657 * float(x[26]))+ (0.71200293 * float(x[27]))+ (0.24544215 * float(x[28]))+ (10.229149 * float(x[29]))+ (-1.926806 * float(x[30]))+ (0.53224766 * float(x[31]))+ (4.4218388 * float(x[32]))+ (-7.69503 * float(x[33]))+ (4.946072 * float(x[34]))+ (3.3618772 * float(x[35]))+ (-2.7351816 * float(x[36]))+ (6.4569373 * float(x[37]))+ (2.7494802 * float(x[38]))+ (3.7169216 * float(x[39]))+ (-5.1998463 * float(x[40]))+ (6.092574 * float(x[41]))+ (4.2885766 * float(x[42]))+ (-5.1133246 * float(x[43]))+ (8.153108 * float(x[44]))+ (0.9373514 * float(x[45]))+ (-2.9544957 * float(x[46]))+ (-2.5261552 * float(x[47]))+ (-1.9168544 * float(x[48]))+ (-0.30837876 * float(x[49]))) + -0.20392798), 0)
    h_1 = max((((-0.18501586 * float(x[0]))+ (6.8087482 * float(x[1]))+ (5.0474644 * float(x[2]))+ (-9.347085 * float(x[3]))+ (-3.022943 * float(x[4]))+ (-10.249089 * float(x[5]))+ (3.6002035 * float(x[6]))+ (-0.12648943 * float(x[7]))+ (3.4187021 * float(x[8]))+ (-5.2206955 * float(x[9]))+ (-3.9660382 * float(x[10]))+ (5.582931 * float(x[11]))+ (1.7369409 * float(x[12]))+ (0.082366094 * float(x[13]))+ (-4.7491984 * float(x[14]))+ (1.2460116 * float(x[15]))+ (1.944419 * float(x[16]))+ (0.33743146 * float(x[17]))+ (-2.6955245 * float(x[18]))+ (-4.705198 * float(x[19]))+ (2.2719054 * float(x[20]))+ (-4.0748863 * float(x[21]))+ (0.4538666 * float(x[22]))+ (-6.5195246 * float(x[23]))+ (-0.36642286 * float(x[24]))+ (-1.1851708 * float(x[25]))+ (1.339348 * float(x[26]))+ (1.0962008 * float(x[27]))+ (-0.95548314 * float(x[28]))+ (2.0620787 * float(x[29]))+ (-0.25427884 * float(x[30]))+ (-1.344128 * float(x[31]))+ (-5.2194576 * float(x[32]))+ (2.079212 * float(x[33]))+ (2.9731932 * float(x[34]))+ (0.049400296 * float(x[35]))+ (-0.5139622 * float(x[36]))+ (6.2891536 * float(x[37]))+ (7.253799 * float(x[38]))+ (5.239847 * float(x[39]))+ (0.4957097 * float(x[40]))+ (-3.193769 * float(x[41]))+ (0.29684475 * float(x[42]))+ (-2.9735408 * float(x[43]))+ (3.4336965 * float(x[44]))+ (-2.8172364 * float(x[45]))+ (1.8361952 * float(x[46]))+ (5.248553 * float(x[47]))+ (-0.76718843 * float(x[48]))+ (0.71712774 * float(x[49]))) + -12.167947), 0)
    h_2 = max((((1.9540308 * float(x[0]))+ (1.874727 * float(x[1]))+ (0.8926832 * float(x[2]))+ (-3.3398857 * float(x[3]))+ (2.4076583 * float(x[4]))+ (-1.0413768 * float(x[5]))+ (1.3193107 * float(x[6]))+ (3.3563504 * float(x[7]))+ (0.6246402 * float(x[8]))+ (1.1725485 * float(x[9]))+ (0.75195915 * float(x[10]))+ (0.59010774 * float(x[11]))+ (1.4623387 * float(x[12]))+ (2.2245986 * float(x[13]))+ (-0.8031834 * float(x[14]))+ (3.1861937 * float(x[15]))+ (2.903057 * float(x[16]))+ (-0.8445534 * float(x[17]))+ (-3.9223638 * float(x[18]))+ (1.1081746 * float(x[19]))+ (1.0575215 * float(x[20]))+ (-3.3553846 * float(x[21]))+ (0.07100038 * float(x[22]))+ (-0.36829358 * float(x[23]))+ (-4.606144 * float(x[24]))+ (-0.5336267 * float(x[25]))+ (1.8536774 * float(x[26]))+ (0.55372065 * float(x[27]))+ (4.3933907 * float(x[28]))+ (2.7917054 * float(x[29]))+ (-1.8812327 * float(x[30]))+ (2.6625614 * float(x[31]))+ (2.0511143 * float(x[32]))+ (1.6906986 * float(x[33]))+ (1.1162422 * float(x[34]))+ (1.3903345 * float(x[35]))+ (-2.006131 * float(x[36]))+ (3.808406 * float(x[37]))+ (2.9406157 * float(x[38]))+ (0.48711234 * float(x[39]))+ (-3.5856605 * float(x[40]))+ (2.423389 * float(x[41]))+ (3.1994605 * float(x[42]))+ (-2.0638804 * float(x[43]))+ (6.104887 * float(x[44]))+ (-2.0104861 * float(x[45]))+ (-2.8184214 * float(x[46]))+ (-2.5916243 * float(x[47]))+ (-2.1358228 * float(x[48]))+ (2.3586586 * float(x[49]))) + -5.015608), 0)
    h_3 = max((((3.4452531 * float(x[0]))+ (3.7563806 * float(x[1]))+ (-1.4103634 * float(x[2]))+ (-3.5889957 * float(x[3]))+ (-7.7950554 * float(x[4]))+ (0.55227727 * float(x[5]))+ (-2.8276153 * float(x[6]))+ (1.7123498 * float(x[7]))+ (-1.1440352 * float(x[8]))+ (-0.28610548 * float(x[9]))+ (-6.572182 * float(x[10]))+ (2.2130804 * float(x[11]))+ (1.4859638 * float(x[12]))+ (1.0240881 * float(x[13]))+ (-1.7178365 * float(x[14]))+ (-1.4461802 * float(x[15]))+ (-3.6345391 * float(x[16]))+ (-1.0778735 * float(x[17]))+ (0.66774505 * float(x[18]))+ (-0.8250268 * float(x[19]))+ (2.103079 * float(x[20]))+ (2.4998484 * float(x[21]))+ (1.929123 * float(x[22]))+ (2.929919 * float(x[23]))+ (-2.296947 * float(x[24]))+ (-2.6764264 * float(x[25]))+ (-2.3304293 * float(x[26]))+ (0.65791994 * float(x[27]))+ (-1.9138991 * float(x[28]))+ (1.6379243 * float(x[29]))+ (0.77622676 * float(x[30]))+ (1.66229 * float(x[31]))+ (4.4169025 * float(x[32]))+ (-4.769559 * float(x[33]))+ (0.08561498 * float(x[34]))+ (-0.61452216 * float(x[35]))+ (-0.5178489 * float(x[36]))+ (2.9672682 * float(x[37]))+ (0.4184274 * float(x[38]))+ (4.760806 * float(x[39]))+ (-0.6075816 * float(x[40]))+ (1.0403862 * float(x[41]))+ (-0.08314898 * float(x[42]))+ (1.3023468 * float(x[43]))+ (1.8537463 * float(x[44]))+ (-3.2960858 * float(x[45]))+ (-1.0867198 * float(x[46]))+ (-3.4253688 * float(x[47]))+ (2.1991005 * float(x[48]))+ (0.474084 * float(x[49]))) + -0.33968058), 0)
    h_4 = max((((2.1843486 * float(x[0]))+ (-2.1761267 * float(x[1]))+ (-2.4631226 * float(x[2]))+ (-6.045092 * float(x[3]))+ (0.86981285 * float(x[4]))+ (0.6756536 * float(x[5]))+ (-0.9842218 * float(x[6]))+ (-0.25656322 * float(x[7]))+ (-1.6075598 * float(x[8]))+ (0.30248287 * float(x[9]))+ (-4.051262 * float(x[10]))+ (1.448329 * float(x[11]))+ (-1.1519153 * float(x[12]))+ (2.1295846 * float(x[13]))+ (0.8864333 * float(x[14]))+ (-1.0562791 * float(x[15]))+ (-0.2417962 * float(x[16]))+ (-0.2534484 * float(x[17]))+ (0.7161964 * float(x[18]))+ (1.114149 * float(x[19]))+ (-0.21722451 * float(x[20]))+ (1.0098656 * float(x[21]))+ (-0.4218086 * float(x[22]))+ (2.4794915 * float(x[23]))+ (0.65593195 * float(x[24]))+ (1.2534382 * float(x[25]))+ (-2.2284865 * float(x[26]))+ (-1.6907719 * float(x[27]))+ (-0.4492603 * float(x[28]))+ (1.3361415 * float(x[29]))+ (1.9049013 * float(x[30]))+ (1.4540583 * float(x[31]))+ (4.1063433 * float(x[32]))+ (-1.551087 * float(x[33]))+ (0.69946134 * float(x[34]))+ (-1.348722 * float(x[35]))+ (1.0530624 * float(x[36]))+ (1.7936486 * float(x[37]))+ (1.3566746 * float(x[38]))+ (4.019266 * float(x[39]))+ (-1.0488492 * float(x[40]))+ (-0.562464 * float(x[41]))+ (-0.0777389 * float(x[42]))+ (2.0585294 * float(x[43]))+ (1.5634317 * float(x[44]))+ (-0.8096899 * float(x[45]))+ (-1.3279111 * float(x[46]))+ (-1.8925828 * float(x[47]))+ (1.6770773 * float(x[48]))+ (0.060202535 * float(x[49]))) + -4.5436883), 0)
    h_5 = max((((-1.1784004 * float(x[0]))+ (-0.06605767 * float(x[1]))+ (1.8305821 * float(x[2]))+ (-3.1432128 * float(x[3]))+ (1.0475433 * float(x[4]))+ (0.6653337 * float(x[5]))+ (0.5568546 * float(x[6]))+ (1.0140519 * float(x[7]))+ (1.1177199 * float(x[8]))+ (0.10106377 * float(x[9]))+ (-1.022568 * float(x[10]))+ (-0.111200586 * float(x[11]))+ (2.086355 * float(x[12]))+ (2.034842 * float(x[13]))+ (-1.088037 * float(x[14]))+ (-0.65606135 * float(x[15]))+ (-4.02323 * float(x[16]))+ (-1.0504373 * float(x[17]))+ (0.24134031 * float(x[18]))+ (-0.35055313 * float(x[19]))+ (0.58308786 * float(x[20]))+ (-0.30817983 * float(x[21]))+ (0.07158178 * float(x[22]))+ (0.57343405 * float(x[23]))+ (-2.3098545 * float(x[24]))+ (-3.2821252 * float(x[25]))+ (2.3607225 * float(x[26]))+ (1.3454248 * float(x[27]))+ (0.2797583 * float(x[28]))+ (1.644612 * float(x[29]))+ (-0.21563223 * float(x[30]))+ (-0.05520764 * float(x[31]))+ (1.8212734 * float(x[32]))+ (-1.0708251 * float(x[33]))+ (1.2735553 * float(x[34]))+ (0.96068287 * float(x[35]))+ (-1.3391039 * float(x[36]))+ (0.83058196 * float(x[37]))+ (-2.130587 * float(x[38]))+ (-0.250766 * float(x[39]))+ (-2.851248 * float(x[40]))+ (2.2076645 * float(x[41]))+ (0.8533821 * float(x[42]))+ (-0.21778782 * float(x[43]))+ (0.26575717 * float(x[44]))+ (-1.7251809 * float(x[45]))+ (0.75979537 * float(x[46]))+ (-0.82867086 * float(x[47]))+ (-0.8981615 * float(x[48]))+ (1.6743981 * float(x[49]))) + -3.362168), 0)
    h_6 = max((((5.677081 * float(x[0]))+ (2.1510715 * float(x[1]))+ (-0.27889976 * float(x[2]))+ (-6.5849175 * float(x[3]))+ (6.3918896 * float(x[4]))+ (0.26948604 * float(x[5]))+ (-4.899568 * float(x[6]))+ (0.22693528 * float(x[7]))+ (4.407349 * float(x[8]))+ (4.6031485 * float(x[9]))+ (2.9551795 * float(x[10]))+ (4.489887 * float(x[11]))+ (1.3187244 * float(x[12]))+ (2.3423712 * float(x[13]))+ (0.115339234 * float(x[14]))+ (1.2882531 * float(x[15]))+ (-3.4706213 * float(x[16]))+ (0.39345363 * float(x[17]))+ (-0.45514947 * float(x[18]))+ (0.15449692 * float(x[19]))+ (-2.122201 * float(x[20]))+ (0.37802395 * float(x[21]))+ (0.8271337 * float(x[22]))+ (-2.2703652 * float(x[23]))+ (1.3266503 * float(x[24]))+ (-4.8226686 * float(x[25]))+ (-1.1434289 * float(x[26]))+ (0.08820851 * float(x[27]))+ (-2.9126291 * float(x[28]))+ (7.255461 * float(x[29]))+ (0.4141525 * float(x[30]))+ (-1.8151586 * float(x[31]))+ (0.86194927 * float(x[32]))+ (-3.1827552 * float(x[33]))+ (3.797162 * float(x[34]))+ (1.9190661 * float(x[35]))+ (-1.8579873 * float(x[36]))+ (1.6780502 * float(x[37]))+ (1.3978982 * float(x[38]))+ (2.0448742 * float(x[39]))+ (-3.6992645 * float(x[40]))+ (2.3332472 * float(x[41]))+ (1.1828684 * float(x[42]))+ (-1.6170461 * float(x[43]))+ (1.6816475 * float(x[44]))+ (1.3721231 * float(x[45]))+ (0.5281401 * float(x[46]))+ (2.1947377 * float(x[47]))+ (-0.4356971 * float(x[48]))+ (-1.4658355 * float(x[49]))) + -1.4182239), 0)
    o[0] = (4.5899224 * h_0)+ (2.561968 * h_1)+ (-5.476658 * h_2)+ (-8.328746 * h_3)+ (8.873351 * h_4)+ (10.356889 * h_5)+ (-7.3423114 * h_6) + -3.142574

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

        model_cap=365

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

