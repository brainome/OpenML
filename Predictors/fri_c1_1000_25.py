#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:48:40
# Invocation: btc -v -v fri_c1_1000_25-10.csv -o fri_c1_1000_25-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                54.60%
Model accuracy:                     76.20% (762/1000 correct)
Improvement over best guess:        21.60% (of possible 45.4%)
Model capacity (MEC):               235 bits
Generalization ratio:               3.24 bits/bit
Model efficiency:                   0.09%/parameter
System behavior
True Negatives:                     42.50% (425/1000)
True Positives:                     33.70% (337/1000)
False Negatives:                    11.70% (117/1000)
False Positives:                    12.10% (121/1000)
True Pos. Rate/Sensitivity/Recall:  0.74
True Neg. Rate/Specificity:         0.78
Precision:                          0.74
F-1 Measure:                        0.74
False Negative Rate/Miss Rate:      0.26
Critical Success Index:             0.59
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
TRAINFILE="fri_c1_1000_25-10.csv"


#Number of attributes
num_attr = 25

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
    if (energy>14.577269000000001):
        return 1.0
    if (energy>13.4396165):
        return 0.0
    if (energy>12.7671545):
        return 1.0
    if (energy>11.8989975):
        return 0.0
    if (energy>11.733244500000001):
        return 1.0
    if (energy>11.223096000000002):
        return 0.0
    if (energy>11.006377):
        return 1.0
    if (energy>10.4249905):
        return 0.0
    if (energy>10.183885):
        return 1.0
    if (energy>9.985963000000002):
        return 0.0
    if (energy>9.784672):
        return 1.0
    if (energy>9.477671):
        return 0.0
    if (energy>9.285057):
        return 1.0
    if (energy>9.0709175):
        return 0.0
    if (energy>8.979600000000001):
        return 1.0
    if (energy>8.9474995):
        return 0.0
    if (energy>8.801547):
        return 1.0
    if (energy>8.697852000000001):
        return 0.0
    if (energy>8.584824000000001):
        return 1.0
    if (energy>8.531683000000001):
        return 0.0
    if (energy>8.483554000000002):
        return 1.0
    if (energy>7.9224085):
        return 0.0
    if (energy>7.8813075):
        return 1.0
    if (energy>7.8050455):
        return 0.0
    if (energy>7.698264):
        return 1.0
    if (energy>7.427598):
        return 0.0
    if (energy>7.0592665):
        return 1.0
    if (energy>6.716417999999999):
        return 0.0
    if (energy>6.582113):
        return 1.0
    if (energy>6.472748500000001):
        return 0.0
    if (energy>6.286915500000001):
        return 1.0
    if (energy>6.2363505):
        return 0.0
    if (energy>6.071764):
        return 1.0
    if (energy>5.922089000000001):
        return 0.0
    if (energy>5.781087499999999):
        return 1.0
    if (energy>5.762505999999999):
        return 0.0
    if (energy>5.724942):
        return 1.0
    if (energy>5.714852499999999):
        return 0.0
    if (energy>5.5667255):
        return 1.0
    if (energy>5.528305499999999):
        return 0.0
    if (energy>5.4463065):
        return 1.0
    if (energy>5.3364265):
        return 0.0
    if (energy>5.2798535):
        return 1.0
    if (energy>5.2100355):
        return 0.0
    if (energy>5.178990000000001):
        return 1.0
    if (energy>5.1587914999999995):
        return 0.0
    if (energy>5.108495):
        return 1.0
    if (energy>4.928668500000001):
        return 0.0
    if (energy>4.8745674999999995):
        return 1.0
    if (energy>4.827953):
        return 0.0
    if (energy>4.749518500000001):
        return 1.0
    if (energy>4.640137):
        return 0.0
    if (energy>4.533825499999999):
        return 1.0
    if (energy>4.498434):
        return 0.0
    if (energy>4.4709985):
        return 1.0
    if (energy>4.2634):
        return 0.0
    if (energy>4.217131):
        return 1.0
    if (energy>4.100091):
        return 0.0
    if (energy>3.8887055000000004):
        return 1.0
    if (energy>3.7574665000000014):
        return 0.0
    if (energy>3.5901255000000005):
        return 1.0
    if (energy>3.4861515):
        return 0.0
    if (energy>3.4309149999999997):
        return 1.0
    if (energy>3.3027135000000003):
        return 0.0
    if (energy>3.2769440000000003):
        return 1.0
    if (energy>3.2592215):
        return 0.0
    if (energy>3.2156979999999997):
        return 1.0
    if (energy>3.1816109999999993):
        return 0.0
    if (energy>3.1791814999999994):
        return 1.0
    if (energy>3.17028):
        return 0.0
    if (energy>3.123786):
        return 1.0
    if (energy>3.0723480000000003):
        return 0.0
    if (energy>3.065395000000001):
        return 1.0
    if (energy>2.8631414999999993):
        return 0.0
    if (energy>2.8598729999999994):
        return 1.0
    if (energy>2.831989):
        return 0.0
    if (energy>2.710346500000001):
        return 1.0
    if (energy>2.6966390000000002):
        return 0.0
    if (energy>2.6829345):
        return 1.0
    if (energy>2.654019999999999):
        return 0.0
    if (energy>2.6331249999999997):
        return 1.0
    if (energy>2.4407050000000003):
        return 0.0
    if (energy>2.43185):
        return 1.0
    if (energy>2.3594035000000004):
        return 0.0
    if (energy>2.240474999999999):
        return 1.0
    if (energy>2.1786855000000003):
        return 0.0
    if (energy>2.1027025000000004):
        return 1.0
    if (energy>2.076556):
        return 0.0
    if (energy>2.044792):
        return 1.0
    if (energy>1.9961070000000005):
        return 0.0
    if (energy>1.894199):
        return 1.0
    if (energy>1.8574659999999998):
        return 0.0
    if (energy>1.8526529999999997):
        return 1.0
    if (energy>1.7544725000000003):
        return 0.0
    if (energy>1.7385455):
        return 1.0
    if (energy>1.7201274999999998):
        return 0.0
    if (energy>1.7115685000000003):
        return 1.0
    if (energy>1.6912845):
        return 0.0
    if (energy>1.6714215000000001):
        return 1.0
    if (energy>1.6489549999999997):
        return 0.0
    if (energy>1.6462409999999998):
        return 1.0
    if (energy>1.6198919999999997):
        return 0.0
    if (energy>1.4404279999999998):
        return 1.0
    if (energy>1.3731860000000005):
        return 0.0
    if (energy>1.3233925000000002):
        return 1.0
    if (energy>1.2007769999999998):
        return 0.0
    if (energy>1.1416985000000006):
        return 1.0
    if (energy>1.1192010000000003):
        return 0.0
    if (energy>1.038468):
        return 1.0
    if (energy>0.9826329999999994):
        return 0.0
    if (energy>0.9403149999999999):
        return 1.0
    if (energy>0.906326):
        return 0.0
    if (energy>0.8622585):
        return 1.0
    if (energy>0.7263899999999992):
        return 0.0
    if (energy>0.7082534999999998):
        return 1.0
    if (energy>0.6850495000000005):
        return 0.0
    if (energy>0.655545):
        return 1.0
    if (energy>0.599321):
        return 0.0
    if (energy>0.5372284999999999):
        return 1.0
    if (energy>0.4595484999999999):
        return 0.0
    if (energy>0.3588715000000001):
        return 1.0
    if (energy>0.3351304999999998):
        return 0.0
    if (energy>0.2955904999999993):
        return 1.0
    if (energy>0.1446984999999999):
        return 0.0
    if (energy>0.11360600000000082):
        return 1.0
    if (energy>-0.07946349999999987):
        return 0.0
    if (energy>-0.08988000000000028):
        return 1.0
    if (energy>-0.11265700000000001):
        return 0.0
    if (energy>-0.2051424999999999):
        return 1.0
    if (energy>-0.41874199999999956):
        return 0.0
    if (energy>-0.43130499999999955):
        return 1.0
    if (energy>-0.43640899999999977):
        return 0.0
    if (energy>-0.43850850000000013):
        return 1.0
    if (energy>-0.5190239999999998):
        return 0.0
    if (energy>-0.5493279999999997):
        return 1.0
    if (energy>-0.5809395000000002):
        return 0.0
    if (energy>-0.6257084999999997):
        return 1.0
    if (energy>-0.6679470000000003):
        return 0.0
    if (energy>-0.6937760000000001):
        return 1.0
    if (energy>-0.7549165000000003):
        return 0.0
    if (energy>-0.7602455000000001):
        return 1.0
    if (energy>-0.7975199999999998):
        return 0.0
    if (energy>-0.8203485000000007):
        return 1.0
    if (energy>-0.86063):
        return 0.0
    if (energy>-0.8992385000000002):
        return 1.0
    if (energy>-0.9188435000000005):
        return 0.0
    if (energy>-0.9684205000000006):
        return 1.0
    if (energy>-1.0404585000000002):
        return 0.0
    if (energy>-1.0764559999999987):
        return 1.0
    if (energy>-1.1705255):
        return 0.0
    if (energy>-1.3620190000000005):
        return 1.0
    if (energy>-1.4460865):
        return 0.0
    if (energy>-1.6930215000000004):
        return 1.0
    if (energy>-1.7155765):
        return 0.0
    if (energy>-1.7457009999999997):
        return 1.0
    if (energy>-1.7840469999999995):
        return 0.0
    if (energy>-1.9163814999999982):
        return 1.0
    if (energy>-1.9227179999999984):
        return 0.0
    if (energy>-1.9922975000000003):
        return 1.0
    if (energy>-2.2035864999999992):
        return 0.0
    if (energy>-2.2569405000000007):
        return 1.0
    if (energy>-2.316671500000001):
        return 0.0
    if (energy>-2.4013044999999997):
        return 1.0
    if (energy>-2.424846):
        return 0.0
    if (energy>-2.4918389999999997):
        return 1.0
    if (energy>-2.590905499999999):
        return 0.0
    if (energy>-2.609285999999999):
        return 1.0
    if (energy>-2.7220464999999994):
        return 0.0
    if (energy>-2.7548895):
        return 1.0
    if (energy>-2.946115):
        return 0.0
    if (energy>-2.9625335):
        return 1.0
    if (energy>-2.9735385):
        return 0.0
    if (energy>-3.0446009999999992):
        return 1.0
    if (energy>-3.0674080000000004):
        return 0.0
    if (energy>-3.2828125000000004):
        return 1.0
    if (energy>-3.3133365):
        return 0.0
    if (energy>-3.3597355):
        return 1.0
    if (energy>-3.4500474999999993):
        return 0.0
    if (energy>-3.455934):
        return 1.0
    if (energy>-3.48769):
        return 0.0
    if (energy>-3.5238125000000005):
        return 1.0
    if (energy>-3.550328499999999):
        return 0.0
    if (energy>-3.555086499999999):
        return 1.0
    if (energy>-3.5729249999999997):
        return 0.0
    if (energy>-3.7806105000000008):
        return 1.0
    if (energy>-3.8755935):
        return 0.0
    if (energy>-3.940146500000001):
        return 1.0
    if (energy>-3.9619609999999996):
        return 0.0
    if (energy>-3.9784829999999993):
        return 1.0
    if (energy>-4.051337):
        return 0.0
    if (energy>-4.089886):
        return 1.0
    if (energy>-4.122131000000001):
        return 0.0
    if (energy>-4.127907):
        return 1.0
    if (energy>-4.230624000000001):
        return 0.0
    if (energy>-4.307169999999999):
        return 1.0
    if (energy>-4.357175999999999):
        return 0.0
    if (energy>-4.507452000000001):
        return 1.0
    if (energy>-4.6114175):
        return 0.0
    if (energy>-4.6705765):
        return 1.0
    if (energy>-4.711717999999999):
        return 0.0
    if (energy>-4.900368500000001):
        return 1.0
    if (energy>-5.5026285):
        return 0.0
    if (energy>-5.743008500000002):
        return 1.0
    if (energy>-5.748801):
        return 0.0
    if (energy>-5.921627500000001):
        return 1.0
    if (energy>-6.034587500000001):
        return 0.0
    if (energy>-6.0826705):
        return 1.0
    if (energy>-6.18551):
        return 0.0
    if (energy>-6.2253905):
        return 1.0
    if (energy>-6.291769500000001):
        return 0.0
    if (energy>-6.326157):
        return 1.0
    if (energy>-6.582767500000001):
        return 0.0
    if (energy>-6.713361500000001):
        return 1.0
    if (energy>-6.8990855):
        return 0.0
    if (energy>-7.2896095):
        return 1.0
    if (energy>-7.594317):
        return 0.0
    if (energy>-7.626077499999999):
        return 1.0
    if (energy>-7.784962):
        return 0.0
    if (energy>-7.845174999999999):
        return 1.0
    if (energy>-7.921361999999999):
        return 0.0
    if (energy>-7.979367999999999):
        return 1.0
    if (energy>-8.0797925):
        return 0.0
    if (energy>-8.271590500000002):
        return 1.0
    if (energy>-8.6348795):
        return 0.0
    if (energy>-8.768677999999998):
        return 1.0
    if (energy>-8.967960000000001):
        return 0.0
    if (energy>-9.092576000000003):
        return 1.0
    if (energy>-9.211457499999998):
        return 0.0
    if (energy>-10.2399185):
        return 1.0
    if (energy>-10.604245500000001):
        return 0.0
    if (energy>-10.659533499999998):
        return 1.0
    if (energy>-10.7589675):
        return 0.0
    if (energy>-11.1635985):
        return 1.0
    if (energy>-11.378958):
        return 0.0
    if (energy>-13.068005499999998):
        return 1.0
    return 0.0

numthresholds=235


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

