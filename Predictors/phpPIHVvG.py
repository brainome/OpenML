#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:27:59
# Invocation: btc -v -v phpPIHVvG-10.csv -o phpPIHVvG-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                86.00%
Model accuracy:                     94.32% (3093/3279 correct)
Improvement over best guess:        8.32% (of possible 14.0%)
Model capacity (MEC):               226 bits
Generalization ratio:               13.68 bits/bit
Model efficiency:                   0.03%/parameter
System behavior
True Negatives:                     83.96% (2753/3279)
True Positives:                     10.37% (340/3279)
False Negatives:                    3.63% (119/3279)
False Positives:                    2.04% (67/3279)
True Pos. Rate/Sensitivity/Recall:  0.74
True Neg. Rate/Specificity:         0.98
Precision:                          0.84
F-1 Measure:                        0.79
False Negative Rate/Miss Rate:      0.26
Critical Success Index:             0.65
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
TRAINFILE="phpPIHVvG-10.csv"


#Number of attributes
num_attr = 1558

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
    if (energy>711.4834000000001):
        return 0.0
    if (energy>696.5):
        return 1.0
    if (energy>633.5101999999999):
        return 0.0
    if (energy>613.4794999999999):
        return 1.0
    if (energy>601.2384):
        return 0.0
    if (energy>587.55905):
        return 1.0
    if (energy>585.55905):
        return 0.0
    if (energy>571.32705):
        return 1.0
    if (energy>570.32705):
        return 0.0
    if (energy>547.8):
        return 1.0
    if (energy>547.8):
        return 0.0
    if (energy>546.7013):
        return 1.0
    if (energy>546.2013):
        return 0.0
    if (energy>536.50585):
        return 1.0
    if (energy>526.85745):
        return 0.0
    if (energy>523.6647):
        return 1.0
    if (energy>514.6128):
        return 0.0
    if (energy>503.51265):
        return 1.0
    if (energy>497.65155):
        return 0.0
    if (energy>489.02025000000003):
        return 1.0
    if (energy>475.6395):
        return 0.0
    if (energy>474.12694999999997):
        return 1.0
    if (energy>471.8333):
        return 0.0
    if (energy>471.42420000000004):
        return 1.0
    if (energy>471.09090000000003):
        return 0.0
    if (energy>468.0):
        return 1.0
    if (energy>464.08095000000003):
        return 0.0
    if (energy>463.90815):
        return 1.0
    if (energy>462.40255):
        return 0.0
    if (energy>459.8696):
        return 1.0
    if (energy>433.69759999999997):
        return 0.0
    if (energy>432.95615):
        return 1.0
    if (energy>401.4137):
        return 0.0
    if (energy>400.21614999999997):
        return 1.0
    if (energy>390.75):
        return 0.0
    if (energy>388.0):
        return 1.0
    if (energy>360.28905):
        return 0.0
    if (energy>359.3989):
        return 1.0
    if (energy>358.829):
        return 0.0
    if (energy>357.944):
        return 1.0
    if (energy>351.1821):
        return 0.0
    if (energy>350.6821):
        return 1.0
    if (energy>333.06425):
        return 0.0
    if (energy>332.84389999999996):
        return 1.0
    if (energy>332.34389999999996):
        return 0.0
    if (energy>330.499):
        return 1.0
    if (energy>328.999):
        return 0.0
    if (energy>325.8441):
        return 1.0
    if (energy>325.2593):
        return 0.0
    if (energy>324.91650000000004):
        return 1.0
    if (energy>322.9045):
        return 0.0
    if (energy>322.85405000000003):
        return 1.0
    if (energy>314.733):
        return 0.0
    if (energy>313.26835):
        return 1.0
    if (energy>311.3132):
        return 0.0
    if (energy>310.05185):
        return 1.0
    if (energy>309.55185):
        return 0.0
    if (energy>308.5625):
        return 1.0
    if (energy>307.03625):
        return 0.0
    if (energy>306.909):
        return 1.0
    if (energy>295.69165):
        return 0.0
    if (energy>292.66665):
        return 1.0
    if (energy>292.16665):
        return 0.0
    if (energy>291.37185):
        return 1.0
    if (energy>289.11620000000005):
        return 0.0
    if (energy>288.6487):
        return 1.0
    if (energy>283.2781):
        return 0.0
    if (energy>282.9133):
        return 1.0
    if (energy>281.19065):
        return 0.0
    if (energy>280.61505):
        return 1.0
    if (energy>277.9738):
        return 0.0
    if (energy>277.69845):
        return 1.0
    if (energy>277.21565):
        return 0.0
    if (energy>276.53285000000005):
        return 1.0
    if (energy>274.04075):
        return 0.0
    if (energy>273.3482):
        return 1.0
    if (energy>272.33500000000004):
        return 0.0
    if (energy>271.4924):
        return 1.0
    if (energy>265.8):
        return 0.0
    if (energy>264.97055):
        return 1.0
    if (energy>257.0):
        return 0.0
    if (energy>256.5):
        return 1.0
    if (energy>256.0):
        return 0.0
    if (energy>255.87095):
        return 1.0
    if (energy>251.84795):
        return 0.0
    if (energy>250.96134999999998):
        return 1.0
    if (energy>245.69535):
        return 0.0
    if (energy>244.92835000000002):
        return 1.0
    if (energy>243.7243):
        return 0.0
    if (energy>243.5833):
        return 1.0
    if (energy>243.0833):
        return 0.0
    if (energy>241.87875):
        return 1.0
    if (energy>239.7369):
        return 0.0
    if (energy>239.3333):
        return 1.0
    if (energy>231.95695):
        return 0.0
    if (energy>231.87815):
        return 1.0
    if (energy>231.31965):
        return 0.0
    if (energy>230.50455):
        return 1.0
    if (energy>221.05505):
        return 0.0
    if (energy>220.7817):
        return 1.0
    if (energy>218.16665):
        return 0.0
    if (energy>217.875):
        return 1.0
    if (energy>217.54165):
        return 0.0
    if (energy>217.3333):
        return 1.0
    if (energy>217.3333):
        return 0.0
    if (energy>217.29165):
        return 1.0
    if (energy>216.16665):
        return 0.0
    if (energy>216.0):
        return 1.0
    if (energy>212.1611):
        return 0.0
    if (energy>212.0):
        return 1.0
    if (energy>212.0):
        return 0.0
    if (energy>211.8846):
        return 1.0
    if (energy>210.07365):
        return 0.0
    if (energy>209.4245):
        return 1.0
    if (energy>207.0217):
        return 0.0
    if (energy>206.72854999999998):
        return 1.0
    if (energy>206.1695):
        return 0.0
    if (energy>206.13885):
        return 1.0
    if (energy>202.2031):
        return 0.0
    if (energy>201.31345):
        return 1.0
    if (energy>197.5944):
        return 0.0
    if (energy>197.2779):
        return 1.0
    if (energy>196.20125000000002):
        return 0.0
    if (energy>195.9444):
        return 1.0
    if (energy>194.714):
        return 0.0
    if (energy>194.20775):
        return 1.0
    if (energy>194.10795000000002):
        return 0.0
    if (energy>193.64260000000002):
        return 1.0
    if (energy>190.03125):
        return 0.0
    if (energy>189.63330000000002):
        return 1.0
    if (energy>186.26045):
        return 0.0
    if (energy>186.12984999999998):
        return 1.0
    if (energy>186.01875):
        return 0.0
    if (energy>186.0):
        return 1.0
    if (energy>184.94315):
        return 0.0
    if (energy>184.61645):
        return 1.0
    if (energy>181.68329999999997):
        return 0.0
    if (energy>181.6094):
        return 1.0
    if (energy>180.7063):
        return 0.0
    if (energy>180.31585):
        return 1.0
    if (energy>179.59165000000002):
        return 0.0
    if (energy>179.4708):
        return 1.0
    if (energy>179.22915):
        return 0.0
    if (energy>178.96625):
        return 1.0
    if (energy>174.81349999999998):
        return 0.0
    if (energy>174.50310000000002):
        return 1.0
    if (energy>174.07235):
        return 0.0
    if (energy>173.9651):
        return 1.0
    if (energy>172.0):
        return 0.0
    if (energy>171.98805):
        return 1.0
    if (energy>163.27965):
        return 0.0
    if (energy>163.07):
        return 1.0
    if (energy>162.23665):
        return 0.0
    if (energy>161.83085):
        return 1.0
    if (energy>160.67915):
        return 0.0
    if (energy>160.5525):
        return 1.0
    if (energy>159.74020000000002):
        return 0.0
    if (energy>159.44295):
        return 1.0
    if (energy>157.57945):
        return 0.0
    if (energy>157.39995):
        return 1.0
    if (energy>150.01605):
        return 0.0
    if (energy>149.8476):
        return 1.0
    if (energy>148.2226):
        return 0.0
    if (energy>147.8362):
        return 1.0
    if (energy>137.1583):
        return 0.0
    if (energy>137.10775):
        return 1.0
    if (energy>131.91935):
        return 0.0
    if (energy>131.83870000000002):
        return 1.0
    if (energy>131.73985):
        return 0.0
    if (energy>131.69815):
        return 1.0
    if (energy>114.7333):
        return 0.0
    if (energy>114.39994999999999):
        return 1.0
    if (energy>33.5):
        return 0.0
    if (energy>27.0):
        return 1.0
    if (energy>27.0):
        return 0.0
    if (energy>27.0):
        return 1.0
    if (energy>26.0):
        return 0.0
    if (energy>26.0):
        return 1.0
    if (energy>23.0):
        return 0.0
    if (energy>23.0):
        return 1.0
    if (energy>23.0):
        return 0.0
    if (energy>23.0):
        return 1.0
    if (energy>22.0):
        return 0.0
    if (energy>21.5):
        return 1.0
    if (energy>21.0):
        return 0.0
    if (energy>21.0):
        return 1.0
    if (energy>16.5):
        return 0.0
    if (energy>16.0):
        return 1.0
    if (energy>16.0):
        return 0.0
    if (energy>16.0):
        return 1.0
    if (energy>16.0):
        return 0.0
    if (energy>16.0):
        return 1.0
    if (energy>16.0):
        return 0.0
    if (energy>16.0):
        return 1.0
    if (energy>15.0):
        return 0.0
    if (energy>15.0):
        return 1.0
    if (energy>14.0):
        return 0.0
    if (energy>14.0):
        return 1.0
    if (energy>12.0):
        return 0.0
    if (energy>12.0):
        return 1.0
    if (energy>12.0):
        return 0.0
    if (energy>12.0):
        return 1.0
    if (energy>9.5):
        return 0.0
    if (energy>9.0):
        return 1.0
    if (energy>9.0):
        return 0.0
    if (energy>9.0):
        return 1.0
    if (energy>9.0):
        return 0.0
    if (energy>9.0):
        return 1.0
    if (energy>9.0):
        return 0.0
    if (energy>9.0):
        return 1.0
    if (energy>8.0):
        return 0.0
    if (energy>8.0):
        return 1.0
    if (energy>7.0):
        return 0.0
    if (energy>7.0):
        return 1.0
    if (energy>6.0):
        return 0.0
    if (energy>6.0):
        return 1.0
    if (energy>6.0):
        return 0.0
    if (energy>6.0):
        return 1.0
    if (energy>6.0):
        return 0.0
    if (energy>6.0):
        return 1.0
    if (energy>5.5):
        return 0.0
    if (energy>5.0):
        return 1.0
    if (energy>4.0):
        return 0.0
    if (energy>4.0):
        return 1.0
    if (energy>4.0):
        return 0.0
    if (energy>4.0):
        return 1.0
    return 0.0

numthresholds=226


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

