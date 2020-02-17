#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:48:00
# Invocation: btc -v -v dataset_37_diabetes-1.csv -o dataset_37_diabetes-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                65.10%
Model accuracy:                     79.03% (607/768 correct)
Improvement over best guess:        13.93% (of possible 34.9%)
Model capacity (MEC):               158 bits
Generalization ratio:               3.84 bits/bit
Model efficiency:                   0.08%/parameter
System behavior
True Negatives:                     23.96% (184/768)
True Positives:                     55.08% (423/768)
False Negatives:                    10.03% (77/768)
False Positives:                    10.94% (84/768)
True Pos. Rate/Sensitivity/Recall:  0.85
True Neg. Rate/Specificity:         0.69
Precision:                          0.83
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.15
Critical Success Index:             0.72
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
TRAINFILE="dataset_37_diabetes-1.csv"


#Number of attributes
num_attr = 8

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
    if (energy>992.1924999999999):
        return 1.0
    if (energy>853.001):
        return 0.0
    if (energy>826.1225000000001):
        return 1.0
    if (energy>692.6415):
        return 0.0
    if (energy>684.8299999999999):
        return 1.0
    if (energy>682.3):
        return 0.0
    if (energy>678.639):
        return 1.0
    if (energy>673.436):
        return 0.0
    if (energy>660.1765):
        return 1.0
    if (energy>632.6205):
        return 0.0
    if (energy>625.509):
        return 1.0
    if (energy>619.156):
        return 0.0
    if (energy>603.9594999999999):
        return 1.0
    if (energy>595.3354999999999):
        return 0.0
    if (energy>579.8154999999999):
        return 1.0
    if (energy>560.8344999999999):
        return 0.0
    if (energy>557.406):
        return 1.0
    if (energy>542.8675000000001):
        return 0.0
    if (energy>537.895):
        return 1.0
    if (energy>532.2579999999999):
        return 0.0
    if (energy>525.935):
        return 1.0
    if (energy>523.779):
        return 0.0
    if (energy>521.274):
        return 1.0
    if (energy>511.9335):
        return 0.0
    if (energy>506.91700000000003):
        return 1.0
    if (energy>498.2215):
        return 0.0
    if (energy>495.8705):
        return 1.0
    if (energy>490.07899999999995):
        return 0.0
    if (energy>489.23400000000004):
        return 1.0
    if (energy>487.7105):
        return 0.0
    if (energy>486.64750000000004):
        return 1.0
    if (energy>475.482):
        return 0.0
    if (energy>467.736):
        return 1.0
    if (energy>466.502):
        return 0.0
    if (energy>465.77799999999996):
        return 1.0
    if (energy>460.461):
        return 0.0
    if (energy>452.403):
        return 1.0
    if (energy>450.567):
        return 0.0
    if (energy>449.7345):
        return 1.0
    if (energy>448.8005):
        return 0.0
    if (energy>446.68):
        return 1.0
    if (energy>443.639):
        return 0.0
    if (energy>440.672):
        return 1.0
    if (energy>440.34900000000005):
        return 0.0
    if (energy>436.21450000000004):
        return 1.0
    if (energy>434.305):
        return 0.0
    if (energy>430.48699999999997):
        return 1.0
    if (energy>429.5635):
        return 0.0
    if (energy>429.136):
        return 1.0
    if (energy>420.212):
        return 0.0
    if (energy>406.4435):
        return 1.0
    if (energy>403.90549999999996):
        return 0.0
    if (energy>399.0655):
        return 1.0
    if (energy>397.05899999999997):
        return 0.0
    if (energy>395.352):
        return 1.0
    if (energy>389.1915):
        return 0.0
    if (energy>385.678):
        return 1.0
    if (energy>384.3635):
        return 0.0
    if (energy>382.56899999999996):
        return 1.0
    if (energy>381.53700000000003):
        return 0.0
    if (energy>378.30150000000003):
        return 1.0
    if (energy>375.687):
        return 0.0
    if (energy>374.73749999999995):
        return 1.0
    if (energy>371.218):
        return 0.0
    if (energy>362.01800000000003):
        return 1.0
    if (energy>360.573):
        return 0.0
    if (energy>357.323):
        return 1.0
    if (energy>353.534):
        return 0.0
    if (energy>351.9155):
        return 1.0
    if (energy>351.7065):
        return 0.0
    if (energy>350.8845):
        return 1.0
    if (energy>350.38):
        return 0.0
    if (energy>350.26):
        return 1.0
    if (energy>350.1465):
        return 0.0
    if (energy>345.37):
        return 1.0
    if (energy>344.755):
        return 0.0
    if (energy>342.3315):
        return 1.0
    if (energy>341.374):
        return 0.0
    if (energy>338.755):
        return 1.0
    if (energy>336.827):
        return 0.0
    if (energy>334.4):
        return 1.0
    if (energy>333.7595):
        return 0.0
    if (energy>332.53200000000004):
        return 1.0
    if (energy>331.524):
        return 0.0
    if (energy>321.8505):
        return 1.0
    if (energy>321.33299999999997):
        return 0.0
    if (energy>318.868):
        return 1.0
    if (energy>318.51750000000004):
        return 0.0
    if (energy>317.56050000000005):
        return 1.0
    if (energy>316.919):
        return 0.0
    if (energy>312.894):
        return 1.0
    if (energy>312.14649999999995):
        return 0.0
    if (energy>311.142):
        return 1.0
    if (energy>310.59900000000005):
        return 0.0
    if (energy>309.23699999999997):
        return 1.0
    if (energy>308.2425):
        return 0.0
    if (energy>304.9535):
        return 1.0
    if (energy>302.573):
        return 0.0
    if (energy>299.7395):
        return 1.0
    if (energy>299.0195):
        return 0.0
    if (energy>298.7875):
        return 1.0
    if (energy>298.2545):
        return 0.0
    if (energy>295.617):
        return 1.0
    if (energy>293.797):
        return 0.0
    if (energy>293.51700000000005):
        return 1.0
    if (energy>292.96450000000004):
        return 0.0
    if (energy>292.35200000000003):
        return 1.0
    if (energy>291.67600000000004):
        return 0.0
    if (energy>289.2505):
        return 1.0
    if (energy>288.55899999999997):
        return 0.0
    if (energy>284.349):
        return 1.0
    if (energy>283.5315):
        return 0.0
    if (energy>278.4855):
        return 1.0
    if (energy>278.41099999999994):
        return 0.0
    if (energy>271.9325):
        return 1.0
    if (energy>271.17):
        return 0.0
    if (energy>269.06050000000005):
        return 1.0
    if (energy>267.725):
        return 0.0
    if (energy>266.79150000000004):
        return 1.0
    if (energy>266.265):
        return 0.0
    if (energy>265.6595):
        return 1.0
    if (energy>265.236):
        return 0.0
    if (energy>264.7105):
        return 1.0
    if (energy>263.63249999999994):
        return 0.0
    if (energy>262.161):
        return 1.0
    if (energy>261.966):
        return 0.0
    if (energy>259.37399999999997):
        return 1.0
    if (energy>258.418):
        return 0.0
    if (energy>257.361):
        return 1.0
    if (energy>256.8515):
        return 0.0
    if (energy>252.12400000000002):
        return 1.0
    if (energy>251.4595):
        return 0.0
    if (energy>250.881):
        return 1.0
    if (energy>249.98399999999998):
        return 0.0
    if (energy>249.18349999999998):
        return 1.0
    if (energy>248.2235):
        return 0.0
    if (energy>244.8335):
        return 1.0
    if (energy>243.95):
        return 0.0
    if (energy>238.892):
        return 1.0
    if (energy>238.72050000000002):
        return 0.0
    if (energy>237.211):
        return 1.0
    if (energy>236.19299999999998):
        return 0.0
    if (energy>227.3995):
        return 1.0
    if (energy>226.07150000000001):
        return 0.0
    if (energy>220.91049999999998):
        return 1.0
    if (energy>220.286):
        return 0.0
    if (energy>203.839):
        return 1.0
    if (energy>201.576):
        return 0.0
    if (energy>200.5805):
        return 1.0
    if (energy>200.2055):
        return 0.0
    if (energy>197.00349999999997):
        return 1.0
    if (energy>190.81799999999998):
        return 0.0
    if (energy>175.9495):
        return 1.0
    if (energy>174.064):
        return 0.0
    if (energy>171.0355):
        return 1.0
    if (energy>168.07850000000002):
        return 0.0
    if (energy>156.781):
        return 1.0
    if (energy>150.725):
        return 0.0
    return 1.0

numthresholds=158


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

