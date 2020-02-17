#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 12:57:43
# Invocation: btc -target Home/Away -v -v profb-3.csv -o profb-3.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                66.66%
Model accuracy:                     78.42% (527/672 correct)
Improvement over best guess:        11.76% (of possible 33.34%)
Model capacity (MEC):               174 bits
Generalization ratio:               3.02 bits/bit
Model efficiency:                   0.06%/parameter
System behavior
True Negatives:                     21.28% (143/672)
True Positives:                     57.14% (384/672)
False Negatives:                    9.52% (64/672)
False Positives:                    12.05% (81/672)
True Pos. Rate/Sensitivity/Recall:  0.86
True Neg. Rate/Specificity:         0.64
Precision:                          0.83
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.14
Critical Success Index:             0.73
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
TRAINFILE="profb-3.csv"


#Number of attributes
num_attr = 9

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
    if (energy>13653609042.25):
        return 0.0
    if (energy>12785708212.75):
        return 1.0
    if (energy>12071597805.0):
        return 0.0
    if (energy>11932616725.25):
        return 1.0
    if (energy>11760672262.0):
        return 0.0
    if (energy>11592070213.0):
        return 1.0
    if (energy>11474761589.0):
        return 0.0
    if (energy>11326419526.75):
        return 1.0
    if (energy>11129093128.75):
        return 0.0
    if (energy>10977614315.25):
        return 1.0
    if (energy>10972031978.25):
        return 0.0
    if (energy>10922338620.5):
        return 1.0
    if (energy>10806762391.0):
        return 0.0
    if (energy>10743729092.0):
        return 1.0
    if (energy>10719689610.0):
        return 0.0
    if (energy>10603875805.0):
        return 1.0
    if (energy>10509691462.75):
        return 0.0
    if (energy>10246792939.75):
        return 1.0
    if (energy>10170300834.75):
        return 0.0
    if (energy>10090617956.75):
        return 1.0
    if (energy>10059161272.25):
        return 0.0
    if (energy>9972414641.25):
        return 1.0
    if (energy>9964308841.5):
        return 0.0
    if (energy>9905725117.25):
        return 1.0
    if (energy>9860889683.75):
        return 0.0
    if (energy>9828937157.5):
        return 1.0
    if (energy>9815242196.75):
        return 0.0
    if (energy>9736656318.25):
        return 1.0
    if (energy>9718965598.75):
        return 0.0
    if (energy>9701274865.75):
        return 1.0
    if (energy>9673579198.5):
        return 0.0
    if (energy>9625507125.0):
        return 1.0
    if (energy>9601296606.0):
        return 0.0
    if (energy>9600228195.5):
        return 1.0
    if (energy>9600228182.75):
        return 0.0
    if (energy>9533113979.5):
        return 1.0
    if (energy>9502307590.0):
        return 0.0
    if (energy>9477197210.5):
        return 1.0
    if (energy>9443173780.75):
        return 0.0
    if (energy>9392004680.5):
        return 1.0
    if (energy>9370847622.0):
        return 0.0
    if (energy>9361060087.0):
        return 1.0
    if (energy>9357084386.75):
        return 0.0
    if (energy>9293887180.25):
        return 1.0
    if (energy>9273989034.5):
        return 0.0
    if (energy>9241748018.75):
        return 1.0
    if (energy>9209288362.25):
        return 0.0
    if (energy>9100055735.0):
        return 1.0
    if (energy>9080725410.25):
        return 0.0
    if (energy>9038625886.25):
        return 1.0
    if (energy>9009231990.75):
        return 0.0
    if (energy>8945109575.0):
        return 1.0
    if (energy>8945109569.5):
        return 0.0
    if (energy>8898107998.5):
        return 1.0
    if (energy>8884436274.75):
        return 0.0
    if (energy>8872331713.25):
        return 1.0
    if (energy>8869903656.25):
        return 0.0
    if (energy>8848822554.5):
        return 1.0
    if (energy>8848822543.75):
        return 0.0
    if (energy>8758790622.25):
        return 1.0
    if (energy>8740212674.0):
        return 0.0
    if (energy>8720594379.0):
        return 1.0
    if (energy>8710409692.25):
        return 0.0
    if (energy>8682014854.0):
        return 1.0
    if (energy>8648231815.0):
        return 0.0
    if (energy>8621240088.75):
        return 1.0
    if (energy>8615777923.0):
        return 0.0
    if (energy>8565106641.5):
        return 1.0
    if (energy>8545323895.0):
        return 0.0
    if (energy>8442474615.5):
        return 1.0
    if (energy>8442474598.5):
        return 0.0
    if (energy>8442474595.75):
        return 1.0
    if (energy>8440545935.25):
        return 0.0
    if (energy>8321496606.0):
        return 1.0
    if (energy>8295355056.75):
        return 0.0
    if (energy>8193703964.0):
        return 1.0
    if (energy>8177248949.0):
        return 0.0
    if (energy>8144246364.5):
        return 1.0
    if (energy>8133413642.5):
        return 0.0
    if (energy>8125189701.0):
        return 1.0
    if (energy>8125189694.75):
        return 0.0
    if (energy>8094744865.75):
        return 1.0
    if (energy>8086268709.25):
        return 0.0
    if (energy>8051245665.25):
        return 1.0
    if (energy>8015563337.5):
        return 0.0
    if (energy>7954509893.5):
        return 1.0
    if (energy>7948576820.0):
        return 0.0
    if (energy>7939867324.5):
        return 1.0
    if (energy>7918306519.75):
        return 0.0
    if (energy>7845410726.5):
        return 1.0
    if (energy>7808704816.5):
        return 0.0
    if (energy>7776296974.5):
        return 1.0
    if (energy>7773948425.75):
        return 0.0
    if (energy>7752487240.75):
        return 1.0
    if (energy>7752487235.5):
        return 0.0
    if (energy>7717147842.5):
        return 1.0
    if (energy>7717147836.25):
        return 0.0
    if (energy>7639669855.25):
        return 1.0
    if (energy>7632853603.25):
        return 0.0
    if (energy>7626037352.5):
        return 1.0
    if (energy>7626037344.0):
        return 0.0
    if (energy>7590831845.0):
        return 1.0
    if (energy>7554399381.5):
        return 0.0
    if (energy>7503127582.25):
        return 1.0
    if (energy>7412149215.25):
        return 0.0
    if (energy>7387779433.5):
        return 1.0
    if (energy>7369851959.5):
        return 0.0
    if (energy>7357989190.0):
        return 1.0
    if (energy>7352673052.5):
        return 0.0
    if (energy>7251529138.0):
        return 1.0
    if (energy>7242287826.25):
        return 0.0
    if (energy>7234092573.25):
        return 1.0
    if (energy>7217021068.25):
        return 0.0
    if (energy>7137928038.75):
        return 1.0
    if (energy>7091259734.75):
        return 0.0
    if (energy>6994186212.25):
        return 1.0
    if (energy>6977142121.0):
        return 0.0
    if (energy>6875805005.25):
        return 1.0
    if (energy>6874631706.25):
        return 0.0
    if (energy>6866954520.5):
        return 1.0
    if (energy>6859277332.75):
        return 0.0
    if (energy>6814340554.0):
        return 1.0
    if (energy>6758310029.25):
        return 0.0
    if (energy>6747216265.75):
        return 1.0
    if (energy>6747216259.75):
        return 0.0
    if (energy>6708974328.5):
        return 1.0
    if (energy>6682661522.5):
        return 0.0
    if (energy>6625167040.0):
        return 1.0
    if (energy>6592178578.5):
        return 0.0
    if (energy>6562757086.25):
        return 1.0
    if (energy>6424521282.75):
        return 0.0
    if (energy>6348219782.0):
        return 1.0
    if (energy>6338909584.5):
        return 0.0
    if (energy>6263584017.25):
        return 1.0
    if (energy>6256699418.5):
        return 0.0
    if (energy>6249814830.0):
        return 1.0
    if (energy>6211119553.0):
        return 0.0
    if (energy>6141339184.5):
        return 1.0
    if (energy>6096699925.0):
        return 0.0
    if (energy>6083084980.25):
        return 1.0
    if (energy>6083084960.5):
        return 0.0
    if (energy>6081787878.25):
        return 1.0
    if (energy>6020518085.75):
        return 0.0
    if (energy>5982218810.75):
        return 1.0
    if (energy>5943919523.5):
        return 0.0
    if (energy>5932590300.0):
        return 1.0
    if (energy>5920496190.25):
        return 0.0
    if (energy>5681431898.5):
        return 1.0
    if (energy>5642706222.5):
        return 0.0
    if (energy>5513234581.25):
        return 1.0
    if (energy>5509163504.75):
        return 0.0
    if (energy>5509163474.0):
        return 1.0
    if (energy>5458343680.0):
        return 0.0
    if (energy>5352403194.5):
        return 1.0
    if (energy>5286891573.75):
        return 0.0
    if (energy>5276500658.5):
        return 1.0
    if (energy>5276500653.5):
        return 0.0
    if (energy>5209427707.5):
        return 1.0
    if (energy>5154392425.25):
        return 0.0
    if (energy>4673606758.5):
        return 1.0
    if (energy>4626091826.75):
        return 0.0
    if (energy>4575347405.5):
        return 1.0
    if (energy>4544215244.0):
        return 0.0
    if (energy>4517032801.25):
        return 1.0
    if (energy>4426457171.75):
        return 0.0
    if (energy>4357522987.5):
        return 1.0
    if (energy>4243225624.25):
        return 0.0
    if (energy>4181121591.5):
        return 1.0
    if (energy>4096928670.25):
        return 0.0
    if (energy>4018192356.75):
        return 1.0
    if (energy>4006908128.25):
        return 0.0
    if (energy>3995623896.5):
        return 1.0
    if (energy>3983694768.25):
        return 0.0
    if (energy>3790825671.5):
        return 1.0
    return 0.0

numthresholds=174


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

