#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 21:58:29
# Invocation: btc -server brain.brainome.ai Data/pbcseq.csv -o Models/pbcseq.py -v -v -v -stopat 84.37 -port 8100 -f QC -e 100
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.02%
Model accuracy:                     84.37% (1641/1945 correct)
Improvement over best guess:        34.35% (of possible 49.98%)
Model capacity (MEC):               471 bits
Generalization ratio:               3.48 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     41.85% (814/1945)
True Positives:                     42.52% (827/1945)
False Negatives:                    7.51% (146/1945)
False Positives:                    8.12% (158/1945)
True Pos. Rate/Sensitivity/Recall:  0.85
True Neg. Rate/Specificity:         0.84
Precision:                          0.84
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.15
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
import faulthandler

from bisect import bisect_left

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="pbcseq.csv"


#Number of attributes
num_attr = 18
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



# Calculate energy

import numpy as np
energy_thresholds=np.array([4700143925.36, 4727576509.87, 4727576770.690001, 4727577754.110001, 4727578963.76, 4727579239.245, 4727579649.475, 4727580164.87, 4727580456.950001, 4727581275.535, 4727581652.675, 4727582112.62, 4727583002.424999, 4727583273.559999, 4727583458.915, 4727583755.395, 4727583904.34, 4727584013.48, 4727584135.44, 4727584258.940001, 4727584293.29, 4727584372.705, 4727584468.725, 4727584547.435, 4727584647.585, 4727584690.65, 4727584931.38, 4727585014.120001, 4727585376.915, 4727585406.51, 4727585605.549999, 4727585636.190001, 4727585667.055, 4727585683.65, 4727585717.759999, 4727585765.924999, 4727585781.724999, 4727585951.014999, 4727586078.33, 4727586168.845, 4727586190.06, 4727586298.995, 4727586319.620001, 4727586349.51, 4727586489.695, 4727586537.254999, 4727586561.635, 4727586645.870001, 4727586717.71, 4727586887.15, 4727586908.074999, 4727587265.684999, 4727587374.790001, 4727587589.885, 4727587830.93, 4727587965.309999, 4727588052.63, 4727588175.24, 4727591041.955, 4727591381.105, 4820695215.355, 4913796267.835, 4913796666.33, 4913798157.335, 4966459747.110001, 4966459894.844999, 4966460414.210001, 4966460569.85, 4966460730.040001, 4966460840.065001, 4966461097.78, 4966461179.285, 4966462628.674999, 4966462885.054999, 4966463092.734999, 4966463335.33, 4966463875.155, 4966464125.85, 4966465466.95, 4966465525.905, 4966466111.6449995, 4966466388.59, 4966466634.25, 4966466735.56, 4966467509.125, 4966467819.970001, 4966468155.75, 4966468339.865, 4966469088.255, 4966469464.475, 4966469722.014999, 4966470036.79, 4966470132.31, 4966470165.190001, 4966470322.18, 4966470520.824999, 4966471006.355, 4966471090.974999, 4966471306.38, 4966471983.490001, 4966473603.15, 5152676433.125, 5152679053.809999, 5152679612.615, 5205344804.135, 5205345808.95, 5874296235.445, 6384465697.950001, 6411902451.844999, 6411903905.17, 6411904256.42, 6411905384.705, 6411905688.315, 6411907039.440001, 6411907216.58, 6411907257.184999, 6411907289.92, 6411907786.54, 6411907818.11, 6411908411.935, 6411908456.725, 6411909409.455, 6411909433.035, 6411909556.065, 6411909668.84, 6411909709.0, 6411909739.935, 6411910721.05, 6411910856.755, 6411911056.875, 6411911544.955, 6411911588.155, 6411911962.785, 6411912169.225, 6411912348.035, 6411912538.525, 6411912672.84, 6411912983.005, 6411913920.315001, 6411913987.295, 6411914130.119999, 6411914866.584999, 6411915749.525, 6439349150.475, 6466781623.115, 6466782545.674999, 6466782775.424999, 6466785573.309999, 6466786723.165, 6466788351.735001, 6466788575.235001, 6466790649.65, 6466790969.904999, 6598120623.799999, 6598121012.934999, 6598122716.375, 6624453695.635, 6650784333.514999, 6650784538.514999, 6650784882.93, 6650785532.085, 6650785647.23, 6650785896.48, 6650786033.105, 6650787080.63, 6650788028.92, 6650788407.46, 6650788551.995, 6650788727.445, 6650789280.095, 6650789320.77, 6650789683.845, 6650789740.15, 6650790086.2, 6650790588.045, 6650790905.96, 6650791137.940001, 6650791482.09, 6650791573.125, 6650791654.335, 6650791950.725, 6650792243.18, 6650792583.469999, 6650792771.190001, 6650793074.045, 6650793087.315001, 6650793159.58, 6650793223.055, 6650793258.225, 6650793602.59, 6650794391.725, 6650794474.695, 6650794719.469999, 6650795092.275, 6650796164.18, 6650796326.77, 6650798561.68, 6650800145.755, 6651899013.690001, 6652999943.645, 6653005963.575, 6679336779.905, 6705664083.88, 6705664218.71, 6705664321.799999, 6705664552.719999, 6705665033.275, 6705665523.83, 6705665844.59, 6705667033.325, 6705668264.145, 6705668353.530001, 6705668460.525, 6705668654.82, 6705670221.764999, 6705670651.07, 6705671057.64, 6705672297.24, 6705672558.605, 6705672733.335, 6705672900.764999, 6705673308.924999, 6705673605.09, 6705673675.435, 6705673814.49, 6705674567.76, 6705676925.025, 6705679099.085, 6771339056.82, 6836997696.07, 6836999592.48, 6837001752.505, 6837003274.465, 6837005051.449999, 6837008228.860001, 6863337487.91, 6891879142.065001, 6918215201.72, 7517087911.21, 7903411490.45, 7903411659.804999, 7903411941.55, 7903412459.929999, 7903413579.105, 7903413771.33, 7903414070.85, 7903414086.54, 7903414225.125, 7903414460.255001, 7903414571.91, 7903415507.785001, 7903417030.32, 7903417153.48, 7903417799.974999, 7903417838.629999, 7903418226.754999, 7903418331.34, 7903418715.030001, 7903418876.21, 7903419541.995, 7903419696.795, 7903420175.16, 7903420621.765, 7903420707.33, 7903420798.095, 7903420923.355, 7903421027.865, 7903421157.23, 7903421169.23, 7903421268.119999, 7903421447.12, 7903421505.82, 7903421809.63, 7903421847.645, 7903422220.25, 7903422373.665, 7903422561.815, 7903422644.309999, 7903423043.625, 7903424193.29, 7903424717.535, 7903424808.74, 7903425385.325, 7903425573.459999, 8089622805.365, 8089626999.145, 8089628776.389999, 8089629161.165, 8096232742.45, 8119262626.84, 8142293476.66, 8142295505.815001, 8142296463.715, 8142296990.38, 8142297293.67, 8142297516.375, 8142297861.559999, 8142297932.045, 8142298525.885, 8142299075.9, 8142300827.465, 8142301014.110001, 8142301189.264999, 8142301409.280001, 8142301476.17, 8142301691.325001, 8142302063.905, 8142302373.75, 8142302640.545, 8142302711.44, 8142303061.09, 8142303210.795, 8142303390.16, 8142303652.295, 8142303825.825001, 8142304022.08, 8142304300.839999, 8142304448.985001, 8142304671.875, 8142304820.98, 8142304896.065001, 8142305119.935, 8142306437.76, 8142306584.01, 8142306784.035, 8239808648.045, 8328504455.325001, 8328507376.375, 8328510065.849999, 8328511976.035, 8381181571.440001, 8381181935.365, 8381182404.665, 8381183448.01, 8381186491.725, 8381187454.215, 8381187824.905001, 8381188629.295, 8389992852.57, 8389995199.914999, 8390000079.110001, 8567397458.135, 8601438044.895, 8721980366.105, 9201411888.474998, 9587739951.455, 9587740008.33, 9587740571.04, 9587740898.515, 9587742282.68, 9587742818.825, 9587744081.735, 9587744136.289999, 9587744210.21, 9587744283.435001, 9587744590.204998, 9587744719.64, 9587744824.664997, 9587744923.98, 9587745175.625, 9587745378.975002, 9587745673.305, 9587745762.650002, 9587745901.39, 9587746777.435, 9587747288.755, 9587747928.265, 9587748988.27, 9587749376.365, 9642618066.37, 9642618780.755001, 9642620832.27, 9642620915.095001, 9642621559.255001, 9642621650.335001, 9707187446.245, 9772844440.195, 9773953135.59, 9773953639.205002, 9773953996.095001, 9800285605.915, 9826617119.505001, 9826618300.735, 9826618747.33, 9826619057.274998, 9826619280.375, 9826619653.22, 9826619974.675, 9826620170.18, 9826620430.369999, 9826620536.305, 9826620630.244999, 9826621660.904999, 9826622025.849998, 9826622092.39, 9826622938.255001, 9826623070.619999, 9826623248.565, 9826623361.9, 9826623539.35, 9826623742.33, 9826623823.88, 9826624105.955, 9826624276.905, 9826624425.735, 9826624507.48, 9826625087.345, 9826625493.8, 9826626317.705, 9826627012.154999, 9826627164.275, 9826627272.945, 9826627293.82, 9826627304.585, 9826627318.725, 9826627358.050001, 9826627458.83, 9826627685.285, 9826629156.869999, 9826629450.255001, 9826629586.67, 9826629724.779999, 9826630071.009998, 9826630272.595001, 9826630446.43, 9826630838.775002, 9827728290.785, 9881495531.155, 9881497044.244999, 9881500197.815, 9881502343.849998, 9881502535.83, 9881502837.965, 9881504345.46, 9881504507.25, 9881506222.275, 9881508718.26, 9881512767.220001, 9881513814.420002, 9947171338.940002, 10012829654.64, 10012834096.895, 10012834232.955002, 10012834622.225, 10012834956.169998, 10012835258.21, 10012838417.384998, 10012839022.855, 10012839833.380001, 10042466659.335001, 10065500223.16, 10065502287.349998, 10065505762.865002, 10065506569.225, 10065507069.425, 10065507255.575, 10065512196.755001, 10067714444.565, 10067717059.39, 10067718744.43, 10816773025.810001, 11326942891.23, 11326946286.895, 11326951296.005001, 11355119229.425, 11355120752.4, 11512057312.895, 11527246393.79, 11565827863.724998, 13167836483.73, 14717172000.2])
def eqenergy(rows):
    return np.sum(rows,axis=1)
def classify(rows):
    energys=eqenergy(rows)
    start_label=1
    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys=np.argwhere(np.logical_and(numers<len(energy_thresholds),numers>=0)).reshape(-1)
        defaultindys=np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds),numers>=0))).reshape(-1)
        outputs=np.zeros(input_energys.shape[0])
        outputs[indys]=(numers[indys]+start_label)%2
        outputs[defaultindys]=0
        return outputs
    return thresh_search(energys)

numthresholds=471


# Main method
model_cap=numthresholds
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    if numthresholds<10:
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
                    for i,valrow in enumerate(valcsvreader):
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








    else:
        if not args.validate: # Then predict
            if args.cleanfile:
                cleanarr=np.loadtxt(args.csvfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr)
                for k,o in enumerate(outputs):

                    print(str(','.join(str(j) for j in ([i for i in cleanarr[k]])))+','+str(int(o)))
            else:
                tempdir=tempfile.gettempdir()
                cleanfile=tempdir+os.sep+"clean.csv"
                clean(args.csvfile,cleanfile, -1, args.headerless, True)
                with open(args.csvfile,'r') as dirtycsvfile:
                    dirtycsvreader = csv.reader(dirtycsvfile)
                    if (not args.headerless):
                            print(','.join(next(dirtycsvreader, None)+['Prediction']))
                    cleanarr=np.loadtxt(cleanfile,delimiter=',',dtype='float64')
                    outputs=classify(cleanarr)
                    for k,dirtyrow in enumerate(dirtycsvreader):

                        print(str(','.join(str(j) for j in ([i for i in dirtyrow])))+','+str(int(outputs[k])))
                os.remove(cleanfile)
                
        else: # Then validate this predictor
            if n_classes==2:
                tempdir=tempfile.gettempdir()
                temp_name = next(tempfile._get_candidate_names())
                cleanvalfile=tempdir+os.sep+temp_name

                clean(args.csvfile,cleanvalfile, -1, args.headerless)
                cleanarr=np.loadtxt(cleanvalfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr[:,:-1])
                count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0=0,0,0,0,0,0,0,0
                correct_count=int(np.sum(outputs.reshape(-1)==cleanarr[:,-1].reshape(-1)))
                count=outputs.shape[0]
                num_TP=int(np.sum(np.logical_and(outputs.reshape(-1)==1,cleanarr[:,-1].reshape(-1)==1)))
                num_TN=int(np.sum(np.logical_and(outputs.reshape(-1)==0,cleanarr[:,-1].reshape(-1)==0)))
                num_FN=int(np.sum(np.logical_and(outputs.reshape(-1)==0,cleanarr[:,-1].reshape(-1)==1)))
                num_FP=int(np.sum(np.logical_and(outputs.reshape(-1)==1,cleanarr[:,-1].reshape(-1)==0)))
                num_class_0=int(np.sum(cleanarr[:,-1].reshape(-1)==0))
                num_class_1=int(np.sum(cleanarr[:,-1].reshape(-1)==1))
            else:
                tempdir=tempfile.gettempdir()
                temp_name = next(tempfile._get_candidate_names())
                cleanvalfile=tempdir+os.sep+temp_name

                clean(args.csvfile,cleanvalfile, -1, args.headerless)
                cleanarr=np.loadtxt(cleanvalfile,delimiter=',',dtype='float64')
                outputs=classify(cleanarr[:,:-1])
                count,correct_count=0,0
                numeachclass={}
                for k,o in enumerate(outputs):
                    if int(o)==int(float(cleanarr[k,-1])):
                        correct_count+=1
                    if int(float(cleanarr[k,-1])) in numeachclass.keys():
                        numeachclass[int(float(cleanarr[k,-1]))]+=1
                    else:
                        numeachclass[int(float(cleanarr[k,-1]))]=0
                    count+=1


    

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
    

