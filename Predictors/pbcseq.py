#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 12:49:07
# Invocation: btc -v -v pbcseq-2.csv -o pbcseq-2.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.02%
Model accuracy:                     84.37% (1641/1945 correct)
Improvement over best guess:        34.35% (of possible 49.98%)
Model capacity (MEC):               479 bits
Generalization ratio:               3.42 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     42.26% (822/1945)
True Positives:                     42.11% (819/1945)
False Negatives:                    7.87% (153/1945)
False Positives:                    7.76% (151/1945)
True Pos. Rate/Sensitivity/Recall:  0.84
True Neg. Rate/Specificity:         0.84
Precision:                          0.84
F-1 Measure:                        0.84
False Negative Rate/Miss Rate:      0.16
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
TRAINFILE="pbcseq-2.csv"


#Number of attributes
num_attr = 18

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
    if (energy>14717171236.385):
        return 0.0
    if (energy>14717169417.0):
        return 1.0
    if (energy>14717167568.635):
        return 0.0
    if (energy>13167835551.150002):
        return 1.0
    if (energy>11685277803.5):
        return 0.0
    if (energy>11326951617.779999):
        return 1.0
    if (energy>11326946286.895):
        return 0.0
    if (energy>11326942891.23):
        return 1.0
    if (energy>10668788206.060001):
        return 0.0
    if (energy>10094051922.720001):
        return 1.0
    if (energy>10067716491.994999):
        return 0.0
    if (energy>10067714673.849998):
        return 1.0
    if (energy>10067714291.525):
        return 0.0
    if (energy>10066613432.59):
        return 1.0
    if (energy>10065512260.61):
        return 0.0
    if (energy>10065507094.66):
        return 1.0
    if (energy>10065507069.425):
        return 0.0
    if (energy>10065506361.075):
        return 1.0
    if (energy>10065505171.725002):
        return 0.0
    if (energy>10019437873.41):
        return 1.0
    if (energy>10012839833.380001):
        return 0.0
    if (energy>10012838996.855):
        return 1.0
    if (energy>10012838324.119999):
        return 0.0
    if (energy>10012835465.235):
        return 1.0
    if (energy>10012835338.285):
        return 0.0
    if (energy>10012835258.21):
        return 1.0
    if (energy>10012835062.01):
        return 0.0
    if (energy>10012834368.265):
        return 1.0
    if (energy>10012831524.210001):
        return 0.0
    if (energy>9947171834.86):
        return 1.0
    if (energy>9881511774.235):
        return 0.0
    if (energy>9881509444.66):
        return 1.0
    if (energy>9881508718.259998):
        return 0.0
    if (energy>9881507295.845):
        return 1.0
    if (energy>9881506737.045):
        return 0.0
    if (energy>9881506172.18):
        return 1.0
    if (energy>9881504603.515):
        return 0.0
    if (energy>9881504345.46):
        return 1.0
    if (energy>9881503000.235):
        return 0.0
    if (energy>9881502826.920002):
        return 1.0
    if (energy>9881502665.455002):
        return 0.0
    if (energy>9881502535.829998):
        return 1.0
    if (energy>9881502343.849998):
        return 0.0
    if (energy>9881500197.815002):
        return 1.0
    if (energy>9881498323.32):
        return 0.0
    if (energy>9881497634.96):
        return 1.0
    if (energy>9881497044.244999):
        return 0.0
    if (energy>9881495531.155):
        return 1.0
    if (energy>9827738166.515):
        return 0.0
    if (energy>9826631052.905):
        return 1.0
    if (energy>9826630718.740002):
        return 0.0
    if (energy>9826630272.595001):
        return 1.0
    if (energy>9826629810.09):
        return 0.0
    if (energy>9826629724.779999):
        return 1.0
    if (energy>9826629586.669998):
        return 0.0
    if (energy>9826629450.255):
        return 1.0
    if (energy>9826629166.005001):
        return 0.0
    if (energy>9826628962.165):
        return 1.0
    if (energy>9826628691.68):
        return 0.0
    if (energy>9826627975.080002):
        return 1.0
    if (energy>9826627395.145):
        return 0.0
    if (energy>9826627343.91):
        return 1.0
    if (energy>9826627151.235):
        return 0.0
    if (energy>9826626999.115002):
        return 1.0
    if (energy>9826626923.77):
        return 0.0
    if (energy>9826626673.23):
        return 1.0
    if (energy>9826626317.704998):
        return 0.0
    if (energy>9826625557.76):
        return 1.0
    if (energy>9826624928.289999):
        return 0.0
    if (energy>9826624604.27):
        return 1.0
    if (energy>9826624425.735):
        return 0.0
    if (energy>9826624318.400002):
        return 1.0
    if (energy>9826624094.93):
        return 0.0
    if (energy>9826623823.88):
        return 1.0
    if (energy>9826623607.455):
        return 0.0
    if (energy>9826623244.85):
        return 1.0
    if (energy>9826623145.460001):
        return 0.0
    if (energy>9826622854.665):
        return 1.0
    if (energy>9826622584.08):
        return 0.0
    if (energy>9826622313.189999):
        return 1.0
    if (energy>9826622092.39):
        return 0.0
    if (energy>9826622004.125):
        return 1.0
    if (energy>9826621896.125):
        return 0.0
    if (energy>9826621456.904999):
        return 1.0
    if (energy>9826621370.49):
        return 0.0
    if (energy>9826620892.075):
        return 1.0
    if (energy>9826620759.62):
        return 0.0
    if (energy>9826620630.245):
        return 1.0
    if (energy>9826620536.305):
        return 0.0
    if (energy>9826620339.08):
        return 1.0
    if (energy>9826620170.18):
        return 0.0
    if (energy>9826619974.675):
        return 1.0
    if (energy>9826619580.66):
        return 0.0
    if (energy>9826619280.375):
        return 1.0
    if (energy>9826618862.239998):
        return 0.0
    if (energy>9826618747.33):
        return 1.0
    if (energy>9826618020.48):
        return 0.0
    if (energy>9800285509.685001):
        return 1.0
    if (energy>9773953639.205):
        return 0.0
    if (energy>9773952695.759998):
        return 1.0
    if (energy>9773952549.755):
        return 0.0
    if (energy>9773952358.24):
        return 1.0
    if (energy>9642624482.98):
        return 0.0
    if (energy>9642624190.244999):
        return 1.0
    if (energy>9642621650.335):
        return 0.0
    if (energy>9642621402.365):
        return 1.0
    if (energy>9642618836.4):
        return 0.0
    if (energy>9642617799.465):
        return 1.0
    if (energy>9587750047.650002):
        return 0.0
    if (energy>9587748813.02):
        return 1.0
    if (energy>9587748401.14):
        return 0.0
    if (energy>9587747332.66):
        return 1.0
    if (energy>9587746673.105):
        return 0.0
    if (energy>9587746585.045002):
        return 1.0
    if (energy>9587746264.275):
        return 0.0
    if (energy>9587745901.39):
        return 1.0
    if (energy>9587745739.46):
        return 0.0
    if (energy>9587745673.305):
        return 1.0
    if (energy>9587745289.535):
        return 0.0
    if (energy>9587745179.905):
        return 1.0
    if (energy>9587744893.945):
        return 0.0
    if (energy>9587744835.795):
        return 1.0
    if (energy>9587744719.64):
        return 0.0
    if (energy>9587744608.485):
        return 1.0
    if (energy>9587743046.05):
        return 0.0
    if (energy>9587742393.03):
        return 1.0
    if (energy>9587740008.33):
        return 0.0
    if (energy>9587739951.455):
        return 1.0
    if (energy>9587739523.84):
        return 0.0
    if (energy>9587739483.480001):
        return 1.0
    if (energy>9587738316.235):
        return 0.0
    if (energy>9587738251.325):
        return 1.0
    if (energy>9587735564.404999):
        return 0.0
    if (energy>9587734426.060001):
        return 1.0
    if (energy>9201410750.130001):
        return 0.0
    if (energy>8721980366.105):
        return 1.0
    if (energy>8598137730.375):
        return 0.0
    if (energy>8567390002.764999):
        return 1.0
    if (energy>8390000079.110001):
        return 0.0
    if (energy>8381188629.295):
        return 1.0
    if (energy>8381187916.84):
        return 0.0
    if (energy>8381182794.264999):
        return 1.0
    if (energy>8381182404.665):
        return 0.0
    if (energy>8381181935.365):
        return 1.0
    if (energy>8381181312.110001):
        return 0.0
    if (energy>8328512048.985):
        return 1.0
    if (energy>8328510065.849999):
        return 0.0
    if (energy>8328507376.375):
        return 1.0
    if (energy>8328504750.610001):
        return 0.0
    if (energy>8239808541.48):
        return 1.0
    if (energy>8142307294.405):
        return 0.0
    if (energy>8142307236.38):
        return 1.0
    if (energy>8142307155.265):
        return 0.0
    if (energy>8142307043.58):
        return 1.0
    if (energy>8142306863.120001):
        return 0.0
    if (energy>8142306593.26):
        return 1.0
    if (energy>8142306447.01):
        return 0.0
    if (energy>8142305761.465):
        return 1.0
    if (energy>8142305354.925):
        return 0.0
    if (energy>8142305152.355):
        return 1.0
    if (energy>8142304896.065001):
        return 0.0
    if (energy>8142304445.76):
        return 1.0
    if (energy>8142304292.115):
        return 0.0
    if (energy>8142303983.75):
        return 1.0
    if (energy>8142303811.234999):
        return 0.0
    if (energy>8142303625.925001):
        return 1.0
    if (energy>8142303503.985001):
        return 0.0
    if (energy>8142303227.665):
        return 1.0
    if (energy>8142303030.18):
        return 0.0
    if (energy>8142302810.174999):
        return 1.0
    if (energy>8142302665.28):
        return 0.0
    if (energy>8142302646.965):
        return 1.0
    if (energy>8142302640.545):
        return 0.0
    if (energy>8142302319.235001):
        return 1.0
    if (energy>8142302157.870001):
        return 0.0
    if (energy>8142301753.139999):
        return 1.0
    if (energy>8142301476.17):
        return 0.0
    if (energy>8142301398.065):
        return 1.0
    if (energy>8142301189.264999):
        return 0.0
    if (energy>8142301014.110001):
        return 1.0
    if (energy>8142300321.325001):
        return 0.0
    if (energy>8142299358.275):
        return 1.0
    if (energy>8142298881.549999):
        return 0.0
    if (energy>8142298623.245):
        return 1.0
    if (energy>8142298092.8):
        return 0.0
    if (energy>8142297932.045):
        return 1.0
    if (energy>8142297772.345):
        return 0.0
    if (energy>8142297516.375):
        return 1.0
    if (energy>8142297293.67):
        return 0.0
    if (energy>8142297058.07):
        return 1.0
    if (energy>8142296556.735):
        return 0.0
    if (energy>8142295402.05):
        return 1.0
    if (energy>8142293477.924999):
        return 0.0
    if (energy>8119263399.1449995):
        return 1.0
    if (energy>8096231301.265001):
        return 0.0
    if (energy>8092931238.59):
        return 1.0
    if (energy>8089631401.52):
        return 0.0
    if (energy>8089629566.41):
        return 1.0
    if (energy>8089628990.535):
        return 0.0
    if (energy>8089628682.84):
        return 1.0
    if (energy>7903425927.799999):
        return 0.0
    if (energy>7903424395.755):
        return 1.0
    if (energy>7903424006.309999):
        return 0.0
    if (energy>7903423995.794999):
        return 1.0
    if (energy>7903423896.17):
        return 0.0
    if (energy>7903422635.235):
        return 1.0
    if (energy>7903422561.815):
        return 0.0
    if (energy>7903422342.73):
        return 1.0
    if (energy>7903422238.844999):
        return 0.0
    if (energy>7903421941.42):
        return 1.0
    if (energy>7903421894.050001):
        return 0.0
    if (energy>7903421847.645):
        return 1.0
    if (energy>7903421737.18):
        return 0.0
    if (energy>7903421580.94):
        return 1.0
    if (energy>7903421473.844999):
        return 0.0
    if (energy>7903421291.285):
        return 1.0
    if (energy>7903421169.23):
        return 0.0
    if (energy>7903421152.59):
        return 1.0
    if (energy>7903421071.905):
        return 0.0
    if (energy>7903420528.26):
        return 1.0
    if (energy>7903420216.620001):
        return 0.0
    if (energy>7903420100.035):
        return 1.0
    if (energy>7903419908.38):
        return 0.0
    if (energy>7903419786.360001):
        return 1.0
    if (energy>7903419079.255):
        return 0.0
    if (energy>7903418777.105):
        return 1.0
    if (energy>7903417850.54):
        return 0.0
    if (energy>7903417799.974999):
        return 1.0
    if (energy>7903417284.35):
        return 0.0
    if (energy>7903417030.32):
        return 1.0
    if (energy>7903415389.485001):
        return 0.0
    if (energy>7903414719.965):
        return 1.0
    if (energy>7903414460.255):
        return 0.0
    if (energy>7903414225.125):
        return 1.0
    if (energy>7903414102.58):
        return 0.0
    if (energy>7903414070.85):
        return 1.0
    if (energy>7903413983.855):
        return 0.0
    if (energy>7903413645.965):
        return 1.0
    if (energy>7903412291.705):
        return 0.0
    if (energy>7903412087.6):
        return 1.0
    if (energy>7903411659.804999):
        return 0.0
    if (energy>7903411517.379999):
        return 1.0
    if (energy>7489646329.25):
        return 0.0
    if (energy>6891883449.85):
        return 1.0
    if (energy>6891879476.684999):
        return 0.0
    if (energy>6889671328.99):
        return 1.0
    if (energy>6889670908.825001):
        return 0.0
    if (energy>6837005051.450001):
        return 1.0
    if (energy>6837002857.02):
        return 0.0
    if (energy>6837000923.385):
        return 1.0
    if (energy>6836999592.48):
        return 0.0
    if (energy>6836997696.07):
        return 1.0
    if (energy>6771339056.820001):
        return 0.0
    if (energy>6705678982.84):
        return 1.0
    if (energy>6705676925.025):
        return 0.0
    if (energy>6705674543.07):
        return 1.0
    if (energy>6705672863.41):
        return 0.0
    if (energy>6705672733.335):
        return 1.0
    if (energy>6705672558.605):
        return 0.0
    if (energy>6705672054.035):
        return 1.0
    if (energy>6705671419.545):
        return 0.0
    if (energy>6705670701.125):
        return 1.0
    if (energy>6705670158.75):
        return 0.0
    if (energy>6705669981.53):
        return 1.0
    if (energy>6705669782.69):
        return 0.0
    if (energy>6705668721.66):
        return 1.0
    if (energy>6705666467.889999):
        return 0.0
    if (energy>6705666161.165):
        return 1.0
    if (energy>6705665356.455):
        return 0.0
    if (energy>6705664552.719999):
        return 1.0
    if (energy>6705664332.674999):
        return 0.0
    if (energy>6653001140.05):
        return 1.0
    if (energy>6651899013.69):
        return 0.0
    if (energy>6650800145.754999):
        return 1.0
    if (energy>6650797461.775):
        return 0.0
    if (energy>6650796326.77):
        return 1.0
    if (energy>6650796164.18):
        return 0.0
    if (energy>6650795100.415001):
        return 1.0
    if (energy>6650794599.934999):
        return 0.0
    if (energy>6650794474.695):
        return 1.0
    if (energy>6650794379.6):
        return 0.0
    if (energy>6650793682.97):
        return 1.0
    if (energy>6650793638.15):
        return 0.0
    if (energy>6650793602.59):
        return 1.0
    if (energy>6650793481.06):
        return 0.0
    if (energy>6650793285.125):
        return 1.0
    if (energy>6650793258.224999):
        return 0.0
    if (energy>6650793206.545):
        return 1.0
    if (energy>6650793143.070001):
        return 0.0
    if (energy>6650792771.190001):
        return 1.0
    if (energy>6650792724.29):
        return 0.0
    if (energy>6650792285.46):
        return 1.0
    if (energy>6650791950.725):
        return 0.0
    if (energy>6650791636.165):
        return 1.0
    if (energy>6650791573.125):
        return 0.0
    if (energy>6650791440.295):
        return 1.0
    if (energy>6650791131.7699995):
        return 0.0
    if (energy>6650790926.655):
        return 1.0
    if (energy>6650790588.045):
        return 0.0
    if (energy>6650790154.8):
        return 1.0
    if (energy>6650789740.15):
        return 0.0
    if (energy>6650789657.715):
        return 1.0
    if (energy>6650789320.77):
        return 0.0
    if (energy>6650788819.84):
        return 1.0
    if (energy>6650788360.035):
        return 0.0
    if (energy>6650787972.01):
        return 1.0
    if (energy>6650787279.299999):
        return 0.0
    if (energy>6650785826.514999):
        return 1.0
    if (energy>6650785532.084999):
        return 0.0
    if (energy>6650784728.209999):
        return 1.0
    if (energy>6650784443.849998):
        return 0.0
    if (energy>6624452914.540001):
        return 1.0
    if (energy>6466793389.320001):
        return 0.0
    if (energy>6466792899.695):
        return 1.0
    if (energy>6466791621.365):
        return 0.0
    if (energy>6466790433.1):
        return 1.0
    if (energy>6466788575.235001):
        return 0.0
    if (energy>6466788351.735001):
        return 1.0
    if (energy>6466786723.165):
        return 0.0
    if (energy>6466786343.635):
        return 1.0
    if (energy>6466782656.725):
        return 0.0
    if (energy>6466782545.674999):
        return 1.0
    if (energy>6411915749.525001):
        return 0.0
    if (energy>6411915073.530001):
        return 1.0
    if (energy>6411914130.119999):
        return 0.0
    if (energy>6411913987.295):
        return 1.0
    if (energy>6411913764.38):
        return 0.0
    if (energy>6411912806.49):
        return 1.0
    if (energy>6411912672.84):
        return 0.0
    if (energy>6411912608.509998):
        return 1.0
    if (energy>6411912348.035):
        return 0.0
    if (energy>6411912327.190001):
        return 1.0
    if (energy>6411911962.785):
        return 0.0
    if (energy>6411911588.154999):
        return 1.0
    if (energy>6411911544.955):
        return 0.0
    if (energy>6411911503.869999):
        return 1.0
    if (energy>6411911254.925001):
        return 0.0
    if (energy>6411911057.755):
        return 1.0
    if (energy>6411910452.365):
        return 0.0
    if (energy>6411910361.49):
        return 1.0
    if (energy>6411909739.934999):
        return 0.0
    if (energy>6411909612.625):
        return 1.0
    if (energy>6411909433.035):
        return 0.0
    if (energy>6411909382.49):
        return 1.0
    if (energy>6411909309.905001):
        return 0.0
    if (energy>6411909089.32):
        return 1.0
    if (energy>6411908605.27):
        return 0.0
    if (energy>6411908536.165):
        return 1.0
    if (energy>6411908456.725):
        return 0.0
    if (energy>6411908332.085):
        return 1.0
    if (energy>6411907995.505):
        return 0.0
    if (energy>6411907898.959999):
        return 1.0
    if (energy>6411907818.11):
        return 0.0
    if (energy>6411907675.25):
        return 1.0
    if (energy>6411907216.58):
        return 0.0
    if (energy>6411907184.33):
        return 1.0
    if (energy>6411907015.985001):
        return 0.0
    if (energy>6411906871.094999):
        return 1.0
    if (energy>6411906005.225):
        return 0.0
    if (energy>6411905833.22):
        return 1.0
    if (energy>6411905611.635):
        return 0.0
    if (energy>6411905557.8550005):
        return 1.0
    if (energy>6411904256.42):
        return 0.0
    if (energy>6411903843.594999):
        return 1.0
    if (energy>6411902307.895):
        return 0.0
    if (energy>6411902118.295):
        return 1.0
    if (energy>6411901531.744999):
        return 0.0
    if (energy>6384465697.95):
        return 1.0
    if (energy>5874296235.445):
        return 0.0
    if (energy>5205353732.91):
        return 1.0
    if (energy>5205352974.595):
        return 0.0
    if (energy>5205346814.56):
        return 1.0
    if (energy>5205345346.72):
        return 0.0
    if (energy>5152679515.735001):
        return 1.0
    if (energy>5152679450.77):
        return 0.0
    if (energy>5059575672.32):
        return 1.0
    if (energy>4966473822.37):
        return 0.0
    if (energy>4966472825.2300005):
        return 1.0
    if (energy>4966472086.115):
        return 0.0
    if (energy>4966471833.125):
        return 1.0
    if (energy>4966471361.969999):
        return 0.0
    if (energy>4966471090.974999):
        return 1.0
    if (energy>4966471006.355):
        return 0.0
    if (energy>4966470601.235001):
        return 1.0
    if (energy>4966470118.47):
        return 0.0
    if (energy>4966469989.005001):
        return 1.0
    if (energy>4966469825.110001):
        return 0.0
    if (energy>4966469464.475):
        return 1.0
    if (energy>4966469289.559999):
        return 0.0
    if (energy>4966468763.775001):
        return 1.0
    if (energy>4966468660.825001):
        return 0.0
    if (energy>4966468296.295):
        return 1.0
    if (energy>4966468155.749999):
        return 0.0
    if (energy>4966467819.97):
        return 1.0
    if (energy>4966467498.984999):
        return 0.0
    if (energy>4966467413.99):
        return 1.0
    if (energy>4966467383.690001):
        return 0.0
    if (energy>4966467026.875):
        return 1.0
    if (energy>4966466858.395):
        return 0.0
    if (energy>4966466735.56):
        return 1.0
    if (energy>4966466634.950001):
        return 0.0
    if (energy>4966466442.559999):
        return 1.0
    if (energy>4966466182.94):
        return 0.0
    if (energy>4966465523.655001):
        return 1.0
    if (energy>4966465466.950001):
        return 0.0
    if (energy>4966464495.7300005):
        return 1.0
    if (energy>4966464091.43):
        return 0.0
    if (energy>4966463952.3949995):
        return 1.0
    if (energy>4966463814.985001):
        return 0.0
    if (energy>4966463084.754999):
        return 1.0
    if (energy>4966462641.23):
        return 0.0
    if (energy>4966460887.275):
        return 1.0
    if (energy>4966460777.25):
        return 0.0
    if (energy>4966460557.965):
        return 1.0
    if (energy>4966460414.210001):
        return 0.0
    if (energy>4913799311.17):
        return 1.0
    if (energy>4913798458.684999):
        return 0.0
    if (energy>4913798157.334999):
        return 1.0
    if (energy>4820694003.389999):
        return 0.0
    if (energy>4727591381.1050005):
        return 1.0
    if (energy>4727591041.955):
        return 0.0
    if (energy>4727588394.190001):
        return 1.0
    if (energy>4727588044.08):
        return 0.0
    if (energy>4727587965.309999):
        return 1.0
    if (energy>4727587830.93):
        return 0.0
    if (energy>4727587606.37):
        return 1.0
    if (energy>4727587379.934999):
        return 0.0
    if (energy>4727587265.684999):
        return 1.0
    if (energy>4727587222.785):
        return 0.0
    if (energy>4727587120.195):
        return 1.0
    if (energy>4727586908.074999):
        return 0.0
    if (energy>4727586732.4):
        return 1.0
    if (energy>4727586561.635):
        return 0.0
    if (energy>4727586537.254999):
        return 1.0
    if (energy>4727586489.695):
        return 0.0
    if (energy>4727586355.025):
        return 1.0
    if (energy>4727586319.620001):
        return 0.0
    if (energy>4727586298.995001):
        return 1.0
    if (energy>4727585859.83):
        return 0.0
    if (energy>4727585826.065001):
        return 1.0
    if (energy>4727585781.725):
        return 0.0
    if (energy>4727585765.925):
        return 1.0
    if (energy>4727585740.96):
        return 0.0
    if (energy>4727585706.85):
        return 1.0
    if (energy>4727585671.66):
        return 0.0
    if (energy>4727585657.1):
        return 1.0
    if (energy>4727585591.895):
        return 0.0
    if (energy>4727585406.51):
        return 1.0
    if (energy>4727585335.889999):
        return 0.0
    if (energy>4727584827.36):
        return 1.0
    if (energy>4727584719.6050005):
        return 0.0
    if (energy>4727584690.65):
        return 1.0
    if (energy>4727584647.584999):
        return 0.0
    if (energy>4727584560.655):
        return 1.0
    if (energy>4727584501.004999):
        return 0.0
    if (energy>4727584487.785):
        return 1.0
    if (energy>4727584468.725):
        return 0.0
    if (energy>4727584372.705):
        return 1.0
    if (energy>4727584299.415001):
        return 0.0
    if (energy>4727584258.940001):
        return 1.0
    if (energy>4727583830.130001):
        return 0.0
    if (energy>4727583609.360001):
        return 1.0
    if (energy>4727583521.3):
        return 0.0
    if (energy>4727583301.495001):
        return 1.0
    if (energy>4727582640.005):
        return 0.0
    if (energy>4727582471.309999):
        return 1.0
    if (energy>4727582145.945):
        return 0.0
    if (energy>4727582112.62):
        return 1.0
    if (energy>4727582071.094999):
        return 0.0
    if (energy>4727581850.169999):
        return 1.0
    if (energy>4727581652.674999):
        return 0.0
    if (energy>4727581583.035):
        return 1.0
    if (energy>4727580872.05):
        return 0.0
    if (energy>4727580546.955):
        return 1.0
    if (energy>4727579721.855):
        return 0.0
    if (energy>4727579430.705):
        return 1.0
    if (energy>4727578963.759999):
        return 0.0
    if (energy>4727577943.360001):
        return 1.0
    if (energy>4727576891.0):
        return 0.0
    return 1.0

numthresholds=479


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

