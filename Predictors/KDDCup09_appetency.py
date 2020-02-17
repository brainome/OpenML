#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:34:08
# Invocation: btc -v -v KDDCup09_appetency-1.csv -o KDDCup09_appetency-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                98.22%
Model accuracy:                     98.23% (49117/50000 correct)
Improvement over best guess:        0.01% (of possible 1.78%)
Model capacity (MEC):               896 bits
Generalization ratio:               54.81 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     97.31% (48654/50000)
True Positives:                     0.93% (463/50000)
False Negatives:                    0.85% (427/50000)
False Positives:                    0.91% (456/50000)
True Pos. Rate/Sensitivity/Recall:  0.52
True Neg. Rate/Specificity:         0.99
Precision:                          0.50
F-1 Measure:                        0.51
False Negative Rate/Miss Rate:      0.48
Critical Success Index:             0.34
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
TRAINFILE="KDDCup09_appetency-1.csv"


#Number of attributes
num_attr = 230

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
    if (energy>368689734619.7571):
        return 0.0
    if (energy>368605082875.7377):
        return 1.0
    if (energy>367552168739.3156):
        return 0.0
    if (energy>367523390670.5725):
        return 1.0
    if (energy>365945227557.30493):
        return 0.0
    if (energy>365927408955.84174):
        return 1.0
    if (energy>365416776098.21985):
        return 0.0
    if (energy>365389096442.01886):
        return 1.0
    if (energy>365346621466.9719):
        return 0.0
    if (energy>365336313721.5884):
        return 1.0
    if (energy>364600034383.204):
        return 0.0
    if (energy>364595681985.38904):
        return 1.0
    if (energy>364471060249.6515):
        return 0.0
    if (energy>364430223672.02673):
        return 1.0
    if (energy>363165203554.80695):
        return 0.0
    if (energy>363146838147.1841):
        return 1.0
    if (energy>361495803442.16516):
        return 0.0
    if (energy>361470485137.23474):
        return 1.0
    if (energy>361295447683.5508):
        return 0.0
    if (energy>361284578656.04517):
        return 1.0
    if (energy>361238032669.33484):
        return 0.0
    if (energy>361229204118.6178):
        return 1.0
    if (energy>360047523002.1499):
        return 0.0
    if (energy>360038177014.255):
        return 1.0
    if (energy>359877420694.1782):
        return 0.0
    if (energy>359871622590.8899):
        return 1.0
    if (energy>359066900090.6919):
        return 0.0
    if (energy>359065390741.38416):
        return 1.0
    if (energy>359038349761.73303):
        return 0.0
    if (energy>359035832089.20715):
        return 1.0
    if (energy>359029625339.4159):
        return 0.0
    if (energy>359024085291.0371):
        return 1.0
    if (energy>358561129918.90784):
        return 0.0
    if (energy>358556892326.568):
        return 1.0
    if (energy>357580608086.34216):
        return 0.0
    if (energy>357575943028.78265):
        return 1.0
    if (energy>357383618764.71814):
        return 0.0
    if (energy>357377893437.1661):
        return 1.0
    if (energy>357289216503.20197):
        return 0.0
    if (energy>357285200967.8286):
        return 1.0
    if (energy>356894920067.38007):
        return 0.0
    if (energy>356886647858.79016):
        return 1.0
    if (energy>356501123021.9178):
        return 0.0
    if (energy>356488655200.7337):
        return 1.0
    if (energy>356336998611.7948):
        return 0.0
    if (energy>356334445980.6255):
        return 1.0
    if (energy>356303652959.5959):
        return 0.0
    if (energy>356300011299.87384):
        return 1.0
    if (energy>356264629022.02295):
        return 0.0
    if (energy>356262919909.69415):
        return 1.0
    if (energy>355628515576.6679):
        return 0.0
    if (energy>355625694066.0438):
        return 1.0
    if (energy>355364167614.14215):
        return 0.0
    if (energy>355362527472.47107):
        return 1.0
    if (energy>355353361756.2565):
        return 0.0
    if (energy>355348544557.9865):
        return 1.0
    if (energy>355132952021.397):
        return 0.0
    if (energy>355131035077.85876):
        return 1.0
    if (energy>354696763091.75305):
        return 0.0
    if (energy>354695535923.7565):
        return 1.0
    if (energy>354629684219.3008):
        return 0.0
    if (energy>354620925805.93665):
        return 1.0
    if (energy>354529393984.6194):
        return 0.0
    if (energy>354528889881.2117):
        return 1.0
    if (energy>354294656165.3506):
        return 0.0
    if (energy>354289983917.4197):
        return 1.0
    if (energy>354259636299.79205):
        return 0.0
    if (energy>354259298586.7321):
        return 1.0
    if (energy>353924358685.7138):
        return 0.0
    if (energy>353920556312.43506):
        return 1.0
    if (energy>353667204770.72906):
        return 0.0
    if (energy>353661388977.7188):
        return 1.0
    if (energy>353647959979.7123):
        return 0.0
    if (energy>353646184798.7506):
        return 1.0
    if (energy>353477063359.1178):
        return 0.0
    if (energy>353472997529.838):
        return 1.0
    if (energy>353412231923.67834):
        return 0.0
    if (energy>353409376560.54706):
        return 1.0
    if (energy>353406298481.2803):
        return 0.0
    if (energy>353403237996.7335):
        return 1.0
    if (energy>353247489268.98315):
        return 0.0
    if (energy>353244903297.4205):
        return 1.0
    if (energy>353064608974.8987):
        return 0.0
    if (energy>353062611163.3384):
        return 1.0
    if (energy>352814374294.18835):
        return 0.0
    if (energy>352813970736.29987):
        return 1.0
    if (energy>352670862486.823):
        return 0.0
    if (energy>352669202865.3187):
        return 1.0
    if (energy>352608355554.46643):
        return 0.0
    if (energy>352605993060.9215):
        return 1.0
    if (energy>352127425808.57):
        return 0.0
    if (energy>352125582614.8612):
        return 1.0
    if (energy>352099674011.1497):
        return 0.0
    if (energy>352098352253.30255):
        return 1.0
    if (energy>352057560618.07294):
        return 0.0
    if (energy>352057019477.9457):
        return 1.0
    if (energy>351986852338.40356):
        return 0.0
    if (energy>351986448830.98566):
        return 1.0
    if (energy>351964602533.6803):
        return 0.0
    if (energy>351961213200.6692):
        return 1.0
    if (energy>351937751032.00977):
        return 0.0
    if (energy>351935808069.2625):
        return 1.0
    if (energy>351854286101.196):
        return 0.0
    if (energy>351853822544.6592):
        return 1.0
    if (energy>351801034422.8987):
        return 0.0
    if (energy>351800262424.9534):
        return 1.0
    if (energy>351780284010.3009):
        return 0.0
    if (energy>351780237744.87354):
        return 1.0
    if (energy>351743255380.66675):
        return 0.0
    if (energy>351741164104.27893):
        return 1.0
    if (energy>351724999059.1663):
        return 0.0
    if (energy>351723610354.9011):
        return 1.0
    if (energy>351661227510.067):
        return 0.0
    if (energy>351659552775.4037):
        return 1.0
    if (energy>351476300594.3846):
        return 0.0
    if (energy>351475028913.9216):
        return 1.0
    if (energy>351095462238.9116):
        return 0.0
    if (energy>351094234280.11426):
        return 1.0
    if (energy>351090823413.2626):
        return 0.0
    if (energy>351090280618.182):
        return 1.0
    if (energy>350990264620.1424):
        return 0.0
    if (energy>350989709263.1289):
        return 1.0
    if (energy>350943695086.53955):
        return 0.0
    if (energy>350943533151.01575):
        return 1.0
    if (energy>350882679658.7758):
        return 0.0
    if (energy>350881599501.4503):
        return 1.0
    if (energy>350837801062.76953):
        return 0.0
    if (energy>350836206289.98987):
        return 1.0
    if (energy>350767289609.73816):
        return 0.0
    if (energy>350766284309.0168):
        return 1.0
    if (energy>350722026043.93915):
        return 0.0
    if (energy>350721189304.8008):
        return 1.0
    if (energy>350652645856.98773):
        return 0.0
    if (energy>350651643625.55414):
        return 1.0
    if (energy>350619521318.6498):
        return 0.0
    if (energy>350619106596.04297):
        return 1.0
    if (energy>350618483412.08374):
        return 0.0
    if (energy>350617389631.5895):
        return 1.0
    if (energy>350592363941.05383):
        return 0.0
    if (energy>350591945900.7476):
        return 1.0
    if (energy>350487489761.81287):
        return 0.0
    if (energy>350485865621.9752):
        return 1.0
    if (energy>350430881572.2607):
        return 0.0
    if (energy>350430168668.4205):
        return 1.0
    if (energy>350315671980.1051):
        return 0.0
    if (energy>350314553509.3694):
        return 1.0
    if (energy>350112130208.5928):
        return 0.0
    if (energy>350111634550.0542):
        return 1.0
    if (energy>349959646309.1609):
        return 0.0
    if (energy>349958791698.95215):
        return 1.0
    if (energy>349949670306.23364):
        return 0.0
    if (energy>349948548916.4381):
        return 1.0
    if (energy>349944417863.39636):
        return 0.0
    if (energy>349943935139.334):
        return 1.0
    if (energy>349936830292.2678):
        return 0.0
    if (energy>349936522971.55115):
        return 1.0
    if (energy>349726425838.24133):
        return 0.0
    if (energy>349723984121.87915):
        return 1.0
    if (energy>349435270736.4794):
        return 0.0
    if (energy>349435080225.18384):
        return 1.0
    if (energy>349406832467.0909):
        return 0.0
    if (energy>349406514343.5875):
        return 1.0
    if (energy>349403304317.7211):
        return 0.0
    if (energy>349402534047.1176):
        return 1.0
    if (energy>349293774038.4674):
        return 0.0
    if (energy>349292428247.80725):
        return 1.0
    if (energy>349230433328.7889):
        return 0.0
    if (energy>349228322238.5844):
        return 1.0
    if (energy>349225279148.32477):
        return 0.0
    if (energy>349224736831.11774):
        return 1.0
    if (energy>349196793023.52246):
        return 0.0
    if (energy>349196023365.5701):
        return 1.0
    if (energy>349112227428.1504):
        return 0.0
    if (energy>349110242760.72705):
        return 1.0
    if (energy>349103915721.95605):
        return 0.0
    if (energy>349103546646.7314):
        return 1.0
    if (energy>349103053527.11017):
        return 0.0
    if (energy>349102096730.76794):
        return 1.0
    if (energy>348962331173.55273):
        return 0.0
    if (energy>348960938006.0188):
        return 1.0
    if (energy>348960191049.1686):
        return 0.0
    if (energy>348959514627.61426):
        return 1.0
    if (energy>348914114636.9098):
        return 0.0
    if (energy>348913020415.2621):
        return 1.0
    if (energy>348905383756.28174):
        return 0.0
    if (energy>348902849318.4293):
        return 1.0
    if (energy>348880399135.44946):
        return 0.0
    if (energy>348878730507.7914):
        return 1.0
    if (energy>348835984474.4853):
        return 0.0
    if (energy>348834078644.3702):
        return 1.0
    if (energy>348831507732.8971):
        return 0.0
    if (energy>348829576057.01306):
        return 1.0
    if (energy>348820404107.47705):
        return 0.0
    if (energy>348820201920.0388):
        return 1.0
    if (energy>348808082602.9452):
        return 0.0
    if (energy>348807841086.3502):
        return 1.0
    if (energy>348724007682.1061):
        return 0.0
    if (energy>348722348555.33966):
        return 1.0
    if (energy>348629749978.47974):
        return 0.0
    if (energy>348628827292.5938):
        return 1.0
    if (energy>348576077678.6155):
        return 0.0
    if (energy>348575199898.3318):
        return 1.0
    if (energy>348560678010.81647):
        return 0.0
    if (energy>348559644070.3691):
        return 1.0
    if (energy>348543947616.0417):
        return 0.0
    if (energy>348543385500.1279):
        return 1.0
    if (energy>348392763339.54395):
        return 0.0
    if (energy>348392487347.1542):
        return 1.0
    if (energy>348313023030.2804):
        return 0.0
    if (energy>348310759085.09607):
        return 1.0
    if (energy>348291513659.95544):
        return 0.0
    if (energy>348290161080.23254):
        return 1.0
    if (energy>348197627580.7395):
        return 0.0
    if (energy>348196625655.0087):
        return 1.0
    if (energy>348079366260.22266):
        return 0.0
    if (energy>348078338350.6432):
        return 1.0
    if (energy>348014216625.8884):
        return 0.0
    if (energy>348014133038.3572):
        return 1.0
    if (energy>348010708866.9955):
        return 0.0
    if (energy>348010505303.4729):
        return 1.0
    if (energy>347956757508.3331):
        return 0.0
    if (energy>347955682444.8791):
        return 1.0
    if (energy>347925451362.1911):
        return 0.0
    if (energy>347924778847.43005):
        return 1.0
    if (energy>347840907772.90576):
        return 0.0
    if (energy>347839778118.55634):
        return 1.0
    if (energy>347801075856.057):
        return 0.0
    if (energy>347800139340.98584):
        return 1.0
    if (energy>347763076229.1306):
        return 0.0
    if (energy>347761999654.3682):
        return 1.0
    if (energy>347743037949.0397):
        return 0.0
    if (energy>347742927649.00745):
        return 1.0
    if (energy>347741547405.5069):
        return 0.0
    if (energy>347740630498.4236):
        return 1.0
    if (energy>347556525631.82825):
        return 0.0
    if (energy>347555810345.94946):
        return 1.0
    if (energy>347473484201.74023):
        return 0.0
    if (energy>347472981934.43115):
        return 1.0
    if (energy>347395787301.3802):
        return 0.0
    if (energy>347394255210.2867):
        return 1.0
    if (energy>347310341004.0012):
        return 0.0
    if (energy>347309393834.6349):
        return 1.0
    if (energy>347264047419.38696):
        return 0.0
    if (energy>347263581616.5831):
        return 1.0
    if (energy>347201236936.18994):
        return 0.0
    if (energy>347200502917.11707):
        return 1.0
    if (energy>347191117439.6542):
        return 0.0
    if (energy>347190966039.6145):
        return 1.0
    if (energy>347153407411.9253):
        return 0.0
    if (energy>347152393894.5714):
        return 1.0
    if (energy>347124931432.0039):
        return 0.0
    if (energy>347123198159.4785):
        return 1.0
    if (energy>347113618508.47546):
        return 0.0
    if (energy>347112529793.96454):
        return 1.0
    if (energy>347092063074.9979):
        return 0.0
    if (energy>347091764471.2347):
        return 1.0
    if (energy>347090962637.1334):
        return 0.0
    if (energy>347089527572.2738):
        return 1.0
    if (energy>347061828463.75635):
        return 0.0
    if (energy>347061500967.7714):
        return 1.0
    if (energy>347020162965.64795):
        return 0.0
    if (energy>347019775937.07996):
        return 1.0
    if (energy>346977734501.9975):
        return 0.0
    if (energy>346976532765.5775):
        return 1.0
    if (energy>346929621079.9302):
        return 0.0
    if (energy>346929024307.43805):
        return 1.0
    if (energy>346885545685.88464):
        return 0.0
    if (energy>346884344654.4741):
        return 1.0
    if (energy>346853702286.2914):
        return 0.0
    if (energy>346853197343.3566):
        return 1.0
    if (energy>346779349704.4789):
        return 0.0
    if (energy>346778786285.5498):
        return 1.0
    if (energy>346733325386.72974):
        return 0.0
    if (energy>346732655120.7455):
        return 1.0
    if (energy>346727655730.15967):
        return 0.0
    if (energy>346727331929.45386):
        return 1.0
    if (energy>346659841861.8172):
        return 0.0
    if (energy>346658730951.28455):
        return 1.0
    if (energy>346650582189.6345):
        return 0.0
    if (energy>346649403572.19543):
        return 1.0
    if (energy>346562662716.3511):
        return 0.0
    if (energy>346561910639.594):
        return 1.0
    if (energy>346425897480.6076):
        return 0.0
    if (energy>346425171422.6521):
        return 1.0
    if (energy>346407434878.4418):
        return 0.0
    if (energy>346406288308.38153):
        return 1.0
    if (energy>346402388095.4447):
        return 0.0
    if (energy>346402175893.78723):
        return 1.0
    if (energy>346368271096.06744):
        return 0.0
    if (energy>346367696126.5487):
        return 1.0
    if (energy>346366230399.55774):
        return 0.0
    if (energy>346365941436.6311):
        return 1.0
    if (energy>346350661122.1247):
        return 0.0
    if (energy>346348946993.89874):
        return 1.0
    if (energy>346341564061.4839):
        return 0.0
    if (energy>346340674101.7956):
        return 1.0
    if (energy>346283520043.98865):
        return 0.0
    if (energy>346281915363.0033):
        return 1.0
    if (energy>346214423292.1155):
        return 0.0
    if (energy>346214159364.24097):
        return 1.0
    if (energy>346208302958.04663):
        return 0.0
    if (energy>346207983524.08093):
        return 1.0
    if (energy>346154027840.6578):
        return 0.0
    if (energy>346152514653.31586):
        return 1.0
    if (energy>346135116114.46375):
        return 0.0
    if (energy>346134599069.81946):
        return 1.0
    if (energy>346095738786.6876):
        return 0.0
    if (energy>346094403270.4431):
        return 1.0
    if (energy>345986409313.4922):
        return 0.0
    if (energy>345985454622.9989):
        return 1.0
    if (energy>345948428991.657):
        return 0.0
    if (energy>345948320258.802):
        return 1.0
    if (energy>345901616812.34216):
        return 0.0
    if (energy>345900994424.28107):
        return 1.0
    if (energy>345845196770.98535):
        return 0.0
    if (energy>345844464816.3977):
        return 1.0
    if (energy>345789164016.11633):
        return 0.0
    if (energy>345788010089.47076):
        return 1.0
    if (energy>345786606167.74744):
        return 0.0
    if (energy>345786429129.5781):
        return 1.0
    if (energy>345771547792.1532):
        return 0.0
    if (energy>345770222478.38635):
        return 1.0
    if (energy>345741304850.4274):
        return 0.0
    if (energy>345740191289.89075):
        return 1.0
    if (energy>345663425839.65936):
        return 0.0
    if (energy>345662000921.89136):
        return 1.0
    if (energy>345627450271.7174):
        return 0.0
    if (energy>345625861608.0475):
        return 1.0
    if (energy>345613363711.2667):
        return 0.0
    if (energy>345612388256.5625):
        return 1.0
    if (energy>345602382032.59845):
        return 0.0
    if (energy>345601438962.7843):
        return 1.0
    if (energy>345600999444.3109):
        return 0.0
    if (energy>345600204201.1128):
        return 1.0
    if (energy>345528040080.38794):
        return 0.0
    if (energy>345527765194.9455):
        return 1.0
    if (energy>345432242128.4969):
        return 0.0
    if (energy>345431948804.5874):
        return 1.0
    if (energy>345407580925.0059):
        return 0.0
    if (energy>345406684602.7709):
        return 1.0
    if (energy>345399118942.5485):
        return 0.0
    if (energy>345397667755.02185):
        return 1.0
    if (energy>345324406392.2226):
        return 0.0
    if (energy>345324130817.1204):
        return 1.0
    if (energy>345323154816.29004):
        return 0.0
    if (energy>345322258346.8321):
        return 1.0
    if (energy>345234760642.29297):
        return 0.0
    if (energy>345233695416.8517):
        return 1.0
    if (energy>345228329980.579):
        return 0.0
    if (energy>345225957761.5373):
        return 1.0
    if (energy>345161999685.2273):
        return 0.0
    if (energy>345161527111.3356):
        return 1.0
    if (energy>344976812460.69324):
        return 0.0
    if (energy>344975964224.6367):
        return 1.0
    if (energy>344975533096.0473):
        return 0.0
    if (energy>344974343795.9797):
        return 1.0
    if (energy>344963832127.7136):
        return 0.0
    if (energy>344963580582.5989):
        return 1.0
    if (energy>344900247659.2905):
        return 0.0
    if (energy>344899467107.57117):
        return 1.0
    if (energy>344888508479.0239):
        return 0.0
    if (energy>344887990301.758):
        return 1.0
    if (energy>344881618753.8647):
        return 0.0
    if (energy>344880917299.8154):
        return 1.0
    if (energy>344839060531.57904):
        return 0.0
    if (energy>344838619138.3125):
        return 1.0
    if (energy>344832095724.0553):
        return 0.0
    if (energy>344831112803.7564):
        return 1.0
    if (energy>344729276440.6054):
        return 0.0
    if (energy>344727374127.18726):
        return 1.0
    if (energy>344708013198.8036):
        return 0.0
    if (energy>344707507141.26355):
        return 1.0
    if (energy>344642742243.4523):
        return 0.0
    if (energy>344642489286.73193):
        return 1.0
    if (energy>344568440507.9548):
        return 0.0
    if (energy>344567345627.68317):
        return 1.0
    if (energy>344514834630.16003):
        return 0.0
    if (energy>344513085812.199):
        return 1.0
    if (energy>344494935076.73267):
        return 0.0
    if (energy>344493925550.65173):
        return 1.0
    if (energy>344473329283.1614):
        return 0.0
    if (energy>344472424504.55774):
        return 1.0
    if (energy>344470838989.88184):
        return 0.0
    if (energy>344470661206.84546):
        return 1.0
    if (energy>344469948818.5021):
        return 0.0
    if (energy>344469451923.5331):
        return 1.0
    if (energy>344461445333.4618):
        return 0.0
    if (energy>344460610705.8716):
        return 1.0
    if (energy>344368561053.08545):
        return 0.0
    if (energy>344367970232.67676):
        return 1.0
    if (energy>344360635627.26697):
        return 0.0
    if (energy>344359965220.3042):
        return 1.0
    if (energy>344327837363.02454):
        return 0.0
    if (energy>344326473304.6012):
        return 1.0
    if (energy>344320317867.666):
        return 0.0
    if (energy>344320044414.01697):
        return 1.0
    if (energy>344317828637.01636):
        return 0.0
    if (energy>344317595494.8912):
        return 1.0
    if (energy>344305752367.4988):
        return 0.0
    if (energy>344305198425.2062):
        return 1.0
    if (energy>344283756432.89026):
        return 0.0
    if (energy>344283687750.5784):
        return 1.0
    if (energy>344257652329.8828):
        return 0.0
    if (energy>344256031087.867):
        return 1.0
    if (energy>344208194972.84753):
        return 0.0
    if (energy>344207682601.37244):
        return 1.0
    if (energy>344057668678.54614):
        return 0.0
    if (energy>344057480565.05493):
        return 1.0
    if (energy>344020004160.9929):
        return 0.0
    if (energy>344019555755.44617):
        return 1.0
    if (energy>343985153335.8911):
        return 0.0
    if (energy>343984300945.7056):
        return 1.0
    if (energy>343946366130.26324):
        return 0.0
    if (energy>343944511250.0418):
        return 1.0
    if (energy>343930436315.9428):
        return 0.0
    if (energy>343927932164.70337):
        return 1.0
    if (energy>343917439000.26556):
        return 0.0
    if (energy>343916499969.84705):
        return 1.0
    if (energy>343912859337.8049):
        return 0.0
    if (energy>343912329742.5684):
        return 1.0
    if (energy>343899705706.90857):
        return 0.0
    if (energy>343898196335.8417):
        return 1.0
    if (energy>343879543094.25244):
        return 0.0
    if (energy>343878319915.3052):
        return 1.0
    if (energy>343801834819.9881):
        return 0.0
    if (energy>343801020929.09143):
        return 1.0
    if (energy>343672380255.70154):
        return 0.0
    if (energy>343671857108.3163):
        return 1.0
    if (energy>343626874848.43756):
        return 0.0
    if (energy>343626133054.12305):
        return 1.0
    if (energy>343613259115.146):
        return 0.0
    if (energy>343612564386.54407):
        return 1.0
    if (energy>343591187794.1571):
        return 0.0
    if (energy>343590886526.14233):
        return 1.0
    if (energy>343563972625.2873):
        return 0.0
    if (energy>343562835660.1731):
        return 1.0
    if (energy>343542002633.3165):
        return 0.0
    if (energy>343541529927.70197):
        return 1.0
    if (energy>343508128653.2317):
        return 0.0
    if (energy>343507820711.1156):
        return 1.0
    if (energy>343312081651.53455):
        return 0.0
    if (energy>343311673510.8657):
        return 1.0
    if (energy>343309563148.82794):
        return 0.0
    if (energy>343308761599.0608):
        return 1.0
    if (energy>343301376028.6811):
        return 0.0
    if (energy>343300216455.1804):
        return 1.0
    if (energy>343235376747.21814):
        return 0.0
    if (energy>343235155095.97925):
        return 1.0
    if (energy>343227700753.23096):
        return 0.0
    if (energy>343226131188.9381):
        return 1.0
    if (energy>343188161950.379):
        return 0.0
    if (energy>343187591873.7594):
        return 1.0
    if (energy>343174388485.4282):
        return 0.0
    if (energy>343173921453.08716):
        return 1.0
    if (energy>343135388653.0408):
        return 0.0
    if (energy>343134777330.0187):
        return 1.0
    if (energy>343134009531.9601):
        return 0.0
    if (energy>343133551460.15015):
        return 1.0
    if (energy>343121032937.06165):
        return 0.0
    if (energy>343120316026.3319):
        return 1.0
    if (energy>343105984473.0469):
        return 0.0
    if (energy>343104847042.3805):
        return 1.0
    if (energy>343095267416.58575):
        return 0.0
    if (energy>343094977972.7042):
        return 1.0
    if (energy>343089660724.3146):
        return 0.0
    if (energy>343089230367.2129):
        return 1.0
    if (energy>343054869281.04785):
        return 0.0
    if (energy>343054256036.09):
        return 1.0
    if (energy>343032596783.81555):
        return 0.0
    if (energy>343031953490.4691):
        return 1.0
    if (energy>343029345411.4049):
        return 0.0
    if (energy>343028750990.458):
        return 1.0
    if (energy>343023519223.33344):
        return 0.0
    if (energy>343022440604.96094):
        return 1.0
    if (energy>343013687616.269):
        return 0.0
    if (energy>343012668092.2962):
        return 1.0
    if (energy>343009445643.5638):
        return 0.0
    if (energy>343008633005.2935):
        return 1.0
    if (energy>342935871304.36646):
        return 0.0
    if (energy>342935225642.08124):
        return 1.0
    if (energy>342874879210.3075):
        return 0.0
    if (energy>342874019650.16833):
        return 1.0
    if (energy>342867302343.00134):
        return 0.0
    if (energy>342866794118.68036):
        return 1.0
    if (energy>342815422368.5973):
        return 0.0
    if (energy>342814310738.62915):
        return 1.0
    if (energy>342795988163.79614):
        return 0.0
    if (energy>342794540914.5696):
        return 1.0
    if (energy>342721295094.2899):
        return 0.0
    if (energy>342720056276.3561):
        return 1.0
    if (energy>342710415337.1603):
        return 0.0
    if (energy>342710129205.00494):
        return 1.0
    if (energy>342702463278.02747):
        return 0.0
    if (energy>342702392755.007):
        return 1.0
    if (energy>342551007254.48):
        return 0.0
    if (energy>342549869165.0178):
        return 1.0
    if (energy>342529758864.5891):
        return 0.0
    if (energy>342529342892.6461):
        return 1.0
    if (energy>342495822688.24744):
        return 0.0
    if (energy>342495356947.9641):
        return 1.0
    if (energy>342482499913.2514):
        return 0.0
    if (energy>342482209459.01526):
        return 1.0
    if (energy>342424131562.7263):
        return 0.0
    if (energy>342423782407.98413):
        return 1.0
    if (energy>342269193423.8751):
        return 0.0
    if (energy>342267189082.9724):
        return 1.0
    if (energy>342263439973.655):
        return 0.0
    if (energy>342262916884.95447):
        return 1.0
    if (energy>342259089814.4318):
        return 0.0
    if (energy>342257816824.99805):
        return 1.0
    if (energy>342249300347.5855):
        return 0.0
    if (energy>342248768514.8164):
        return 1.0
    if (energy>342201802265.1742):
        return 0.0
    if (energy>342201526861.9396):
        return 1.0
    if (energy>342173886012.41205):
        return 0.0
    if (energy>342173164591.4019):
        return 1.0
    if (energy>342134504736.07837):
        return 0.0
    if (energy>342134143459.53143):
        return 1.0
    if (energy>342072879294.36115):
        return 0.0
    if (energy>342072560236.46924):
        return 1.0
    if (energy>342066405653.527):
        return 0.0
    if (energy>342066307616.17285):
        return 1.0
    if (energy>342065304745.36554):
        return 0.0
    if (energy>342063196896.96906):
        return 1.0
    if (energy>342052586937.26416):
        return 0.0
    if (energy>342051836059.9903):
        return 1.0
    if (energy>342024642239.9853):
        return 0.0
    if (energy>342024309268.0375):
        return 1.0
    if (energy>341915861628.9652):
        return 0.0
    if (energy>341915786602.0255):
        return 1.0
    if (energy>341889951928.625):
        return 0.0
    if (energy>341888976850.4517):
        return 1.0
    if (energy>341855933556.9981):
        return 0.0
    if (energy>341855606969.4637):
        return 1.0
    if (energy>341844028243.91174):
        return 0.0
    if (energy>341843419513.421):
        return 1.0
    if (energy>341822614435.0973):
        return 0.0
    if (energy>341821718642.0853):
        return 1.0
    if (energy>341810981373.091):
        return 0.0
    if (energy>341810432832.75385):
        return 1.0
    if (energy>341792283315.4198):
        return 0.0
    if (energy>341791864941.83777):
        return 1.0
    if (energy>341711393348.71936):
        return 0.0
    if (energy>341710942059.60876):
        return 1.0
    if (energy>341445742963.4618):
        return 0.0
    if (energy>341444887064.9306):
        return 1.0
    if (energy>341426236996.615):
        return 0.0
    if (energy>341426192298.1978):
        return 1.0
    if (energy>341344528821.9127):
        return 0.0
    if (energy>341344209150.1113):
        return 1.0
    if (energy>341310302261.01495):
        return 0.0
    if (energy>341308643407.05756):
        return 1.0
    if (energy>341302665076.6332):
        return 0.0
    if (energy>341302528538.0515):
        return 1.0
    if (energy>341237135066.8157):
        return 0.0
    if (energy>341236505151.4607):
        return 1.0
    if (energy>341226629755.0159):
        return 0.0
    if (energy>341226017302.3951):
        return 1.0
    if (energy>341097182697.051):
        return 0.0
    if (energy>341096702572.5351):
        return 1.0
    if (energy>341091250811.6503):
        return 0.0
    if (energy>341090411412.5541):
        return 1.0
    if (energy>341088589857.35284):
        return 0.0
    if (energy>341088303672.8975):
        return 1.0
    if (energy>341066132946.0142):
        return 0.0
    if (energy>341065710507.1589):
        return 1.0
    if (energy>341015452961.4902):
        return 0.0
    if (energy>341014603166.0811):
        return 1.0
    if (energy>340946962600.37683):
        return 0.0
    if (energy>340945851587.1382):
        return 1.0
    if (energy>340738750309.20935):
        return 0.0
    if (energy>340738653260.8463):
        return 1.0
    if (energy>340614297471.20874):
        return 0.0
    if (energy>340614144051.5289):
        return 1.0
    if (energy>340599428102.38904):
        return 0.0
    if (energy>340598091432.03015):
        return 1.0
    if (energy>340525025974.51483):
        return 0.0
    if (energy>340524438657.324):
        return 1.0
    if (energy>340495263555.47363):
        return 0.0
    if (energy>340494690036.1704):
        return 1.0
    if (energy>340456816679.33826):
        return 0.0
    if (energy>340455174468.5046):
        return 1.0
    if (energy>340448703088.202):
        return 0.0
    if (energy>340448176065.8185):
        return 1.0
    if (energy>340444784549.6819):
        return 0.0
    if (energy>340444646174.0517):
        return 1.0
    if (energy>340434587006.2644):
        return 0.0
    if (energy>340433429647.64966):
        return 1.0
    if (energy>340372169469.0298):
        return 0.0
    if (energy>340372030087.47754):
        return 1.0
    if (energy>340326364967.0702):
        return 0.0
    if (energy>340326239451.23145):
        return 1.0
    if (energy>340309087417.4193):
        return 0.0
    if (energy>340308678776.37115):
        return 1.0
    if (energy>340263936778.20483):
        return 0.0
    if (energy>340262992867.6543):
        return 1.0
    if (energy>340218552711.604):
        return 0.0
    if (energy>340217113258.1823):
        return 1.0
    if (energy>340169060313.9641):
        return 0.0
    if (energy>340168757586.64825):
        return 1.0
    if (energy>340127717835.4807):
        return 0.0
    if (energy>340126390069.2416):
        return 1.0
    if (energy>340112773073.64514):
        return 0.0
    if (energy>340112499294.3943):
        return 1.0
    if (energy>340059371834.7068):
        return 0.0
    if (energy>340059012965.7954):
        return 1.0
    if (energy>340048905225.1498):
        return 0.0
    if (energy>340047487765.7995):
        return 1.0
    if (energy>340034194133.7554):
        return 0.0
    if (energy>340030814095.2693):
        return 1.0
    if (energy>340020950379.77844):
        return 0.0
    if (energy>340020354250.6024):
        return 1.0
    if (energy>339981739397.8896):
        return 0.0
    if (energy>339981236869.67444):
        return 1.0
    if (energy>339975228394.4617):
        return 0.0
    if (energy>339975046375.4927):
        return 1.0
    if (energy>339952523370.7928):
        return 0.0
    if (energy>339952431535.84125):
        return 1.0
    if (energy>339900658052.302):
        return 0.0
    if (energy>339900640420.11505):
        return 1.0
    if (energy>339865720528.89636):
        return 0.0
    if (energy>339864782682.8944):
        return 1.0
    if (energy>339833441240.18036):
        return 0.0
    if (energy>339831718884.98804):
        return 1.0
    if (energy>339828056293.52325):
        return 0.0
    if (energy>339826936876.8967):
        return 1.0
    if (energy>339734014813.6106):
        return 0.0
    if (energy>339733698461.184):
        return 1.0
    if (energy>339725677545.5713):
        return 0.0
    if (energy>339724870510.36365):
        return 1.0
    if (energy>339672874170.43024):
        return 0.0
    if (energy>339672596024.4163):
        return 1.0
    if (energy>339661674883.5241):
        return 0.0
    if (energy>339660921233.0029):
        return 1.0
    if (energy>339638467701.1521):
        return 0.0
    if (energy>339637933988.238):
        return 1.0
    if (energy>339533818195.62103):
        return 0.0
    if (energy>339533588737.32385):
        return 1.0
    if (energy>339514249566.2657):
        return 0.0
    if (energy>339514056724.6477):
        return 1.0
    if (energy>339453247144.3535):
        return 0.0
    if (energy>339451400643.5045):
        return 1.0
    if (energy>339410578169.89355):
        return 0.0
    if (energy>339410278634.432):
        return 1.0
    if (energy>339278817569.8312):
        return 0.0
    if (energy>339278189751.3262):
        return 1.0
    if (energy>339192990152.90356):
        return 0.0
    if (energy>339192505765.5941):
        return 1.0
    if (energy>339133431698.32886):
        return 0.0
    if (energy>339131568236.9153):
        return 1.0
    if (energy>339048050256.55383):
        return 0.0
    if (energy>339047419046.05347):
        return 1.0
    if (energy>339000785596.0759):
        return 0.0
    if (energy>338999443554.6435):
        return 1.0
    if (energy>338912496395.42053):
        return 0.0
    if (energy>338912014795.48553):
        return 1.0
    if (energy>338881595307.07263):
        return 0.0
    if (energy>338880591617.8395):
        return 1.0
    if (energy>338871822699.70483):
        return 0.0
    if (energy>338871275658.20605):
        return 1.0
    if (energy>338809346355.11847):
        return 0.0
    if (energy>338808858321.9831):
        return 1.0
    if (energy>338793521514.897):
        return 0.0
    if (energy>338792334204.60583):
        return 1.0
    if (energy>338755985201.11005):
        return 0.0
    if (energy>338755512697.0952):
        return 1.0
    if (energy>338715943073.9332):
        return 0.0
    if (energy>338715576175.1903):
        return 1.0
    if (energy>338705481927.79944):
        return 0.0
    if (energy>338704573582.8255):
        return 1.0
    if (energy>338701799709.6594):
        return 0.0
    if (energy>338700651093.8838):
        return 1.0
    if (energy>338588241290.08105):
        return 0.0
    if (energy>338587046607.23737):
        return 1.0
    if (energy>338577702619.40845):
        return 0.0
    if (energy>338575700726.2531):
        return 1.0
    if (energy>338556343526.5832):
        return 0.0
    if (energy>338555820501.54626):
        return 1.0
    if (energy>338251558438.9852):
        return 0.0
    if (energy>338250675118.62103):
        return 1.0
    if (energy>338226995530.46533):
        return 0.0
    if (energy>338225107917.7302):
        return 1.0
    if (energy>338192358369.19543):
        return 0.0
    if (energy>338191214364.2003):
        return 1.0
    if (energy>338190832842.39734):
        return 0.0
    if (energy>338190439114.67236):
        return 1.0
    if (energy>338129364888.3973):
        return 0.0
    if (energy>338128372673.8652):
        return 1.0
    if (energy>338119437525.12854):
        return 0.0
    if (energy>338118692304.39514):
        return 1.0
    if (energy>338029270513.9298):
        return 0.0
    if (energy>338027934245.8562):
        return 1.0
    if (energy>338023558760.8644):
        return 0.0
    if (energy>338023177447.9773):
        return 1.0
    if (energy>337992033041.7407):
        return 0.0
    if (energy>337990024519.2728):
        return 1.0
    if (energy>337977282757.49786):
        return 0.0
    if (energy>337976776198.6072):
        return 1.0
    if (energy>337958653883.21826):
        return 0.0
    if (energy>337958358106.725):
        return 1.0
    if (energy>337859388862.08276):
        return 0.0
    if (energy>337858742346.26184):
        return 1.0
    if (energy>337820295442.58496):
        return 0.0
    if (energy>337817901982.84):
        return 1.0
    if (energy>337749745022.3053):
        return 0.0
    if (energy>337749207746.75793):
        return 1.0
    if (energy>337718435919.8939):
        return 0.0
    if (energy>337718038591.77136):
        return 1.0
    if (energy>337712778360.61115):
        return 0.0
    if (energy>337711811922.874):
        return 1.0
    if (energy>337576783820.98804):
        return 0.0
    if (energy>337575333939.0822):
        return 1.0
    if (energy>337566824951.6758):
        return 0.0
    if (energy>337566604669.9221):
        return 1.0
    if (energy>337476593406.8611):
        return 0.0
    if (energy>337476388401.9786):
        return 1.0
    if (energy>337258931668.8075):
        return 0.0
    if (energy>337258184053.7459):
        return 1.0
    if (energy>337246650096.9757):
        return 0.0
    if (energy>337246002012.64026):
        return 1.0
    if (energy>337196084754.3918):
        return 0.0
    if (energy>337195616932.6538):
        return 1.0
    if (energy>337131159627.55914):
        return 0.0
    if (energy>337130558712.50244):
        return 1.0
    if (energy>337129318874.23474):
        return 0.0
    if (energy>337128182195.36255):
        return 1.0
    if (energy>337123765131.9155):
        return 0.0
    if (energy>337121572507.9866):
        return 1.0
    if (energy>337071391620.4269):
        return 0.0
    if (energy>337070239520.1123):
        return 1.0
    if (energy>337031709677.3326):
        return 0.0
    if (energy>337031344294.2936):
        return 1.0
    if (energy>337015020551.9574):
        return 0.0
    if (energy>337014137004.6376):
        return 1.0
    if (energy>336998233204.4745):
        return 0.0
    if (energy>336996879161.3465):
        return 1.0
    if (energy>336987818357.96924):
        return 0.0
    if (energy>336985501557.4093):
        return 1.0
    if (energy>336957085697.15454):
        return 0.0
    if (energy>336953808992.3898):
        return 1.0
    if (energy>336897592635.9715):
        return 0.0
    if (energy>336897312965.2605):
        return 1.0
    if (energy>336713967173.55164):
        return 0.0
    if (energy>336713354617.2571):
        return 1.0
    if (energy>336661851746.8032):
        return 0.0
    if (energy>336660855498.9319):
        return 1.0
    if (energy>336631375272.09595):
        return 0.0
    if (energy>336630163801.806):
        return 1.0
    if (energy>336589590348.9711):
        return 0.0
    if (energy>336589416757.7101):
        return 1.0
    if (energy>336480704973.92786):
        return 0.0
    if (energy>336480517007.1104):
        return 1.0
    if (energy>336397686793.22186):
        return 0.0
    if (energy>336397530129.55115):
        return 1.0
    if (energy>336384651929.7295):
        return 0.0
    if (energy>336383954307.3833):
        return 1.0
    if (energy>336377842524.949):
        return 0.0
    if (energy>336377005045.9756):
        return 1.0
    if (energy>336371185664.4791):
        return 0.0
    if (energy>336370114424.9408):
        return 1.0
    if (energy>336278095907.0762):
        return 0.0
    if (energy>336277744717.5608):
        return 1.0
    if (energy>336212493688.4453):
        return 0.0
    if (energy>336210606958.66797):
        return 1.0
    if (energy>336156610298.96484):
        return 0.0
    if (energy>336155426311.24805):
        return 1.0
    if (energy>336090440609.73694):
        return 0.0
    if (energy>336090373938.8636):
        return 1.0
    if (energy>336068258259.1393):
        return 0.0
    if (energy>336065791015.80664):
        return 1.0
    if (energy>336001282641.66473):
        return 0.0
    if (energy>335999458865.5232):
        return 1.0
    if (energy>335955711116.8082):
        return 0.0
    if (energy>335954762020.3358):
        return 1.0
    if (energy>335937606697.90186):
        return 0.0
    if (energy>335934591772.0825):
        return 1.0
    if (energy>335902957454.1098):
        return 0.0
    if (energy>335902008146.643):
        return 1.0
    if (energy>335777513019.3165):
        return 0.0
    if (energy>335777249686.2987):
        return 1.0
    if (energy>335615206449.276):
        return 0.0
    if (energy>335613433829.5078):
        return 1.0
    if (energy>335455269037.3515):
        return 0.0
    if (energy>335454775109.24066):
        return 1.0
    if (energy>335355148123.0502):
        return 0.0
    if (energy>335354157381.6343):
        return 1.0
    if (energy>335342051777.59924):
        return 0.0
    if (energy>335339909781.4132):
        return 1.0
    if (energy>335283962720.4722):
        return 0.0
    if (energy>335283309857.13184):
        return 1.0
    if (energy>335197448726.42554):
        return 0.0
    if (energy>335195907027.06323):
        return 1.0
    if (energy>335174159802.59534):
        return 0.0
    if (energy>335172296095.23065):
        return 1.0
    if (energy>335169607784.95874):
        return 0.0
    if (energy>335167725490.2678):
        return 1.0
    if (energy>335160069707.59033):
        return 0.0
    if (energy>335157659963.5298):
        return 1.0
    if (energy>335113178141.7288):
        return 0.0
    if (energy>335112780355.098):
        return 1.0
    if (energy>335058527876.39966):
        return 0.0
    if (energy>335056627786.47205):
        return 1.0
    if (energy>335050343729.15906):
        return 0.0
    if (energy>335049890203.03516):
        return 1.0
    if (energy>335013255435.6638):
        return 0.0
    if (energy>335012446150.37085):
        return 1.0
    if (energy>334909797037.03424):
        return 0.0
    if (energy>334908289437.67944):
        return 1.0
    if (energy>334877352247.6474):
        return 0.0
    if (energy>334876981528.03796):
        return 1.0
    if (energy>334774592971.6946):
        return 0.0
    if (energy>334771866231.07043):
        return 1.0
    if (energy>334717098032.0574):
        return 0.0
    if (energy>334714898385.0645):
        return 1.0
    if (energy>334495550855.6547):
        return 0.0
    if (energy>334494769924.2694):
        return 1.0
    if (energy>334414947055.893):
        return 0.0
    if (energy>334414914726.5971):
        return 1.0
    if (energy>334336204994.15344):
        return 0.0
    if (energy>334333780412.01514):
        return 1.0
    if (energy>334262262177.61584):
        return 0.0
    if (energy>334259522734.12646):
        return 1.0
    if (energy>334199532615.5191):
        return 0.0
    if (energy>334196486146.6655):
        return 1.0
    if (energy>334079532701.43835):
        return 0.0
    if (energy>334079095428.31445):
        return 1.0
    if (energy>333956932090.01917):
        return 0.0
    if (energy>333954940385.8648):
        return 1.0
    if (energy>333902982536.1107):
        return 0.0
    if (energy>333901936368.0886):
        return 1.0
    if (energy>333865822853.35486):
        return 0.0
    if (energy>333863045937.20135):
        return 1.0
    if (energy>333842787827.2501):
        return 0.0
    if (energy>333838185977.8569):
        return 1.0
    if (energy>333567571610.83923):
        return 0.0
    if (energy>333564136159.82434):
        return 1.0
    if (energy>333492350977.666):
        return 0.0
    if (energy>333488528856.8998):
        return 1.0
    if (energy>333481849034.058):
        return 0.0
    if (energy>333481236634.5796):
        return 1.0
    if (energy>333262959388.9471):
        return 0.0
    if (energy>333261110238.1793):
        return 1.0
    if (energy>333214001280.6494):
        return 0.0
    if (energy>333212733986.8142):
        return 1.0
    if (energy>333153264882.85547):
        return 0.0
    if (energy>333151988889.2848):
        return 1.0
    if (energy>333143461502.42914):
        return 0.0
    if (energy>333142408329.67633):
        return 1.0
    if (energy>333073586926.89197):
        return 0.0
    if (energy>333071962825.6334):
        return 1.0
    if (energy>332752197543.90015):
        return 0.0
    if (energy>332751086405.1849):
        return 1.0
    if (energy>332688453837.7137):
        return 0.0
    if (energy>332687666980.96954):
        return 1.0
    if (energy>332674129056.04517):
        return 0.0
    if (energy>332670296412.57745):
        return 1.0
    if (energy>332656751843.62634):
        return 0.0
    if (energy>332652232764.4774):
        return 1.0
    if (energy>332040219586.2147):
        return 0.0
    if (energy>332039045237.8273):
        return 1.0
    if (energy>331827609093.7018):
        return 0.0
    if (energy>331824129748.52893):
        return 1.0
    if (energy>331822735973.8566):
        return 0.0
    if (energy>331821959367.834):
        return 1.0
    if (energy>331730234931.89514):
        return 0.0
    if (energy>331726661226.9388):
        return 1.0
    if (energy>331609742147.16455):
        return 0.0
    if (energy>331607066029.3671):
        return 1.0
    if (energy>331606019269.575):
        return 0.0
    if (energy>331604295372.0121):
        return 1.0
    if (energy>330907046963.5681):
        return 0.0
    if (energy>330903747501.1119):
        return 1.0
    if (energy>330632521503.55334):
        return 0.0
    if (energy>330629530697.1843):
        return 1.0
    if (energy>330253542753.0232):
        return 0.0
    if (energy>330250561313.4678):
        return 1.0
    if (energy>330181636393.10803):
        return 0.0
    if (energy>330180798074.2436):
        return 1.0
    if (energy>330170785432.78625):
        return 0.0
    if (energy>330165548804.7935):
        return 1.0
    if (energy>330141182485.02454):
        return 0.0
    if (energy>330133033538.7396):
        return 1.0
    if (energy>329887671312.0152):
        return 0.0
    if (energy>329884886713.48126):
        return 1.0
    if (energy>329805593373.742):
        return 0.0
    if (energy>329801546644.1974):
        return 1.0
    if (energy>329426342034.60693):
        return 0.0
    if (energy>329421989237.02454):
        return 1.0
    if (energy>329332983969.8396):
        return 0.0
    if (energy>329321363630.8346):
        return 1.0
    if (energy>327572114312.3235):
        return 0.0
    if (energy>327559176620.567):
        return 1.0
    if (energy>327540141893.1026):
        return 0.0
    if (energy>327525789698.00824):
        return 1.0
    if (energy>326719819773.798):
        return 0.0
    if (energy>326689051021.8807):
        return 1.0
    return 0.0

numthresholds=896


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

