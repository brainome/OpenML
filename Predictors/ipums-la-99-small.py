#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 13:00:08
# Invocation: btc -v -v ipums-la-99-small-1.csv -o ipums-la-99-small-1.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                93.57%
Model accuracy:                     96.78% (8560/8844 correct)
Improvement over best guess:        3.21% (of possible 6.43%)
Model capacity (MEC):               690 bits
Generalization ratio:               12.40 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     92.16% (8151/8844)
True Positives:                     4.62% (409/8844)
False Negatives:                    1.80% (159/8844)
False Positives:                    1.41% (125/8844)
True Pos. Rate/Sensitivity/Recall:  0.72
True Neg. Rate/Specificity:         0.98
Precision:                          0.77
F-1 Measure:                        0.74
False Negative Rate/Miss Rate:      0.28
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
TRAINFILE="ipums-la-99-small-1.csv"


#Number of attributes
num_attr = 56

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
    if (energy>59303373407.0):
        return 0.0
    if (energy>59298035639.5):
        return 1.0
    if (energy>58330053809.0):
        return 0.0
    if (energy>58291150057.0):
        return 1.0
    if (energy>58223615024.5):
        return 0.0
    if (energy>58203097601.5):
        return 1.0
    if (energy>58177122179.0):
        return 0.0
    if (energy>58164552271.0):
        return 1.0
    if (energy>58017706963.0):
        return 0.0
    if (energy>57998664768.0):
        return 1.0
    if (energy>57958490478.0):
        return 0.0
    if (energy>57925233369.0):
        return 1.0
    if (energy>57376045838.5):
        return 0.0
    if (energy>57371919215.5):
        return 1.0
    if (energy>57056353612.0):
        return 0.0
    if (energy>57029855003.0):
        return 1.0
    if (energy>57000263089.5):
        return 0.0
    if (energy>56996306081.0):
        return 1.0
    if (energy>56665661329.0):
        return 0.0
    if (energy>56662701861.0):
        return 1.0
    if (energy>56622715613.0):
        return 0.0
    if (energy>56622116410.5):
        return 1.0
    if (energy>56442159068.5):
        return 0.0
    if (energy>56431340818.0):
        return 1.0
    if (energy>56318481099.5):
        return 0.0
    if (energy>56308637222.5):
        return 1.0
    if (energy>56262611602.5):
        return 0.0
    if (energy>56248078327.0):
        return 1.0
    if (energy>55995062389.0):
        return 0.0
    if (energy>55994661650.0):
        return 1.0
    if (energy>55976607078.5):
        return 0.0
    if (energy>55973985232.0):
        return 1.0
    if (energy>55883328338.0):
        return 0.0
    if (energy>55879716455.5):
        return 1.0
    if (energy>55876698801.5):
        return 0.0
    if (energy>55872240481.0):
        return 1.0
    if (energy>55780231114.0):
        return 0.0
    if (energy>55778173695.5):
        return 1.0
    if (energy>55775613402.0):
        return 0.0
    if (energy>55769484324.0):
        return 1.0
    if (energy>55658768027.0):
        return 0.0
    if (energy>55650287055.0):
        return 1.0
    if (energy>55615777785.5):
        return 0.0
    if (energy>55613021517.0):
        return 1.0
    if (energy>55538054179.5):
        return 0.0
    if (energy>55527248013.5):
        return 1.0
    if (energy>55507590899.5):
        return 0.0
    if (energy>55502505612.0):
        return 1.0
    if (energy>55463485576.5):
        return 0.0
    if (energy>55456979737.0):
        return 1.0
    if (energy>55322244761.0):
        return 0.0
    if (energy>55322088123.5):
        return 1.0
    if (energy>55322052871.0):
        return 0.0
    if (energy>55321491172.0):
        return 1.0
    if (energy>55109185340.0):
        return 0.0
    if (energy>55102835803.0):
        return 1.0
    if (energy>55058280181.5):
        return 0.0
    if (energy>55050569300.0):
        return 1.0
    if (energy>55022883207.0):
        return 0.0
    if (energy>55015628113.0):
        return 1.0
    if (energy>55005714959.0):
        return 0.0
    if (energy>54987974482.5):
        return 1.0
    if (energy>54945149681.5):
        return 0.0
    if (energy>54944385432.0):
        return 1.0
    if (energy>54921880472.5):
        return 0.0
    if (energy>54917484418.0):
        return 1.0
    if (energy>54813051660.5):
        return 0.0
    if (energy>54810267153.0):
        return 1.0
    if (energy>54739268250.5):
        return 0.0
    if (energy>54736793820.5):
        return 1.0
    if (energy>54730036080.0):
        return 0.0
    if (energy>54728378429.5):
        return 1.0
    if (energy>54563817811.5):
        return 0.0
    if (energy>54562545328.5):
        return 1.0
    if (energy>54549691817.0):
        return 0.0
    if (energy>54543171639.5):
        return 1.0
    if (energy>54492903791.5):
        return 0.0
    if (energy>54492877434.0):
        return 1.0
    if (energy>54417621970.5):
        return 0.0
    if (energy>54416674064.0):
        return 1.0
    if (energy>54382475955.5):
        return 0.0
    if (energy>54378858528.5):
        return 1.0
    if (energy>54354637676.0):
        return 0.0
    if (energy>54352805959.0):
        return 1.0
    if (energy>54320605085.0):
        return 0.0
    if (energy>54318463138.0):
        return 1.0
    if (energy>54301606330.5):
        return 0.0
    if (energy>54295170740.0):
        return 1.0
    if (energy>54280636422.5):
        return 0.0
    if (energy>54269489594.0):
        return 1.0
    if (energy>54207781153.5):
        return 0.0
    if (energy>54206924643.0):
        return 1.0
    if (energy>54122948702.5):
        return 0.0
    if (energy>54121300752.0):
        return 1.0
    if (energy>54109442476.0):
        return 0.0
    if (energy>54108176884.5):
        return 1.0
    if (energy>54062877685.0):
        return 0.0
    if (energy>54062306475.5):
        return 1.0
    if (energy>54051933962.0):
        return 0.0
    if (energy>54051172048.5):
        return 1.0
    if (energy>54045039270.5):
        return 0.0
    if (energy>54042062323.5):
        return 1.0
    if (energy>54018636389.0):
        return 0.0
    if (energy>54016116326.0):
        return 1.0
    if (energy>54003140546.0):
        return 0.0
    if (energy>53998746544.0):
        return 1.0
    if (energy>53994925669.5):
        return 0.0
    if (energy>53992767984.0):
        return 1.0
    if (energy>53986620694.5):
        return 0.0
    if (energy>53983163534.0):
        return 1.0
    if (energy>53960614699.5):
        return 0.0
    if (energy>53951607933.0):
        return 1.0
    if (energy>53944871403.0):
        return 0.0
    if (energy>53944794637.5):
        return 1.0
    if (energy>53870415487.0):
        return 0.0
    if (energy>53868272412.0):
        return 1.0
    if (energy>53819099648.0):
        return 0.0
    if (energy>53807448665.0):
        return 1.0
    if (energy>53764404675.0):
        return 0.0
    if (energy>53758908775.0):
        return 1.0
    if (energy>53707302727.0):
        return 0.0
    if (energy>53706214244.0):
        return 1.0
    if (energy>53679156719.5):
        return 0.0
    if (energy>53673797451.0):
        return 1.0
    if (energy>53647771483.0):
        return 0.0
    if (energy>53646358099.5):
        return 1.0
    if (energy>53615676494.5):
        return 0.0
    if (energy>53611000026.0):
        return 1.0
    if (energy>53607268524.0):
        return 0.0
    if (energy>53606928589.5):
        return 1.0
    if (energy>53603250300.5):
        return 0.0
    if (energy>53597998191.0):
        return 1.0
    if (energy>53596032756.5):
        return 0.0
    if (energy>53596028015.0):
        return 1.0
    if (energy>53589010019.5):
        return 0.0
    if (energy>53576629383.0):
        return 1.0
    if (energy>53515621215.5):
        return 0.0
    if (energy>53513497483.5):
        return 1.0
    if (energy>53486996157.0):
        return 0.0
    if (energy>53486980695.5):
        return 1.0
    if (energy>53455945617.0):
        return 0.0
    if (energy>53454880782.5):
        return 1.0
    if (energy>53403053353.5):
        return 0.0
    if (energy>53397816757.5):
        return 1.0
    if (energy>53374874777.5):
        return 0.0
    if (energy>53368157886.5):
        return 1.0
    if (energy>53360283896.5):
        return 0.0
    if (energy>53359083595.5):
        return 1.0
    if (energy>53354607237.0):
        return 0.0
    if (energy>53350314549.5):
        return 1.0
    if (energy>53339381099.5):
        return 0.0
    if (energy>53333789490.0):
        return 1.0
    if (energy>53332062332.0):
        return 0.0
    if (energy>53325680098.0):
        return 1.0
    if (energy>53171704776.5):
        return 0.0
    if (energy>53166216518.5):
        return 1.0
    if (energy>53157484534.0):
        return 0.0
    if (energy>53157476414.5):
        return 1.0
    if (energy>53139508161.5):
        return 0.0
    if (energy>53138575547.0):
        return 1.0
    if (energy>53132171285.5):
        return 0.0
    if (energy>53131887673.0):
        return 1.0
    if (energy>53120955073.5):
        return 0.0
    if (energy>53114666906.0):
        return 1.0
    if (energy>53108553584.0):
        return 0.0
    if (energy>53107584154.0):
        return 1.0
    if (energy>53106215064.5):
        return 0.0
    if (energy>53103734035.5):
        return 1.0
    if (energy>53100635880.5):
        return 0.0
    if (energy>53099285083.5):
        return 1.0
    if (energy>53063326044.5):
        return 0.0
    if (energy>53059437390.5):
        return 1.0
    if (energy>53050400367.0):
        return 0.0
    if (energy>53047239582.0):
        return 1.0
    if (energy>53022301783.0):
        return 0.0
    if (energy>53021192494.5):
        return 1.0
    if (energy>53001190667.5):
        return 0.0
    if (energy>52998976031.5):
        return 1.0
    if (energy>52973719419.5):
        return 0.0
    if (energy>52972097173.5):
        return 1.0
    if (energy>52961754990.5):
        return 0.0
    if (energy>52960834598.5):
        return 1.0
    if (energy>52869810366.0):
        return 0.0
    if (energy>52867650254.0):
        return 1.0
    if (energy>52821196642.5):
        return 0.0
    if (energy>52816611993.5):
        return 1.0
    if (energy>52793315361.0):
        return 0.0
    if (energy>52788628041.0):
        return 1.0
    if (energy>52766801627.5):
        return 0.0
    if (energy>52765289994.5):
        return 1.0
    if (energy>52742962704.0):
        return 0.0
    if (energy>52741243272.0):
        return 1.0
    if (energy>52736480392.5):
        return 0.0
    if (energy>52735395514.0):
        return 1.0
    if (energy>52673712892.5):
        return 0.0
    if (energy>52672356364.5):
        return 1.0
    if (energy>52644219564.5):
        return 0.0
    if (energy>52641762052.5):
        return 1.0
    if (energy>52607442947.0):
        return 0.0
    if (energy>52604198080.0):
        return 1.0
    if (energy>52529708626.5):
        return 0.0
    if (energy>52528417710.5):
        return 1.0
    if (energy>52524268232.5):
        return 0.0
    if (energy>52520289616.5):
        return 1.0
    if (energy>52506828379.0):
        return 0.0
    if (energy>52505583468.5):
        return 1.0
    if (energy>52501318029.5):
        return 0.0
    if (energy>52500949188.0):
        return 1.0
    if (energy>52500769562.5):
        return 0.0
    if (energy>52500205077.5):
        return 1.0
    if (energy>52497164456.0):
        return 0.0
    if (energy>52494048839.5):
        return 1.0
    if (energy>52478384323.0):
        return 0.0
    if (energy>52477677473.0):
        return 1.0
    if (energy>52412944413.5):
        return 0.0
    if (energy>52409376517.5):
        return 1.0
    if (energy>52403522905.0):
        return 0.0
    if (energy>52401961725.0):
        return 1.0
    if (energy>52352030047.5):
        return 0.0
    if (energy>52351096992.0):
        return 1.0
    if (energy>52348406477.5):
        return 0.0
    if (energy>52347941395.0):
        return 1.0
    if (energy>52329872236.5):
        return 0.0
    if (energy>52329207492.5):
        return 1.0
    if (energy>52323868925.0):
        return 0.0
    if (energy>52323827407.0):
        return 1.0
    if (energy>52272217553.0):
        return 0.0
    if (energy>52271445607.0):
        return 1.0
    if (energy>52252046695.5):
        return 0.0
    if (energy>52250613639.0):
        return 1.0
    if (energy>52209255198.5):
        return 0.0
    if (energy>52207038154.0):
        return 1.0
    if (energy>52206282573.5):
        return 0.0
    if (energy>52202603200.5):
        return 1.0
    if (energy>52193626024.5):
        return 0.0
    if (energy>52190158949.0):
        return 1.0
    if (energy>52169302773.5):
        return 0.0
    if (energy>52168267056.0):
        return 1.0
    if (energy>52148994845.0):
        return 0.0
    if (energy>52138898404.0):
        return 1.0
    if (energy>52126639375.5):
        return 0.0
    if (energy>52124775581.0):
        return 1.0
    if (energy>52117301128.5):
        return 0.0
    if (energy>52114854606.0):
        return 1.0
    if (energy>52109995543.0):
        return 0.0
    if (energy>52109224629.5):
        return 1.0
    if (energy>52079033353.5):
        return 0.0
    if (energy>52078619471.5):
        return 1.0
    if (energy>52078246204.5):
        return 0.0
    if (energy>52076146876.5):
        return 1.0
    if (energy>52045905807.0):
        return 0.0
    if (energy>52038802840.0):
        return 1.0
    if (energy>52019407242.5):
        return 0.0
    if (energy>52018146675.5):
        return 1.0
    if (energy>51942623157.5):
        return 0.0
    if (energy>51940574841.0):
        return 1.0
    if (energy>51938107500.0):
        return 0.0
    if (energy>51935926390.0):
        return 1.0
    if (energy>51881527955.0):
        return 0.0
    if (energy>51881036361.0):
        return 1.0
    if (energy>51760032919.5):
        return 0.0
    if (energy>51759327031.0):
        return 1.0
    if (energy>51708015638.0):
        return 0.0
    if (energy>51706845879.5):
        return 1.0
    if (energy>51700788211.0):
        return 0.0
    if (energy>51699831655.5):
        return 1.0
    if (energy>51674515573.0):
        return 0.0
    if (energy>51673054376.0):
        return 1.0
    if (energy>51656959064.5):
        return 0.0
    if (energy>51656446024.0):
        return 1.0
    if (energy>51654036957.5):
        return 0.0
    if (energy>51653358467.5):
        return 1.0
    if (energy>51619562252.0):
        return 0.0
    if (energy>51619297282.5):
        return 1.0
    if (energy>51619167914.0):
        return 0.0
    if (energy>51619075481.0):
        return 1.0
    if (energy>51600151238.5):
        return 0.0
    if (energy>51595603442.5):
        return 1.0
    if (energy>51576181295.5):
        return 0.0
    if (energy>51574320905.0):
        return 1.0
    if (energy>51564262530.5):
        return 0.0
    if (energy>51562599847.0):
        return 1.0
    if (energy>51546637835.0):
        return 0.0
    if (energy>51544984798.5):
        return 1.0
    if (energy>51521799876.0):
        return 0.0
    if (energy>51519944815.0):
        return 1.0
    if (energy>51492077897.5):
        return 0.0
    if (energy>51490883512.5):
        return 1.0
    if (energy>51485964505.0):
        return 0.0
    if (energy>51483601795.5):
        return 1.0
    if (energy>51463387320.5):
        return 0.0
    if (energy>51461612259.0):
        return 1.0
    if (energy>51438079618.0):
        return 0.0
    if (energy>51437618844.0):
        return 1.0
    if (energy>51371399661.0):
        return 0.0
    if (energy>51371387515.0):
        return 1.0
    if (energy>51293734657.5):
        return 0.0
    if (energy>51293206103.5):
        return 1.0
    if (energy>51282550929.5):
        return 0.0
    if (energy>51282160827.0):
        return 1.0
    if (energy>51269475143.0):
        return 0.0
    if (energy>51268456235.5):
        return 1.0
    if (energy>51201894067.0):
        return 0.0
    if (energy>51201668638.5):
        return 1.0
    if (energy>51176489334.5):
        return 0.0
    if (energy>51169329071.5):
        return 1.0
    if (energy>51155611815.5):
        return 0.0
    if (energy>51149814238.0):
        return 1.0
    if (energy>51128093237.0):
        return 0.0
    if (energy>51126643244.5):
        return 1.0
    if (energy>51121708582.5):
        return 0.0
    if (energy>51121523549.5):
        return 1.0
    if (energy>51121168474.5):
        return 0.0
    if (energy>51121150191.5):
        return 1.0
    if (energy>51109287789.5):
        return 0.0
    if (energy>51108448732.5):
        return 1.0
    if (energy>51105393996.0):
        return 0.0
    if (energy>51104230329.0):
        return 1.0
    if (energy>51071121062.5):
        return 0.0
    if (energy>51069249759.0):
        return 1.0
    if (energy>51064767114.0):
        return 0.0
    if (energy>51064465347.0):
        return 1.0
    if (energy>51014443230.5):
        return 0.0
    if (energy>51013793487.5):
        return 1.0
    if (energy>50993471444.0):
        return 0.0
    if (energy>50984323751.0):
        return 1.0
    if (energy>50916462250.0):
        return 0.0
    if (energy>50915437130.5):
        return 1.0
    if (energy>50887447530.0):
        return 0.0
    if (energy>50886663120.5):
        return 1.0
    if (energy>50845895638.0):
        return 0.0
    if (energy>50844560707.0):
        return 1.0
    if (energy>50831735343.5):
        return 0.0
    if (energy>50829096445.0):
        return 1.0
    if (energy>50808366458.0):
        return 0.0
    if (energy>50805683274.0):
        return 1.0
    if (energy>50771110388.5):
        return 0.0
    if (energy>50770271618.0):
        return 1.0
    if (energy>50732500177.0):
        return 0.0
    if (energy>50731995725.0):
        return 1.0
    if (energy>50711164937.5):
        return 0.0
    if (energy>50708698240.5):
        return 1.0
    if (energy>50698677194.0):
        return 0.0
    if (energy>50697212104.5):
        return 1.0
    if (energy>50678869736.0):
        return 0.0
    if (energy>50678025203.0):
        return 1.0
    if (energy>50663679370.5):
        return 0.0
    if (energy>50660069335.5):
        return 1.0
    if (energy>50627012806.5):
        return 0.0
    if (energy>50624105633.0):
        return 1.0
    if (energy>50612884981.5):
        return 0.0
    if (energy>50612835725.0):
        return 1.0
    if (energy>50604667768.5):
        return 0.0
    if (energy>50603010592.0):
        return 1.0
    if (energy>50597078410.0):
        return 0.0
    if (energy>50596187012.0):
        return 1.0
    if (energy>50581288919.5):
        return 0.0
    if (energy>50579992302.5):
        return 1.0
    if (energy>50545488967.0):
        return 0.0
    if (energy>50537245077.0):
        return 1.0
    if (energy>50511661962.5):
        return 0.0
    if (energy>50511435766.0):
        return 1.0
    if (energy>50511403051.0):
        return 0.0
    if (energy>50511360581.5):
        return 1.0
    if (energy>50494059987.5):
        return 0.0
    if (energy>50483339691.0):
        return 1.0
    if (energy>50470086649.5):
        return 0.0
    if (energy>50467978641.5):
        return 1.0
    if (energy>50461984472.0):
        return 0.0
    if (energy>50460760527.0):
        return 1.0
    if (energy>50442187813.0):
        return 0.0
    if (energy>50440109518.0):
        return 1.0
    if (energy>50412752302.0):
        return 0.0
    if (energy>50412489307.5):
        return 1.0
    if (energy>50412365939.5):
        return 0.0
    if (energy>50412055215.0):
        return 1.0
    if (energy>50404010382.5):
        return 0.0
    if (energy>50402412867.5):
        return 1.0
    if (energy>50382948421.5):
        return 0.0
    if (energy>50382687491.0):
        return 1.0
    if (energy>50373729805.5):
        return 0.0
    if (energy>50370447499.0):
        return 1.0
    if (energy>50364421606.5):
        return 0.0
    if (energy>50363340679.5):
        return 1.0
    if (energy>50354072603.5):
        return 0.0
    if (energy>50346444241.0):
        return 1.0
    if (energy>50310246990.5):
        return 0.0
    if (energy>50304159510.0):
        return 1.0
    if (energy>50229983591.0):
        return 0.0
    if (energy>50224715457.0):
        return 1.0
    if (energy>50195572402.0):
        return 0.0
    if (energy>50193941661.5):
        return 1.0
    if (energy>50187015107.0):
        return 0.0
    if (energy>50186983963.0):
        return 1.0
    if (energy>50183572274.0):
        return 0.0
    if (energy>50183264489.0):
        return 1.0
    if (energy>50139346146.0):
        return 0.0
    if (energy>50138239833.0):
        return 1.0
    if (energy>50121866068.0):
        return 0.0
    if (energy>50121620649.0):
        return 1.0
    if (energy>50092475342.0):
        return 0.0
    if (energy>50089367134.5):
        return 1.0
    if (energy>50083990390.5):
        return 0.0
    if (energy>50081897950.0):
        return 1.0
    if (energy>50065761764.5):
        return 0.0
    if (energy>50063414762.5):
        return 1.0
    if (energy>50052034602.0):
        return 0.0
    if (energy>50049478309.0):
        return 1.0
    if (energy>50047635487.0):
        return 0.0
    if (energy>50045759616.0):
        return 1.0
    if (energy>50020093096.0):
        return 0.0
    if (energy>50019049375.5):
        return 1.0
    if (energy>50011331445.5):
        return 0.0
    if (energy>50010185171.0):
        return 1.0
    if (energy>49964614528.5):
        return 0.0
    if (energy>49961277155.0):
        return 1.0
    if (energy>49950890020.0):
        return 0.0
    if (energy>49950663225.5):
        return 1.0
    if (energy>49917626388.0):
        return 0.0
    if (energy>49916828239.0):
        return 1.0
    if (energy>49895214729.5):
        return 0.0
    if (energy>49886889718.5):
        return 1.0
    if (energy>49851005224.0):
        return 0.0
    if (energy>49850316010.5):
        return 1.0
    if (energy>49845637111.5):
        return 0.0
    if (energy>49844651362.5):
        return 1.0
    if (energy>49836858900.0):
        return 0.0
    if (energy>49835429627.5):
        return 1.0
    if (energy>49768851951.5):
        return 0.0
    if (energy>49766524935.5):
        return 1.0
    if (energy>49743948010.0):
        return 0.0
    if (energy>49735330952.0):
        return 1.0
    if (energy>49692050721.0):
        return 0.0
    if (energy>49686284122.0):
        return 1.0
    if (energy>49665749895.0):
        return 0.0
    if (energy>49660882844.5):
        return 1.0
    if (energy>49629973948.0):
        return 0.0
    if (energy>49629443402.0):
        return 1.0
    if (energy>49626176637.5):
        return 0.0
    if (energy>49623448859.0):
        return 1.0
    if (energy>49571597368.0):
        return 0.0
    if (energy>49569428276.0):
        return 1.0
    if (energy>49553460251.0):
        return 0.0
    if (energy>49553067829.5):
        return 1.0
    if (energy>49549168989.5):
        return 0.0
    if (energy>49547009502.5):
        return 1.0
    if (energy>49494318442.0):
        return 0.0
    if (energy>49493310906.5):
        return 1.0
    if (energy>49489891038.0):
        return 0.0
    if (energy>49489884784.0):
        return 1.0
    if (energy>49489876422.5):
        return 0.0
    if (energy>49489141687.5):
        return 1.0
    if (energy>49400735158.5):
        return 0.0
    if (energy>49398964100.5):
        return 1.0
    if (energy>49373757468.0):
        return 0.0
    if (energy>49372654727.0):
        return 1.0
    if (energy>49332735066.0):
        return 0.0
    if (energy>49327273315.5):
        return 1.0
    if (energy>49306140263.5):
        return 0.0
    if (energy>49291834919.0):
        return 1.0
    if (energy>49259977900.0):
        return 0.0
    if (energy>49258448551.5):
        return 1.0
    if (energy>49188878759.0):
        return 0.0
    if (energy>49186059862.0):
        return 1.0
    if (energy>49183150285.5):
        return 0.0
    if (energy>49181920197.5):
        return 1.0
    if (energy>49175553633.5):
        return 0.0
    if (energy>49172850670.5):
        return 1.0
    if (energy>49149854910.5):
        return 0.0
    if (energy>49146454718.0):
        return 1.0
    if (energy>49117512946.5):
        return 0.0
    if (energy>49115049099.5):
        return 1.0
    if (energy>49112363738.5):
        return 0.0
    if (energy>49109146703.0):
        return 1.0
    if (energy>49105428098.5):
        return 0.0
    if (energy>49102024543.0):
        return 1.0
    if (energy>49069846725.5):
        return 0.0
    if (energy>49069141653.0):
        return 1.0
    if (energy>49053685335.5):
        return 0.0
    if (energy>49052364236.0):
        return 1.0
    if (energy>49042010676.0):
        return 0.0
    if (energy>49039985473.0):
        return 1.0
    if (energy>49022638646.0):
        return 0.0
    if (energy>49022158562.5):
        return 1.0
    if (energy>49013980467.5):
        return 0.0
    if (energy>49013675745.5):
        return 1.0
    if (energy>49005896121.0):
        return 0.0
    if (energy>49004847827.0):
        return 1.0
    if (energy>48991592241.0):
        return 0.0
    if (energy>48990998120.5):
        return 1.0
    if (energy>48979765528.0):
        return 0.0
    if (energy>48975892673.5):
        return 1.0
    if (energy>48961421720.5):
        return 0.0
    if (energy>48960944741.5):
        return 1.0
    if (energy>48919034993.5):
        return 0.0
    if (energy>48918798174.5):
        return 1.0
    if (energy>48829546471.0):
        return 0.0
    if (energy>48828676844.0):
        return 1.0
    if (energy>48799613495.5):
        return 0.0
    if (energy>48799006470.5):
        return 1.0
    if (energy>48780684205.5):
        return 0.0
    if (energy>48778065973.5):
        return 1.0
    if (energy>48763661763.0):
        return 0.0
    if (energy>48762090728.5):
        return 1.0
    if (energy>48716835566.5):
        return 0.0
    if (energy>48715637339.0):
        return 1.0
    if (energy>48689667897.0):
        return 0.0
    if (energy>48688190366.0):
        return 1.0
    if (energy>48589789504.0):
        return 0.0
    if (energy>48586391544.0):
        return 1.0
    if (energy>48581333263.5):
        return 0.0
    if (energy>48576923219.0):
        return 1.0
    if (energy>48505826676.0):
        return 0.0
    if (energy>48504691163.5):
        return 1.0
    if (energy>48471044503.0):
        return 0.0
    if (energy>48468790195.0):
        return 1.0
    if (energy>48466183187.0):
        return 0.0
    if (energy>48457891767.0):
        return 1.0
    if (energy>48413250361.5):
        return 0.0
    if (energy>48408198662.0):
        return 1.0
    if (energy>48390088207.0):
        return 0.0
    if (energy>48390040104.0):
        return 1.0
    if (energy>48378224970.0):
        return 0.0
    if (energy>48377061391.5):
        return 1.0
    if (energy>48371386487.0):
        return 0.0
    if (energy>48368384341.5):
        return 1.0
    if (energy>48319211939.0):
        return 0.0
    if (energy>48307937709.0):
        return 1.0
    if (energy>48249482709.0):
        return 0.0
    if (energy>48248132523.5):
        return 1.0
    if (energy>48167545110.5):
        return 0.0
    if (energy>48167454439.5):
        return 1.0
    if (energy>48144108087.0):
        return 0.0
    if (energy>48140450964.0):
        return 1.0
    if (energy>48128180061.5):
        return 0.0
    if (energy>48128161301.0):
        return 1.0
    if (energy>48098082349.5):
        return 0.0
    if (energy>48095927115.0):
        return 1.0
    if (energy>47829962734.0):
        return 0.0
    if (energy>47828379567.0):
        return 1.0
    if (energy>47801665819.0):
        return 0.0
    if (energy>47798538794.0):
        return 1.0
    if (energy>47779890219.0):
        return 0.0
    if (energy>47776411724.5):
        return 1.0
    if (energy>47771450594.5):
        return 0.0
    if (energy>47770339691.0):
        return 1.0
    if (energy>47745298096.0):
        return 0.0
    if (energy>47742626267.5):
        return 1.0
    if (energy>47720960015.0):
        return 0.0
    if (energy>47714987394.0):
        return 1.0
    if (energy>47641768711.5):
        return 0.0
    if (energy>47640928551.5):
        return 1.0
    if (energy>47633587845.0):
        return 0.0
    if (energy>47632445656.0):
        return 1.0
    if (energy>47619216814.5):
        return 0.0
    if (energy>47618682428.5):
        return 1.0
    if (energy>47614448936.5):
        return 0.0
    if (energy>47613717530.0):
        return 1.0
    if (energy>47609621013.0):
        return 0.0
    if (energy>47605430297.0):
        return 1.0
    if (energy>47574313577.0):
        return 0.0
    if (energy>47569813840.0):
        return 1.0
    if (energy>47565993561.5):
        return 0.0
    if (energy>47562847736.0):
        return 1.0
    if (energy>47536972960.5):
        return 0.0
    if (energy>47535646216.5):
        return 1.0
    if (energy>47462564914.0):
        return 0.0
    if (energy>47462264530.0):
        return 1.0
    if (energy>47456803988.0):
        return 0.0
    if (energy>47451596526.0):
        return 1.0
    if (energy>47419967465.0):
        return 0.0
    if (energy>47414322340.5):
        return 1.0
    if (energy>47398813845.0):
        return 0.0
    if (energy>47397847981.5):
        return 1.0
    if (energy>47356787188.0):
        return 0.0
    if (energy>47355138552.5):
        return 1.0
    if (energy>47351015274.5):
        return 0.0
    if (energy>47347091253.0):
        return 1.0
    if (energy>47337001212.5):
        return 0.0
    if (energy>47334921531.0):
        return 1.0
    if (energy>47241438823.5):
        return 0.0
    if (energy>47241065629.0):
        return 1.0
    if (energy>47105891406.5):
        return 0.0
    if (energy>47103619921.5):
        return 1.0
    if (energy>47084861009.5):
        return 0.0
    if (energy>47084824878.0):
        return 1.0
    if (energy>47004825018.5):
        return 0.0
    if (energy>47004394609.5):
        return 1.0
    if (energy>46993114362.0):
        return 0.0
    if (energy>46991234978.0):
        return 1.0
    if (energy>46955529636.0):
        return 0.0
    if (energy>46954906232.5):
        return 1.0
    if (energy>46746359432.0):
        return 0.0
    if (energy>46746348666.5):
        return 1.0
    if (energy>46718633169.0):
        return 0.0
    if (energy>46713671824.0):
        return 1.0
    if (energy>46679768655.0):
        return 0.0
    if (energy>46671725991.5):
        return 1.0
    if (energy>46623073289.0):
        return 0.0
    if (energy>46617158756.0):
        return 1.0
    if (energy>46592751567.0):
        return 0.0
    if (energy>46592596971.0):
        return 1.0
    if (energy>46592136022.5):
        return 0.0
    if (energy>46590663607.5):
        return 1.0
    if (energy>46512731868.0):
        return 0.0
    if (energy>46512037032.0):
        return 1.0
    if (energy>46473595976.5):
        return 0.0
    if (energy>46468252414.5):
        return 1.0
    if (energy>46460541930.5):
        return 0.0
    if (energy>46457358231.0):
        return 1.0
    if (energy>46412424286.0):
        return 0.0
    if (energy>46409241561.0):
        return 1.0
    if (energy>46405587921.0):
        return 0.0
    if (energy>46398773082.5):
        return 1.0
    if (energy>46385640536.5):
        return 0.0
    if (energy>46381409435.0):
        return 1.0
    if (energy>46350015283.0):
        return 0.0
    if (energy>46349951662.5):
        return 1.0
    if (energy>46237772441.0):
        return 0.0
    if (energy>46234818786.0):
        return 1.0
    if (energy>46207561125.0):
        return 0.0
    if (energy>46204289846.5):
        return 1.0
    if (energy>46193932611.0):
        return 0.0
    if (energy>46192160437.5):
        return 1.0
    if (energy>46111919432.5):
        return 0.0
    if (energy>46110760583.5):
        return 1.0
    if (energy>46073846452.0):
        return 0.0
    if (energy>46066089090.0):
        return 1.0
    if (energy>46060485799.0):
        return 0.0
    if (energy>46051133777.0):
        return 1.0
    if (energy>45965620889.0):
        return 0.0
    if (energy>45965048423.5):
        return 1.0
    if (energy>45954420109.0):
        return 0.0
    if (energy>45952770468.0):
        return 1.0
    if (energy>45908383603.0):
        return 0.0
    if (energy>45908260103.0):
        return 1.0
    if (energy>45700663736.5):
        return 0.0
    if (energy>45688634137.5):
        return 1.0
    if (energy>45647056688.0):
        return 0.0
    if (energy>45638675857.5):
        return 1.0
    if (energy>45622760715.5):
        return 0.0
    if (energy>45619593375.0):
        return 1.0
    if (energy>45589543463.0):
        return 0.0
    if (energy>45578945673.5):
        return 1.0
    if (energy>45532289300.0):
        return 0.0
    if (energy>45527153470.0):
        return 1.0
    if (energy>45521618195.5):
        return 0.0
    if (energy>45518696121.5):
        return 1.0
    if (energy>45512362446.5):
        return 0.0
    if (energy>45509777465.0):
        return 1.0
    if (energy>45364112002.5):
        return 0.0
    if (energy>45358973037.5):
        return 1.0
    if (energy>45332517296.0):
        return 0.0
    if (energy>45331353462.0):
        return 1.0
    if (energy>45323920991.5):
        return 0.0
    if (energy>45319025255.5):
        return 1.0
    if (energy>45313862405.5):
        return 0.0
    if (energy>45310007685.5):
        return 1.0
    if (energy>45306233430.5):
        return 0.0
    if (energy>45297105950.0):
        return 1.0
    if (energy>45017684066.0):
        return 0.0
    if (energy>45009465743.0):
        return 1.0
    if (energy>44980518938.5):
        return 0.0
    if (energy>44977843217.5):
        return 1.0
    if (energy>44952212466.0):
        return 0.0
    if (energy>44945518203.5):
        return 1.0
    if (energy>44924503324.5):
        return 0.0
    if (energy>44920646948.0):
        return 1.0
    if (energy>44912098377.0):
        return 0.0
    if (energy>44908061224.5):
        return 1.0
    if (energy>44691453183.0):
        return 0.0
    if (energy>44679477981.0):
        return 1.0
    if (energy>44547065330.0):
        return 0.0
    if (energy>44540136393.5):
        return 1.0
    if (energy>44499749132.5):
        return 0.0
    if (energy>44492227084.5):
        return 1.0
    if (energy>44110886540.5):
        return 0.0
    if (energy>44107802775.0):
        return 1.0
    if (energy>43364134354.5):
        return 0.0
    if (energy>43354642680.5):
        return 1.0
    if (energy>43296194897.0):
        return 0.0
    if (energy>43294034774.0):
        return 1.0
    if (energy>43046319930.5):
        return 0.0
    if (energy>43037725650.5):
        return 1.0
    if (energy>42487520043.0):
        return 0.0
    if (energy>42445998078.5):
        return 1.0
    if (energy>42161022148.0):
        return 0.0
    if (energy>42150050777.0):
        return 1.0
    if (energy>40555179603.0):
        return 0.0
    if (energy>40469158519.5):
        return 1.0
    return 0.0

numthresholds=690


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

