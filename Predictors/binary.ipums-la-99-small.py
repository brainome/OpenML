#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f QC -target binaryClass ipums-la-99-small.csv -o ipums-la-99-small.py -nsamples 0 --yes -nsamples 0 -e 100
# Total compiler execution time: 0:03:22.94. Finished on: Sep-03-2020 17:36:39.
# This source code requires Python 3.
#
"""
Classifier Type:                     Decision Tree
System Type:                         Binary classifier
Training/Validation Split:           70:30%
Best-guess accuracy:                 93.57%
Training accuracy:                   100.00% (7075/7075 correct)
Validation accuracy:                 88.24% (1561/1769 correct)
Overall Model accuracy:              97.64% (8636/8844 correct)
Overall Improvement over best guess: 4.07% (of possible 6.43%)
Model capacity (MEC):                764 bits
Generalization ratio:                11.30 bits/bit
Model efficiency:                    0.00%/parameter
System behavior
True Negatives:                      92.61% (8190/8844)
True Positives:                      5.04% (446/8844)
False Negatives:                     1.38% (122/8844)
False Positives:                     0.97% (86/8844)
True Pos. Rate/Sensitivity/Recall:   0.79
True Neg. Rate/Specificity:          0.99
Precision:                           0.84
F-1 Measure:                         0.81
False Negative Rate/Miss Rate:       0.21
Critical Success Index:              0.68
Confusion Matrix:
 [92.61% 0.97%]
 [1.38% 5.04%]
Overfitting:                         No
Note: Labels have been remapped to 'P'=0, 'N'=1.
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
try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "ipums-la-99-small.csv"


#Number of attributes
num_attr = 56
n_classes = 2


# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target="binaryClass"


def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"
    if ignorelabels == [] and ignorecolumns == [] and target == "":
        return
    if (testfile):
        target = ''
        hc = -1
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless == False):
                header=next(reader, None)
                try:
                    if not testfile:
                        if (target != ''): 
                            hc = header.index(target)
                        else:
                            hc = len(header) - 1
                            target=header[hc]
                except:
                    raise NameError("Target '" + target + "' not found! Header must be same as in file passed to btc.")
                for i in range(0, len(ignorecolumns)):
                    try:
                        col = header.index(ignorecolumns[i])
                        if not testfile:
                            if (col == hc):
                                raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
                        il = il + [col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
                first = True
                for i in range(0, len(header)):

                    if (i == hc):
                        continue
                    if (i in il):
                        continue
                    if first:
                        first = False
                    else:
                        print(",", end='', file=outputfile)
                    print(header[i], end='', file=outputfile)
                if not testfile:
                    print("," + header[hc], file=outputfile)
                else:
                    print("", file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if target and (row[target] in ignorelabels):
                        continue
                    first = True
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name == target):
                            continue
                        if first:
                            first = False
                        else:
                            print(",", end='', file=outputfile)
                        if (',' in row[name]):
                            print('"' + row[name].replace('"', '') + '"', end='', file=outputfile)
                        else:
                            print(row[name].replace('"', ''), end='', file=outputfile)
                    if not testfile:
                        print("," + row[target], file=outputfile)
                    else:
                        print("", file=outputfile)

            else:
                try:
                    if (target != ""): 
                        hc = int(target)
                    else:
                        hc = -1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0, len(ignorecolumns)):
                    try:
                        col = int(ignorecolumns[i])
                        if (col == hc):
                            raise ValueError("Attribute " + str(col) + " is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il = il + [col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    first = True
                    if (hc == -1) and (not testfile):
                        hc = len(row) - 1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0, len(row)):
                        if (i in il):
                            continue
                        if (i == hc):
                            continue
                        if first:
                            first = False
                        else:
                            print(",", end='', file=outputfile)
                        if (',' in row[i]):
                            print('"' + row[i].replace('"', '') + '"', end='', file=outputfile)
                        else:
                            print(row[i].replace('"', ''), end = '', file=outputfile)
                    if not testfile:
                        print("," + row[hc], file=outputfile)
                    else:
                        print("", file=outputfile)


def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    #This function takes a preprocessed csv and cleans it to real numbers for prediction or validation


    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    clean.mapping={'P': 0, 'N': 1}

    def convert(cell):
        value = str(cell)
        try:
            result = int(value)
            return result
        except:
            try:
                result = float(value)
                if (rounding != -1):
                    result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
                return result
            except:
                result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
                return result

    #Function to return key for any value 
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")


    #Function to convert the class label
    def convertclassid(cell):
        if (clean.testfile):
            return convert(cell)
        value = str(cell)
        if (value == ''):
            raise ValueError("All cells in the target column must contain a class label.")

        if (not clean.mapping == {}):
            result = -1
            try:
                result = clean.mapping[cell]
            except:
                raise ValueError("Class label '" + value + "' encountered in input not defined in user-provided mapping.")
            if (not result == int(result)):
                raise ValueError("Class labels must be mapped to integer.")
            if (not str(result) in clean.classlist):
                clean.classlist = clean.classlist + [str(result)]
            return result
        try:
            result = float(cell)
            if (rounding != -1):
                result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
            else:
                result = int(int(result * 100) / 100)  # round classes to two digits

            if (not str(result) in clean.classlist):
                clean.classlist = clean.classlist + [str(result)]
        except:
            result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
            if (result in clean.classlist):
                result = clean.classlist.index(result)
            else:
                clean.classlist = clean.classlist + [result]
                result = clean.classlist.index(result)
            if (not result == int(result)):
                raise ValueError("Class labels must be mappable to integer.")
        finally:
            if (result < 0):
                raise ValueError("Integer class labels must be positive and contiguous.")

        return result


    #Main Cleaning Code
    rowcount = 0
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f = open(outfile, "w+")
        if (headerless == False):
            next(reader, None)
        outbuf = []
        for row in reader:
            if (row == []):  # Skip empty rows
                continue
            rowcount = rowcount + 1
            rowlen = num_attr
            if (not testfile):
                rowlen = rowlen + 1    
            if (not len(row) == rowlen):
                raise ValueError("Column count must match trained predictor. Row " + str(rowcount) + " differs.")
            i = 0
            for elem in row:
                if(i + 1 < len(row)):
                    outbuf.append(str(convert(elem)))
                    outbuf.append(',')
                else:
                    classid = str(convertclassid(elem))
                    outbuf.append(classid)
                i = i + 1
            if (len(outbuf) < IOBUF):
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf = []
        print(''.join(outbuf), end="", file=f)
        f.close()

        if (testfile == False and not len(clean.classlist) >= 2):
            raise ValueError("Number of classes must be at least 2.")

        return get_key, clean.mapping


# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([38818263062.5, 39052625924.5, 40464268163.5, 40555111316.5, 43354642680.5, 43364134354.5, 43459698000.0, 43481008720.5, 43674640613.0, 43678246746.5, 44105953922.5, 44110886540.5, 44170287476.0, 44176532735.5, 44540136393.5, 44553669549.0, 44679477981.0, 44691295022.0, 44920646948.0, 44924503324.5, 44929422748.5, 44930282350.0, 44945518203.5, 44955923055.5, 44977843217.5, 44985688296.0, 45297105950.0, 45306233430.5, 45310007685.5, 45313859108.0, 45319025255.5, 45323920991.5, 45331353462.0, 45332363942.5, 45509985757.0, 45512362446.5, 45517826084.5, 45521618195.5, 45528455356.0, 45532289300.0, 45566671869.0, 45568384626.5, 45578945673.5, 45589543463.0, 45615604770.5, 45616889349.5, 45619612719.0, 45622760715.5, 45638675857.5, 45647056688.0, 45688634137.5, 45700663736.5, 45744389277.5, 45744999749.5, 45908260103.0, 45910145898.0, 45919710265.5, 45921531266.5, 45952770468.0, 45956587814.5, 45965023033.5, 45965606341.0, 46001105460.5, 46002008346.0, 46058161208.5, 46060485799.0, 46066089090.0, 46068037721.0, 46110760583.5, 46111919432.5, 46129288653.5, 46135207289.0, 46192160437.5, 46195629211.5, 46204289846.5, 46207561125.0, 46234818786.0, 46237772441.0, 46252702178.5, 46257874563.0, 46381409435.0, 46385640536.5, 46395312995.5, 46405587921.0, 46409241561.0, 46411915798.5, 46430441744.5, 46436683030.0, 46467847764.5, 46472579361.5, 46512037032.0, 46512731868.0, 46590645772.5, 46592118651.0, 46592596971.0, 46592751567.0, 46617158756.0, 46622789780.0, 46671725991.5, 46679768655.0, 46715320008.0, 46715894358.0, 46746348666.5, 46748904635.0, 46781114307.5, 46784513188.5, 46835466720.5, 46836938399.0, 46892082620.0, 46899685982.0, 46953625988.0, 46956333064.5, 47103619921.5, 47105891406.5, 47239377932.5, 47241438823.5, 47319607941.5, 47319892640.5, 47334969793.5, 47337001212.5, 47355138552.5, 47356840139.0, 47397847981.5, 47398813845.0, 47415557320.5, 47419967465.0, 47451890527.0, 47456803988.0, 47460828407.5, 47462649351.0, 47535646216.5, 47536693962.5, 47571236136.5, 47574313577.0, 47581021345.0, 47583299096.5, 47614107454.0, 47614448936.5, 47618682428.5, 47619154889.5, 47621534511.0, 47622060312.0, 47632445656.0, 47633587845.0, 47675168496.5, 47677640619.0, 47714987394.0, 47720960015.0, 47770282082.5, 47771450594.5, 47798538794.0, 47803870233.0, 47828152299.0, 47829193474.5, 47832151638.5, 47833996020.5, 48032212295.0, 48033641805.5, 48059712471.0, 48063657578.5, 48087304003.0, 48087359219.0, 48095927115.0, 48100772705.0, 48128167071.5, 48131035600.5, 48142379845.0, 48144108087.0, 48234556180.0, 48240121206.0, 48248132523.5, 48249482709.0, 48306940374.5, 48319708491.5, 48360096554.0, 48363739220.5, 48368384341.5, 48371386487.0, 48390040104.0, 48391514884.0, 48451573902.5, 48454381039.5, 48460698904.0, 48466183187.0, 48468790195.0, 48472085323.5, 48505070983.0, 48506036093.5, 48525529655.5, 48527626657.5, 48530016516.0, 48532916801.5, 48576923219.0, 48581333263.5, 48715897841.0, 48716233745.0, 48754426742.5, 48759289942.5, 48795576256.5, 48795864456.0, 48799006470.5, 48799613495.5, 48802217163.0, 48806505933.5, 48828676844.0, 48829546471.0, 48918798174.5, 48919034993.5, 48942991332.5, 48944030814.5, 48960944741.5, 48960974921.5, 48975892673.5, 48979765528.0, 48990998120.5, 48991592241.0, 49004847827.0, 49005896121.0, 49013675745.5, 49013980467.5, 49022463696.0, 49022638646.0, 49052391973.0, 49053685335.5, 49069141653.0, 49071249919.0, 49109406727.0, 49112781223.0, 49116694713.5, 49117512946.5, 49117740588.5, 49120107404.5, 49173064772.0, 49173915894.0, 49178434197.0, 49179432006.0, 49181920197.5, 49183153514.5, 49259217163.0, 49259977900.0, 49291524958.0, 49293376305.5, 49370998164.5, 49371581154.0, 49372654727.0, 49373757468.0, 49398964100.5, 49400631572.5, 49407378092.0, 49409715599.5, 49466067279.0, 49472182781.0, 49478463363.5, 49479821490.0, 49489179067.5, 49489876422.5, 49489888388.5, 49489891038.0, 49547009502.5, 49550019895.5, 49553074320.0, 49553460251.0, 49623936791.5, 49626176637.5, 49629443402.0, 49629973948.0, 49632387564.5, 49632484115.0, 49663078424.0, 49667862094.5, 49736205249.5, 49743908487.0, 49749148505.0, 49749378865.0, 49767385659.0, 49769452388.0, 49798741174.0, 49800785024.0, 49840106426.0, 49843326756.5, 49883959929.5, 49884004405.5, 49887482058.5, 49895178087.0, 49907477898.5, 49912364718.0, 49916828239.0, 49917626388.0, 49938022305.5, 49939435403.5, 49950663225.5, 49950890020.0, 49984189400.0, 49985720671.0, 49996529476.0, 49997619994.0, 50019049375.5, 50020093096.0, 50044173162.0, 50044811826.0, 50046164568.0, 50046809175.5, 50063052073.5, 50065941570.5, 50081555691.0, 50084170652.0, 50089367134.5, 50092475342.0, 50121620649.0, 50121866068.0, 50138907612.5, 50139346146.0, 50163273919.5, 50165365300.5, 50174368018.0, 50177478757.0, 50179213866.5, 50180128644.5, 50183264489.0, 50183572274.0, 50186983963.0, 50187015107.0, 50227624178.0, 50228900051.5, 50305418956.5, 50307962057.5, 50309242627.5, 50309646604.0, 50317798462.5, 50317943735.5, 50318104080.0, 50319150703.5, 50363340679.5, 50364421606.5, 50370829149.5, 50373345055.0, 50377555463.5, 50378367918.0, 50382687491.0, 50383163138.5, 50402412867.5, 50404010382.5, 50412336349.5, 50412365939.5, 50412500388.0, 50412826191.5, 50459096376.0, 50459497056.0, 50460866669.5, 50461984472.0, 50467978641.5, 50470086649.5, 50484779551.5, 50492812586.0, 50494889782.0, 50495868017.5, 50511363925.0, 50511629247.5, 50538044407.5, 50553989345.0, 50595782651.5, 50597078410.0, 50612872157.5, 50612884981.5, 50624575365.0, 50627012806.5, 50660423769.5, 50662267427.5, 50678026556.5, 50678869736.0, 50697212104.5, 50698677194.0, 50731995725.0, 50732199582.0, 50770271618.0, 50772479985.0, 50784411861.0, 50786266233.5, 50807006458.5, 50808366458.0, 50828560977.0, 50829148083.0, 50844560707.0, 50846874361.5, 50849591760.0, 50849921952.5, 50886663120.5, 50887447530.0, 50909888186.5, 50912404133.0, 50915437130.5, 50916462250.0, 50952631028.5, 50956510368.5, 50969934130.0, 50970916479.0, 50984833166.0, 50991656001.5, 50992153610.5, 50993220290.0, 51007206701.5, 51008600573.5, 51074563294.5, 51078187267.0, 51104230329.0, 51105393996.0, 51108448732.5, 51109287789.5, 51121150191.5, 51121168086.0, 51121678008.0, 51121708582.5, 51126643237.0, 51129081493.5, 51169329071.5, 51175991368.5, 51197530970.5, 51201076249.5, 51252516329.5, 51257953567.5, 51262411767.5, 51262442586.0, 51268456235.5, 51269475143.0, 51282160827.0, 51282554562.5, 51293206103.5, 51293734657.5, 51347917670.5, 51349825758.0, 51381050890.0, 51384605207.0, 51391611090.0, 51394169019.5, 51401610187.0, 51403602947.5, 51437618844.0, 51438233246.0, 51485043719.0, 51485964505.0, 51499212807.0, 51501026693.0, 51519944815.0, 51521799876.0, 51543673429.5, 51546583428.0, 51562599847.0, 51564192706.0, 51574029353.0, 51576181295.5, 51582434897.5, 51582804764.0, 51584471567.5, 51586313546.0, 51595603442.5, 51600151238.5, 51619075481.0, 51619153989.0, 51619292237.5, 51620076986.0, 51656446024.0, 51656959064.5, 51673355176.0, 51678364976.5, 51699831655.5, 51700579000.5, 51716616671.0, 51722270643.5, 51759327031.0, 51760032919.5, 51810720215.5, 51811244869.0, 51834523002.0, 51837708489.0, 51881040514.5, 51881527955.0, 51916891674.5, 51920201743.5, 51935870153.0, 51937615891.5, 51940083232.5, 51942623157.5, 52018874214.5, 52019406176.0, 52038802840.0, 52045905807.0, 52076150475.5, 52078246204.5, 52078619471.5, 52079033353.5, 52094516349.5, 52097200724.0, 52109224629.5, 52109995543.0, 52114854606.0, 52117301128.5, 52124775581.0, 52126334060.5, 52138898404.0, 52148994845.0, 52162334033.0, 52163037371.5, 52168058120.0, 52169298732.5, 52192682351.5, 52193981448.0, 52202603200.5, 52206282573.5, 52207092425.0, 52209255198.5, 52251198649.0, 52252211080.5, 52271445607.0, 52271622306.0, 52287749129.5, 52288111363.0, 52323828220.0, 52323863134.0, 52351096992.0, 52352030047.5, 52379627696.5, 52386128086.0, 52410787482.0, 52412649418.0, 52447791782.5, 52447839308.0, 52458215007.5, 52458781290.0, 52470294091.0, 52473939003.0, 52477677473.0, 52478384323.0, 52500205077.5, 52500769562.5, 52500949188.0, 52502819758.0, 52520291255.0, 52524268232.5, 52598732609.5, 52600231014.5, 52605260664.0, 52606835721.0, 52641762052.5, 52643314787.5, 52672356364.5, 52673712892.5, 52678671867.0, 52678711133.5, 52697393276.5, 52702508473.0, 52735395514.0, 52735690964.5, 52741243272.0, 52742931635.5, 52765289994.5, 52766801627.5, 52788628041.0, 52793315361.0, 52837585201.0, 52839353989.0, 52867650254.0, 52869810366.0, 52969405656.5, 52972339368.5, 52998943677.0, 53001178986.5, 53048097160.5, 53050400367.0, 53059437390.5, 53063906049.0, 53099285083.5, 53100635880.5, 53103734035.5, 53106215064.5, 53107584154.0, 53108553584.0, 53115351237.5, 53118495432.5, 53131887673.0, 53132171285.5, 53137998293.5, 53139508161.5, 53157476414.5, 53157484534.0, 53166216518.5, 53175011482.5, 53234173763.5, 53234246311.0, 53263648802.5, 53264939713.5, 53325383229.5, 53330535632.5, 53333798969.5, 53335399301.0, 53337132970.0, 53344086319.5, 53350612168.0, 53353434482.0, 53368157886.5, 53370539732.5, 53372902953.0, 53374874777.5, 53397816757.5, 53403053353.5, 53454880782.5, 53455945617.0, 53486987553.5, 53486996157.0, 53512236755.5, 53515621215.5, 53579027290.0, 53592193168.5, 53596028015.0, 53596032756.5, 53597497855.5, 53603250300.5, 53606921534.0, 53607268524.0, 53611000026.0, 53615676494.5, 53646358099.5, 53647652160.5, 53673797451.0, 53679156719.5, 53706214244.0, 53707302727.0, 53752227961.5, 53753520402.0, 53759354733.5, 53764404675.0, 53807453968.5, 53819099648.0, 53868272412.0, 53870415487.0, 53882723754.0, 53885553788.5, 53893316167.0, 53911218782.5, 53944794637.5, 53944871403.0, 53951607933.0, 53959495663.5, 53983163534.0, 53986620694.5, 53992767984.0, 53996123182.0, 54000362874.5, 54003140546.0, 54011279560.0, 54014122070.5, 54016116326.0, 54018636389.0, 54051415995.5, 54051929043.0, 54061953739.5, 54063141942.5, 54092656544.5, 54100731153.5, 54108176884.5, 54109442476.0, 54121300752.0, 54122948702.5, 54206924643.0, 54207787189.5, 54266451230.0, 54282893381.5, 54318463138.0, 54320605085.0, 54378427168.5, 54382475955.5, 54416674064.0, 54417621970.5, 54492877434.0, 54492903791.5, 54543171639.5, 54549691817.0, 54557883613.0, 54560256533.0, 54563079353.5, 54563817811.5, 54630312766.0, 54631363940.5, 54708937599.0, 54713289770.0, 54728318782.5, 54730036080.0, 54736793820.5, 54739234434.5, 54757988444.0, 54758879795.0, 54777468975.5, 54785545240.0, 54792002949.0, 54794566229.5, 54810270564.5, 54813051660.5, 54917484418.0, 54921880472.5, 54944385432.0, 54945149681.5, 54994741343.5, 55005715592.0, 55015628113.0, 55022883207.0, 55050569300.0, 55058280181.5, 55102875978.5, 55109185340.0, 55321983971.0, 55322052871.0, 55322092217.0, 55329599070.0, 55456979737.0, 55463340363.5, 55504394414.0, 55507590899.5, 55650283504.5, 55658768027.0, 55771522020.0, 55775613402.0, 55778173695.5, 55780231114.0, 55801677857.0, 55806632859.5, 55874129466.0, 55876698801.5, 55879716455.5, 55881139876.0, 56192181294.5, 56197573012.5, 56248078327.0, 56262611602.5, 56308637222.5, 56318481099.5, 56326109995.5, 56328251889.0, 56431340818.0, 56442159068.5, 56566118737.0, 56575964740.5, 56662701861.0, 56663754107.5, 56767501405.5, 56774499831.5, 56996306081.0, 57000263089.5, 57029855003.0, 57040080466.0, 57046312945.0, 57054733305.5, 57371919215.5, 57376045838.5, 57493728092.0, 57505190684.0, 57925233369.0, 57958490478.0, 57998664768.0, 58017706963.0, 58164552271.0, 58177122179.0, 58291150057.0, 58330053809.0, 59298035639.5, 59303373407.0])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)
    start_label = 1
    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = (numers[indys] + start_label) % 2
        outputs[defaultindys]=0
        return outputs
    return thresh_search(energys)

numthresholds = 764



# Main method
model_cap = numthresholds


def Validate(file):
    #Load Array
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')


    if n_classes == 2:
        #note that classification is a single line of code
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        correct_count = int(np.sum(outputs.reshape(-1) == cleanarr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(cleanarr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(cleanarr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs, cleanarr[:, -1]


    else:
        #validation
        outputs = classify(cleanarr[:, :-1])


        #metrics
        count, correct_count = 0, 0
        numeachclass = {}
        for k, o in enumerate(outputs):
            if int(o) == int(float(cleanarr[k, -1])):
                correct_count += 1
            if int(float(cleanarr[k, -1])) in numeachclass.keys():
                numeachclass[int(float(cleanarr[k, -1]))] += 1
            else:
                numeachclass[int(float(cleanarr[k, -1]))] = 1
            count += 1
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    cleanarr = cleanarr.reshape(cleanarr.shape[0], -1)
    with open(preprocessedfile, 'r') as csvinput:
        dirtyreader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(dirtyreader, None) + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            print(str(','.join(str(j) for j in (['"' + i + '"' if ',' in i else i for i in row]))) + ',' + str(get_key(int(outputs[k]), classmapping)))



#Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    args = parser.parse_args()
    faulthandler.enable()

    #clean if not already clean
    if not args.cleanfile:
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}

    #Predict or Validate?
    if not args.validate:
        Predict(cleanfile, get_key, args.headerless, preprocessedfile, classmapping)


    else:
        classifier_type = 'DT'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
        if args.json:
            import json
        if n_classes == 2:
            #Base metrics
            FN = float(num_FN) * 100.0 / float(count)
            FP = float(num_FP) * 100.0 / float(count)
            TN = float(num_TN) * 100.0 / float(count)
            TP = float(num_TP) * 100.0 / float(count)
            num_correct = correct_count
        
            #Calculated Metrics
            if int(num_TP + num_FN) != 0:
                TPR = num_TP / (num_TP + num_FN) # Sensitivity, Recall
            if int(num_TN + num_FP) != 0:
                TNR = num_TN / (num_TN + num_FP) # Specificity
            if int(num_TP + num_FP) != 0:
                PPV = num_TP / (num_TP + num_FP) # Recall
            if int(num_FN + num_TP) != 0:
                FNR = num_FN / (num_FN + num_TP) # Miss rate
            if int(2 * num_TP + num_FP + num_FN) != 0:
                FONE = 2 * num_TP / (2 * num_TP + num_FP + num_FN) # F1 Score
            if int(num_TP + num_FN + num_FP) != 0:
                TS = num_TP / (num_TP + num_FN + num_FP) # Critical Success Index
            #Best Guess Accuracy
            randguess = int(float(10000.0 * max(num_class_1, num_class_0)) / count) / 100.0
            #Model Accuracy
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            #Report
            if args.json:
                #                json_dict = {'Instance Count':count, 'classifier_type':classifier_type, 'n_classes':2, 'Number of False Negative Instances': num_FN, 'Number of False Positive Instances': num_FP, 'Number of True Positive Instances': num_TP, 'Number of True Negative Instances': num_TN,   'False Negatives': FN, 'False Positives': FP, 'True Negatives': TN, 'True Positives': TP, 'Number Correct': num_correct, 'Best Guess': randguess, 'Model Accuracy': modelacc, 'Model Capacity': model_cap, 'Generalization Ratio': int(float(num_correct * 100) / model_cap) / 100.0, 'Model Efficiency': int(100 * (modelacc - randguess) / model_cap) / 100.0}
                json_dict = {'instance_count':                        count ,
                            'classifier_type':                        classifier_type ,
                            'n_classes':                            2 ,
                            'number_of_false_negative_instances':    num_FN ,
                            'number_of_false_positive_instances':    num_FP ,
                            'number_of_true_positive_instances':    num_TP ,
                            'number_of_true_negative_instances':    num_TN,
                            'false_negatives':                        FN ,
                            'false_positives':                        FP ,
                            'true_negatives':                        TN ,
                            'true_positives':                        TP ,
                            'number_correct':                        num_correct ,
                            'best_guess':                            randguess ,
                            'model_accuracy':                        modelacc ,
                            'model_capacity':                        model_cap ,
                            'generalization_ratio':                int(float(num_correct * 100) / model_cap) / 100.0,
                            'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0
                             }
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                else:
                    print("Classifier Type:                    Decision Tree")
                print("System Type:                        Binary classifier")
                print("Best-guess accuracy:                {:.2f}%".format(randguess))
                print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
                print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
                print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
                print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
                print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))
                print("System behavior")
                print("True Negatives:                     {:.2f}%".format(TN) + " (" + str(int(num_TN)) + "/" + str(count) + ")")
                print("True Positives:                     {:.2f}%".format(TP) + " (" + str(int(num_TP)) + "/" + str(count) + ")")
                print("False Negatives:                    {:.2f}%".format(FN) + " (" + str(int(num_FN)) + "/" + str(count) + ")")
                print("False Positives:                    {:.2f}%".format(FP) + " (" + str(int(num_FP)) + "/" + str(count) + ")")
                if int(num_TP + num_FN) != 0:
                    print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
                if int(num_TN + num_FP) != 0:
                    print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
                if int(num_TP + num_FP) != 0:
                    print("Precision:                          {:.2f}".format(PPV))
                if int(2 * num_TP + num_FP + num_FN) != 0:
                    print("F-1 Measure:                        {:.2f}".format(FONE))
                if int(num_TP + num_FN) != 0:
                    print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
                if int(num_TP + num_FN + num_FP) != 0:
                    print("Critical Success Index:             {:.2f}".format(TS))
        
        #Multiclass
        else:
            num_correct = correct_count
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            randguess = round(max(numeachclass.values()) / sum(numeachclass.values()) * 100, 2)
            if args.json:
        #        json_dict = {'Instance Count':count, 'classifier_type':classifier_type, 'Number Correct': num_correct, 'Best Guess': randguess, 'Model Accuracy': modelacc, 'Model Capacity': model_cap, 'Generalization Ratio': int(float(num_correct * 100) / model_cap) / 100.0, 'Model Efficiency': int(100 * (modelacc - randguess) / model_cap) / 100.0, 'n_classes': n_classes}
                json_dict = {'instance_count':                        count,
                            'classifier_type':                        classifier_type,
                            'n_classes':                            n_classes,
                            'number_correct':                        num_correct,
                            'best_guess':                            randguess,
                            'model_accuracy':                        modelacc,
                            'model_capacity':                        model_cap,
                            'generalization_ratio':                int(float(num_correct * 100) / model_cap) / 100.0,
                            'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0
                            }
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                else:
                    print("Classifier Type:                    Decision Tree")
                print("System Type:                        " + str(n_classes) + "-way classifier")
                print("Best-guess accuracy:                {:.2f}%".format(randguess))
                print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
                print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
                print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
                print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
                print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))

        try:
            import numpy as np # For numpy see: http://numpy.org
            from numpy import array
        except:
            print("Note: If you install numpy (https://www.numpy.org) and scipy (https://www.scipy.org) this predictor generates a confusion matrix")

        def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
            #check for numpy/scipy is imported
            try:
                from scipy.sparse import coo_matrix #required for multiclass metrics
            except:
                print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix")
                sys.exit()
            # Compute confusion matrix to evaluate the accuracy of a classification.
            # By definition a confusion matrix :math:C is such that :math:C_{i, j}
            # is equal to the number of observations known to be in group :math:i and
            # predicted to be in group :math:j.
            # Thus in binary classification, the count of true negatives is
            # :math:C_{0,0}, false negatives is :math:C_{1,0}, true positives is
            # :math:C_{1,1} and false positives is :math:C_{0,1}.
            # Read more in the :ref:User Guide <confusion_matrix>.
            # Parameters
            # ----------
            # y_true : array-like of shape (n_samples,)
            # Ground truth (correct) target values.
            # y_pred : array-like of shape (n_samples,)
            # Estimated targets as returned by a classifier.
            # labels : array-like of shape (n_classes), default=None
            # List of labels to index the matrix. This may be used to reorder
            # or select a subset of labels.
            # If None is given, those that appear at least once
            # in y_true or y_pred are used in sorted order.
            # sample_weight : array-like of shape (n_samples,), default=None
            # Sample weights.
            # normalize : {'true', 'pred', 'all'}, default=None
            # Normalizes confusion matrix over the true (rows), predicted (columns)
            # conditions or all the population. If None, confusion matrix will not be
            # normalized.
            # Returns
            # -------
            # C : ndarray of shape (n_classes, n_classes)
            # Confusion matrix.
            # References
            # ----------
            if labels is None:
                labels = np.array(list(set(list(y_true.astype('int')))))
            else:
                labels = np.asarray(labels)
                if np.all([l not in y_true for l in labels]):
                    raise ValueError("At least one label specified must be in y_true")


            if sample_weight is None:
                sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
            else:
                sample_weight = np.asarray(sample_weight)
            if y_true.shape[0]!=y_pred.shape[0]:
                raise ValueError("y_true and y_pred must be of the same length")

            if normalize not in ['true', 'pred', 'all', None]:
                raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


            n_labels = labels.size
            label_to_ind = {y: x for x, y in enumerate(labels)}
            # convert yt, yp into index
            y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
            y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
            # intersect y_pred, y_true with labels, eliminate items not in labels
            ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
            y_pred = y_pred[ind]
            y_true = y_true[ind]
            # also eliminate weights of eliminated items
            sample_weight = sample_weight[ind]
            # Choose the accumulator dtype to always have high precision
            if sample_weight.dtype.kind in {'i', 'u', 'b'}:
                dtype = np.int64
            else:
                dtype = np.float64
            cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype,).toarray()


            with np.errstate(all='ignore'):
                if normalize == 'true':
                    cm = cm / cm.sum(axis=1, keepdims=True)
                elif normalize == 'pred':
                    cm = cm / cm.sum(axis=0, keepdims=True)
                elif normalize == 'all':
                    cm = cm / cm.sum()
                cm = np.nan_to_num(cm)
            return cm
        mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
            print(json.dumps(json_dict))
        else:
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print("Confusion Matrix:")
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])


    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)


