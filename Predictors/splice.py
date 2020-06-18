#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc https://www.openml.org/data/get_csv/46/dataset_46_splice.arff -o Predictors/splice_QC.py -target Class -stopat 96.18 -f QC -e 100 --yes --runlocalonly
# Total compiler execution time: 0:01:01.22. Finished on: May-28-2020 00:12:19.
# This source code requires Python 3.
#
"""
Classifier Type: Quick Clustering
System Type:                        3-way classifier
Best-guess accuracy:                51.90%
Model accuracy:                     71.28% (2274/3190 correct)
Improvement over best guess:        19.38% (of possible 48.1%)
Model capacity (MEC):               974 bits
Generalization ratio:               2.33 bits/bit
Confusion Matrix:
 [15.80% 2.26% 5.99%]
 [2.29% 15.33% 6.46%]
 [6.08% 5.64% 40.16%]

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
TRAINFILE = "dataset_46_splice.csv"


#Number of attributes
num_attr = 61
n_classes = 3


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="Class"


    if (testfile):
        target = ''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless == False):
                header=next(reader, None)
                try:
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
                        if (col == hc):
                            raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
                for i in range(0, len(header)):      
                    if (i == hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i] + ",", end='', file=outputfile)
                print(header[hc], file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"' + row[name] + '"' + ",", end='', file=outputfile)
                        else:
                            print (row[name] + ",", end='', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target != ""): 
                        hc = int(target)
                    else:
                        hc =- 1
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
                    if (hc == -1):
                        hc = len(row) - 1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0, len(row)):
                        if (i in il):
                            continue
                        if (i == hc):
                            continue
                        if (',' in row[i]):
                            print ('"' + row[i] + '"'+",", end='', file=outputfile)
                        else:
                            print(row[i]+",", end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = {}
    clean.mapping={'EI': 0, 'IE': 1, 'N': 2}

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

    # function to return key for any value 
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")

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
energy_thresholds = array([82345047264.0, 84060086921.5, 87193244934.0, 89377576472.0, 89878395674.5, 90473884156.0, 90771308443.5, 91818545742.5, 92042876557.5, 92580364381.5, 93973611041.0, 94463618717.5, 94994538484.0, 95348865398.0, 95865804049.0, 96761793258.0, 96996023829.5, 97496320161.0, 98027582860.5, 98222425465.5, 98679061410.0, 98875974501.0, 99171069650.5, 99446854048.0, 99795948982.5, 99960271311.0, 100106148314.5, 101005121341.5, 101087980257.5, 101153000886.5, 101757753170.5, 101925141870.0, 102032536048.0, 102119947983.0, 102611836521.5, 102731225482.5, 102851470210.0, 103649677353.5, 103942998371.0, 103986269115.5, 104069486475.5, 104145153948.0, 104392213027.5, 104405101845.5, 104826049297.5, 105066749847.0, 105141138304.5, 105223711192.5, 105378575458.5, 105785332966.5, 105806095051.0, 105991014438.0, 106277607499.0, 106367788388.0, 106380441489.0, 106552588175.0, 106613858283.0, 106698022142.0, 106925858216.5, 106982723230.5, 107016112911.0, 107137178711.0, 107258192657.5, 107281105057.5, 107430218880.0, 107473990143.5, 107644513264.0, 107939843807.0, 108013161440.0, 108055957995.5, 108122695588.0, 108152706692.5, 108217801098.5, 108277652372.5, 108308234494.5, 108331215606.0, 108370492004.5, 108508285297.0, 108612007432.5, 108638469009.5, 108815120211.0, 108853075733.0, 108885183593.0, 108934475845.0, 109073000630.0, 109132865260.5, 109155839706.5, 109380791802.5, 109417373155.0, 109451568128.5, 109499422326.5, 109534684697.0, 109566882608.5, 109666845384.0, 109693170971.5, 109701167072.0, 109743780346.0, 109805493936.0, 109814114621.0, 109864224034.5, 109947915943.0, 109957427363.0, 110101924583.0, 110302358740.0, 110354981410.0, 110389635519.5, 110469253544.0, 110550673980.5, 110567796025.5, 110622535445.5, 110695359951.0, 110757999702.5, 110797561426.0, 110848409467.5, 110891771961.5, 110901490687.5, 110985731376.5, 111124602145.5, 111271855994.5, 111378790338.0, 111441646902.5, 111544782426.0, 111553426490.5, 111570537628.5, 111581905497.5, 111660244585.0, 111694895747.5, 111753559683.5, 111837783257.0, 111959477984.0, 112229736767.5, 112385269430.0, 112410734408.0, 112531207371.5, 112585723954.0, 112649551380.0, 112699168351.5, 112728637359.5, 112765512805.5, 112810048690.5, 112860141048.0, 112959813205.5, 113048499133.0, 113060922923.0, 113101367642.5, 113138230741.0, 113185140319.0, 113246471080.5, 113273354954.0, 113310523643.5, 113347289879.0, 113368355685.0, 113406208192.0, 113458668116.5, 113505301843.5, 113568476448.5, 113765950499.5, 113902527555.0, 113925178219.0, 113940684864.5, 113952830766.5, 113980803576.5, 114009394878.5, 114024058417.0, 114036507396.5, 114105049644.5, 114182540343.0, 114233926310.5, 114260377546.0, 114293434051.5, 114327259989.5, 114374018444.5, 114414230763.0, 114446837991.0, 114449336725.5, 114483567312.0, 114536420035.0, 114577374023.0, 114607072487.0, 114631352294.0, 114654815572.0, 114731323137.5, 114783604164.0, 114853246937.5, 114889229331.0, 114932836454.0, 114955444097.0, 114991706439.0, 115040610893.0, 115073413056.5, 115113395301.5, 115125703495.5, 115134509336.0, 115162995003.5, 115197643234.5, 115204290849.0, 115215977266.0, 115244043253.0, 115309003480.0, 115360164679.5, 115381075527.0, 115401104008.0, 115415667512.0, 115499870597.5, 115589465405.5, 115610978332.0, 115660582729.5, 115689655775.5, 115767625087.0, 115878663059.0, 115925439523.5, 115998892071.5, 116024024667.5, 116068884422.0, 116147488405.0, 116168307806.5, 116192563392.5, 116284524412.0, 116350101636.5, 116361580963.0, 116382496433.5, 116423312589.0, 116476486794.5, 116685242866.0, 116716378808.0, 116758053276.0, 116819732512.0, 116853460718.0, 116909690296.5, 116930667330.0, 116955837697.5, 116986666334.0, 117003507740.5, 117013771794.5, 117015209721.5, 117164208974.5, 117320081508.5, 117414952443.0, 117480297821.5, 117520870083.0, 117530126939.0, 117579050072.0, 117642908670.5, 117666402875.0, 117852916877.5, 117886408686.0, 117936165696.5, 117973540311.5, 118020385409.5, 118074429315.5, 118166645003.0, 118174402682.5, 118195291634.0, 118210592643.5, 118233699592.5, 118263503093.5, 118281948406.5, 118293230229.5, 118399324659.0, 118448069820.0, 118458177479.5, 118590556275.0, 118616961396.0, 118651760286.5, 118702741558.5, 118752408871.0, 118783952141.5, 118812781431.5, 118859691467.5, 118915757261.5, 118978928983.0, 119119302817.5, 119134079697.0, 119144337008.5, 119152747412.0, 119161819577.0, 119183370808.5, 119209812608.5, 119230493775.5, 119248029958.0, 119293349393.0, 119330009658.5, 119349953408.5, 119381621698.0, 119492332195.0, 119512894495.0, 119528022041.5, 119553157897.0, 119599079512.0, 119644261907.5, 119665189524.0, 119719717168.0, 119778982145.5, 119818081226.5, 119840279073.0, 119905691005.0, 119923763470.0, 119972911913.0, 120124492400.0, 120189566856.0, 120218070679.5, 120233912756.0, 120250129173.0, 120283867596.5, 120344157942.0, 120379261872.0, 120412420341.5, 120490220390.5, 120523439396.0, 120530434662.0, 120538276663.5, 120552538583.0, 120573806184.0, 120672737930.0, 120710023091.0, 120760851570.0, 120793596368.5, 120803506407.0, 120842884081.0, 120855833644.0, 120863010911.5, 120909747140.0, 120983915665.0, 120998755291.5, 121010576555.0, 121032034982.0, 121100686726.5, 121172645826.0, 121209388497.5, 121222173811.0, 121231378258.5, 121264034084.5, 121288180411.0, 121292330651.0, 121308498727.0, 121375780325.0, 121443084299.0, 121465653839.5, 121481391894.5, 121489587777.5, 121493328166.5, 121498722419.0, 121515720199.0, 121549738569.5, 121634622800.5, 121668966244.0, 121677611233.5, 121714476845.5, 121729247842.0, 121734506332.0, 121769923857.5, 121782621579.5, 121801758028.0, 121836950338.5, 121861638566.5, 121876424842.5, 121920005316.0, 121954220966.5, 122035628866.5, 122058803311.0, 122081929548.0, 122097283530.5, 122117755340.0, 122134103011.5, 122188897823.5, 122213094213.5, 122227047499.5, 122247818509.0, 122255944777.5, 122398326508.0, 122441631931.5, 122445159662.0, 122457839633.0, 122490334547.0, 122524791611.0, 122548551135.0, 122572382221.5, 122638884272.0, 122695835441.5, 122748065877.0, 122758625324.0, 122784238445.5, 122795819467.5, 122842628888.5, 122894927399.0, 122911037217.5, 122921691789.5, 122937888820.0, 122951756346.0, 123021735417.5, 123070877354.0, 123076434920.5, 123089471426.0, 123107479217.5, 123131070752.5, 123147602391.5, 123169991659.5, 123255026004.0, 123293011627.0, 123306546081.5, 123312027695.5, 123338569740.5, 123397544505.5, 123414080467.0, 123429129760.0, 123460640897.0, 123483201747.0, 123528000464.0, 123542824435.0, 123546169266.0, 123600731141.5, 123616599243.5, 123745166993.5, 123827362413.5, 123882711130.5, 123899319552.5, 123929129599.0, 123958041543.0, 123974539676.5, 123983327099.5, 123995740488.5, 124006696826.5, 124008180555.5, 124061055532.0, 124104080448.0, 124115175213.5, 124127474475.5, 124195469486.0, 124198298948.0, 124205625972.0, 124249282291.0, 124264415117.0, 124316862292.0, 124362332851.5, 124376334887.5, 124394088500.5, 124453644913.0, 124520319310.5, 124553823655.5, 124570094562.0, 124588356440.5, 124689830441.5, 124699336484.5, 124707723407.5, 124816718736.0, 124824614405.0, 124832465515.0, 124844307254.5, 124855845125.5, 124877235221.5, 125064569389.5, 125129697618.5, 125159982789.0, 125189151501.5, 125349743915.5, 125358061240.0, 125396316903.0, 125403000159.0, 125410530843.5, 125426696313.5, 125457279514.5, 125478321352.5, 125486127201.0, 125657097466.5, 125702608459.5, 125736880527.0, 125764636999.0, 125787699483.0, 125794160541.5, 125800800945.5, 125827258859.0, 125860937569.5, 125978517399.0, 126006669697.0, 126027561738.5, 126066049734.0, 126151307710.5, 126202415941.5, 126254693512.0, 126343973993.0, 126352821522.5, 126357041322.0, 126365116559.5, 126373317622.5, 126387765082.0, 126398895054.0, 126419719941.0, 126432384603.0, 126443264845.0, 126465898823.5, 126496192554.0, 126513338320.5, 126528272846.5, 126583556142.5, 126606705848.5, 126654206454.0, 126691722930.0, 126817929606.5, 126830433857.5, 126866219322.0, 126928954267.0, 126972266152.5, 127052251875.0, 127207160598.0, 127245694155.5, 127256683521.5, 127265565685.0, 127275288487.5, 127357380651.0, 127427678578.5, 127478580631.5, 127495450975.5, 127567897419.5, 127634148753.5, 127656646436.5, 127680026315.0, 127751494273.5, 127814856075.0, 127825112496.0, 127857404129.5, 127897762596.5, 128028082030.5, 128079108939.0, 128158212332.5, 128192481126.5, 128219845776.5, 128296484790.5, 128306657420.5, 128383450192.0, 128427455113.5, 128633064886.0, 128690112509.0, 128733313372.0, 128774074120.0, 128832437590.0, 128859873330.5, 128891125910.0, 128971124778.0, 129042469055.5, 129081474159.0, 129112876256.0, 129179139825.5, 129217249668.0, 129250666212.0, 129290759325.0, 129375709795.0, 129436558391.5, 129532784277.5, 129618173085.5, 129623634231.0, 129632128037.0, 129818210357.5, 129829576886.5, 129879719350.5, 129966692745.0, 129999438712.5, 130032485993.0, 130083067933.5, 130139913987.5, 130160182578.0, 130216856148.5, 130305092199.0, 130402424322.5, 130434798365.0, 130525459305.0, 130547824951.5, 130570271196.0, 130575899335.5, 130591870094.5, 130628367032.5, 130668006995.5, 130699384445.0, 130703217172.5, 130711754392.5, 130751569263.0, 130815297360.0, 130905515304.5, 130963960690.5, 130994251889.0, 131021395711.5, 131056519042.0, 131101828040.0, 131142093223.0, 131156495463.5, 131233799401.5, 131281185481.0, 131308842158.5, 131326419137.5, 131334041015.5, 131343944121.0, 131347509562.0, 131350186269.5, 131374540272.0, 131401740721.5, 131407799978.5, 131446721389.5, 131532102768.0, 131624000798.0, 131703453812.0, 131777084204.0, 131824039286.5, 131891464089.0, 131913478547.5, 131927864402.0, 131934971502.5, 131946562075.0, 131969101142.5, 131989718191.5, 132109726144.0, 132141846598.5, 132257250269.5, 132276905034.5, 132281839126.0, 132288897150.5, 132554254333.0, 132554786603.5, 132558597124.5, 132597586629.5, 132662940493.5, 132706234145.5, 132747669345.0, 132778104189.0, 132791330797.5, 132866359253.5, 133094591458.0, 133155894768.5, 133222779000.5, 133248240693.0, 133273931020.5, 133294928272.0, 133331865249.0, 133362886823.5, 133501213638.0, 133607169444.5, 133741033660.5, 133805976006.5, 133850536160.0, 133855481767.0, 133882974546.5, 133909747613.0, 133996265807.0, 134090899743.5, 134140124837.0, 134244705848.5, 134327552318.5, 134365024020.0, 134412098066.5, 134507265729.0, 134550390923.5, 134578511066.0, 134613581059.0, 134628541530.5, 134632347738.0, 134656353633.5, 134690762298.5, 134716981618.5, 134759843511.5, 134862901241.5, 134908957315.0, 135041011557.0, 135062196118.0, 135072579435.5, 135129008005.0, 135185099413.0, 135204207065.5, 135242751762.0, 135284768356.0, 135335712697.0, 135347859906.5, 135361681982.5, 135467840267.0, 135550228233.0, 135609717689.5, 135622442239.0, 135724125305.5, 135738730056.0, 135816339797.0, 135918935872.0, 135951281733.5, 135958598145.5, 135964795359.5, 135978730158.0, 136041273665.0, 136152769047.0, 136303857324.0, 136366470156.5, 136419749823.0, 136478898732.0, 136505474121.0, 136523345752.0, 136572490754.5, 136610559989.5, 136667035254.0, 136719440589.5, 136769976958.0, 136793288046.0, 136823512313.0, 136875901512.0, 136948710738.0, 137001695817.0, 137071934130.5, 137165258996.5, 137186377588.5, 137206969529.5, 137442929313.5, 137494688778.5, 137588268730.5, 137668326369.0, 137746856412.5, 137994208924.0, 138012934212.0, 138035607675.5, 138184415489.5, 138235456734.5, 138292330201.5, 138502828028.5, 138551319799.5, 138607509607.0, 138655013283.5, 138732168897.5, 138882967363.5, 138965827814.0, 139062121494.0, 139267403353.5, 139298489126.5, 139332637400.0, 139352230001.5, 139370214293.5, 139388753951.5, 139462934512.0, 139512886317.5, 139563707532.0, 139655460480.0, 140164371233.5, 140187164157.0, 140254512571.5, 140415260911.5, 140445001605.5, 140599030096.5, 140648717710.5, 140718879026.5, 140743923631.5, 140769062698.5, 140846185969.0, 140962355909.0, 140988046551.0, 141011360296.5, 141093783962.0, 141097072451.0, 141196258096.0, 141220185553.5, 141264433108.0, 141288708370.5, 141314397699.5, 141347901477.5, 141598291433.0, 141628596596.5, 141742077472.0, 141771814944.0, 141801393390.5, 141836703912.5, 141921868364.0, 141929997562.5, 142131289460.5, 142197510088.5, 142285153715.5, 142353587022.5, 142398399306.5, 142417306851.0, 142592780445.0, 142772295145.5, 143000172382.0, 143059626088.5, 143069907911.0, 143084989600.0, 143100631605.5, 143116548919.5, 143155448414.0, 143165324809.0, 143171490726.5, 143437663549.5, 143451860868.5, 143453451123.0, 143460710885.0, 143623637272.0, 143773526063.0, 143864137052.0, 143924496572.5, 143981555821.5, 143990709588.0, 144049285976.0, 144126638953.0, 144165874696.5, 144228862205.5, 144372426582.5, 144421775140.5, 144449396871.5, 144455351068.0, 144484694746.0, 144514519306.5, 144530761982.0, 144558150783.0, 144907591304.0, 145003106572.5, 145188536071.0, 145217690139.0, 145232691925.5, 145252707325.0, 145395135451.5, 145485155217.5, 145563456385.5, 145744875782.0, 146112249617.5, 146185016181.0, 146213584728.5, 146267241954.5, 146412305756.5, 146524336910.5, 146546239254.0, 146630270251.5, 146995352754.0, 147026048224.0, 147497034523.0, 147534289014.5, 148155134494.0, 148194044244.0, 148309573582.5, 148369337048.5, 148979967893.5, 149035609392.5, 149088827000.0, 149725923126.0, 149779942237.5, 149846866304.0, 149874309892.0, 149899238305.5, 150038328592.0, 150048082984.0, 150161574735.5, 150264481527.0, 150298135391.5, 150387294613.5, 150402577276.5, 150571125278.5, 150646633422.5, 150771010428.0, 150884919382.0, 150938595602.0, 150988298030.5, 151025023172.5, 151118089461.5, 151173573544.5, 151278735746.0, 151349529269.5, 151371689891.0, 151435568809.5, 151487150787.0, 151516978220.5, 151634021402.0, 151779863901.0, 151856484908.5, 151921414951.5, 151955297034.0, 152106514490.0, 152317752490.5, 152331049339.5, 152356170301.0, 152423789628.5, 152487063569.5, 152701782283.0, 152717947881.5, 152846389474.0, 152907606984.0, 153079993788.0, 153098459673.0, 153270658620.0, 153350440189.0, 153397225628.5, 153424726977.5, 153476092567.0, 154149529327.0, 154219055080.0, 154323922078.5, 154370537912.0, 154835281011.0, 154917940627.5, 154987571756.5, 155056538329.0, 155172759035.5, 155255095575.5, 155300503979.5, 155328466360.5, 155495190727.5, 155643707644.0, 155696477115.5, 155748741447.5, 155776183356.0, 155807090483.5, 155828343326.5, 155853157882.0, 155911660340.0, 156020441820.5, 156374174134.5, 156421829905.5, 156464682910.0, 156588169747.0, 156592853695.0, 156666337622.0, 156812225610.5, 156924555872.0, 156970642177.0, 157389399487.0, 157643500753.5, 157685546401.5, 157730427208.0, 157940274177.0, 158081155205.0, 158174993104.5, 158200565154.5, 158245205836.0, 158374825622.0, 158559111513.0, 158734808387.0, 158796513608.5, 159068051112.5, 159166733520.0, 159421864814.0, 160056821582.0, 160143073363.5, 160241620083.5, 160289905104.0, 160345462633.0, 160596175206.0, 160625716627.0, 160916453975.5, 161071409090.0, 161475283050.0, 161577782087.0, 161643934911.5, 161769120840.5, 161886947017.0, 162047645916.0, 162359716726.5, 162432009744.5, 162507932484.5, 162698832017.5, 163469136837.5, 163526918865.0, 163622151653.0, 163641933514.0, 163726685488.5, 164885100575.0, 165266708342.5, 165445457766.0, 165458125536.0, 165726358059.5, 166108131690.0, 166289047424.0, 166354820039.5, 167030904344.0, 167730767416.5, 168295679426.0, 168464041095.0, 168583631737.5, 168844119845.0, 170002109869.5, 170113939529.5, 171508740002.0, 172150405701.5, 172691418339.0, 173444692796.0, 173737441764.0, 173973165002.5, 174274676467.0])
labels = array([0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0])
def eqenergy(rows):
    return np.sum(rows, axis=1)
def classify(rows):
    energys = eqenergy(rows)

    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = labels[numers[indys]]
        outputs[defaultindys] = 2.0
        return outputs
    return thresh_search(energys)

numthresholds = 974



# Main method
model_cap = numthresholds


def Validate(file):
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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0


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
                numeachclass[int(float(cleanarr[k, -1]))] = 0
            count += 1
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    with open(preprocessedfile, 'r') as csvinput:
        dirtyreader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(dirtyreader, None) + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            print(str(','.join(str(j) for j in ([i for i in row]))) + ',' + str(get_key(int(outputs[k]), classmapping)))



#Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
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
        print("Classifier Type: Quick Clustering")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
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
            print("System Type:                        " + str(n_classes) + "-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
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


            print("Confusion Matrix:")
            mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])



    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        os.remove(preprocessedfile)


