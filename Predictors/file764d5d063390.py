#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-20-2020 04:19:49
# Invocation: btc -server brain.brainome.ai Data/file764d5d063390.csv -o Models/file764d5d063390.py -v -v -v -stopat 64.44 -port 8100 -f QC -e 100 -target class -cm {'0':0,'1':1}
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                50.00%
Model accuracy:                     76.06% (4121/5418 correct)
Improvement over best guess:        26.06% (of possible 50.0%)
Model capacity (MEC):               1258 bits
Generalization ratio:               3.27 bits/bit
Model efficiency:                   0.02%/parameter
System behavior
True Negatives:                     38.32% (2076/5418)
True Positives:                     37.74% (2045/5418)
False Negatives:                    12.26% (664/5418)
False Positives:                    11.68% (633/5418)
True Pos. Rate/Sensitivity/Recall:  0.75
True Neg. Rate/Specificity:         0.77
Precision:                          0.76
F-1 Measure:                        0.76
False Negative Rate/Miss Rate:      0.25
Critical Success Index:             0.61

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
TRAINFILE="file764d5d063390.csv"


#Number of attributes
num_attr = 1636
n_classes = 2


# Preprocessor for CSV files
def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
    clean.classlist=[]
    clean.testfile=testfile
    clean.mapping={}
    clean.mapping={'0':0,'1':1}

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
energy_thresholds=np.array([411023.0, 411276.5, 412236.0, 413386.5, 413590.0, 414173.0, 414610.5, 415115.0, 415208.5, 415897.5, 416497.5, 416692.0, 416932.0, 416974.5, 417081.5, 417872.0, 417953.5, 418063.0, 418189.5, 418282.0, 418491.0, 418622.5, 418737.0, 418962.0, 419966.0, 419984.0, 420176.0, 420367.5, 420472.5, 420507.0, 420625.0, 421377.0, 421476.0, 421698.5, 421717.0, 421798.0, 421936.0, 422262.0, 422276.0, 422313.0, 422330.5, 422626.0, 422900.5, 422941.5, 422969.0, 423527.5, 423705.0, 423857.0, 424059.0, 424097.0, 424235.5, 424432.0, 424485.0, 424524.5, 424572.0, 424603.0, 424649.0, 424708.0, 424745.5, 425840.5, 425941.0, 426020.5, 426045.0, 426354.0, 426493.5, 427094.0, 427140.5, 427186.0, 427197.5, 427281.0, 427346.0, 427774.0, 427888.0, 428166.5, 428219.0, 428386.5, 428449.5, 428871.0, 428979.0, 429129.5, 429161.0, 429285.0, 429423.5, 429605.5, 429843.5, 430195.5, 430217.5, 430272.0, 430320.0, 430354.0, 430401.5, 430648.0, 430673.5, 431002.0, 431067.5, 431099.0, 431172.0, 431398.0, 431517.5, 431682.0, 431744.0, 431750.0, 431767.0, 431829.5, 431847.0, 431857.5, 431862.5, 431870.0, 432089.0, 432216.5, 432225.0, 432258.0, 432308.0, 432385.5, 432422.0, 432479.5, 432531.0, 432571.5, 432608.0, 432628.0, 432630.5, 432640.5, 432671.5, 432748.0, 432788.5, 432874.0, 432907.5, 433033.0, 433127.5, 433212.0, 433299.0, 433351.0, 433387.0, 433565.5, 433624.0, 434062.0, 434161.5, 434194.0, 434217.5, 434256.5, 434263.0, 434276.5, 434321.0, 434424.0, 434435.5, 434541.0, 434616.0, 434705.0, 434816.0, 434961.5, 434970.5, 435010.0, 435053.0, 435159.0, 435232.5, 435398.0, 435457.0, 435525.5, 435578.5, 435812.5, 435821.0, 435909.5, 436045.0, 436062.0, 436069.0, 436075.0, 436080.5, 436091.0, 436119.0, 436164.5, 436338.5, 436354.5, 436361.5, 436372.0, 436401.5, 436429.0, 436429.5, 436436.0, 436588.5, 436772.5, 436827.5, 436934.0, 436981.0, 437010.5, 437079.0, 437195.5, 437225.0, 437299.5, 437327.5, 437356.0, 437675.0, 437753.5, 437765.0, 437795.5, 437840.0, 437883.0, 437901.0, 438181.0, 438272.0, 438342.5, 438358.5, 438510.0, 438527.5, 438586.5, 438624.5, 438665.0, 438775.5, 438964.5, 439014.5, 439259.5, 439403.0, 439412.0, 439416.0, 439420.0, 439427.0, 439437.0, 439460.5, 439490.0, 439520.5, 439605.5, 439624.0, 439755.5, 439796.0, 439840.0, 439881.0, 440052.5, 440179.0, 440231.0, 440417.5, 440460.0, 440511.0, 440561.0, 440574.5, 440620.5, 440636.0, 440857.5, 440882.0, 440887.5, 440896.5, 440935.0, 441039.0, 441078.5, 441149.5, 441159.0, 441165.0, 441173.5, 441198.5, 441237.5, 441392.0, 441491.0, 441567.0, 441622.5, 441639.0, 441725.0, 441775.5, 441808.0, 441898.5, 441935.5, 441958.0, 442047.0, 442097.5, 442132.5, 442159.0, 442327.0, 442372.5, 442437.0, 442477.0, 442490.0, 442491.0, 442501.0, 442538.0, 442561.0, 442639.0, 442759.5, 442816.0, 442867.0, 442872.0, 443005.0, 443028.0, 443052.5, 443125.0, 443216.0, 443245.0, 443280.0, 443301.0, 443316.0, 443345.0, 443384.5, 443407.0, 443465.5, 443501.0, 443593.5, 443687.5, 443721.5, 443744.0, 443816.0, 443835.0, 443879.0, 443894.0, 443943.0, 443964.5, 444096.5, 444112.0, 444114.0, 444117.0, 444153.0, 444173.5, 444196.0, 444215.5, 444229.0, 444242.5, 444262.5, 444305.5, 444339.0, 444389.5, 444392.5, 444424.5, 444439.0, 444461.0, 444497.0, 444523.0, 444628.5, 444664.0, 444687.0, 444708.5, 444746.0, 444777.5, 444802.0, 444809.5, 444837.5, 444855.0, 444862.5, 444867.0, 444941.5, 444990.0, 445079.5, 445128.5, 445174.0, 445201.5, 445214.5, 445223.5, 445237.5, 445258.0, 445300.5, 445367.0, 445403.0, 445433.5, 445483.5, 445493.0, 445519.0, 445552.5, 445587.5, 445611.5, 445636.5, 445654.0, 445661.5, 445665.5, 445698.5, 445729.5, 445782.0, 445793.0, 445798.5, 445805.5, 445821.0, 445834.5, 445844.0, 445850.0, 445865.0, 445906.0, 446004.5, 446033.0, 446055.5, 446080.5, 446094.5, 446116.5, 446156.0, 446181.0, 446223.0, 446251.5, 446259.0, 446281.5, 446320.5, 446341.5, 446370.0, 446388.0, 446402.0, 446420.0, 446440.5, 446475.0, 446517.5, 446532.0, 446586.0, 446625.0, 446669.5, 446691.0, 446707.0, 446732.5, 446764.0, 446794.5, 446867.0, 446904.5, 447034.0, 447077.0, 447113.0, 447127.5, 447287.5, 447301.5, 447315.5, 447375.0, 447416.5, 447423.5, 447435.5, 447442.5, 447445.5, 447462.5, 447524.0, 447532.0, 447580.0, 447686.0, 447762.5, 447906.0, 447930.5, 447941.5, 447954.0, 447958.0, 447983.5, 447984.0, 447991.0, 448034.0, 448058.0, 448075.5, 448203.0, 448247.5, 448295.5, 448301.5, 448314.0, 448351.5, 448378.5, 448381.5, 448389.0, 448392.0, 448396.5, 448406.5, 448416.5, 448435.5, 448487.5, 448501.5, 448524.5, 448603.5, 448707.5, 448713.0, 448770.0, 448800.0, 448813.0, 448826.5, 448837.0, 448986.0, 449099.0, 449107.0, 449231.5, 449344.5, 449357.0, 449374.0, 449407.0, 449457.5, 449494.5, 449498.5, 449503.5, 449508.5, 449546.5, 449562.0, 449632.5, 449641.5, 449719.0, 449734.0, 449764.5, 449815.5, 449833.0, 449843.0, 449860.0, 449886.5, 449920.0, 449963.5, 449980.5, 449990.5, 450043.5, 450102.0, 450112.0, 450122.0, 450139.5, 450157.0, 450218.0, 450234.0, 450275.5, 450296.5, 450300.5, 450316.5, 450368.5, 450442.0, 450468.5, 450476.5, 450553.5, 450618.5, 450651.5, 450748.0, 450750.5, 450761.5, 450790.5, 450813.0, 450831.5, 450851.5, 450923.0, 450936.0, 450937.5, 450939.5, 450992.5, 450997.0, 451058.5, 451072.0, 451090.0, 451162.0, 451172.0, 451182.0, 451189.5, 451206.5, 451261.0, 451265.5, 451298.0, 451447.5, 451473.0, 451488.5, 451520.5, 451610.0, 451668.0, 451715.0, 451837.0, 451867.0, 451880.5, 451907.5, 451945.5, 451957.0, 451973.0, 451994.0, 452039.5, 452055.5, 452194.0, 452264.5, 452434.5, 452470.5, 452495.0, 452522.0, 452570.0, 452592.5, 452611.5, 452619.5, 452634.0, 452647.5, 452661.5, 452681.5, 452684.0, 452691.0, 452733.0, 452774.0, 452787.5, 452823.5, 452861.0, 452898.0, 452909.0, 452922.0, 452942.0, 452952.5, 453005.5, 453022.0, 453031.5, 453044.5, 453064.0, 453075.5, 453107.0, 453121.5, 453214.0, 453238.5, 453283.0, 453416.5, 453439.0, 453473.0, 453538.5, 453567.0, 453580.5, 453594.5, 453730.5, 453750.5, 453765.5, 453778.0, 453798.0, 453830.5, 453846.5, 453860.0, 453873.5, 453884.5, 453912.5, 453933.5, 453951.0, 453997.0, 454052.0, 454068.0, 454083.5, 454099.5, 454119.5, 454145.0, 454169.5, 454193.5, 454220.0, 454234.0, 454258.0, 454270.5, 454280.0, 454284.5, 454298.5, 454318.0, 454419.5, 454458.0, 454560.0, 454613.0, 454629.0, 454655.0, 454725.5, 454738.5, 454761.0, 454787.5, 454836.5, 454850.0, 454876.0, 454892.0, 454985.0, 454999.5, 455025.5, 455104.5, 455176.5, 455188.0, 455228.0, 455307.5, 455351.5, 455374.5, 455381.0, 455387.5, 455403.5, 455412.5, 455425.0, 455435.5, 455447.5, 455458.5, 455466.0, 455553.5, 455572.5, 455616.0, 455630.5, 455640.5, 455691.0, 455695.5, 455703.5, 455715.0, 455723.0, 455726.5, 455733.0, 455738.5, 455781.0, 455791.0, 455861.0, 455868.0, 455876.5, 456013.0, 456039.0, 456062.0, 456079.5, 456128.0, 456165.5, 456206.0, 456336.5, 456356.0, 456412.0, 456540.0, 456586.0, 456599.5, 456703.5, 456776.0, 456784.5, 456871.5, 456891.0, 456915.5, 456943.5, 456989.0, 457026.5, 457045.0, 457063.5, 457066.5, 457117.0, 457131.0, 457187.0, 457209.0, 457288.0, 457443.5, 457493.0, 457517.5, 457589.0, 457638.5, 457664.0, 457665.5, 457667.5, 457676.5, 457705.5, 457727.5, 457739.5, 457805.0, 457835.0, 457957.5, 457973.0, 457999.5, 458024.0, 458042.5, 458051.5, 458059.0, 458085.5, 458115.5, 458125.5, 458147.5, 458160.5, 458175.5, 458196.5, 458217.0, 458231.5, 458345.0, 458360.5, 458361.5, 458363.5, 458455.5, 458487.0, 458495.0, 458501.5, 458527.5, 458544.0, 458551.0, 458561.5, 458574.0, 458593.5, 458602.0, 458619.5, 458746.0, 458752.5, 458807.0, 458820.5, 458822.5, 458831.5, 458866.5, 458897.0, 458913.5, 458939.0, 458961.5, 459125.5, 459160.5, 459175.5, 459190.0, 459281.5, 459297.5, 459304.5, 459315.5, 459370.5, 459482.0, 459555.0, 459561.0, 459660.0, 459690.0, 459766.5, 459789.0, 459821.0, 459858.5, 459927.5, 459992.5, 460030.5, 460091.0, 460123.0, 460125.5, 460145.0, 460156.0, 460219.0, 460231.0, 460306.5, 460383.5, 460487.5, 460521.0, 460599.0, 460644.0, 460657.5, 460687.5, 460711.5, 460774.0, 460816.0, 460882.0, 460895.0, 460935.0, 460974.0, 460981.0, 460987.0, 461037.0, 461073.5, 461106.5, 461121.0, 461126.5, 461215.0, 461351.5, 461401.0, 461415.0, 461484.5, 461485.5, 461517.0, 461534.0, 461544.5, 461575.0, 461596.5, 461633.5, 461649.0, 461736.5, 461742.0, 461749.0, 461823.5, 461868.5, 461948.0, 461975.0, 461983.5, 462103.5, 462120.0, 462151.5, 462168.0, 462203.0, 462224.5, 462251.0, 462291.0, 462326.0, 462362.0, 462395.0, 462407.5, 462595.0, 462610.5, 462622.5, 462643.0, 462695.0, 462726.5, 462739.0, 462775.5, 462799.5, 462826.0, 462838.0, 462876.5, 462926.0, 462962.0, 463009.5, 463061.5, 463103.0, 463109.5, 463114.0, 463115.5, 463184.5, 463270.0, 463274.5, 463301.0, 463320.0, 463327.5, 463337.0, 463356.0, 463441.0, 463474.5, 463534.5, 463570.0, 463612.0, 463631.0, 463664.0, 463682.5, 463696.5, 463755.5, 463895.0, 463922.5, 463941.0, 463968.0, 463987.5, 464000.0, 464015.0, 464032.0, 464048.0, 464145.0, 464184.5, 464209.0, 464241.0, 464249.5, 464260.0, 464296.0, 464348.0, 464418.5, 464520.5, 464529.0, 464542.0, 464556.0, 464618.5, 464638.0, 464682.5, 464741.5, 464747.5, 464773.5, 464809.0, 464826.0, 464886.0, 464894.5, 464925.0, 464940.5, 464942.0, 464983.5, 465068.5, 465097.5, 465204.5, 465257.5, 465322.0, 465330.5, 465397.5, 465461.0, 465525.5, 465532.5, 465751.5, 465781.5, 465851.5, 465857.5, 465885.0, 465942.5, 465984.5, 466030.5, 466034.5, 466077.0, 466121.0, 466162.0, 466202.5, 466208.5, 466278.0, 466327.0, 466345.0, 466418.0, 466502.0, 466603.5, 466616.5, 466669.5, 466811.0, 466868.5, 466967.5, 466996.5, 467022.0, 467037.5, 467074.5, 467086.0, 467145.5, 467159.0, 467229.5, 467254.5, 467308.0, 467397.0, 467467.5, 467476.5, 467599.0, 467643.0, 467716.0, 467762.5, 467789.0, 467850.5, 468043.5, 468077.5, 468210.0, 468227.0, 468313.5, 468336.0, 468385.5, 468424.5, 468474.5, 468484.0, 468503.5, 468529.5, 468548.0, 468565.0, 468676.0, 468706.0, 468772.5, 468777.5, 468781.5, 468797.0, 468855.0, 468863.5, 468869.0, 468900.0, 469099.5, 469123.0, 469146.5, 469180.0, 469277.5, 469299.5, 469331.0, 469343.0, 469346.0, 469380.0, 469417.5, 469446.5, 469645.5, 469745.0, 469760.0, 469781.0, 470007.5, 470053.0, 470060.5, 470086.0, 470137.5, 470170.0, 470228.5, 470309.5, 470474.5, 470482.0, 470709.5, 470744.0, 470763.0, 470770.5, 470801.0, 470874.0, 470913.5, 471071.5, 471081.0, 471088.0, 471126.5, 471134.0, 471140.0, 471165.0, 471170.0, 471209.0, 471326.0, 471335.0, 471368.5, 471403.5, 471422.0, 471477.5, 471574.5, 471589.0, 471593.5, 471633.0, 471980.0, 471984.0, 472025.0, 472044.5, 472121.5, 472134.5, 472152.0, 472198.5, 472204.5, 472209.5, 472216.5, 472226.0, 472312.5, 472331.0, 472551.0, 472565.0, 472617.5, 472641.5, 472664.0, 472724.0, 472732.0, 472751.5, 472985.5, 473007.0, 473013.0, 473059.5, 473104.5, 473109.0, 473117.5, 473124.0, 473142.5, 473176.0, 473179.0, 473180.5, 473345.0, 473375.5, 473385.5, 473448.5, 473566.0, 473592.0, 473721.0, 473814.5, 473871.0, 473907.0, 473984.5, 474015.0, 474033.0, 474054.0, 474082.0, 474102.5, 474241.0, 474273.0, 474310.5, 474367.0, 474616.0, 474638.0, 474733.5, 474804.0, 474846.0, 474886.5, 475001.5, 475045.5, 475139.0, 475162.5, 475224.5, 475296.0, 475355.5, 475371.5, 475404.0, 475413.5, 475419.0, 475511.0, 475575.0, 475617.0, 475671.5, 475740.0, 475797.0, 475830.0, 475930.5, 475968.5, 476021.0, 476049.5, 476061.0, 476093.5, 476246.5, 476329.0, 476368.0, 476378.5, 476495.5, 476603.5, 476625.5, 476657.0, 476677.0, 476805.0, 476913.5, 476978.0, 477032.0, 477063.0, 477134.5, 477263.0, 477347.0, 477417.0, 477453.5, 477551.5, 477610.0, 477665.5, 477752.5, 477791.0, 477793.0, 477812.5, 477943.0, 477973.5, 478198.5, 478220.0, 478457.5, 478538.0, 478568.5, 478581.5, 478588.5, 478670.5, 479073.0, 479099.5, 479361.0, 479569.5, 479616.0, 479739.5, 480008.5, 480083.5, 480119.0, 480145.0, 480167.5, 480199.5, 480349.5, 480373.5, 480421.0, 480494.0, 480657.0, 480705.0, 480865.0, 480873.5, 481129.5, 481189.0, 481267.5, 481275.5, 481763.5, 481825.5, 481892.5, 481927.5, 482072.0, 482182.0, 482422.5, 482492.5, 482609.5, 482646.5, 482664.0, 482716.0, 482782.5, 482816.0, 483194.0, 483218.0, 483422.0, 483448.5, 483510.0, 483676.0, 483842.0, 483894.5, 483904.0, 483914.5, 484070.5, 484163.5, 484457.5, 484899.5, 485030.0, 485046.5, 485511.5, 485724.0, 485800.5, 485813.5, 485825.5, 485900.5, 485993.0, 486157.0, 486207.0, 486245.5, 487449.0, 487502.0, 487662.5, 487817.0, 488069.0, 488105.5, 488293.0, 488463.5, 488685.5, 488718.5, 489062.0, 489139.5, 489940.5, 490013.0, 491098.5, 491397.5, 491414.0, 491417.5, 491524.0, 492046.5, 493330.5, 493376.5, 493850.0, 494170.0, 494660.5, 494796.0, 496382.5, 496833.5, 497701.0, 498496.5])
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

numthresholds=1258


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
    

