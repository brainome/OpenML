#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Feb-28-2020 21:02:22
# Invocation: btc Data/ipums-la-99-small.csv -o Models/ipums-la-99-small.py -v -v -v -stopat 93.86 -port 8090 -f QC -e 100
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
import faulthandler

from bisect import bisect_left

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="ipums-la-99-small.csv"


#Number of attributes
num_attr = 56

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
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mapped to 0 and 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        try:
            result=float(cell)
            if (rounding!=-1):
                result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
            else:
                result=int(result)

            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
        except:
            result=(binascii.crc32(value.encode('utf8')) % (1<<32))
            if (result in clean.classlist):
                result=clean.classlist.index(result)
            else:
                clean.classlist=clean.classlist+[result]
                result=clean.classlist.index(result)
            if (not (result==0 or result==1)):
                raise ValueError("Alpha version restriction: Class labels must be mappable to 0 and 1.")
        finally:
            if (result<0 or result>1):
                raise ValueError("Alpha version restriction: Integer class labels can only be 0 or 1.")
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


# Calculate energy
def eqenergy(row):
    result=0
    for elem in row:
        result = result + float(elem)
    return result

# Classifier 
def classify(row):
    energy=eqenergy(row)
    energy_thresholds=[40469158519.5, 40555179603.0, 42150050777.0, 42161022148.0, 42445998078.5, 42487520043.0, 43037725650.5, 43046319930.5, 43294034774.0, 43296194897.0, 43354642680.5, 43364134354.5, 44107802775.0, 44110886540.5, 44492227084.5, 44499749132.5, 44540136393.5, 44547065330.0, 44679477981.0, 44691453183.0, 44908061224.5, 44912098377.0, 44920646948.0, 44924503324.5, 44945518203.5, 44952212466.0, 44977843217.5, 44980518938.5, 45009465743.0, 45017684066.0, 45297105950.0, 45306233430.5, 45310007685.5, 45313862405.5, 45319025255.5, 45323920991.5, 45331353462.0, 45332517296.0, 45358973037.5, 45364112002.5, 45509777465.0, 45512362446.5, 45518696121.5, 45521618195.5, 45527153470.0, 45532289300.0, 45578945673.5, 45589543463.0, 45619593375.0, 45622760715.5, 45638675857.5, 45647056688.0, 45688634137.5, 45700663736.5, 45908260103.0, 45908383603.0, 45952770468.0, 45954420109.0, 45965048423.5, 45965620889.0, 46051133777.0, 46060485799.0, 46066089090.0, 46073846452.0, 46110760583.5, 46111919432.5, 46192160437.5, 46193932611.0, 46204289846.5, 46207561125.0, 46234818786.0, 46237772441.0, 46349951662.5, 46350015283.0, 46381409435.0, 46385640536.5, 46398773082.5, 46405587921.0, 46409241561.0, 46412424286.0, 46457358231.0, 46460541930.5, 46468252414.5, 46473595976.5, 46512037032.0, 46512731868.0, 46590663607.5, 46592136022.5, 46592596971.0, 46592751567.0, 46617158756.0, 46623073289.0, 46671725991.5, 46679768655.0, 46713671824.0, 46718633169.0, 46746348666.5, 46746359432.0, 46954906232.5, 46955529636.0, 46991234978.0, 46993114362.0, 47004394609.5, 47004825018.5, 47084824878.0, 47084861009.5, 47103619921.5, 47105891406.5, 47241065629.0, 47241438823.5, 47334921531.0, 47337001212.5, 47347091253.0, 47351015274.5, 47355138552.5, 47356787188.0, 47397847981.5, 47398813845.0, 47414322340.5, 47419967465.0, 47451596526.0, 47456803988.0, 47462264530.0, 47462564914.0, 47535646216.5, 47536972960.5, 47562847736.0, 47565993561.5, 47569813840.0, 47574313577.0, 47605430297.0, 47609621013.0, 47613717530.0, 47614448936.5, 47618682428.5, 47619216814.5, 47632445656.0, 47633587845.0, 47640928551.5, 47641768711.5, 47714987394.0, 47720960015.0, 47742626267.5, 47745298096.0, 47770339691.0, 47771450594.5, 47776411724.5, 47779890219.0, 47798538794.0, 47801665819.0, 47828379567.0, 47829962734.0, 48095927115.0, 48098082349.5, 48128161301.0, 48128180061.5, 48140450964.0, 48144108087.0, 48167454439.5, 48167545110.5, 48248132523.5, 48249482709.0, 48307937709.0, 48319211939.0, 48368384341.5, 48371386487.0, 48377061391.5, 48378224970.0, 48390040104.0, 48390088207.0, 48408198662.0, 48413250361.5, 48457891767.0, 48466183187.0, 48468790195.0, 48471044503.0, 48504691163.5, 48505826676.0, 48576923219.0, 48581333263.5, 48586391544.0, 48589789504.0, 48688190366.0, 48689667897.0, 48715637339.0, 48716835566.5, 48762090728.5, 48763661763.0, 48778065973.5, 48780684205.5, 48799006470.5, 48799613495.5, 48828676844.0, 48829546471.0, 48918798174.5, 48919034993.5, 48960944741.5, 48961421720.5, 48975892673.5, 48979765528.0, 48990998120.5, 48991592241.0, 49004847827.0, 49005896121.0, 49013675745.5, 49013980467.5, 49022158562.5, 49022638646.0, 49039985473.0, 49042010676.0, 49052364236.0, 49053685335.5, 49069141653.0, 49069846725.5, 49102024543.0, 49105428098.5, 49109146703.0, 49112363738.5, 49115049099.5, 49117512946.5, 49146454718.0, 49149854910.5, 49172850670.5, 49175553633.5, 49181920197.5, 49183150285.5, 49186059862.0, 49188878759.0, 49258448551.5, 49259977900.0, 49291834919.0, 49306140263.5, 49327273315.5, 49332735066.0, 49372654727.0, 49373757468.0, 49398964100.5, 49400735158.5, 49489141687.5, 49489876422.5, 49489884784.0, 49489891038.0, 49493310906.5, 49494318442.0, 49547009502.5, 49549168989.5, 49553067829.5, 49553460251.0, 49569428276.0, 49571597368.0, 49623448859.0, 49626176637.5, 49629443402.0, 49629973948.0, 49660882844.5, 49665749895.0, 49686284122.0, 49692050721.0, 49735330952.0, 49743948010.0, 49766524935.5, 49768851951.5, 49835429627.5, 49836858900.0, 49844651362.5, 49845637111.5, 49850316010.5, 49851005224.0, 49886889718.5, 49895214729.5, 49916828239.0, 49917626388.0, 49950663225.5, 49950890020.0, 49961277155.0, 49964614528.5, 50010185171.0, 50011331445.5, 50019049375.5, 50020093096.0, 50045759616.0, 50047635487.0, 50049478309.0, 50052034602.0, 50063414762.5, 50065761764.5, 50081897950.0, 50083990390.5, 50089367134.5, 50092475342.0, 50121620649.0, 50121866068.0, 50138239833.0, 50139346146.0, 50183264489.0, 50183572274.0, 50186983963.0, 50187015107.0, 50193941661.5, 50195572402.0, 50224715457.0, 50229983591.0, 50304159510.0, 50310246990.5, 50346444241.0, 50354072603.5, 50363340679.5, 50364421606.5, 50370447499.0, 50373729805.5, 50382687491.0, 50382948421.5, 50402412867.5, 50404010382.5, 50412055215.0, 50412365939.5, 50412489307.5, 50412752302.0, 50440109518.0, 50442187813.0, 50460760527.0, 50461984472.0, 50467978641.5, 50470086649.5, 50483339691.0, 50494059987.5, 50511360581.5, 50511403051.0, 50511435766.0, 50511661962.5, 50537245077.0, 50545488967.0, 50579992302.5, 50581288919.5, 50596187012.0, 50597078410.0, 50603010592.0, 50604667768.5, 50612835725.0, 50612884981.5, 50624105633.0, 50627012806.5, 50660069335.5, 50663679370.5, 50678025203.0, 50678869736.0, 50697212104.5, 50698677194.0, 50708698240.5, 50711164937.5, 50731995725.0, 50732500177.0, 50770271618.0, 50771110388.5, 50805683274.0, 50808366458.0, 50829096445.0, 50831735343.5, 50844560707.0, 50845895638.0, 50886663120.5, 50887447530.0, 50915437130.5, 50916462250.0, 50984323751.0, 50993471444.0, 51013793487.5, 51014443230.5, 51064465347.0, 51064767114.0, 51069249759.0, 51071121062.5, 51104230329.0, 51105393996.0, 51108448732.5, 51109287789.5, 51121150191.5, 51121168474.5, 51121523549.5, 51121708582.5, 51126643244.5, 51128093237.0, 51149814238.0, 51155611815.5, 51169329071.5, 51176489334.5, 51201668638.5, 51201894067.0, 51268456235.5, 51269475143.0, 51282160827.0, 51282550929.5, 51293206103.5, 51293734657.5, 51371387515.0, 51371399661.0, 51437618844.0, 51438079618.0, 51461612259.0, 51463387320.5, 51483601795.5, 51485964505.0, 51490883512.5, 51492077897.5, 51519944815.0, 51521799876.0, 51544984798.5, 51546637835.0, 51562599847.0, 51564262530.5, 51574320905.0, 51576181295.5, 51595603442.5, 51600151238.5, 51619075481.0, 51619167914.0, 51619297282.5, 51619562252.0, 51653358467.5, 51654036957.5, 51656446024.0, 51656959064.5, 51673054376.0, 51674515573.0, 51699831655.5, 51700788211.0, 51706845879.5, 51708015638.0, 51759327031.0, 51760032919.5, 51881036361.0, 51881527955.0, 51935926390.0, 51938107500.0, 51940574841.0, 51942623157.5, 52018146675.5, 52019407242.5, 52038802840.0, 52045905807.0, 52076146876.5, 52078246204.5, 52078619471.5, 52079033353.5, 52109224629.5, 52109995543.0, 52114854606.0, 52117301128.5, 52124775581.0, 52126639375.5, 52138898404.0, 52148994845.0, 52168267056.0, 52169302773.5, 52190158949.0, 52193626024.5, 52202603200.5, 52206282573.5, 52207038154.0, 52209255198.5, 52250613639.0, 52252046695.5, 52271445607.0, 52272217553.0, 52323827407.0, 52323868925.0, 52329207492.5, 52329872236.5, 52347941395.0, 52348406477.5, 52351096992.0, 52352030047.5, 52401961725.0, 52403522905.0, 52409376517.5, 52412944413.5, 52477677473.0, 52478384323.0, 52494048839.5, 52497164456.0, 52500205077.5, 52500769562.5, 52500949188.0, 52501318029.5, 52505583468.5, 52506828379.0, 52520289616.5, 52524268232.5, 52528417710.5, 52529708626.5, 52604198080.0, 52607442947.0, 52641762052.5, 52644219564.5, 52672356364.5, 52673712892.5, 52735395514.0, 52736480392.5, 52741243272.0, 52742962704.0, 52765289994.5, 52766801627.5, 52788628041.0, 52793315361.0, 52816611993.5, 52821196642.5, 52867650254.0, 52869810366.0, 52960834598.5, 52961754990.5, 52972097173.5, 52973719419.5, 52998976031.5, 53001190667.5, 53021192494.5, 53022301783.0, 53047239582.0, 53050400367.0, 53059437390.5, 53063326044.5, 53099285083.5, 53100635880.5, 53103734035.5, 53106215064.5, 53107584154.0, 53108553584.0, 53114666906.0, 53120955073.5, 53131887673.0, 53132171285.5, 53138575547.0, 53139508161.5, 53157476414.5, 53157484534.0, 53166216518.5, 53171704776.5, 53325680098.0, 53332062332.0, 53333789490.0, 53339381099.5, 53350314549.5, 53354607237.0, 53359083595.5, 53360283896.5, 53368157886.5, 53374874777.5, 53397816757.5, 53403053353.5, 53454880782.5, 53455945617.0, 53486980695.5, 53486996157.0, 53513497483.5, 53515621215.5, 53576629383.0, 53589010019.5, 53596028015.0, 53596032756.5, 53597998191.0, 53603250300.5, 53606928589.5, 53607268524.0, 53611000026.0, 53615676494.5, 53646358099.5, 53647771483.0, 53673797451.0, 53679156719.5, 53706214244.0, 53707302727.0, 53758908775.0, 53764404675.0, 53807448665.0, 53819099648.0, 53868272412.0, 53870415487.0, 53944794637.5, 53944871403.0, 53951607933.0, 53960614699.5, 53983163534.0, 53986620694.5, 53992767984.0, 53994925669.5, 53998746544.0, 54003140546.0, 54016116326.0, 54018636389.0, 54042062323.5, 54045039270.5, 54051172048.5, 54051933962.0, 54062306475.5, 54062877685.0, 54108176884.5, 54109442476.0, 54121300752.0, 54122948702.5, 54206924643.0, 54207781153.5, 54269489594.0, 54280636422.5, 54295170740.0, 54301606330.5, 54318463138.0, 54320605085.0, 54352805959.0, 54354637676.0, 54378858528.5, 54382475955.5, 54416674064.0, 54417621970.5, 54492877434.0, 54492903791.5, 54543171639.5, 54549691817.0, 54562545328.5, 54563817811.5, 54728378429.5, 54730036080.0, 54736793820.5, 54739268250.5, 54810267153.0, 54813051660.5, 54917484418.0, 54921880472.5, 54944385432.0, 54945149681.5, 54987974482.5, 55005714959.0, 55015628113.0, 55022883207.0, 55050569300.0, 55058280181.5, 55102835803.0, 55109185340.0, 55321491172.0, 55322052871.0, 55322088123.5, 55322244761.0, 55456979737.0, 55463485576.5, 55502505612.0, 55507590899.5, 55527248013.5, 55538054179.5, 55613021517.0, 55615777785.5, 55650287055.0, 55658768027.0, 55769484324.0, 55775613402.0, 55778173695.5, 55780231114.0, 55872240481.0, 55876698801.5, 55879716455.5, 55883328338.0, 55973985232.0, 55976607078.5, 55994661650.0, 55995062389.0, 56248078327.0, 56262611602.5, 56308637222.5, 56318481099.5, 56431340818.0, 56442159068.5, 56622116410.5, 56622715613.0, 56662701861.0, 56665661329.0, 56996306081.0, 57000263089.5, 57029855003.0, 57056353612.0, 57371919215.5, 57376045838.5, 57925233369.0, 57958490478.0, 57998664768.0, 58017706963.0, 58164552271.0, 58177122179.0, 58203097601.5, 58223615024.5, 58291150057.0, 58330053809.0, 59298035639.5, 59303373407.0]
    start_label=1

    def thresh_search(input_energy):
        i = bisect_left(energy_thresholds, input_energy)-1
        if i in range(len(energy_thresholds)):
            return ((i+start_label)%2)
        else:
            return 0

    return thresh_search(energy)

numthresholds=690


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()

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

