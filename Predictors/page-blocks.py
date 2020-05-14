#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53555/page-blocks.arff -o Predictors/page-blocks_NN.py -target binaryClass -stopat 97.84 -f NN -e 20 --yes
# Total compiler execution time: 0:40:15.88. Finished on: Apr-21-2020 16:30:59.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                89.76%
Model accuracy:                     95.90% (5249/5473 correct)
Improvement over best guess:        6.14% (of possible 10.24%)
Model capacity (MEC):               49 bits
Generalization ratio:               107.12 bits/bit
Model efficiency:                   0.12%/parameter
System behavior
True Negatives:                     88.16% (4825/5473)
True Positives:                     7.75% (424/5473)
False Negatives:                    2.48% (136/5473)
False Positives:                    1.61% (88/5473)
True Pos. Rate/Sensitivity/Recall:  0.76
True Neg. Rate/Specificity:         0.98
Precision:                          0.83
F-1 Measure:                        0.79
False Negative Rate/Miss Rate:      0.24
Critical Success Index:             0.65

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
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "page-blocks.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 10
n_classes = 2

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 11.0: 4, 12.0: 5, 13.0: 6, 14.0: 7, 15.0: 8, 16.0: 9, 17.0: 10, 18.0: 11, 19.0: 12, 20.0: 13, 21.0: 14, 22.0: 15, 23.0: 16, 24.0: 17, 25.0: 18, 26.0: 19, 27.0: 20, 28.0: 21, 29.0: 22, 30.0: 23, 31.0: 24, 32.0: 25, 33.0: 26, 34.0: 27, 35.0: 28, 36.0: 29, 37.0: 30, 38.0: 31, 39.0: 32, 40.0: 33, 42.0: 34, 43.0: 35, 44.0: 36, 45.0: 37, 46.0: 38, 47.0: 39, 48.0: 40, 49.0: 41, 50.0: 42, 52.0: 43, 54.0: 44, 55.0: 45, 56.0: 46, 57.0: 47, 58.0: 48, 60.0: 49, 61.0: 50, 62.0: 51, 63.0: 52, 64.0: 53, 65.0: 54, 66.0: 55, 67.0: 56, 68.0: 57, 69.0: 58, 70.0: 59, 71.0: 60, 72.0: 61, 73.0: 62, 75.0: 63, 76.0: 64, 77.0: 65, 78.0: 66, 79.0: 67, 80.0: 68, 81.0: 69, 84.0: 70, 85.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 94.0: 77, 95.0: 78, 96.0: 79, 98.0: 80, 99.0: 81, 100.0: 82, 102.0: 83, 104.0: 84, 105.0: 85, 106.0: 86, 108.0: 87, 110.0: 88, 112.0: 89, 114.0: 90, 115.0: 91, 117.0: 92, 118.0: 93, 119.0: 94, 120.0: 95, 121.0: 96, 124.0: 97, 126.0: 98, 127.0: 99, 128.0: 100, 130.0: 101, 132.0: 102, 133.0: 103, 135.0: 104, 136.0: 105, 138.0: 106, 140.0: 107, 143.0: 108, 144.0: 109, 145.0: 110, 147.0: 111, 148.0: 112, 150.0: 113, 152.0: 114, 153.0: 115, 154.0: 116, 155.0: 117, 156.0: 118, 159.0: 119, 160.0: 120, 161.0: 121, 162.0: 122, 165.0: 123, 168.0: 124, 169.0: 125, 170.0: 126, 171.0: 127, 174.0: 128, 175.0: 129, 176.0: 130, 180.0: 131, 182.0: 132, 184.0: 133, 186.0: 134, 187.0: 135, 188.0: 136, 189.0: 137, 190.0: 138, 192.0: 139, 193.0: 140, 195.0: 141, 196.0: 142, 198.0: 143, 199.0: 144, 200.0: 145, 203.0: 146, 207.0: 147, 208.0: 148, 209.0: 149, 210.0: 150, 214.0: 151, 215.0: 152, 216.0: 153, 217.0: 154, 220.0: 155, 222.0: 156, 224.0: 157, 225.0: 158, 228.0: 159, 230.0: 160, 231.0: 161, 232.0: 162, 234.0: 163, 238.0: 164, 240.0: 165, 242.0: 166, 243.0: 167, 245.0: 168, 246.0: 169, 248.0: 170, 250.0: 171, 252.0: 172, 255.0: 173, 256.0: 174, 257.0: 175, 259.0: 176, 260.0: 177, 261.0: 178, 264.0: 179, 266.0: 180, 270.0: 181, 272.0: 182, 273.0: 183, 275.0: 184, 276.0: 185, 277.0: 186, 278.0: 187, 280.0: 188, 282.0: 189, 283.0: 190, 286.0: 191, 287.0: 192, 288.0: 193, 290.0: 194, 294.0: 195, 296.0: 196, 297.0: 197, 300.0: 198, 301.0: 199, 304.0: 200, 306.0: 201, 308.0: 202, 310.0: 203, 312.0: 204, 315.0: 205, 318.0: 206, 319.0: 207, 320.0: 208, 322.0: 209, 324.0: 210, 325.0: 211, 328.0: 212, 329.0: 213, 330.0: 214, 333.0: 215, 336.0: 216, 339.0: 217, 340.0: 218, 341.0: 219, 342.0: 220, 343.0: 221, 344.0: 222, 345.0: 223, 348.0: 224, 350.0: 225, 351.0: 226, 352.0: 227, 356.0: 228, 357.0: 229, 360.0: 230, 363.0: 231, 364.0: 232, 368.0: 233, 369.0: 234, 370.0: 235, 371.0: 236, 372.0: 237, 374.0: 238, 375.0: 239, 376.0: 240, 378.0: 241, 379.0: 242, 380.0: 243, 384.0: 244, 385.0: 245, 387.0: 246, 390.0: 247, 392.0: 248, 396.0: 249, 399.0: 250, 400.0: 251, 405.0: 252, 406.0: 253, 407.0: 254, 408.0: 255, 410.0: 256, 413.0: 257, 414.0: 258, 416.0: 259, 418.0: 260, 420.0: 261, 423.0: 262, 424.0: 263, 426.0: 264, 427.0: 265, 430.0: 266, 432.0: 267, 434.0: 268, 435.0: 269, 440.0: 270, 441.0: 271, 444.0: 272, 450.0: 273, 455.0: 274, 456.0: 275, 460.0: 276, 462.0: 277, 464.0: 278, 465.0: 279, 468.0: 280, 469.0: 281, 470.0: 282, 472.0: 283, 473.0: 284, 476.0: 285, 477.0: 286, 480.0: 287, 481.0: 288, 484.0: 289, 486.0: 290, 488.0: 291, 490.0: 292, 492.0: 293, 494.0: 294, 495.0: 295, 496.0: 296, 497.0: 297, 498.0: 298, 500.0: 299, 504.0: 300, 510.0: 301, 512.0: 302, 516.0: 303, 518.0: 304, 520.0: 305, 522.0: 306, 525.0: 307, 530.0: 308, 531.0: 309, 532.0: 310, 534.0: 311, 535.0: 312, 536.0: 313, 539.0: 314, 540.0: 315, 544.0: 316, 548.0: 317, 549.0: 318, 550.0: 319, 552.0: 320, 558.0: 321, 560.0: 322, 564.0: 323, 567.0: 324, 568.0: 325, 570.0: 326, 572.0: 327, 575.0: 328, 576.0: 329, 578.0: 330, 580.0: 331, 582.0: 332, 583.0: 333, 584.0: 334, 585.0: 335, 590.0: 336, 592.0: 337, 594.0: 338, 595.0: 339, 598.0: 340, 600.0: 341, 602.0: 342, 606.0: 343, 608.0: 344, 609.0: 345, 610.0: 346, 612.0: 347, 616.0: 348, 620.0: 349, 621.0: 350, 624.0: 351, 627.0: 352, 632.0: 353, 636.0: 354, 637.0: 355, 639.0: 356, 640.0: 357, 644.0: 358, 648.0: 359, 650.0: 360, 656.0: 361, 657.0: 362, 660.0: 363, 664.0: 364, 665.0: 365, 670.0: 366, 671.0: 367, 672.0: 368, 675.0: 369, 679.0: 370, 680.0: 371, 682.0: 372, 684.0: 373, 686.0: 374, 688.0: 375, 690.0: 376, 693.0: 377, 696.0: 378, 700.0: 379, 702.0: 380, 704.0: 381, 711.0: 382, 715.0: 383, 720.0: 384, 729.0: 385, 730.0: 386, 732.0: 387, 735.0: 388, 737.0: 389, 738.0: 390, 740.0: 391, 741.0: 392, 744.0: 393, 747.0: 394, 749.0: 395, 754.0: 396, 756.0: 397, 760.0: 398, 763.0: 399, 765.0: 400, 768.0: 401, 774.0: 402, 777.0: 403, 780.0: 404, 781.0: 405, 783.0: 406, 784.0: 407, 786.0: 408, 790.0: 409, 792.0: 410, 798.0: 411, 800.0: 412, 801.0: 413, 804.0: 414, 808.0: 415, 810.0: 416, 812.0: 417, 814.0: 418, 816.0: 419, 819.0: 420, 824.0: 421, 825.0: 422, 826.0: 423, 828.0: 424, 830.0: 425, 833.0: 426, 837.0: 427, 840.0: 428, 845.0: 429, 847.0: 430, 848.0: 431, 850.0: 432, 852.0: 433, 855.0: 434, 856.0: 435, 858.0: 436, 861.0: 437, 864.0: 438, 868.0: 439, 869.0: 440, 870.0: 441, 872.0: 442, 873.0: 443, 875.0: 444, 876.0: 445, 880.0: 446, 882.0: 447, 885.0: 448, 891.0: 449, 892.0: 450, 896.0: 451, 899.0: 452, 900.0: 453, 902.0: 454, 909.0: 455, 912.0: 456, 913.0: 457, 915.0: 458, 918.0: 459, 924.0: 460, 927.0: 461, 928.0: 462, 930.0: 463, 935.0: 464, 936.0: 465, 940.0: 466, 945.0: 467, 946.0: 468, 952.0: 469, 954.0: 470, 957.0: 471, 960.0: 472, 963.0: 473, 968.0: 474, 970.0: 475, 972.0: 476, 980.0: 477, 981.0: 478, 984.0: 479, 987.0: 480, 990.0: 481, 996.0: 482, 999.0: 483, 1000.0: 484, 1001.0: 485, 1005.0: 486, 1008.0: 487, 1010.0: 488, 1012.0: 489, 1015.0: 490, 1020.0: 491, 1026.0: 492, 1029.0: 493, 1032.0: 494, 1034.0: 495, 1040.0: 496, 1044.0: 497, 1048.0: 498, 1050.0: 499, 1053.0: 500, 1056.0: 501, 1062.0: 502, 1064.0: 503, 1067.0: 504, 1068.0: 505, 1070.0: 506, 1072.0: 507, 1074.0: 508, 1076.0: 509, 1078.0: 510, 1080.0: 511, 1085.0: 512, 1089.0: 513, 1098.0: 514, 1099.0: 515, 1100.0: 516, 1104.0: 517, 1105.0: 518, 1106.0: 519, 1107.0: 520, 1111.0: 521, 1112.0: 522, 1116.0: 523, 1122.0: 524, 1125.0: 525, 1128.0: 526, 1134.0: 527, 1141.0: 528, 1143.0: 529, 1147.0: 530, 1150.0: 531, 1152.0: 532, 1160.0: 533, 1161.0: 534, 1170.0: 535, 1176.0: 536, 1183.0: 537, 1184.0: 538, 1188.0: 539, 1190.0: 540, 1197.0: 541, 1199.0: 542, 1200.0: 543, 1204.0: 544, 1206.0: 545, 1208.0: 546, 1210.0: 547, 1212.0: 548, 1216.0: 549, 1220.0: 550, 1224.0: 551, 1225.0: 552, 1230.0: 553, 1232.0: 554, 1233.0: 555, 1236.0: 556, 1240.0: 557, 1246.0: 558, 1250.0: 559, 1260.0: 560, 1264.0: 561, 1267.0: 562, 1270.0: 563, 1272.0: 564, 1276.0: 565, 1280.0: 566, 1281.0: 567, 1298.0: 568, 1300.0: 569, 1302.0: 570, 1304.0: 571, 1309.0: 572, 1310.0: 573, 1323.0: 574, 1330.0: 575, 1332.0: 576, 1337.0: 577, 1342.0: 578, 1350.0: 579, 1351.0: 580, 1352.0: 581, 1353.0: 582, 1359.0: 583, 1368.0: 584, 1370.0: 585, 1372.0: 586, 1379.0: 587, 1380.0: 588, 1386.0: 589, 1389.0: 590, 1392.0: 591, 1395.0: 592, 1400.0: 593, 1404.0: 594, 1416.0: 595, 1422.0: 596, 1428.0: 597, 1430.0: 598, 1431.0: 599, 1435.0: 600, 1440.0: 601, 1449.0: 602, 1450.0: 603, 1452.0: 604, 1458.0: 605, 1464.0: 606, 1467.0: 607, 1470.0: 608, 1472.0: 609, 1474.0: 610, 1476.0: 611, 1477.0: 612, 1480.0: 613, 1485.0: 614, 1488.0: 615, 1491.0: 616, 1494.0: 617, 1495.0: 618, 1496.0: 619, 1498.0: 620, 1500.0: 621, 1503.0: 622, 1505.0: 623, 1507.0: 624, 1510.0: 625, 1512.0: 626, 1521.0: 627, 1530.0: 628, 1533.0: 629, 1536.0: 630, 1539.0: 631, 1540.0: 632, 1544.0: 633, 1548.0: 634, 1550.0: 635, 1551.0: 636, 1561.0: 637, 1562.0: 638, 1566.0: 639, 1568.0: 640, 1570.0: 641, 1573.0: 642, 1575.0: 643, 1578.0: 644, 1584.0: 645, 1590.0: 646, 1593.0: 647, 1596.0: 648, 1606.0: 649, 1608.0: 650, 1611.0: 651, 1617.0: 652, 1620.0: 653, 1623.0: 654, 1624.0: 655, 1629.0: 656, 1632.0: 657, 1647.0: 658, 1650.0: 659, 1656.0: 660, 1665.0: 661, 1666.0: 662, 1672.0: 663, 1674.0: 664, 1680.0: 665, 1690.0: 666, 1694.0: 667, 1700.0: 668, 1701.0: 669, 1705.0: 670, 1708.0: 671, 1715.0: 672, 1716.0: 673, 1719.0: 674, 1720.0: 675, 1728.0: 676, 1729.0: 677, 1730.0: 678, 1744.0: 679, 1746.0: 680, 1752.0: 681, 1755.0: 682, 1760.0: 683, 1770.0: 684, 1771.0: 685, 1782.0: 686, 1792.0: 687, 1794.0: 688, 1800.0: 689, 1802.0: 690, 1810.0: 691, 1818.0: 692, 1830.0: 693, 1834.0: 694, 1837.0: 695, 1840.0: 696, 1848.0: 697, 1850.0: 698, 1860.0: 699, 1864.0: 700, 1890.0: 701, 1917.0: 702, 1920.0: 703, 1936.0: 704, 1953.0: 705, 1956.0: 706, 1960.0: 707, 1962.0: 708, 1963.0: 709, 1970.0: 710, 1971.0: 711, 1976.0: 712, 1980.0: 713, 2004.0: 714, 2010.0: 715, 2016.0: 716, 2024.0: 717, 2025.0: 718, 2035.0: 719, 2040.0: 720, 2070.0: 721, 2079.0: 722, 2088.0: 723, 2090.0: 724, 2097.0: 725, 2100.0: 726, 2112.0: 727, 2116.0: 728, 2120.0: 729, 2123.0: 730, 2135.0: 731, 2142.0: 732, 2145.0: 733, 2160.0: 734, 2176.0: 735, 2178.0: 736, 2192.0: 737, 2196.0: 738, 2200.0: 739, 2205.0: 740, 2226.0: 741, 2240.0: 742, 2241.0: 743, 2250.0: 744, 2255.0: 745, 2259.0: 746, 2270.0: 747, 2296.0: 748, 2304.0: 749, 2312.0: 750, 2331.0: 751, 2343.0: 752, 2349.0: 753, 2358.0: 754, 2360.0: 755, 2376.0: 756, 2380.0: 757, 2384.0: 758, 2392.0: 759, 2401.0: 760, 2403.0: 761, 2412.0: 762, 2414.0: 763, 2416.0: 764, 2420.0: 765, 2422.0: 766, 2430.0: 767, 2431.0: 768, 2439.0: 769, 2444.0: 770, 2448.0: 771, 2480.0: 772, 2484.0: 773, 2490.0: 774, 2496.0: 775, 2502.0: 776, 2510.0: 777, 2511.0: 778, 2520.0: 779, 2529.0: 780, 2547.0: 781, 2552.0: 782, 2556.0: 783, 2560.0: 784, 2562.0: 785, 2563.0: 786, 2568.0: 787, 2580.0: 788, 2590.0: 789, 2592.0: 790, 2596.0: 791, 2600.0: 792, 2604.0: 793, 2610.0: 794, 2618.0: 795, 2620.0: 796, 2622.0: 797, 2625.0: 798, 2630.0: 799, 2640.0: 800, 2655.0: 801, 2660.0: 802, 2664.0: 803, 2666.0: 804, 2673.0: 805, 2680.0: 806, 2690.0: 807, 2709.0: 808, 2710.0: 809, 2720.0: 810, 2736.0: 811, 2744.0: 812, 2747.0: 813, 2754.0: 814, 2760.0: 815, 2761.0: 816, 2768.0: 817, 2769.0: 818, 2784.0: 819, 2790.0: 820, 2800.0: 821, 2808.0: 822, 2820.0: 823, 2824.0: 824, 2840.0: 825, 2842.0: 826, 2844.0: 827, 2850.0: 828, 2852.0: 829, 2862.0: 830, 2871.0: 831, 2880.0: 832, 2886.0: 833, 2896.0: 834, 2925.0: 835, 2934.0: 836, 2940.0: 837, 2950.0: 838, 2951.0: 839, 2964.0: 840, 2979.0: 841, 2981.0: 842, 2988.0: 843, 2992.0: 844, 3024.0: 845, 3030.0: 846, 3038.0: 847, 3084.0: 848, 3090.0: 849, 3105.0: 850, 3107.0: 851, 3123.0: 852, 3132.0: 853, 3151.0: 854, 3160.0: 855, 3168.0: 856, 3177.0: 857, 3178.0: 858, 3180.0: 859, 3190.0: 860, 3222.0: 861, 3230.0: 862, 3240.0: 863, 3250.0: 864, 3252.0: 865, 3255.0: 866, 3264.0: 867, 3267.0: 868, 3270.0: 869, 3280.0: 870, 3288.0: 871, 3300.0: 872, 3311.0: 873, 3322.0: 874, 3330.0: 875, 3333.0: 876, 3384.0: 877, 3420.0: 878, 3451.0: 879, 3471.0: 880, 3542.0: 881, 3555.0: 882, 3624.0: 883, 3632.0: 884, 3648.0: 885, 3684.0: 886, 3710.0: 887, 3732.0: 888, 3766.0: 889, 3871.0: 890, 3876.0: 891, 3927.0: 892, 3972.0: 893, 3976.0: 894, 3980.0: 895, 3984.0: 896, 4020.0: 897, 4064.0: 898, 4077.0: 899, 4080.0: 900, 4088.0: 901, 4130.0: 902, 4158.0: 903, 4165.0: 904, 4185.0: 905, 4186.0: 906, 4200.0: 907, 4214.0: 908, 4248.0: 909, 4280.0: 910, 4300.0: 911, 4381.0: 912, 4392.0: 913, 4396.0: 914, 4410.0: 915, 4420.0: 916, 4428.0: 917, 4446.0: 918, 4450.0: 919, 4466.0: 920, 4494.0: 921, 4522.0: 922, 4524.0: 923, 4537.0: 924, 4545.0: 925, 4550.0: 926, 4572.0: 927, 4581.0: 928, 4590.0: 929, 4596.0: 930, 4599.0: 931, 4602.0: 932, 4625.0: 933, 4650.0: 934, 4662.0: 935, 4675.0: 936, 4680.0: 937, 4697.0: 938, 4740.0: 939, 4746.0: 940, 4760.0: 941, 4784.0: 942, 4788.0: 943, 4796.0: 944, 4797.0: 945, 4806.0: 946, 4815.0: 947, 4824.0: 948, 4826.0: 949, 4848.0: 950, 4872.0: 951, 4875.0: 952, 4900.0: 953, 4901.0: 954, 4920.0: 955, 4950.0: 956, 4968.0: 957, 4990.0: 958, 5016.0: 959, 5026.0: 960, 5040.0: 961, 5060.0: 962, 5070.0: 963, 5075.0: 964, 5080.0: 965, 5100.0: 966, 5110.0: 967, 5130.0: 968, 5194.0: 969, 5200.0: 970, 5280.0: 971, 5291.0: 972, 5317.0: 973, 5350.0: 974, 5390.0: 975, 5456.0: 976, 5460.0: 977, 5558.0: 978, 5586.0: 979, 5621.0: 980, 5628.0: 981, 5629.0: 982, 5632.0: 983, 5696.0: 984, 5698.0: 985, 5727.0: 986, 5733.0: 987, 5740.0: 988, 5762.0: 989, 5768.0: 990, 5797.0: 991, 5814.0: 992, 5824.0: 993, 5865.0: 994, 5866.0: 995, 5872.0: 996, 5880.0: 997, 5908.0: 998, 5940.0: 999, 5964.0: 1000, 5976.0: 1001, 6000.0: 1002, 6016.0: 1003, 6062.0: 1004, 6072.0: 1005, 6112.0: 1006, 6224.0: 1007, 6256.0: 1008, 6285.0: 1009, 6384.0: 1010, 6528.0: 1011, 6560.0: 1012, 6640.0: 1013, 6672.0: 1014, 6696.0: 1015, 6720.0: 1016, 6768.0: 1017, 6784.0: 1018, 6912.0: 1019, 7209.0: 1020, 7300.0: 1021, 7416.0: 1022, 7500.0: 1023, 7644.0: 1024, 8016.0: 1025, 8160.0: 1026, 8533.0: 1027, 8640.0: 1028, 8778.0: 1029, 8835.0: 1030, 9240.0: 1031, 9324.0: 1032, 9672.0: 1033, 10092.0: 1034, 11200.0: 1035, 11232.0: 1036, 11304.0: 1037, 12240.0: 1038, 12275.0: 1039, 12367.0: 1040, 12390.0: 1041, 12561.0: 1042, 13442.0: 1043, 13542.0: 1044, 13767.0: 1045, 15580.0: 1046, 19278.0: 1047, 19296.0: 1048, 19789.0: 1049, 19832.0: 1050, 20867.0: 1051, 24360.0: 1052, 24920.0: 1053, 25619.0: 1054, 25748.0: 1055, 25935.0: 1056, 26062.0: 1057, 26145.0: 1058, 26243.0: 1059, 27058.0: 1060, 39006.0: 1061, 45760.0: 1062, 67626.0: 1063, 81954.0: 1064, 87234.0: 1065, 140752.0: 1066, 1870.0: 1067, 3458.0: 1068, 2834.0: 1069, 1410.0: 1070, 4543.0: 1071, 249.0: 1072, 3408.0: 1073, 391.0: 1074, 3087.0: 1075, 2745.0: 1076, 172.0: 1077, 98368.0: 1078, 6331.0: 1079, 5180.0: 1080, 1144.0: 1081, 1375.0: 1082, 1614.0: 1083, 5530.0: 1084, 1377.0: 1085, 1911.0: 1086, 630.0: 1087, 2691.0: 1088, 429.0: 1089, 860.0: 1090, 6480.0: 1091, 125.0: 1092, 2544.0: 1093, 707.0: 1094, 202.0: 1095, 1360.0: 1096, 2889.0: 1097, 561.0: 1098, 2168.0: 1099, 1925.0: 1100, 3861.0: 1101, 1079.0: 1102, 3768.0: 1103, 5726.0: 1104, 1683.0: 1105, 1998.0: 1106, 976.0: 1107, 12996.0: 1108, 247.0: 1109, 2013.0: 1110, 1211.0: 1111, 528.0: 1112, 2706.0: 1113, 7350.0: 1114, 263.0: 1115, 6832.0: 1116, 9999.0: 1117, 537.0: 1118, 2475.0: 1119, 131.0: 1120, 3601.0: 1121, 1664.0: 1122, 893.0: 1123, 1520.0: 1124, 836.0: 1125, 890.0: 1126, 7275.0: 1127, 354.0: 1128, 2821.0: 1129, 5504.0: 1130, 445.0: 1131, 1274.0: 1132, 3969.0: 1133, 323.0: 1134, 6525.0: 1135, 1773.0: 1136, 6233.0: 1137, 805.0: 1138, 253.0: 1139, 1086.0: 1140, 258.0: 1141, 2472.0: 1142, 5838.0: 1143, 1092.0: 1144, 1696.0: 1145, 4488.0: 1146, 206.0: 1147, 1071.0: 1148, 53.0: 1149, 314.0: 1150, 2688.0: 1151, 2997.0: 1152, 8844.0: 1153, 1215.0: 1154, 1326.0: 1155, 4752.0: 1156, 2096.0: 1157, 4043.0: 1158, 752.0: 1159, 7830.0: 1160, 2695.0: 1161, 2752.0: 1162, 167.0: 1163, 451.0: 1164, 3192.0: 1165, 2280.0: 1166, 2340.0: 1167, 4199.0: 1168, 3888.0: 1169, 2050.0: 1170, 87.0: 1171, 2400.0: 1172, 3146.0: 1173, 4500.0: 1174, 770.0: 1175, 4763.0: 1176, 4576.0: 1177, 5340.0: 1178, 1397.0: 1179, 4956.0: 1180, 1969.0: 1181, 279.0: 1182, 553.0: 1183, 3819.0: 1184, 2466.0: 1185, 1926.0: 1186, 1081.0: 1187, 1344.0: 1188, 5304.0: 1189, 8626.0: 1190, 1768.0: 1191, 4037.0: 1192, 72204.0: 1193, 5392.0: 1194, 1560.0: 1195, 513.0: 1196, 1863.0: 1197, 485.0: 1198, 1769.0: 1199, 1790.0: 1200, 618.0: 1201, 2565.0: 1202, 2875.0: 1203, 4256.0: 1204, 5744.0: 1205, 26367.0: 1206, 2150.0: 1207, 4480.0: 1208, 2799.0: 1209, 2928.0: 1210, 3036.0: 1211, 3784.0: 1212, 4564.0: 1213, 2920.0: 1214, 1832.0: 1215, 4113.0: 1216, 5208.0: 1217, 235.0: 1218, 398.0: 1219, 2882.0: 1220, 3456.0: 1221, 6288.0: 1222, 1710.0: 1223, 3141.0: 1224, 10712.0: 1225, 3184.0: 1226, 2232.0: 1227, 3069.0: 1228, 1265.0: 1229, 871.0: 1230, 265.0: 1231, 931.0: 1232, 3825.0: 1233, 3836.0: 1234, 5138.0: 1235, 1413.0: 1236, 2277.0: 1237, 18865.0: 1238, 4225.0: 1239, 26386.0: 1240, 1296.0: 1241, 1712.0: 1242, 22991.0: 1243, 1815.0: 1244, 8645.0: 1245, 1508.0: 1246, 2067.0: 1247, 1036.0: 1248, 1180.0: 1249, 2184.0: 1250, 3822.0: 1251, 123.0: 1252, 23972.0: 1253, 3025.0: 1254, 93.0: 1255, 1057.0: 1256, 4184.0: 1257, 904.0: 1258, 2201.0: 1259, 2730.0: 1260, 776.0: 1261, 950.0: 1262, 4272.0: 1263, 1242.0: 1264, 6851.0: 1265, 3321.0: 1266, 1704.0: 1267, 2613.0: 1268, 4224.0: 1269, 22680.0: 1270, 603.0: 1271, 2930.0: 1272, 2337.0: 1273, 3647.0: 1274, 51.0: 1275, 1248.0: 1276, 12375.0: 1277, 4464.0: 1278, 3737.0: 1279, 2723.0: 1280, 820.0: 1281, 1378.0: 1282, 2470.0: 1283, 5894.0: 1284, 1016.0: 1285, 1589.0: 1286, 5096.0: 1287, 1648.0: 1288, 3692.0: 1289, 3840.0: 1290, 5090.0: 1291, 4908.0: 1292, 448.0: 1293, 1599.0: 1294, 41.0: 1295, 5018.0: 1296, 3633.0: 1297, 8064.0: 1298, 142290.0: 1299, 506.0: 1300, 2104.0: 1301, 2310.0: 1302, 2322.0: 1303, 750.0: 1304, 832.0: 1305, 74.0: 1306, 689.0: 1307, 2772.0: 1308, 109.0: 1309, 2976.0: 1310, 6494.0: 1311, 3170.0: 1312, 107.0: 1313, 6192.0: 1314, 9982.0: 1315, 177.0: 1316, 5382.0: 1317, 1616.0: 1318, 1256.0: 1319, 12350.0: 1320, 2814.0: 1321, 11775.0: 1322, 6613.0: 1323, 4977.0: 1324, 7024.0: 1325, 4172.0: 1326, 4672.0: 1327, 4554.0: 1328, 1148.0: 1329, 3392.0: 1330, 6474.0: 1331, 979.0: 1332, 5950.0: 1333, 1974.0: 1334, 1899.0: 1335, 1133.0: 1336, 3212.0: 1337, 78352.0: 1338, 1809.0: 1339, 2001.0: 1340, 1526.0: 1341, 721.0: 1342, 5852.0: 1343, 6160.0: 1344, 6156.0: 1345, 2907.0: 1346, 4176.0: 1347, 6090.0: 1348, 3336.0: 1349, 2826.0: 1350, 459.0: 1351, 3707.0: 1352, 2048.0: 1353, 483.0: 1354, 3360.0: 1355, 24174.0: 1356, 8088.0: 1357, 3556.0: 1358, 803.0: 1359, 2190.0: 1360, 1775.0: 1361, 2500.0: 1362, 2714.0: 1363, 10250.0: 1364, 143993.0: 1365, 1750.0: 1366, 854.0: 1367, 736.0: 1368, 475.0: 1369, 8500.0: 1370, 7868.0: 1371, 44416.0: 1372, 3000.0: 1373, 1928.0: 1374, 5720.0: 1375, 1638.0: 1376, 992.0: 1377, 9828.0: 1378, 1030.0: 1379, 2483.0: 1380, 4944.0: 1381, 3276.0: 1382, 6766.0: 1383, 2938.0: 1384, 86.0: 1385, 3783.0: 1386, 5499.0: 1387, 2170.0: 1388, 2508.0: 1389, 1518.0: 1390, 1740.0: 1391, 1673.0: 1392, 1592.0: 1393, 6744.0: 1394}]
list_of_cols_to_normalize = [2]

transform_true = False

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values()))+1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize,mappings):
            if i>=data_arr.shape[1]:
                break
            col = data_arr[:,i]
            normcol = column_norm(col,mapping)
            data_arr[:,i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([])
        components = np.array([])
        whiten = None
        explained_variance = np.array([])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target="binaryClass"


    if (testfile):
        target=''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless==False):
                header=next(reader, None)
                try:
                    if (target!=''): 
                        hc=header.index(target)
                    else:
                        hc=len(header)-1
                        target=header[hc]
                except:
                    raise NameError("Target '"+target+"' not found! Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=header.index(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute '"+ignorecolumns[i]+"' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '"+ignorecolumns[i]+"' not found in header. Header must be same as in file passed to btc.")
                for i in range(0,len(header)):      
                    if (i==hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i]+",", end = '', file=outputfile)
                print(header[hc],file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"'+row[name]+'"'+",",end = '', file=outputfile)
                        else:
                            print (row[name]+",",end = '', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target!=""): 
                        hc=int(target)
                    else:
                        hc=-1
                except:
                    raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
                for i in range(0,len(ignorecolumns)):
                    try:
                        col=int(ignorecolumns[i])
                        if (col==hc):
                            raise ValueError("Attribute "+str(col)+" is the target. Cannot ignore. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
                for row in reader:
                    if (hc==-1):
                        hc=len(row)-1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0,len(row)):
                        if (i in il):
                            continue
                        if (i==hc):
                            continue
                        if (',' in row[i]):
                            print ('"'+row[i]+'"'+",",end = '', file=outputfile)
                        else:
                            print(row[i]+",",end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
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

# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def classify(row):
    #inits
    x=row
    o=[0]*num_output_logits


    #Nueron Equations
    h_0 = max((((-6.419584 * float(x[0]))+ (-8.420922 * float(x[1]))+ (-58.22376 * float(x[2]))+ (-1.2579473 * float(x[3]))+ (0.08781621 * float(x[4]))+ (-0.91758835 * float(x[5]))+ (-0.6989698 * float(x[6]))+ (-15.356886 * float(x[7]))+ (-63.683044 * float(x[8]))+ (-8.848855 * float(x[9]))) + -0.75969934), 0)
    h_1 = max((((0.31499213 * float(x[0]))+ (0.8175266 * float(x[1]))+ (0.047608346 * float(x[2]))+ (-0.45413327 * float(x[3]))+ (1.4042038 * float(x[4]))+ (1.6086459 * float(x[5]))+ (-1.5849361 * float(x[6]))+ (-0.7553689 * float(x[7]))+ (-0.3267186 * float(x[8]))+ (0.56736004 * float(x[9]))) + 0.17678359), 0)
    h_2 = max((((-0.3766391 * float(x[0]))+ (0.017307473 * float(x[1]))+ (0.010849183 * float(x[2]))+ (-0.331042 * float(x[3]))+ (-4.9515476 * float(x[4]))+ (6.8976345 * float(x[5]))+ (-0.032331582 * float(x[6]))+ (0.00055543595 * float(x[7]))+ (-0.0023860198 * float(x[8]))+ (0.02044259 * float(x[9]))) + 2.8351853), 0)
    h_3 = max((((-1.113746 * float(x[0]))+ (0.33367857 * float(x[1]))+ (-0.8912636 * float(x[2]))+ (-0.91794 * float(x[3]))+ (0.052677047 * float(x[4]))+ (0.25152868 * float(x[5]))+ (-1.1225314 * float(x[6]))+ (-0.055540692 * float(x[7]))+ (0.0632154 * float(x[8]))+ (-0.11383775 * float(x[9]))) + -0.46633625), 0)
    o[0] = (0.7503999 * h_0)+ (-14.631229 * h_1)+ (-1.3294061 * h_2)+ (-1.910657 * h_3) + 1.7700306

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        writer = csv.writer(sys.stdout, lineterminator=os.linesep)
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            writer.writerow(','.join(next(reader, None) + ["Prediction"]))
        
        
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            writer.writerow(row)


def Validate(arr):
    if n_classes == 2:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
        outputs=[]
        for i, row in enumerate(arr):
            outputs.append(int(classify(arr[i, :-1].tolist())))
        outputs=np.array(outputs)
        correct_count = int(np.sum(outputs.reshape(-1) == arr[:, -1].reshape(-1)))
        count = outputs.shape[0]
        num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 1)))
        num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 0)))
        num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, arr[:, -1].reshape(-1) == 1)))
        num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, arr[:, -1].reshape(-1) == 0)))
        num_class_0 = int(np.sum(arr[:, -1].reshape(-1) == 0))
        num_class_1 = int(np.sum(arr[:, -1].reshape(-1) == 1))
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0
    else:
        numeachclass = {}
        count, correct_count = 0, 0
        preds = []
        for i, row in enumerate(arr):
            pred = int(classify(arr[i].tolist()))
            preds.append(pred)
            if pred == int(float(arr[i, -1])):
                correct_count += 1
                if int(float(arr[i, -1])) in numeachclass.keys():
                    numeachclass[int(float(arr[i, -1]))] += 1
                else:
                    numeachclass[int(float(arr[i, -1]))] = 0
            count += 1
        return count, correct_count, numeachclass, preds
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()


    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempdir + os.sep + "clean.csv"
        preprocessedfile = tempdir + os.sep + "prep.csv"
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x,y: x
        classmapping = {}


    #load file
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')


    #Normalize
    cleanarr = Normalize(cleanarr)


    #Transform
    if transform_true:
        if args.validate:
            trans = transform(cleanarr[:, :-1])
            cleanarr = np.concatenate((trans, cleanarr[:, -1].reshape(-1, 1)), axis = 1)
        else:
            cleanarr = transform(cleanarr)


    #Predict
    if not args.validate:
        Predict(cleanarr, args.headerless, preprocessedfile, get_key, classmapping)


    #Validate
    else: 
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
            #Correct Labels
            true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap=49
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





            def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
                #check for numpy/scipy is imported
                try:
                    from scipy.sparse import coo_matrix #required for multiclass metrics
                    try:
                        np.array
                    except:
                        import numpy as np
                except:
                    raise ValueError("Scipy and Numpy Required for Multiclass Metrics")
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


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)
