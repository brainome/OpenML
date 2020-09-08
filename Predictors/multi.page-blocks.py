#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target binaryClass page-blocks.csv -o page-blocks_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:16:47.69. Finished on: Sep-04-2020 12:02:40.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 89.76%
Training accuracy:                   93.99% (3086/3283 correct)
Validation accuracy:                 94.79% (2076/2190 correct)
Overall Model accuracy:              94.31% (5162/5473 correct)
Overall Improvement over best guess: 4.55% (of possible 10.24%)
Model capacity (MEC):                25 bits
Generalization ratio:                206.48 bits/bit
Model efficiency:                    0.18%/parameter
System behavior
True Negatives:                      88.93% (4867/5473)
True Positives:                      5.39% (295/5473)
False Negatives:                     4.84% (265/5473)
False Positives:                     0.84% (46/5473)
True Pos. Rate/Sensitivity/Recall:   0.53
True Neg. Rate/Specificity:          0.99
Precision:                           0.87
F-1 Measure:                         0.65
False Negative Rate/Miss Rate:       0.47
Critical Success Index:              0.49
Confusion Matrix:
 [88.93% 0.84%]
 [4.84% 5.39%]
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
TRAINFILE = "page-blocks.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 10
n_classes = 2

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 11.0: 4, 12.0: 5, 13.0: 6, 14.0: 7, 15.0: 8, 16.0: 9, 17.0: 10, 18.0: 11, 19.0: 12, 20.0: 13, 21.0: 14, 22.0: 15, 23.0: 16, 24.0: 17, 25.0: 18, 26.0: 19, 27.0: 20, 28.0: 21, 29.0: 22, 30.0: 23, 31.0: 24, 32.0: 25, 33.0: 26, 34.0: 27, 35.0: 28, 36.0: 29, 37.0: 30, 38.0: 31, 39.0: 32, 40.0: 33, 41.0: 34, 42.0: 35, 43.0: 36, 44.0: 37, 45.0: 38, 47.0: 39, 48.0: 40, 49.0: 41, 50.0: 42, 52.0: 43, 53.0: 44, 54.0: 45, 55.0: 46, 56.0: 47, 57.0: 48, 60.0: 49, 61.0: 50, 63.0: 51, 64.0: 52, 65.0: 53, 66.0: 54, 67.0: 55, 68.0: 56, 69.0: 57, 70.0: 58, 72.0: 59, 74.0: 60, 75.0: 61, 76.0: 62, 77.0: 63, 78.0: 64, 79.0: 65, 80.0: 66, 81.0: 67, 84.0: 68, 85.0: 69, 86.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 93.0: 77, 94.0: 78, 95.0: 79, 96.0: 80, 98.0: 81, 99.0: 82, 100.0: 83, 102.0: 84, 104.0: 85, 105.0: 86, 106.0: 87, 107.0: 88, 108.0: 89, 109.0: 90, 110.0: 91, 112.0: 92, 114.0: 93, 115.0: 94, 117.0: 95, 119.0: 96, 120.0: 97, 121.0: 98, 123.0: 99, 124.0: 100, 125.0: 101, 126.0: 102, 127.0: 103, 128.0: 104, 130.0: 105, 131.0: 106, 132.0: 107, 133.0: 108, 135.0: 109, 136.0: 110, 138.0: 111, 140.0: 112, 143.0: 113, 144.0: 114, 145.0: 115, 147.0: 116, 148.0: 117, 150.0: 118, 152.0: 119, 153.0: 120, 154.0: 121, 155.0: 122, 156.0: 123, 159.0: 124, 160.0: 125, 161.0: 126, 162.0: 127, 165.0: 128, 168.0: 129, 169.0: 130, 170.0: 131, 171.0: 132, 174.0: 133, 175.0: 134, 176.0: 135, 177.0: 136, 180.0: 137, 182.0: 138, 184.0: 139, 186.0: 140, 187.0: 141, 188.0: 142, 189.0: 143, 190.0: 144, 192.0: 145, 193.0: 146, 195.0: 147, 196.0: 148, 198.0: 149, 199.0: 150, 200.0: 151, 202.0: 152, 203.0: 153, 207.0: 154, 208.0: 155, 209.0: 156, 210.0: 157, 215.0: 158, 216.0: 159, 217.0: 160, 220.0: 161, 222.0: 162, 224.0: 163, 225.0: 164, 230.0: 165, 231.0: 166, 232.0: 167, 234.0: 168, 235.0: 169, 238.0: 170, 240.0: 171, 242.0: 172, 243.0: 173, 245.0: 174, 246.0: 175, 248.0: 176, 250.0: 177, 252.0: 178, 253.0: 179, 255.0: 180, 256.0: 181, 258.0: 182, 259.0: 183, 260.0: 184, 261.0: 185, 264.0: 186, 265.0: 187, 266.0: 188, 270.0: 189, 272.0: 190, 273.0: 191, 275.0: 192, 276.0: 193, 277.0: 194, 280.0: 195, 282.0: 196, 286.0: 197, 287.0: 198, 288.0: 199, 290.0: 200, 294.0: 201, 296.0: 202, 297.0: 203, 300.0: 204, 301.0: 205, 304.0: 206, 306.0: 207, 308.0: 208, 310.0: 209, 312.0: 210, 315.0: 211, 319.0: 212, 320.0: 213, 322.0: 214, 324.0: 215, 325.0: 216, 328.0: 217, 329.0: 218, 330.0: 219, 333.0: 220, 336.0: 221, 339.0: 222, 340.0: 223, 342.0: 224, 343.0: 225, 344.0: 226, 348.0: 227, 350.0: 228, 351.0: 229, 352.0: 230, 354.0: 231, 357.0: 232, 360.0: 233, 363.0: 234, 364.0: 235, 368.0: 236, 369.0: 237, 370.0: 238, 371.0: 239, 372.0: 240, 374.0: 241, 376.0: 242, 378.0: 243, 379.0: 244, 380.0: 245, 384.0: 246, 385.0: 247, 387.0: 248, 390.0: 249, 391.0: 250, 392.0: 251, 396.0: 252, 399.0: 253, 400.0: 254, 405.0: 255, 406.0: 256, 408.0: 257, 410.0: 258, 413.0: 259, 414.0: 260, 416.0: 261, 418.0: 262, 420.0: 263, 423.0: 264, 424.0: 265, 426.0: 266, 427.0: 267, 429.0: 268, 430.0: 269, 432.0: 270, 435.0: 271, 440.0: 272, 441.0: 273, 444.0: 274, 445.0: 275, 448.0: 276, 450.0: 277, 455.0: 278, 456.0: 279, 460.0: 280, 462.0: 281, 465.0: 282, 468.0: 283, 469.0: 284, 470.0: 285, 472.0: 286, 473.0: 287, 475.0: 288, 477.0: 289, 480.0: 290, 481.0: 291, 483.0: 292, 484.0: 293, 485.0: 294, 486.0: 295, 488.0: 296, 490.0: 297, 492.0: 298, 494.0: 299, 495.0: 300, 496.0: 301, 497.0: 302, 498.0: 303, 500.0: 304, 504.0: 305, 510.0: 306, 512.0: 307, 513.0: 308, 516.0: 309, 518.0: 310, 520.0: 311, 522.0: 312, 525.0: 313, 528.0: 314, 530.0: 315, 531.0: 316, 532.0: 317, 534.0: 318, 535.0: 319, 536.0: 320, 537.0: 321, 540.0: 322, 544.0: 323, 548.0: 324, 549.0: 325, 550.0: 326, 552.0: 327, 558.0: 328, 560.0: 329, 561.0: 330, 564.0: 331, 567.0: 332, 568.0: 333, 570.0: 334, 572.0: 335, 576.0: 336, 578.0: 337, 580.0: 338, 582.0: 339, 583.0: 340, 584.0: 341, 585.0: 342, 590.0: 343, 592.0: 344, 594.0: 345, 598.0: 346, 600.0: 347, 602.0: 348, 603.0: 349, 606.0: 350, 608.0: 351, 609.0: 352, 610.0: 353, 612.0: 354, 616.0: 355, 620.0: 356, 621.0: 357, 624.0: 358, 627.0: 359, 630.0: 360, 632.0: 361, 636.0: 362, 637.0: 363, 639.0: 364, 640.0: 365, 644.0: 366, 650.0: 367, 656.0: 368, 657.0: 369, 660.0: 370, 664.0: 371, 665.0: 372, 670.0: 373, 671.0: 374, 672.0: 375, 675.0: 376, 679.0: 377, 682.0: 378, 684.0: 379, 686.0: 380, 688.0: 381, 693.0: 382, 696.0: 383, 700.0: 384, 702.0: 385, 704.0: 386, 707.0: 387, 711.0: 388, 715.0: 389, 720.0: 390, 729.0: 391, 730.0: 392, 735.0: 393, 737.0: 394, 738.0: 395, 740.0: 396, 744.0: 397, 747.0: 398, 750.0: 399, 752.0: 400, 754.0: 401, 756.0: 402, 760.0: 403, 765.0: 404, 768.0: 405, 770.0: 406, 774.0: 407, 777.0: 408, 780.0: 409, 781.0: 410, 783.0: 411, 784.0: 412, 786.0: 413, 790.0: 414, 792.0: 415, 798.0: 416, 800.0: 417, 804.0: 418, 808.0: 419, 810.0: 420, 812.0: 421, 814.0: 422, 816.0: 423, 819.0: 424, 820.0: 425, 824.0: 426, 826.0: 427, 828.0: 428, 830.0: 429, 832.0: 430, 833.0: 431, 837.0: 432, 840.0: 433, 845.0: 434, 847.0: 435, 850.0: 436, 852.0: 437, 855.0: 438, 856.0: 439, 860.0: 440, 861.0: 441, 864.0: 442, 868.0: 443, 869.0: 444, 870.0: 445, 871.0: 446, 872.0: 447, 873.0: 448, 876.0: 449, 880.0: 450, 882.0: 451, 885.0: 452, 891.0: 453, 892.0: 454, 896.0: 455, 900.0: 456, 902.0: 457, 904.0: 458, 909.0: 459, 912.0: 460, 913.0: 461, 915.0: 462, 918.0: 463, 924.0: 464, 927.0: 465, 928.0: 466, 930.0: 467, 931.0: 468, 935.0: 469, 936.0: 470, 940.0: 471, 945.0: 472, 946.0: 473, 950.0: 474, 952.0: 475, 954.0: 476, 960.0: 477, 963.0: 478, 972.0: 479, 976.0: 480, 980.0: 481, 981.0: 482, 984.0: 483, 987.0: 484, 990.0: 485, 996.0: 486, 999.0: 487, 1000.0: 488, 1001.0: 489, 1005.0: 490, 1008.0: 491, 1010.0: 492, 1015.0: 493, 1020.0: 494, 1029.0: 495, 1032.0: 496, 1034.0: 497, 1036.0: 498, 1040.0: 499, 1044.0: 500, 1050.0: 501, 1053.0: 502, 1056.0: 503, 1064.0: 504, 1068.0: 505, 1070.0: 506, 1071.0: 507, 1072.0: 508, 1074.0: 509, 1076.0: 510, 1078.0: 511, 1080.0: 512, 1085.0: 513, 1086.0: 514, 1089.0: 515, 1092.0: 516, 1098.0: 517, 1099.0: 518, 1100.0: 519, 1104.0: 520, 1105.0: 521, 1107.0: 522, 1111.0: 523, 1116.0: 524, 1122.0: 525, 1128.0: 526, 1134.0: 527, 1141.0: 528, 1147.0: 529, 1148.0: 530, 1150.0: 531, 1152.0: 532, 1160.0: 533, 1161.0: 534, 1170.0: 535, 1176.0: 536, 1184.0: 537, 1190.0: 538, 1197.0: 539, 1200.0: 540, 1206.0: 541, 1208.0: 542, 1210.0: 543, 1211.0: 544, 1215.0: 545, 1220.0: 546, 1224.0: 547, 1225.0: 548, 1232.0: 549, 1240.0: 550, 1242.0: 551, 1246.0: 552, 1250.0: 553, 1256.0: 554, 1260.0: 555, 1264.0: 556, 1265.0: 557, 1267.0: 558, 1270.0: 559, 1274.0: 560, 1276.0: 561, 1280.0: 562, 1281.0: 563, 1296.0: 564, 1298.0: 565, 1300.0: 566, 1309.0: 567, 1310.0: 568, 1323.0: 569, 1332.0: 570, 1337.0: 571, 1342.0: 572, 1344.0: 573, 1350.0: 574, 1353.0: 575, 1359.0: 576, 1360.0: 577, 1372.0: 578, 1375.0: 579, 1377.0: 580, 1379.0: 581, 1380.0: 582, 1386.0: 583, 1392.0: 584, 1400.0: 585, 1404.0: 586, 1413.0: 587, 1416.0: 588, 1430.0: 589, 1431.0: 590, 1435.0: 591, 1440.0: 592, 1449.0: 593, 1450.0: 594, 1464.0: 595, 1467.0: 596, 1472.0: 597, 1474.0: 598, 1480.0: 599, 1485.0: 600, 1488.0: 601, 1494.0: 602, 1495.0: 603, 1496.0: 604, 1498.0: 605, 1500.0: 606, 1505.0: 607, 1508.0: 608, 1510.0: 609, 1512.0: 610, 1520.0: 611, 1521.0: 612, 1530.0: 613, 1533.0: 614, 1536.0: 615, 1539.0: 616, 1540.0: 617, 1544.0: 618, 1548.0: 619, 1550.0: 620, 1560.0: 621, 1561.0: 622, 1562.0: 623, 1566.0: 624, 1568.0: 625, 1573.0: 626, 1575.0: 627, 1578.0: 628, 1584.0: 629, 1589.0: 630, 1590.0: 631, 1592.0: 632, 1596.0: 633, 1608.0: 634, 1611.0: 635, 1617.0: 636, 1620.0: 637, 1623.0: 638, 1624.0: 639, 1629.0: 640, 1632.0: 641, 1638.0: 642, 1647.0: 643, 1648.0: 644, 1650.0: 645, 1656.0: 646, 1665.0: 647, 1666.0: 648, 1672.0: 649, 1673.0: 650, 1674.0: 651, 1680.0: 652, 1694.0: 653, 1701.0: 654, 1704.0: 655, 1705.0: 656, 1708.0: 657, 1710.0: 658, 1715.0: 659, 1719.0: 660, 1720.0: 661, 1728.0: 662, 1729.0: 663, 1744.0: 664, 1746.0: 665, 1750.0: 666, 1752.0: 667, 1755.0: 668, 1760.0: 669, 1768.0: 670, 1769.0: 671, 1770.0: 672, 1792.0: 673, 1794.0: 674, 1810.0: 675, 1815.0: 676, 1818.0: 677, 1832.0: 678, 1837.0: 679, 1848.0: 680, 1850.0: 681, 1860.0: 682, 1863.0: 683, 1890.0: 684, 1899.0: 685, 1911.0: 686, 1920.0: 687, 1926.0: 688, 1936.0: 689, 1956.0: 690, 1962.0: 691, 1963.0: 692, 1969.0: 693, 1970.0: 694, 1971.0: 695, 1976.0: 696, 1998.0: 697, 2001.0: 698, 2010.0: 699, 2016.0: 700, 2024.0: 701, 2025.0: 702, 2035.0: 703, 2040.0: 704, 2050.0: 705, 2067.0: 706, 2070.0: 707, 2079.0: 708, 2088.0: 709, 2090.0: 710, 2096.0: 711, 2100.0: 712, 2104.0: 713, 2112.0: 714, 2123.0: 715, 2142.0: 716, 2145.0: 717, 2150.0: 718, 2160.0: 719, 2168.0: 720, 2170.0: 721, 2176.0: 722, 2178.0: 723, 2190.0: 724, 2192.0: 725, 2200.0: 726, 2201.0: 727, 2205.0: 728, 2240.0: 729, 2241.0: 730, 2250.0: 731, 2255.0: 732, 2259.0: 733, 2277.0: 734, 2304.0: 735, 2310.0: 736, 2322.0: 737, 2331.0: 738, 2337.0: 739, 2340.0: 740, 2343.0: 741, 2349.0: 742, 2358.0: 743, 2360.0: 744, 2376.0: 745, 2384.0: 746, 2392.0: 747, 2400.0: 748, 2401.0: 749, 2403.0: 750, 2412.0: 751, 2416.0: 752, 2420.0: 753, 2422.0: 754, 2430.0: 755, 2431.0: 756, 2439.0: 757, 2444.0: 758, 2448.0: 759, 2466.0: 760, 2472.0: 761, 2475.0: 762, 2480.0: 763, 2484.0: 764, 2490.0: 765, 2500.0: 766, 2502.0: 767, 2508.0: 768, 2510.0: 769, 2511.0: 770, 2520.0: 771, 2544.0: 772, 2547.0: 773, 2552.0: 774, 2556.0: 775, 2560.0: 776, 2563.0: 777, 2565.0: 778, 2580.0: 779, 2590.0: 780, 2592.0: 781, 2600.0: 782, 2604.0: 783, 2610.0: 784, 2613.0: 785, 2618.0: 786, 2620.0: 787, 2622.0: 788, 2625.0: 789, 2640.0: 790, 2655.0: 791, 2660.0: 792, 2664.0: 793, 2666.0: 794, 2680.0: 795, 2688.0: 796, 2691.0: 797, 2695.0: 798, 2706.0: 799, 2709.0: 800, 2710.0: 801, 2714.0: 802, 2720.0: 803, 2723.0: 804, 2730.0: 805, 2736.0: 806, 2747.0: 807, 2754.0: 808, 2760.0: 809, 2761.0: 810, 2768.0: 811, 2799.0: 812, 2800.0: 813, 2808.0: 814, 2834.0: 815, 2844.0: 816, 2850.0: 817, 2852.0: 818, 2871.0: 819, 2875.0: 820, 2880.0: 821, 2882.0: 822, 2886.0: 823, 2896.0: 824, 2907.0: 825, 2934.0: 826, 2940.0: 827, 2950.0: 828, 2951.0: 829, 2964.0: 830, 2976.0: 831, 2979.0: 832, 2981.0: 833, 2988.0: 834, 2992.0: 835, 2997.0: 836, 3000.0: 837, 3024.0: 838, 3030.0: 839, 3036.0: 840, 3084.0: 841, 3087.0: 842, 3090.0: 843, 3105.0: 844, 3107.0: 845, 3132.0: 846, 3151.0: 847, 3168.0: 848, 3170.0: 849, 3184.0: 850, 3192.0: 851, 3222.0: 852, 3230.0: 853, 3250.0: 854, 3252.0: 855, 3264.0: 856, 3267.0: 857, 3276.0: 858, 3280.0: 859, 3288.0: 860, 3311.0: 861, 3321.0: 862, 3322.0: 863, 3330.0: 864, 3333.0: 865, 3336.0: 866, 3392.0: 867, 3408.0: 868, 3420.0: 869, 3451.0: 870, 3456.0: 871, 3458.0: 872, 3542.0: 873, 3556.0: 874, 3624.0: 875, 3633.0: 876, 3647.0: 877, 3692.0: 878, 3707.0: 879, 3737.0: 880, 3766.0: 881, 3783.0: 882, 3784.0: 883, 3822.0: 884, 3825.0: 885, 3836.0: 886, 3840.0: 887, 3871.0: 888, 3876.0: 889, 3888.0: 890, 3927.0: 891, 3976.0: 892, 3980.0: 893, 3984.0: 894, 4020.0: 895, 4037.0: 896, 4043.0: 897, 4077.0: 898, 4080.0: 899, 4113.0: 900, 4130.0: 901, 4158.0: 902, 4165.0: 903, 4184.0: 904, 4185.0: 905, 4186.0: 906, 4200.0: 907, 4248.0: 908, 4256.0: 909, 4280.0: 910, 4300.0: 911, 4392.0: 912, 4410.0: 913, 4428.0: 914, 4446.0: 915, 4450.0: 916, 4480.0: 917, 4494.0: 918, 4500.0: 919, 4522.0: 920, 4524.0: 921, 4537.0: 922, 4543.0: 923, 4554.0: 924, 4572.0: 925, 4576.0: 926, 4590.0: 927, 4599.0: 928, 4625.0: 929, 4650.0: 930, 4662.0: 931, 4672.0: 932, 4675.0: 933, 4680.0: 934, 4697.0: 935, 4740.0: 936, 4746.0: 937, 4752.0: 938, 4760.0: 939, 4763.0: 940, 4788.0: 941, 4796.0: 942, 4797.0: 943, 4806.0: 944, 4815.0: 945, 4826.0: 946, 4872.0: 947, 4875.0: 948, 4908.0: 949, 4944.0: 950, 4956.0: 951, 4990.0: 952, 5018.0: 953, 5026.0: 954, 5070.0: 955, 5075.0: 956, 5080.0: 957, 5096.0: 958, 5110.0: 959, 5130.0: 960, 5138.0: 961, 5180.0: 962, 5194.0: 963, 5200.0: 964, 5208.0: 965, 5291.0: 966, 5304.0: 967, 5317.0: 968, 5340.0: 969, 5350.0: 970, 5382.0: 971, 5390.0: 972, 5392.0: 973, 5504.0: 974, 5530.0: 975, 5621.0: 976, 5629.0: 977, 5698.0: 978, 5727.0: 979, 5762.0: 980, 5768.0: 981, 5797.0: 982, 5838.0: 983, 5852.0: 984, 5880.0: 985, 5894.0: 986, 5908.0: 987, 5940.0: 988, 5964.0: 989, 6000.0: 990, 6016.0: 991, 6072.0: 992, 6112.0: 993, 6156.0: 994, 6160.0: 995, 6233.0: 996, 6256.0: 997, 6285.0: 998, 6331.0: 999, 6480.0: 1000, 6494.0: 1001, 6525.0: 1002, 6613.0: 1003, 6640.0: 1004, 6672.0: 1005, 6696.0: 1006, 6720.0: 1007, 6744.0: 1008, 6768.0: 1009, 6784.0: 1010, 6832.0: 1011, 6851.0: 1012, 6912.0: 1013, 7024.0: 1014, 7209.0: 1015, 7275.0: 1016, 7300.0: 1017, 7416.0: 1018, 7868.0: 1019, 8064.0: 1020, 8088.0: 1021, 8160.0: 1022, 8626.0: 1023, 8778.0: 1024, 8835.0: 1025, 9324.0: 1026, 9672.0: 1027, 9828.0: 1028, 9999.0: 1029, 10712.0: 1030, 11304.0: 1031, 11775.0: 1032, 12240.0: 1033, 12275.0: 1034, 12367.0: 1035, 12561.0: 1036, 13442.0: 1037, 13767.0: 1038, 18865.0: 1039, 19278.0: 1040, 19296.0: 1041, 19789.0: 1042, 22680.0: 1043, 23972.0: 1044, 24360.0: 1045, 24920.0: 1046, 25619.0: 1047, 26062.0: 1048, 26243.0: 1049, 26386.0: 1050, 27058.0: 1051, 39006.0: 1052, 67626.0: 1053, 72204.0: 1054, 81954.0: 1055, 87234.0: 1056, 98368.0: 1057, 318.0: 1058, 2824.0: 1059, 5720.0: 1060, 1775.0: 1061, 51.0: 1062, 4464.0: 1063, 970.0: 1064, 464.0: 1065, 5628.0: 1066, 8645.0: 1067, 2889.0: 1068, 2232.0: 1069, 1410.0: 1070, 78352.0: 1071, 1683.0: 1072, 2752.0: 1073, 2928.0: 1074, 1012.0: 1075, 3240.0: 1076, 2744.0: 1077, 1143.0: 1078, 3123.0: 1079, 172.0: 1080, 893.0: 1081, 1048.0: 1082, 1233.0: 1083, 8844.0: 1084, 1771.0: 1085, 73.0: 1086, 3384.0: 1087, 4901.0: 1088, 2596.0: 1089, 1183.0: 1090, 1112.0: 1091, 8016.0: 1092, 979.0: 1093, 15580.0: 1094, 4824.0: 1095, 4581.0: 1096, 3471.0: 1097, 1712.0: 1098, 8500.0: 1099, 6090.0: 1100, 4199.0: 1101, 7500.0: 1102, 62.0: 1103, 875.0: 1104, 1351.0: 1105, 1057.0: 1106, 2826.0: 1107, 3255.0: 1108, 3177.0: 1109, 3632.0: 1110, 836.0: 1111, 5726.0: 1112, 2013.0: 1113, 46.0: 1114, 9240.0: 1115, 434.0: 1116, 5872.0: 1117, 5280.0: 1118, 1696.0: 1119, 5744.0: 1120, 575.0: 1121, 1773.0: 1122, 4920.0: 1123, 2790.0: 1124, 890.0: 1125, 4224.0: 1126, 167.0: 1127, 1378.0: 1128, 118.0: 1129, 5456.0: 1130, 1928.0: 1131, 899.0: 1132, 12996.0: 1133, 2120.0: 1134, 1953.0: 1135, 5865.0: 1136, 553.0: 1137, 1081.0: 1138, 3732.0: 1139, 3141.0: 1140, 8640.0: 1141, 736.0: 1142, 1302.0: 1143, 375.0: 1144, 2562.0: 1145, 8533.0: 1146, 1428.0: 1147, 1395.0: 1148, 476.0: 1149, 1458.0: 1150, 5976.0: 1151, 689.0: 1152, 4488.0: 1153, 4900.0: 1154, 1062.0: 1155, 763.0: 1156, 3861.0: 1157, 1304.0: 1158, 2925.0: 1159, 10250.0: 1160, 825.0: 1161, 5824.0: 1162, 4848.0: 1163, 2270.0: 1164, 1330.0: 1165, 1188.0: 1166, 11200.0: 1167, 44416.0: 1168, 398.0: 1169, 1700.0: 1170, 4950.0: 1171, 4214.0: 1172, 1026.0: 1173, 7644.0: 1174, 4784.0: 1175, 2920.0: 1176, 6288.0: 1177, 3300.0: 1178, 1790.0: 1179, 1079.0: 1180, 3684.0: 1181, 3819.0: 1182, 2673.0: 1183, 1144.0: 1184, 314.0: 1185, 4172.0: 1186, 2745.0: 1187, 2184.0: 1188, 2004.0: 1189, 1216.0: 1190, 5460.0: 1191, 1352.0: 1192, 140752.0: 1193, 1067.0: 1194, 2470.0: 1195, 19832.0: 1196, 801.0: 1197, 1730.0: 1198, 1230.0: 1199, 3360.0: 1200, 1248.0: 1201, 1834.0: 1202, 2380.0: 1203, 2226.0: 1204, 2814.0: 1205, 1570.0: 1206, 957.0: 1207, 1800.0: 1208, 214.0: 1209, 3969.0: 1210, 9982.0: 1211, 1370.0: 1212, 142290.0: 1213, 3768.0: 1214, 4225.0: 1215, 4564.0: 1216, 206.0: 1217, 1491.0: 1218, 1389.0: 1219, 3270.0: 1220, 4420.0: 1221, 2097.0: 1222, 506.0: 1223, 5696.0: 1224, 4396.0: 1225, 1606.0: 1226, 5586.0: 1227, 992.0: 1228, 3178.0: 1229, 1840.0: 1230, 1397.0: 1231, 4977.0: 1232, 1470.0: 1233, 12375.0: 1234, 1925.0: 1235, 2630.0: 1236, 1782.0: 1237, 2414.0: 1238, 1180.0: 1239, 1809.0: 1240, 2862.0: 1241, 4064.0: 1242, 2483.0: 1243, 4088.0: 1244, 6474.0: 1245, 2116.0: 1246, 12390.0: 1247, 1690.0: 1248, 848.0: 1249, 6766.0: 1250, 1016.0: 1251, 2784.0: 1252, 1614.0: 1253, 5499.0: 1254, 2496.0: 1255, 1272.0: 1256, 407.0: 1257, 1422.0: 1258, 71.0: 1259, 2930.0: 1260, 5950.0: 1261, 2690.0: 1262, 6384.0: 1263, 459.0: 1264, 279.0: 1265, 805.0: 1266, 20867.0: 1267, 2820.0: 1268, 249.0: 1269, 1980.0: 1270, 2296.0: 1271, 776.0: 1272, 5090.0: 1273, 1368.0: 1274, 1870.0: 1275, 1125.0: 1276, 3648.0: 1277, 1616.0: 1278, 2938.0: 1279, 5040.0: 1280, 1599.0: 1281, 6528.0: 1282, 749.0: 1283, 858.0: 1284, 1106.0: 1285, 10092.0: 1286, 3038.0: 1287, 1030.0: 1288, 4272.0: 1289, 1802.0: 1290, 263.0: 1291, 539.0: 1292, 1477.0: 1293, 1830.0: 1294, 5060.0: 1295, 356.0: 1296, 341.0: 1297, 3601.0: 1298, 803.0: 1299, 5740.0: 1300, 1236.0: 1301, 6224.0: 1302, 1452.0: 1303, 45760.0: 1304, 1960.0: 1305, 11232.0: 1306, 1507.0: 1307, 2842.0: 1308, 648.0: 1309, 741.0: 1310, 4602.0: 1311, 1716.0: 1312, 968.0: 1313, 4176.0: 1314, 854.0: 1315, 247.0: 1316, 3212.0: 1317, 283.0: 1318, 1974.0: 1319, 13542.0: 1320, 3146.0: 1321, 24174.0: 1322, 1551.0: 1323, 4968.0: 1324, 228.0: 1325, 3025.0: 1326, 257.0: 1327, 1518.0: 1328, 1917.0: 1329, 1133.0: 1330, 3972.0: 1331, 1864.0: 1332, 4550.0: 1333, 1526.0: 1334, 6062.0: 1335, 6560.0: 1336, 1326.0: 1337, 3190.0: 1338, 721.0: 1339, 2568.0: 1340, 1740.0: 1341, 4466.0: 1342, 618.0: 1343, 595.0: 1344, 278.0: 1345, 3069.0: 1346, 323.0: 1347, 2196.0: 1348, 2280.0: 1349, 26367.0: 1350, 2772.0: 1351, 4381.0: 1352, 1503.0: 1353, 7350.0: 1354, 2135.0: 1355, 12350.0: 1356, 5100.0: 1357, 2048.0: 1358, 1593.0: 1359, 5733.0: 1360, 2840.0: 1361, 451.0: 1362, 1476.0: 1363, 1204.0: 1364, 3710.0: 1365, 4596.0: 1366, 4545.0: 1367, 1664.0: 1368, 3160.0: 1369, 2769.0: 1370, 5558.0: 1371, 25748.0: 1372, 345.0: 1373, 22991.0: 1374, 2312.0: 1375, 5016.0: 1376, 7830.0: 1377, 680.0: 1378, 5866.0: 1379, 2529.0: 1380, 3180.0: 1381, 1199.0: 1382, 58.0: 1383, 143993.0: 1384, 6192.0: 1385, 26145.0: 1386, 5632.0: 1387, 1212.0: 1388, 3555.0: 1389, 25935.0: 1390, 732.0: 1391, 2821.0: 1392, 690.0: 1393, 5814.0: 1394}]
list_of_cols_to_normalize = [2]

transform_true = False

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values())) + 1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize, mappings):
            if i >= data_arr.shape[1]:
                break
            col = data_arr[:, i]
            normcol = column_norm(col,mapping)
            data_arr[:, i] = normcol
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


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def single_classify(row):
    #inits
    x = row
    o = [0] * num_output_logits


    #Nueron Equations
    h_0 = max((((2.849244 * float(x[0]))+ (2.210225 * float(x[1]))+ (-1.8044579 * float(x[2]))+ (0.35044885 * float(x[3]))+ (17.107914 * float(x[4]))+ (13.334761 * float(x[5]))+ (4.6017694 * float(x[6]))+ (-0.16140108 * float(x[7]))+ (0.26590794 * float(x[8]))+ (-2.2991278 * float(x[9]))) + 12.099317), 0)
    h_1 = max((((-1.8276204 * float(x[0]))+ (-8.039448 * float(x[1]))+ (-35.513466 * float(x[2]))+ (-1.0553124 * float(x[3]))+ (0.22128238 * float(x[4]))+ (0.25969815 * float(x[5]))+ (-0.7196689 * float(x[6]))+ (-19.070955 * float(x[7]))+ (-52.741283 * float(x[8]))+ (-12.250555 * float(x[9]))) + -0.30357552), 0)
    o[0] = (0.036602017 * h_0)+ (-0.19740844 * h_1) + -3.3676455

    

    #Output Decision Rule
    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)


def classify(arr, transform=False):
    #apply transformation if necessary
    if transform:
        arr[:,:-1] = transform(arr[:,:-1])
    #init
    w_h = np.array([[2.8492441177368164, 2.2102251052856445, -1.8044579029083252, 0.3504488468170166, 17.107913970947266, 13.334760665893555, 4.60176944732666, -0.16140107810497284, 0.2659079432487488, -2.2991278171539307], [-1.8276203870773315, -8.039447784423828, -35.513465881347656, -1.0553123950958252, 0.22128237783908844, 0.25969815254211426, -0.7196689248085022, -19.070955276489258, -52.74128341674805, -12.250555038452148]])
    b_h = np.array([12.099316596984863, -0.3035755157470703])
    w_o = np.array([[0.03660201653838158, -0.19740843772888184]])
    b_o = np.array(-3.367645502090454)

    #Hidden Layer
    h = np.dot(arr, w_h.T) + b_h
    
    relu = np.maximum(h, np.zeros_like(h))


    #Output
    out = np.dot(relu, w_o.T) + b_o
    if num_output_logits == 1:
        return (out >= 0).astype('int').reshape(-1)
    else:
        return (np.argmax(out, axis=1)).reshape(-1)



def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(reader, None) + ["Prediction"]))
        
        
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(single_classify(arr[i])), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            print(','.join(row))


def Validate(cleanarr):
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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs


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
        return count, correct_count, numeachclass, outputs
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
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
        classifier_type = 'NN'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
        #Correct Labels
        true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap = 25
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

    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)

