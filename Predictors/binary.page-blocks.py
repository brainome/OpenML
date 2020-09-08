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
# Total compiler execution time: 0:17:10.47. Finished on: Sep-03-2020 17:57:35.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           60:40%
Best-guess accuracy:                 89.76%
Training accuracy:                   94.24% (3094/3283 correct)
Validation accuracy:                 94.88% (2078/2190 correct)
Overall Model accuracy:              94.50% (5172/5473 correct)
Overall Improvement over best guess: 4.74% (of possible 10.24%)
Model capacity (MEC):                25 bits
Generalization ratio:                206.88 bits/bit
Model efficiency:                    0.18%/parameter
System behavior
True Negatives:                      88.64% (4851/5473)
True Positives:                      5.87% (321/5473)
False Negatives:                     4.37% (239/5473)
False Positives:                     1.13% (62/5473)
True Pos. Rate/Sensitivity/Recall:   0.57
True Neg. Rate/Specificity:          0.99
Precision:                           0.84
F-1 Measure:                         0.68
False Negative Rate/Miss Rate:       0.43
Critical Success Index:              0.52
Confusion Matrix:
 [88.64% 1.13%]
 [4.37% 5.87%]
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

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 11.0: 4, 12.0: 5, 13.0: 6, 14.0: 7, 15.0: 8, 16.0: 9, 17.0: 10, 18.0: 11, 19.0: 12, 20.0: 13, 22.0: 14, 23.0: 15, 24.0: 16, 25.0: 17, 26.0: 18, 27.0: 19, 28.0: 20, 29.0: 21, 30.0: 22, 31.0: 23, 32.0: 24, 33.0: 25, 34.0: 26, 35.0: 27, 36.0: 28, 37.0: 29, 38.0: 30, 39.0: 31, 40.0: 32, 41.0: 33, 42.0: 34, 43.0: 35, 44.0: 36, 45.0: 37, 47.0: 38, 48.0: 39, 49.0: 40, 50.0: 41, 51.0: 42, 52.0: 43, 53.0: 44, 54.0: 45, 55.0: 46, 56.0: 47, 60.0: 48, 61.0: 49, 62.0: 50, 63.0: 51, 64.0: 52, 65.0: 53, 66.0: 54, 67.0: 55, 68.0: 56, 69.0: 57, 70.0: 58, 71.0: 59, 72.0: 60, 73.0: 61, 74.0: 62, 75.0: 63, 76.0: 64, 77.0: 65, 78.0: 66, 80.0: 67, 81.0: 68, 84.0: 69, 85.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 95.0: 77, 96.0: 78, 98.0: 79, 99.0: 80, 100.0: 81, 102.0: 82, 104.0: 83, 105.0: 84, 106.0: 85, 107.0: 86, 108.0: 87, 110.0: 88, 112.0: 89, 114.0: 90, 115.0: 91, 117.0: 92, 118.0: 93, 119.0: 94, 120.0: 95, 121.0: 96, 123.0: 97, 124.0: 98, 126.0: 99, 128.0: 100, 130.0: 101, 132.0: 102, 133.0: 103, 135.0: 104, 136.0: 105, 138.0: 106, 140.0: 107, 143.0: 108, 144.0: 109, 145.0: 110, 147.0: 111, 148.0: 112, 150.0: 113, 152.0: 114, 153.0: 115, 154.0: 116, 155.0: 117, 156.0: 118, 159.0: 119, 160.0: 120, 161.0: 121, 162.0: 122, 165.0: 123, 167.0: 124, 168.0: 125, 169.0: 126, 170.0: 127, 171.0: 128, 174.0: 129, 175.0: 130, 176.0: 131, 177.0: 132, 180.0: 133, 182.0: 134, 184.0: 135, 186.0: 136, 187.0: 137, 188.0: 138, 189.0: 139, 190.0: 140, 192.0: 141, 195.0: 142, 196.0: 143, 198.0: 144, 199.0: 145, 200.0: 146, 203.0: 147, 207.0: 148, 208.0: 149, 209.0: 150, 210.0: 151, 214.0: 152, 215.0: 153, 216.0: 154, 217.0: 155, 220.0: 156, 224.0: 157, 225.0: 158, 228.0: 159, 230.0: 160, 231.0: 161, 232.0: 162, 234.0: 163, 238.0: 164, 240.0: 165, 242.0: 166, 243.0: 167, 245.0: 168, 246.0: 169, 248.0: 170, 250.0: 171, 252.0: 172, 253.0: 173, 256.0: 174, 259.0: 175, 260.0: 176, 261.0: 177, 263.0: 178, 264.0: 179, 266.0: 180, 270.0: 181, 272.0: 182, 273.0: 183, 275.0: 184, 276.0: 185, 277.0: 186, 278.0: 187, 279.0: 188, 280.0: 189, 287.0: 190, 288.0: 191, 290.0: 192, 294.0: 193, 296.0: 194, 297.0: 195, 300.0: 196, 301.0: 197, 304.0: 198, 306.0: 199, 308.0: 200, 310.0: 201, 312.0: 202, 315.0: 203, 319.0: 204, 320.0: 205, 322.0: 206, 324.0: 207, 328.0: 208, 329.0: 209, 330.0: 210, 333.0: 211, 336.0: 212, 339.0: 213, 340.0: 214, 342.0: 215, 343.0: 216, 344.0: 217, 345.0: 218, 348.0: 219, 350.0: 220, 351.0: 221, 352.0: 222, 354.0: 223, 356.0: 224, 357.0: 225, 360.0: 226, 363.0: 227, 368.0: 228, 369.0: 229, 370.0: 230, 371.0: 231, 374.0: 232, 375.0: 233, 376.0: 234, 378.0: 235, 380.0: 236, 384.0: 237, 385.0: 238, 387.0: 239, 390.0: 240, 391.0: 241, 392.0: 242, 396.0: 243, 399.0: 244, 400.0: 245, 405.0: 246, 406.0: 247, 407.0: 248, 408.0: 249, 410.0: 250, 413.0: 251, 414.0: 252, 416.0: 253, 420.0: 254, 423.0: 255, 424.0: 256, 426.0: 257, 427.0: 258, 429.0: 259, 432.0: 260, 434.0: 261, 440.0: 262, 441.0: 263, 448.0: 264, 450.0: 265, 455.0: 266, 456.0: 267, 459.0: 268, 460.0: 269, 462.0: 270, 464.0: 271, 465.0: 272, 468.0: 273, 469.0: 274, 470.0: 275, 472.0: 276, 473.0: 277, 475.0: 278, 476.0: 279, 477.0: 280, 480.0: 281, 481.0: 282, 484.0: 283, 485.0: 284, 486.0: 285, 488.0: 286, 490.0: 287, 492.0: 288, 494.0: 289, 495.0: 290, 496.0: 291, 497.0: 292, 498.0: 293, 500.0: 294, 504.0: 295, 506.0: 296, 510.0: 297, 512.0: 298, 516.0: 299, 518.0: 300, 520.0: 301, 522.0: 302, 525.0: 303, 528.0: 304, 530.0: 305, 531.0: 306, 532.0: 307, 534.0: 308, 537.0: 309, 539.0: 310, 540.0: 311, 544.0: 312, 548.0: 313, 549.0: 314, 550.0: 315, 552.0: 316, 560.0: 317, 561.0: 318, 564.0: 319, 567.0: 320, 568.0: 321, 570.0: 322, 572.0: 323, 575.0: 324, 576.0: 325, 580.0: 326, 582.0: 327, 584.0: 328, 585.0: 329, 590.0: 330, 592.0: 331, 594.0: 332, 595.0: 333, 598.0: 334, 600.0: 335, 602.0: 336, 603.0: 337, 606.0: 338, 608.0: 339, 609.0: 340, 610.0: 341, 612.0: 342, 616.0: 343, 620.0: 344, 621.0: 345, 624.0: 346, 630.0: 347, 632.0: 348, 637.0: 349, 639.0: 350, 640.0: 351, 644.0: 352, 648.0: 353, 650.0: 354, 656.0: 355, 657.0: 356, 660.0: 357, 664.0: 358, 665.0: 359, 670.0: 360, 671.0: 361, 672.0: 362, 675.0: 363, 680.0: 364, 684.0: 365, 690.0: 366, 693.0: 367, 696.0: 368, 700.0: 369, 702.0: 370, 704.0: 371, 707.0: 372, 711.0: 373, 715.0: 374, 720.0: 375, 721.0: 376, 729.0: 377, 730.0: 378, 735.0: 379, 737.0: 380, 738.0: 381, 744.0: 382, 747.0: 383, 752.0: 384, 756.0: 385, 760.0: 386, 768.0: 387, 770.0: 388, 774.0: 389, 776.0: 390, 777.0: 391, 780.0: 392, 781.0: 393, 783.0: 394, 790.0: 395, 792.0: 396, 798.0: 397, 800.0: 398, 801.0: 399, 804.0: 400, 805.0: 401, 808.0: 402, 810.0: 403, 812.0: 404, 816.0: 405, 819.0: 406, 820.0: 407, 824.0: 408, 826.0: 409, 828.0: 410, 830.0: 411, 832.0: 412, 837.0: 413, 840.0: 414, 845.0: 415, 847.0: 416, 850.0: 417, 852.0: 418, 854.0: 419, 855.0: 420, 856.0: 421, 858.0: 422, 860.0: 423, 861.0: 424, 864.0: 425, 868.0: 426, 869.0: 427, 870.0: 428, 871.0: 429, 872.0: 430, 873.0: 431, 875.0: 432, 876.0: 433, 880.0: 434, 882.0: 435, 885.0: 436, 890.0: 437, 891.0: 438, 896.0: 439, 899.0: 440, 900.0: 441, 902.0: 442, 909.0: 443, 912.0: 444, 913.0: 445, 918.0: 446, 924.0: 447, 927.0: 448, 928.0: 449, 930.0: 450, 931.0: 451, 935.0: 452, 936.0: 453, 945.0: 454, 946.0: 455, 950.0: 456, 952.0: 457, 954.0: 458, 957.0: 459, 960.0: 460, 963.0: 461, 968.0: 462, 972.0: 463, 976.0: 464, 980.0: 465, 981.0: 466, 984.0: 467, 987.0: 468, 990.0: 469, 996.0: 470, 999.0: 471, 1000.0: 472, 1001.0: 473, 1005.0: 474, 1008.0: 475, 1010.0: 476, 1012.0: 477, 1016.0: 478, 1020.0: 479, 1026.0: 480, 1034.0: 481, 1036.0: 482, 1040.0: 483, 1044.0: 484, 1048.0: 485, 1056.0: 486, 1057.0: 487, 1062.0: 488, 1068.0: 489, 1070.0: 490, 1072.0: 491, 1074.0: 492, 1076.0: 493, 1078.0: 494, 1081.0: 495, 1085.0: 496, 1089.0: 497, 1092.0: 498, 1100.0: 499, 1104.0: 500, 1106.0: 501, 1107.0: 502, 1111.0: 503, 1112.0: 504, 1116.0: 505, 1122.0: 506, 1125.0: 507, 1128.0: 508, 1134.0: 509, 1144.0: 510, 1147.0: 511, 1150.0: 512, 1160.0: 513, 1161.0: 514, 1170.0: 515, 1176.0: 516, 1180.0: 517, 1184.0: 518, 1188.0: 519, 1200.0: 520, 1206.0: 521, 1210.0: 522, 1212.0: 523, 1215.0: 524, 1216.0: 525, 1220.0: 526, 1224.0: 527, 1225.0: 528, 1230.0: 529, 1232.0: 530, 1242.0: 531, 1246.0: 532, 1250.0: 533, 1256.0: 534, 1260.0: 535, 1264.0: 536, 1265.0: 537, 1267.0: 538, 1274.0: 539, 1276.0: 540, 1280.0: 541, 1296.0: 542, 1298.0: 543, 1300.0: 544, 1304.0: 545, 1309.0: 546, 1310.0: 547, 1326.0: 548, 1332.0: 549, 1337.0: 550, 1344.0: 551, 1353.0: 552, 1359.0: 553, 1372.0: 554, 1378.0: 555, 1380.0: 556, 1386.0: 557, 1392.0: 558, 1395.0: 559, 1397.0: 560, 1400.0: 561, 1404.0: 562, 1410.0: 563, 1416.0: 564, 1422.0: 565, 1435.0: 566, 1440.0: 567, 1449.0: 568, 1464.0: 569, 1467.0: 570, 1472.0: 571, 1474.0: 572, 1480.0: 573, 1485.0: 574, 1488.0: 575, 1491.0: 576, 1494.0: 577, 1495.0: 578, 1496.0: 579, 1498.0: 580, 1508.0: 581, 1510.0: 582, 1512.0: 583, 1520.0: 584, 1521.0: 585, 1526.0: 586, 1533.0: 587, 1536.0: 588, 1539.0: 589, 1540.0: 590, 1548.0: 591, 1550.0: 592, 1561.0: 593, 1562.0: 594, 1566.0: 595, 1568.0: 596, 1570.0: 597, 1573.0: 598, 1575.0: 599, 1584.0: 600, 1590.0: 601, 1596.0: 602, 1606.0: 603, 1611.0: 604, 1617.0: 605, 1620.0: 606, 1624.0: 607, 1629.0: 608, 1632.0: 609, 1647.0: 610, 1650.0: 611, 1656.0: 612, 1664.0: 613, 1665.0: 614, 1666.0: 615, 1673.0: 616, 1674.0: 617, 1680.0: 618, 1683.0: 619, 1694.0: 620, 1696.0: 621, 1700.0: 622, 1701.0: 623, 1705.0: 624, 1708.0: 625, 1710.0: 626, 1712.0: 627, 1715.0: 628, 1728.0: 629, 1729.0: 630, 1744.0: 631, 1760.0: 632, 1768.0: 633, 1769.0: 634, 1770.0: 635, 1771.0: 636, 1775.0: 637, 1782.0: 638, 1800.0: 639, 1802.0: 640, 1810.0: 641, 1815.0: 642, 1818.0: 643, 1830.0: 644, 1832.0: 645, 1848.0: 646, 1860.0: 647, 1864.0: 648, 1870.0: 649, 1899.0: 650, 1917.0: 651, 1928.0: 652, 1936.0: 653, 1953.0: 654, 1960.0: 655, 1962.0: 656, 1963.0: 657, 1969.0: 658, 1970.0: 659, 1971.0: 660, 1976.0: 661, 1980.0: 662, 1998.0: 663, 2004.0: 664, 2013.0: 665, 2016.0: 666, 2024.0: 667, 2035.0: 668, 2040.0: 669, 2070.0: 670, 2079.0: 671, 2088.0: 672, 2090.0: 673, 2097.0: 674, 2100.0: 675, 2104.0: 676, 2120.0: 677, 2135.0: 678, 2142.0: 679, 2168.0: 680, 2170.0: 681, 2176.0: 682, 2196.0: 683, 2201.0: 684, 2205.0: 685, 2250.0: 686, 2255.0: 687, 2270.0: 688, 2277.0: 689, 2280.0: 690, 2296.0: 691, 2312.0: 692, 2331.0: 693, 2343.0: 694, 2349.0: 695, 2360.0: 696, 2376.0: 697, 2380.0: 698, 2384.0: 699, 2403.0: 700, 2412.0: 701, 2416.0: 702, 2420.0: 703, 2422.0: 704, 2439.0: 705, 2448.0: 706, 2480.0: 707, 2483.0: 708, 2484.0: 709, 2490.0: 710, 2502.0: 711, 2510.0: 712, 2511.0: 713, 2520.0: 714, 2529.0: 715, 2544.0: 716, 2547.0: 717, 2560.0: 718, 2565.0: 719, 2568.0: 720, 2580.0: 721, 2604.0: 722, 2610.0: 723, 2618.0: 724, 2620.0: 725, 2630.0: 726, 2640.0: 727, 2655.0: 728, 2664.0: 729, 2666.0: 730, 2680.0: 731, 2690.0: 732, 2691.0: 733, 2695.0: 734, 2709.0: 735, 2710.0: 736, 2720.0: 737, 2730.0: 738, 2736.0: 739, 2744.0: 740, 2745.0: 741, 2747.0: 742, 2760.0: 743, 2761.0: 744, 2768.0: 745, 2769.0: 746, 2784.0: 747, 2790.0: 748, 2808.0: 749, 2842.0: 750, 2852.0: 751, 2871.0: 752, 2875.0: 753, 2880.0: 754, 2882.0: 755, 2886.0: 756, 2889.0: 757, 2920.0: 758, 2928.0: 759, 2930.0: 760, 2938.0: 761, 2950.0: 762, 2951.0: 763, 2976.0: 764, 2979.0: 765, 2981.0: 766, 2988.0: 767, 2992.0: 768, 2997.0: 769, 3025.0: 770, 3030.0: 771, 3036.0: 772, 3069.0: 773, 3084.0: 774, 3087.0: 775, 3090.0: 776, 3107.0: 777, 3132.0: 778, 3141.0: 779, 3160.0: 780, 3170.0: 781, 3192.0: 782, 3222.0: 783, 3230.0: 784, 3250.0: 785, 3252.0: 786, 3255.0: 787, 3267.0: 788, 3270.0: 789, 3300.0: 790, 3321.0: 791, 3330.0: 792, 3360.0: 793, 3384.0: 794, 3392.0: 795, 3408.0: 796, 3420.0: 797, 3471.0: 798, 3555.0: 799, 3556.0: 800, 3624.0: 801, 3633.0: 802, 3647.0: 803, 3684.0: 804, 3707.0: 805, 3783.0: 806, 3784.0: 807, 3819.0: 808, 3825.0: 809, 3861.0: 810, 3876.0: 811, 3888.0: 812, 3927.0: 813, 3969.0: 814, 3984.0: 815, 4020.0: 816, 4037.0: 817, 4077.0: 818, 4080.0: 819, 4088.0: 820, 4130.0: 821, 4158.0: 822, 4165.0: 823, 4176.0: 824, 4184.0: 825, 4186.0: 826, 4199.0: 827, 4256.0: 828, 4272.0: 829, 4280.0: 830, 4300.0: 831, 4392.0: 832, 4396.0: 833, 4410.0: 834, 4420.0: 835, 4450.0: 836, 4464.0: 837, 4466.0: 838, 4488.0: 839, 4500.0: 840, 4524.0: 841, 4543.0: 842, 4554.0: 843, 4564.0: 844, 4572.0: 845, 4576.0: 846, 4581.0: 847, 4590.0: 848, 4596.0: 849, 4602.0: 850, 4625.0: 851, 4650.0: 852, 4675.0: 853, 4697.0: 854, 4740.0: 855, 4746.0: 856, 4760.0: 857, 4796.0: 858, 4806.0: 859, 4815.0: 860, 4824.0: 861, 4848.0: 862, 4875.0: 863, 4901.0: 864, 4908.0: 865, 4944.0: 866, 4950.0: 867, 4956.0: 868, 4968.0: 869, 4977.0: 870, 4990.0: 871, 5018.0: 872, 5026.0: 873, 5070.0: 874, 5075.0: 875, 5080.0: 876, 5096.0: 877, 5100.0: 878, 5110.0: 879, 5280.0: 880, 5304.0: 881, 5317.0: 882, 5340.0: 883, 5350.0: 884, 5382.0: 885, 5392.0: 886, 5456.0: 887, 5504.0: 888, 5530.0: 889, 5586.0: 890, 5621.0: 891, 5629.0: 892, 5696.0: 893, 5698.0: 894, 5720.0: 895, 5727.0: 896, 5733.0: 897, 5740.0: 898, 5744.0: 899, 5762.0: 900, 5865.0: 901, 5872.0: 902, 5908.0: 903, 5940.0: 904, 5950.0: 905, 5964.0: 906, 6000.0: 907, 6016.0: 908, 6062.0: 909, 6072.0: 910, 6090.0: 911, 6160.0: 912, 6192.0: 913, 6233.0: 914, 6285.0: 915, 6384.0: 916, 6474.0: 917, 6613.0: 918, 6640.0: 919, 6720.0: 920, 6784.0: 921, 7024.0: 922, 7275.0: 923, 7300.0: 924, 7500.0: 925, 7644.0: 926, 7830.0: 927, 8016.0: 928, 8533.0: 929, 8645.0: 930, 8778.0: 931, 8844.0: 932, 9240.0: 933, 9324.0: 934, 9828.0: 935, 9982.0: 936, 10712.0: 937, 11232.0: 938, 11304.0: 939, 11775.0: 940, 12375.0: 941, 12390.0: 942, 12561.0: 943, 12996.0: 944, 13442.0: 945, 13767.0: 946, 15580.0: 947, 19278.0: 948, 22680.0: 949, 22991.0: 950, 25748.0: 951, 25935.0: 952, 26145.0: 953, 26243.0: 954, 27058.0: 955, 39006.0: 956, 72204.0: 957, 78352.0: 958, 87234.0: 959, 98368.0: 960, 143993.0: 961, 5726.0: 962, 1080.0: 963, 1270.0: 964, 3451.0: 965, 1926.0: 966, 1834.0: 967, 1616.0: 968, 558.0: 969, 3024.0: 970, 398.0: 971, 6494.0: 972, 1690.0: 973, 2625.0: 974, 2259.0: 975, 2304.0: 976, 1544.0: 977, 686.0: 978, 86.0: 979, 4672.0: 980, 3123.0: 981, 79.0: 982, 3976.0: 983, 2673.0: 984, 2241.0: 985, 5208.0: 986, 1720.0: 987, 2964.0: 988, 1599.0: 989, 1351.0: 990, 21.0: 991, 4522.0: 992, 2552.0: 993, 4185.0: 994, 1240.0: 995, 1773.0: 996, 2310.0: 997, 2431.0: 998, 445.0: 999, 1079.0: 1000, 904.0: 1001, 1431.0: 1002, 6331.0: 1003, 1477.0: 1004, 2414.0: 1005, 44416.0: 1006, 5060.0: 1007, 202.0: 1008, 1152.0: 1009, 2862.0: 1010, 2556.0: 1011, 1752.0: 1012, 3822.0: 1013, 257.0: 1014, 5090.0: 1015, 1507.0: 1016, 7416.0: 1017, 4428.0: 1018, 81954.0: 1019, 1067.0: 1020, 2563.0: 1021, 6832.0: 1022, 2622.0: 1023, 140752.0: 1024, 1375.0: 1025, 255.0: 1026, 6768.0: 1027, 5390.0: 1028, 2001.0: 1029, 1956.0: 1030, 1360.0: 1031, 1452.0: 1032, 3038.0: 1033, 1211.0: 1034, 1920.0: 1035, 2160.0: 1036, 814.0: 1037, 323.0: 1038, 57.0: 1039, 3184.0: 1040, 682.0: 1041, 5200.0: 1042, 19789.0: 1043, 5460.0: 1044, 618.0: 1045, 2590.0: 1046, 2444.0: 1047, 283.0: 1048, 1578.0: 1049, 3264.0: 1050, 1809.0: 1051, 3456.0: 1052, 4545.0: 1053, 5499.0: 1054, 3311.0: 1055, 1190.0: 1056, 1730.0: 1057, 7868.0: 1058, 2850.0: 1059, 435.0: 1060, 1505.0: 1061, 45760.0: 1062, 2562.0: 1063, 265.0: 1064, 627.0: 1065, 325.0: 1066, 7209.0: 1067, 2010.0: 1068, 1204.0: 1069, 1342.0: 1070, 1323.0: 1071, 341.0: 1072, 2240.0: 1073, 6224.0: 1074, 784.0: 1075, 20867.0: 1076, 1623.0: 1077, 12275.0: 1078, 2814.0: 1079, 2592.0: 1080, 1428.0: 1081, 1197.0: 1082, 1050.0: 1083, 689.0: 1084, 1530.0: 1085, 3178.0: 1086, 1790.0: 1087, 5852.0: 1088, 2337.0: 1089, 5180.0: 1090, 2688.0: 1091, 26367.0: 1092, 786.0: 1093, 1450.0: 1094, 2192.0: 1095, 2706.0: 1096, 2145.0: 1097, 583.0: 1098, 2470.0: 1099, 1302.0: 1100, 4900.0: 1101, 418.0: 1102, 12350.0: 1103, 3972.0: 1104, 2050.0: 1105, 5194.0: 1106, 2401.0: 1107, 4662.0: 1108, 1755.0: 1109, 444.0: 1110, 19832.0: 1111, 1704.0: 1112, 6256.0: 1113, 892.0: 1114, 1746.0: 1115, 4225.0: 1116, 235.0: 1117, 578.0: 1118, 732.0: 1119, 4446.0: 1120, 6480.0: 1121, 754.0: 1122, 3732.0: 1123, 2821.0: 1124, 1911.0: 1125, 6912.0: 1126, 2358.0: 1127, 2723.0: 1128, 1248.0: 1129, 12367.0: 1130, 1352.0: 1131, 1183.0: 1132, 1071.0: 1133, 6112.0: 1134, 6288.0: 1135, 8640.0: 1136, 1199.0: 1137, 1105.0: 1138, 5040.0: 1139, 536.0: 1140, 1272.0: 1141, 1350.0: 1142, 8088.0: 1143, 8500.0: 1144, 513.0: 1145, 2392.0: 1146, 2596.0: 1147, 1648.0: 1148, 763.0: 1149, 4172.0: 1150, 5866.0: 1151, 8064.0: 1152, 5894.0: 1153, 2826.0: 1154, 1840.0: 1155, 1672.0: 1156, 688.0: 1157, 3458.0: 1158, 636.0: 1159, 26062.0: 1160, 372.0: 1161, 193.0: 1162, 24360.0: 1163, 483.0: 1164, 2340.0: 1165, 4550.0: 1166, 970.0: 1167, 314.0: 1168, 1794.0: 1169, 2500.0: 1170, 4381.0: 1171, 249.0: 1172, 2940.0: 1173, 4537.0: 1174, 286.0: 1175, 258.0: 1176, 803.0: 1177, 2400.0: 1178, 765.0: 1179, 1476.0: 1180, 1015.0: 1181, 3240.0: 1182, 379.0: 1183, 1086.0: 1184, 125.0: 1185, 1589.0: 1186, 318.0: 1187, 109.0: 1188, 4200.0: 1189, 8160.0: 1190, 8626.0: 1191, 4763.0: 1192, 1413.0: 1193, 3632.0: 1194, 3871.0: 1195, 4920.0: 1196, 3151.0: 1197, 3710.0: 1198, 4480.0: 1199, 3288.0: 1200, 3190.0: 1201, 553.0: 1202, 6525.0: 1203, 2123.0: 1204, 172.0: 1205, 67626.0: 1206, 3692.0: 1207, 2150.0: 1208, 750.0: 1209, 3280.0: 1210, 893.0: 1211, 1281.0: 1212, 1143.0: 1213, 2067.0: 1214, 1458.0: 1215, 749.0: 1216, 1379.0: 1217, 24920.0: 1218, 5768.0: 1219, 4784.0: 1220, 3212.0: 1221, 282.0: 1222, 1430.0: 1223, 4752.0: 1224, 364.0: 1225, 1029.0: 1226, 2907.0: 1227, 9672.0: 1228, 7350.0: 1229, 1500.0: 1230, 1792.0: 1231, 740.0: 1232, 5291.0: 1233, 2844.0: 1234, 3333.0: 1235, 1389.0: 1236, 2472.0: 1237, 94.0: 1238, 2466.0: 1239, 2613.0: 1240, 2934.0: 1241, 3322.0: 1242, 4680.0: 1243, 1032.0: 1244, 6528.0: 1245, 2752.0: 1246, 1850.0: 1247, 2896.0: 1248, 4826.0: 1249, 1030.0: 1250, 1740.0: 1251, 2840.0: 1252, 1233.0: 1253, 222.0: 1254, 848.0: 1255, 4797.0: 1256, 1592.0: 1257, 5628.0: 1258, 1064.0: 1259, 1614.0: 1260, 1863.0: 1261, 2714.0: 1262, 25619.0: 1263, 12240.0: 1264, 5558.0: 1265, 915.0: 1266, 131.0: 1267, 4494.0: 1268, 3836.0: 1269, 2508.0: 1270, 5130.0: 1271, 2322.0: 1272, 1133.0: 1273, 2496.0: 1274, 1925.0: 1275, 979.0: 1276, 1518.0: 1277, 2184.0: 1278, 2660.0: 1279, 3648.0: 1280, 2430.0: 1281, 1716.0: 1282, 3168.0: 1283, 1330.0: 1284, 940.0: 1285, 5138.0: 1286, 3146.0: 1287, 6744.0: 1288, 2226.0: 1289, 127.0: 1290, 4248.0: 1291, 4064.0: 1292, 825.0: 1293, 5632.0: 1294, 1368.0: 1295, 1560.0: 1296, 4043.0: 1297, 142290.0: 1298, 3000.0: 1299, 1503.0: 1300, 18865.0: 1301, 2116.0: 1302, 3105.0: 1303, 1098.0: 1304, 9999.0: 1305, 247.0: 1306, 10250.0: 1307, 2475.0: 1308, 736.0: 1309, 4214.0: 1310, 3276.0: 1311, 1208.0: 1312, 1099.0: 1313, 1236.0: 1314, 4224.0: 1315, 46.0: 1316, 4872.0: 1317, 6696.0: 1318, 992.0: 1319, 1141.0: 1320, 93.0: 1321, 836.0: 1322, 26386.0: 1323, 679.0: 1324, 2824.0: 1325, 741.0: 1326, 1377.0: 1327, 1148.0: 1328, 1719.0: 1329, 4599.0: 1330, 2048.0: 1331, 1974.0: 1332, 10092.0: 1333, 5814.0: 1334, 6672.0: 1335, 3766.0: 1336, 430.0: 1337, 5880.0: 1338, 5797.0: 1339, 5976.0: 1340, 2754.0: 1341, 5838.0: 1342, 451.0: 1343, 19296.0: 1344, 24174.0: 1345, 206.0: 1346, 2112.0: 1347, 2772.0: 1348, 3336.0: 1349, 1593.0: 1350, 2178.0: 1351, 3737.0: 1352, 2799.0: 1353, 3177.0: 1354, 1750.0: 1355, 1053.0: 1356, 2925.0: 1357, 58.0: 1358, 6766.0: 1359, 3601.0: 1360, 3768.0: 1361, 5824.0: 1362, 535.0: 1363, 4788.0: 1364, 3840.0: 1365, 3542.0: 1366, 1470.0: 1367, 1370.0: 1368, 2190.0: 1369, 1608.0: 1370, 2025.0: 1371, 2200.0: 1372, 6851.0: 1373, 11200.0: 1374, 8835.0: 1375, 1551.0: 1376, 3180.0: 1377, 1837.0: 1378, 2834.0: 1379, 13542.0: 1380, 23972.0: 1381, 1890.0: 1382, 4113.0: 1383, 2800.0: 1384, 1638.0: 1385, 3980.0: 1386, 833.0: 1387, 2096.0: 1388, 2232.0: 1389, 6156.0: 1390, 2820.0: 1391, 5016.0: 1392, 2600.0: 1393, 6560.0: 1394}]
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
    h_0 = max((((0.05190091 * float(x[0]))+ (-1.0186452 * float(x[1]))+ (-1.0662311 * float(x[2]))+ (-0.417866 * float(x[3]))+ (0.08951963 * float(x[4]))+ (-0.11410615 * float(x[5]))+ (-0.64011836 * float(x[6]))+ (-1.2635275 * float(x[7]))+ (-1.0136654 * float(x[8]))+ (-0.32285398 * float(x[9]))) + -0.11445105), 0)
    h_1 = max((((0.76251733 * float(x[0]))+ (1.200921 * float(x[1]))+ (-0.6106404 * float(x[2]))+ (-0.3566037 * float(x[3]))+ (17.073706 * float(x[4]))+ (9.194285 * float(x[5]))+ (1.2023046 * float(x[6]))+ (0.06736264 * float(x[7]))+ (0.0032226585 * float(x[8]))+ (-1.0073316 * float(x[9]))) + 9.149457), 0)
    o[0] = (0.9164909 * h_0)+ (0.085821055 * h_1) + -3.5081928

    

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
    w_h = np.array([[0.05190090835094452, -1.018645167350769, -1.06623113155365, -0.4178659915924072, 0.08951962739229202, -0.11410614848136902, -0.6401183605194092, -1.263527512550354, -1.0136654376983643, -0.3228539824485779], [0.7625173330307007, 1.2009210586547852, -0.6106404066085815, -0.3566037118434906, 17.073705673217773, 9.19428539276123, 1.2023046016693115, 0.06736263632774353, 0.003222658531740308, -1.0073316097259521]])
    b_h = np.array([-0.11445105075836182, 9.149456977844238])
    w_o = np.array([[0.916490912437439, 0.08582105487585068]])
    b_o = np.array(-3.508192777633667)

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

