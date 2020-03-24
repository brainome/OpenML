#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 22:28:56
# Invocation: btc -server brain.brainome.ai Data/page-blocks.csv -o Models/page-blocks.py -v -v -v -stopat 97.84 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                89.76%
Model accuracy:                     95.46% (5225/5473 correct)
Improvement over best guess:        5.70% (of possible 10.24%)
Model capacity (MEC):               25 bits
Generalization ratio:               209.00 bits/bit
Model efficiency:                   0.22%/parameter
System behavior
True Negatives:                     87.81% (4806/5473)
True Positives:                     7.66% (419/5473)
False Negatives:                    2.58% (141/5473)
False Positives:                    1.96% (107/5473)
True Pos. Rate/Sensitivity/Recall:  0.75
True Neg. Rate/Specificity:         0.98
Precision:                          0.80
F-1 Measure:                        0.77
False Negative Rate/Miss Rate:      0.25
Critical Success Index:             0.63

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
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="page-blocks.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 10
n_classes = 2

mappings = [{7.0: 0, 8.0: 1, 9.0: 2, 10.0: 3, 11.0: 4, 12.0: 5, 13.0: 6, 14.0: 7, 15.0: 8, 16.0: 9, 17.0: 10, 18.0: 11, 19.0: 12, 20.0: 13, 21.0: 14, 22.0: 15, 23.0: 16, 24.0: 17, 25.0: 18, 26.0: 19, 27.0: 20, 28.0: 21, 29.0: 22, 30.0: 23, 31.0: 24, 32.0: 25, 33.0: 26, 34.0: 27, 35.0: 28, 36.0: 29, 37.0: 30, 38.0: 31, 39.0: 32, 40.0: 33, 41.0: 34, 42.0: 35, 43.0: 36, 44.0: 37, 45.0: 38, 46.0: 39, 47.0: 40, 48.0: 41, 49.0: 42, 50.0: 43, 52.0: 44, 53.0: 45, 54.0: 46, 55.0: 47, 56.0: 48, 57.0: 49, 58.0: 50, 60.0: 51, 61.0: 52, 62.0: 53, 63.0: 54, 64.0: 55, 65.0: 56, 66.0: 57, 68.0: 58, 69.0: 59, 70.0: 60, 71.0: 61, 72.0: 62, 74.0: 63, 75.0: 64, 76.0: 65, 77.0: 66, 78.0: 67, 79.0: 68, 80.0: 69, 81.0: 70, 84.0: 71, 85.0: 72, 86.0: 73, 87.0: 74, 88.0: 75, 89.0: 76, 90.0: 77, 91.0: 78, 92.0: 79, 93.0: 80, 94.0: 81, 95.0: 82, 96.0: 83, 98.0: 84, 99.0: 85, 100.0: 86, 102.0: 87, 104.0: 88, 105.0: 89, 106.0: 90, 108.0: 91, 109.0: 92, 110.0: 93, 112.0: 94, 114.0: 95, 115.0: 96, 117.0: 97, 119.0: 98, 120.0: 99, 121.0: 100, 123.0: 101, 124.0: 102, 125.0: 103, 126.0: 104, 127.0: 105, 128.0: 106, 130.0: 107, 131.0: 108, 132.0: 109, 133.0: 110, 135.0: 111, 136.0: 112, 138.0: 113, 140.0: 114, 143.0: 115, 144.0: 116, 145.0: 117, 147.0: 118, 148.0: 119, 150.0: 120, 152.0: 121, 153.0: 122, 154.0: 123, 155.0: 124, 156.0: 125, 159.0: 126, 160.0: 127, 161.0: 128, 162.0: 129, 165.0: 130, 167.0: 131, 168.0: 132, 169.0: 133, 170.0: 134, 171.0: 135, 172.0: 136, 174.0: 137, 175.0: 138, 176.0: 139, 177.0: 140, 180.0: 141, 182.0: 142, 184.0: 143, 186.0: 144, 187.0: 145, 188.0: 146, 189.0: 147, 190.0: 148, 192.0: 149, 195.0: 150, 196.0: 151, 198.0: 152, 199.0: 153, 200.0: 154, 202.0: 155, 203.0: 156, 207.0: 157, 208.0: 158, 209.0: 159, 210.0: 160, 215.0: 161, 216.0: 162, 217.0: 163, 220.0: 164, 222.0: 165, 224.0: 166, 225.0: 167, 228.0: 168, 230.0: 169, 231.0: 170, 232.0: 171, 234.0: 172, 235.0: 173, 238.0: 174, 240.0: 175, 242.0: 176, 243.0: 177, 245.0: 178, 246.0: 179, 248.0: 180, 250.0: 181, 252.0: 182, 253.0: 183, 256.0: 184, 257.0: 185, 258.0: 186, 259.0: 187, 260.0: 188, 261.0: 189, 263.0: 190, 264.0: 191, 266.0: 192, 270.0: 193, 272.0: 194, 273.0: 195, 275.0: 196, 276.0: 197, 277.0: 198, 278.0: 199, 279.0: 200, 280.0: 201, 282.0: 202, 286.0: 203, 287.0: 204, 288.0: 205, 290.0: 206, 294.0: 207, 296.0: 208, 297.0: 209, 300.0: 210, 301.0: 211, 304.0: 212, 306.0: 213, 308.0: 214, 310.0: 215, 312.0: 216, 314.0: 217, 315.0: 218, 318.0: 219, 319.0: 220, 320.0: 221, 322.0: 222, 323.0: 223, 324.0: 224, 328.0: 225, 329.0: 226, 330.0: 227, 333.0: 228, 336.0: 229, 339.0: 230, 340.0: 231, 342.0: 232, 343.0: 233, 344.0: 234, 345.0: 235, 348.0: 236, 350.0: 237, 351.0: 238, 352.0: 239, 356.0: 240, 357.0: 241, 360.0: 242, 363.0: 243, 364.0: 244, 368.0: 245, 369.0: 246, 370.0: 247, 371.0: 248, 374.0: 249, 375.0: 250, 376.0: 251, 378.0: 252, 380.0: 253, 384.0: 254, 385.0: 255, 387.0: 256, 390.0: 257, 391.0: 258, 392.0: 259, 396.0: 260, 398.0: 261, 399.0: 262, 400.0: 263, 405.0: 264, 406.0: 265, 408.0: 266, 410.0: 267, 413.0: 268, 414.0: 269, 416.0: 270, 418.0: 271, 420.0: 272, 423.0: 273, 424.0: 274, 426.0: 275, 427.0: 276, 429.0: 277, 432.0: 278, 434.0: 279, 440.0: 280, 441.0: 281, 445.0: 282, 448.0: 283, 450.0: 284, 451.0: 285, 455.0: 286, 456.0: 287, 459.0: 288, 460.0: 289, 462.0: 290, 465.0: 291, 468.0: 292, 470.0: 293, 472.0: 294, 473.0: 295, 476.0: 296, 477.0: 297, 480.0: 298, 481.0: 299, 483.0: 300, 484.0: 301, 485.0: 302, 486.0: 303, 488.0: 304, 490.0: 305, 492.0: 306, 494.0: 307, 495.0: 308, 496.0: 309, 497.0: 310, 500.0: 311, 504.0: 312, 506.0: 313, 510.0: 314, 512.0: 315, 516.0: 316, 518.0: 317, 520.0: 318, 522.0: 319, 525.0: 320, 528.0: 321, 530.0: 322, 531.0: 323, 532.0: 324, 534.0: 325, 535.0: 326, 537.0: 327, 540.0: 328, 544.0: 329, 548.0: 330, 549.0: 331, 550.0: 332, 552.0: 333, 553.0: 334, 558.0: 335, 560.0: 336, 561.0: 337, 564.0: 338, 567.0: 339, 568.0: 340, 570.0: 341, 572.0: 342, 575.0: 343, 576.0: 344, 580.0: 345, 582.0: 346, 583.0: 347, 584.0: 348, 585.0: 349, 590.0: 350, 592.0: 351, 594.0: 352, 595.0: 353, 598.0: 354, 600.0: 355, 602.0: 356, 603.0: 357, 606.0: 358, 608.0: 359, 609.0: 360, 610.0: 361, 616.0: 362, 618.0: 363, 620.0: 364, 621.0: 365, 624.0: 366, 627.0: 367, 630.0: 368, 632.0: 369, 636.0: 370, 637.0: 371, 639.0: 372, 640.0: 373, 644.0: 374, 648.0: 375, 650.0: 376, 656.0: 377, 657.0: 378, 660.0: 379, 664.0: 380, 665.0: 381, 670.0: 382, 671.0: 383, 672.0: 384, 675.0: 385, 680.0: 386, 682.0: 387, 684.0: 388, 686.0: 389, 688.0: 390, 689.0: 391, 690.0: 392, 693.0: 393, 696.0: 394, 700.0: 395, 702.0: 396, 704.0: 397, 707.0: 398, 711.0: 399, 715.0: 400, 720.0: 401, 729.0: 402, 732.0: 403, 735.0: 404, 736.0: 405, 737.0: 406, 738.0: 407, 740.0: 408, 744.0: 409, 747.0: 410, 750.0: 411, 756.0: 412, 760.0: 413, 763.0: 414, 765.0: 415, 768.0: 416, 770.0: 417, 774.0: 418, 777.0: 419, 780.0: 420, 781.0: 421, 783.0: 422, 784.0: 423, 790.0: 424, 792.0: 425, 798.0: 426, 800.0: 427, 801.0: 428, 803.0: 429, 804.0: 430, 808.0: 431, 810.0: 432, 812.0: 433, 814.0: 434, 819.0: 435, 820.0: 436, 824.0: 437, 828.0: 438, 830.0: 439, 832.0: 440, 833.0: 441, 837.0: 442, 840.0: 443, 845.0: 444, 847.0: 445, 848.0: 446, 850.0: 447, 852.0: 448, 854.0: 449, 855.0: 450, 856.0: 451, 858.0: 452, 860.0: 453, 861.0: 454, 864.0: 455, 868.0: 456, 870.0: 457, 871.0: 458, 872.0: 459, 873.0: 460, 875.0: 461, 876.0: 462, 880.0: 463, 882.0: 464, 885.0: 465, 890.0: 466, 891.0: 467, 896.0: 468, 900.0: 469, 902.0: 470, 904.0: 471, 909.0: 472, 912.0: 473, 913.0: 474, 915.0: 475, 918.0: 476, 924.0: 477, 927.0: 478, 928.0: 479, 935.0: 480, 936.0: 481, 940.0: 482, 945.0: 483, 946.0: 484, 952.0: 485, 954.0: 486, 957.0: 487, 960.0: 488, 963.0: 489, 968.0: 490, 970.0: 491, 972.0: 492, 976.0: 493, 980.0: 494, 981.0: 495, 984.0: 496, 990.0: 497, 996.0: 498, 999.0: 499, 1001.0: 500, 1005.0: 501, 1008.0: 502, 1010.0: 503, 1012.0: 504, 1015.0: 505, 1016.0: 506, 1020.0: 507, 1029.0: 508, 1030.0: 509, 1032.0: 510, 1034.0: 511, 1040.0: 512, 1048.0: 513, 1050.0: 514, 1053.0: 515, 1056.0: 516, 1062.0: 517, 1064.0: 518, 1067.0: 519, 1068.0: 520, 1070.0: 521, 1071.0: 522, 1072.0: 523, 1074.0: 524, 1076.0: 525, 1078.0: 526, 1079.0: 527, 1080.0: 528, 1081.0: 529, 1085.0: 530, 1086.0: 531, 1089.0: 532, 1092.0: 533, 1098.0: 534, 1099.0: 535, 1100.0: 536, 1104.0: 537, 1105.0: 538, 1106.0: 539, 1107.0: 540, 1111.0: 541, 1122.0: 542, 1125.0: 543, 1128.0: 544, 1133.0: 545, 1134.0: 546, 1141.0: 547, 1143.0: 548, 1144.0: 549, 1147.0: 550, 1148.0: 551, 1152.0: 552, 1160.0: 553, 1176.0: 554, 1180.0: 555, 1183.0: 556, 1184.0: 557, 1190.0: 558, 1199.0: 559, 1208.0: 560, 1210.0: 561, 1211.0: 562, 1212.0: 563, 1216.0: 564, 1220.0: 565, 1224.0: 566, 1225.0: 567, 1230.0: 568, 1236.0: 569, 1240.0: 570, 1242.0: 571, 1248.0: 572, 1250.0: 573, 1260.0: 574, 1264.0: 575, 1265.0: 576, 1267.0: 577, 1272.0: 578, 1274.0: 579, 1280.0: 580, 1296.0: 581, 1298.0: 582, 1300.0: 583, 1304.0: 584, 1309.0: 585, 1310.0: 586, 1323.0: 587, 1326.0: 588, 1330.0: 589, 1332.0: 590, 1342.0: 591, 1344.0: 592, 1350.0: 593, 1352.0: 594, 1353.0: 595, 1359.0: 596, 1360.0: 597, 1368.0: 598, 1370.0: 599, 1372.0: 600, 1375.0: 601, 1377.0: 602, 1378.0: 603, 1379.0: 604, 1386.0: 605, 1392.0: 606, 1395.0: 607, 1400.0: 608, 1404.0: 609, 1410.0: 610, 1416.0: 611, 1422.0: 612, 1428.0: 613, 1430.0: 614, 1435.0: 615, 1440.0: 616, 1449.0: 617, 1450.0: 618, 1464.0: 619, 1467.0: 620, 1472.0: 621, 1474.0: 622, 1485.0: 623, 1491.0: 624, 1494.0: 625, 1495.0: 626, 1498.0: 627, 1500.0: 628, 1505.0: 629, 1508.0: 630, 1510.0: 631, 1518.0: 632, 1520.0: 633, 1521.0: 634, 1526.0: 635, 1530.0: 636, 1533.0: 637, 1536.0: 638, 1539.0: 639, 1540.0: 640, 1548.0: 641, 1550.0: 642, 1560.0: 643, 1562.0: 644, 1566.0: 645, 1568.0: 646, 1570.0: 647, 1575.0: 648, 1578.0: 649, 1584.0: 650, 1589.0: 651, 1590.0: 652, 1592.0: 653, 1593.0: 654, 1599.0: 655, 1608.0: 656, 1616.0: 657, 1617.0: 658, 1620.0: 659, 1624.0: 660, 1629.0: 661, 1632.0: 662, 1638.0: 663, 1647.0: 664, 1650.0: 665, 1665.0: 666, 1672.0: 667, 1674.0: 668, 1683.0: 669, 1690.0: 670, 1694.0: 671, 1696.0: 672, 1701.0: 673, 1704.0: 674, 1710.0: 675, 1719.0: 676, 1720.0: 677, 1728.0: 678, 1746.0: 679, 1750.0: 680, 1752.0: 681, 1755.0: 682, 1760.0: 683, 1768.0: 684, 1770.0: 685, 1771.0: 686, 1773.0: 687, 1775.0: 688, 1790.0: 689, 1794.0: 690, 1800.0: 691, 1802.0: 692, 1810.0: 693, 1815.0: 694, 1818.0: 695, 1830.0: 696, 1832.0: 697, 1834.0: 698, 1837.0: 699, 1848.0: 700, 1850.0: 701, 1863.0: 702, 1899.0: 703, 1911.0: 704, 1920.0: 705, 1926.0: 706, 1936.0: 707, 1953.0: 708, 1956.0: 709, 1960.0: 710, 1962.0: 711, 1963.0: 712, 1970.0: 713, 1971.0: 714, 1976.0: 715, 1980.0: 716, 1998.0: 717, 2001.0: 718, 2010.0: 719, 2013.0: 720, 2016.0: 721, 2024.0: 722, 2025.0: 723, 2035.0: 724, 2040.0: 725, 2048.0: 726, 2067.0: 727, 2070.0: 728, 2079.0: 729, 2088.0: 730, 2090.0: 731, 2097.0: 732, 2100.0: 733, 2112.0: 734, 2120.0: 735, 2123.0: 736, 2135.0: 737, 2142.0: 738, 2145.0: 739, 2150.0: 740, 2160.0: 741, 2168.0: 742, 2170.0: 743, 2176.0: 744, 2178.0: 745, 2184.0: 746, 2190.0: 747, 2196.0: 748, 2200.0: 749, 2201.0: 750, 2205.0: 751, 2226.0: 752, 2232.0: 753, 2250.0: 754, 2259.0: 755, 2280.0: 756, 2296.0: 757, 2304.0: 758, 2310.0: 759, 2312.0: 760, 2322.0: 761, 2331.0: 762, 2340.0: 763, 2343.0: 764, 2349.0: 765, 2358.0: 766, 2360.0: 767, 2376.0: 768, 2384.0: 769, 2392.0: 770, 2401.0: 771, 2412.0: 772, 2414.0: 773, 2422.0: 774, 2431.0: 775, 2439.0: 776, 2448.0: 777, 2466.0: 778, 2470.0: 779, 2475.0: 780, 2480.0: 781, 2490.0: 782, 2496.0: 783, 2502.0: 784, 2508.0: 785, 2510.0: 786, 2520.0: 787, 2544.0: 788, 2547.0: 789, 2552.0: 790, 2560.0: 791, 2563.0: 792, 2565.0: 793, 2568.0: 794, 2590.0: 795, 2592.0: 796, 2604.0: 797, 2610.0: 798, 2613.0: 799, 2618.0: 800, 2620.0: 801, 2622.0: 802, 2625.0: 803, 2630.0: 804, 2640.0: 805, 2660.0: 806, 2666.0: 807, 2680.0: 808, 2688.0: 809, 2690.0: 810, 2706.0: 811, 2709.0: 812, 2710.0: 813, 2720.0: 814, 2723.0: 815, 2736.0: 816, 2744.0: 817, 2745.0: 818, 2747.0: 819, 2752.0: 820, 2760.0: 821, 2761.0: 822, 2768.0: 823, 2769.0: 824, 2784.0: 825, 2790.0: 826, 2808.0: 827, 2814.0: 828, 2821.0: 829, 2824.0: 830, 2826.0: 831, 2834.0: 832, 2840.0: 833, 2842.0: 834, 2844.0: 835, 2850.0: 836, 2862.0: 837, 2871.0: 838, 2880.0: 839, 2882.0: 840, 2886.0: 841, 2889.0: 842, 2907.0: 843, 2920.0: 844, 2928.0: 845, 2930.0: 846, 2934.0: 847, 2938.0: 848, 2951.0: 849, 2964.0: 850, 2976.0: 851, 2979.0: 852, 2981.0: 853, 2988.0: 854, 2992.0: 855, 3025.0: 856, 3030.0: 857, 3087.0: 858, 3107.0: 859, 3123.0: 860, 3132.0: 861, 3146.0: 862, 3151.0: 863, 3160.0: 864, 3168.0: 865, 3192.0: 866, 3222.0: 867, 3240.0: 868, 3250.0: 869, 3252.0: 870, 3255.0: 871, 3264.0: 872, 3267.0: 873, 3270.0: 874, 3280.0: 875, 3321.0: 876, 3322.0: 877, 3330.0: 878, 3333.0: 879, 3360.0: 880, 3392.0: 881, 3458.0: 882, 3555.0: 883, 3647.0: 884, 3648.0: 885, 3707.0: 886, 3710.0: 887, 3732.0: 888, 3737.0: 889, 3783.0: 890, 3822.0: 891, 3836.0: 892, 3840.0: 893, 3876.0: 894, 3972.0: 895, 3976.0: 896, 4020.0: 897, 4037.0: 898, 4064.0: 899, 4077.0: 900, 4080.0: 901, 4113.0: 902, 4130.0: 903, 4165.0: 904, 4172.0: 905, 4184.0: 906, 4185.0: 907, 4199.0: 908, 4200.0: 909, 4214.0: 910, 4225.0: 911, 4248.0: 912, 4272.0: 913, 4280.0: 914, 4300.0: 915, 4392.0: 916, 4396.0: 917, 4410.0: 918, 4446.0: 919, 4450.0: 920, 4464.0: 921, 4466.0: 922, 4494.0: 923, 4500.0: 924, 4522.0: 925, 4524.0: 926, 4537.0: 927, 4543.0: 928, 4545.0: 929, 4550.0: 930, 4564.0: 931, 4572.0: 932, 4581.0: 933, 4590.0: 934, 4596.0: 935, 4599.0: 936, 4602.0: 937, 4650.0: 938, 4662.0: 939, 4672.0: 940, 4697.0: 941, 4746.0: 942, 4760.0: 943, 4763.0: 944, 4784.0: 945, 4788.0: 946, 4796.0: 947, 4797.0: 948, 4806.0: 949, 4815.0: 950, 4824.0: 951, 4826.0: 952, 4848.0: 953, 4875.0: 954, 4900.0: 955, 4901.0: 956, 4908.0: 957, 4920.0: 958, 4950.0: 959, 4968.0: 960, 4990.0: 961, 5080.0: 962, 5096.0: 963, 5110.0: 964, 5130.0: 965, 5280.0: 966, 5291.0: 967, 5304.0: 968, 5340.0: 969, 5350.0: 970, 5382.0: 971, 5390.0: 972, 5392.0: 973, 5456.0: 974, 5460.0: 975, 5504.0: 976, 5530.0: 977, 5586.0: 978, 5628.0: 979, 5632.0: 980, 5696.0: 981, 5698.0: 982, 5727.0: 983, 5744.0: 984, 5814.0: 985, 5824.0: 986, 5838.0: 987, 5865.0: 988, 5872.0: 989, 5950.0: 990, 5964.0: 991, 6016.0: 992, 6062.0: 993, 6072.0: 994, 6090.0: 995, 6112.0: 996, 6160.0: 997, 6192.0: 998, 6224.0: 999, 6256.0: 1000, 6285.0: 1001, 6331.0: 1002, 6384.0: 1003, 6474.0: 1004, 6480.0: 1005, 6494.0: 1006, 6525.0: 1007, 6560.0: 1008, 6696.0: 1009, 6744.0: 1010, 6768.0: 1011, 6784.0: 1012, 6832.0: 1013, 7024.0: 1014, 7209.0: 1015, 7275.0: 1016, 7300.0: 1017, 7830.0: 1018, 8016.0: 1019, 8064.0: 1020, 8088.0: 1021, 8500.0: 1022, 8533.0: 1023, 8645.0: 1024, 8778.0: 1025, 8835.0: 1026, 9324.0: 1027, 9672.0: 1028, 10250.0: 1029, 10712.0: 1030, 11200.0: 1031, 11232.0: 1032, 11775.0: 1033, 12240.0: 1034, 12367.0: 1035, 12375.0: 1036, 12996.0: 1037, 13442.0: 1038, 13542.0: 1039, 13767.0: 1040, 15580.0: 1041, 18865.0: 1042, 19278.0: 1043, 19296.0: 1044, 19789.0: 1045, 22991.0: 1046, 23972.0: 1047, 24360.0: 1048, 24920.0: 1049, 25748.0: 1050, 26062.0: 1051, 26145.0: 1052, 26243.0: 1053, 26367.0: 1054, 27058.0: 1055, 44416.0: 1056, 67626.0: 1057, 72204.0: 1058, 78352.0: 1059, 87234.0: 1060, 98368.0: 1061, 140752.0: 1062, 143993.0: 1063, 536.0: 1064, 407.0: 1065, 3288.0: 1066, 1188.0: 1067, 11304.0: 1068, 255.0: 1069, 3556.0: 1070, 2192.0: 1071, 1715.0: 1072, 950.0: 1073, 805.0: 1074, 3632.0: 1075, 1729.0: 1076, 749.0: 1077, 430.0: 1078, 6288.0: 1079, 1917.0: 1080, 2730.0: 1081, 5880.0: 1082, 4428.0: 1083, 1680.0: 1084, 1215.0: 1085, 2116.0: 1086, 8626.0: 1087, 107.0: 1088, 354.0: 1089, 3212.0: 1090, 325.0: 1091, 435.0: 1092, 4043.0: 1093, 2104.0: 1094, 1614.0: 1095, 5940.0: 1096, 5499.0: 1097, 5621.0: 1098, 2403.0: 1099, 3766.0: 1100, 826.0: 1101, 3230.0: 1102, 1782.0: 1103, 3420.0: 1104, 2673.0: 1105, 6640.0: 1106, 3141.0: 1107, 2430.0: 1108, 1512.0: 1109, 612.0: 1110, 754.0: 1111, 118.0: 1112, 4740.0: 1113, 987.0: 1114, 3036.0: 1115, 1716.0: 1116, 1507.0: 1117, 6766.0: 1118, 1573.0: 1119, 12390.0: 1120, 193.0: 1121, 5797.0: 1122, 51.0: 1123, 6233.0: 1124, 2556.0: 1125, 1458.0: 1126, 5138.0: 1127, 2240.0: 1128, 1170.0: 1129, 12350.0: 1130, 4680.0: 1131, 7500.0: 1132, 4554.0: 1133, 2004.0: 1134, 2241.0: 1135, 3984.0: 1136, 5208.0: 1137, 836.0: 1138, 992.0: 1139, 1740.0: 1140, 2799.0: 1141, 5317.0: 1142, 1000.0: 1143, 1026.0: 1144, 1380.0: 1145, 1864.0: 1146, 1840.0: 1147, 3871.0: 1148, 3105.0: 1149, 1476.0: 1150, 4256.0: 1151, 539.0: 1152, 1596.0: 1153, 3024.0: 1154, 67.0: 1155, 1769.0: 1156, 6613.0: 1157, 142290.0: 1158, 9240.0: 1159, 5558.0: 1160, 776.0: 1161, 3190.0: 1162, 7350.0: 1163, 3336.0: 1164, 1666.0: 1165, 5200.0: 1166, 1351.0: 1167, 4088.0: 1168, 1200.0: 1169, 341.0: 1170, 9982.0: 1171, 19832.0: 1172, 1256.0: 1173, 444.0: 1174, 6851.0: 1175, 5866.0: 1176, 1611.0: 1177, 3633.0: 1178, 5060.0: 1179, 1561.0: 1180, 2754.0: 1181, 2580.0: 1182, 893.0: 1183, 372.0: 1184, 2691.0: 1185, 5908.0: 1186, 2695.0: 1187, 2772.0: 1188, 469.0: 1189, 4488.0: 1190, 3276.0: 1191, 899.0: 1192, 1431.0: 1193, 3178.0: 1194, 1648.0: 1195, 81954.0: 1196, 2655.0: 1197, 24174.0: 1198, 6720.0: 1199, 1700.0: 1200, 1480.0: 1201, 1969.0: 1202, 1044.0: 1203, 825.0: 1204, 3456.0: 1205, 2875.0: 1206, 2484.0: 1207, 2562.0: 1208, 4956.0: 1209, 3819.0: 1210, 2500.0: 1211, 5075.0: 1212, 2896.0: 1213, 6912.0: 1214, 5629.0: 1215, 4186.0: 1216, 1413.0: 1217, 3684.0: 1218, 1925.0: 1219, 3184.0: 1220, 3000.0: 1221, 1673.0: 1222, 5894.0: 1223, 5976.0: 1224, 1708.0: 1225, 2416.0: 1226, 3180.0: 1227, 6156.0: 1228, 1389.0: 1229, 679.0: 1230, 3451.0: 1231, 475.0: 1232, 8640.0: 1233, 3980.0: 1234, 1397.0: 1235, 4176.0: 1236, 1809.0: 1237, 2255.0: 1238, 2483.0: 1239, 4381.0: 1240, 1477.0: 1241, 3471.0: 1242, 1860.0: 1243, 1656.0: 1244, 4944.0: 1245, 1302.0: 1246, 4420.0: 1247, 3300.0: 1248, 1246.0: 1249, 3384.0: 1250, 22680.0: 1251, 2820.0: 1252, 3090.0: 1253, 283.0: 1254, 1496.0: 1255, 816.0: 1256, 464.0: 1257, 5852.0: 1258, 5194.0: 1259, 7416.0: 1260, 5018.0: 1261, 786.0: 1262, 1197.0: 1263, 5740.0: 1264, 3542.0: 1265, 1116.0: 1266, 2600.0: 1267, 869.0: 1268, 1544.0: 1269, 3692.0: 1270, 730.0: 1271, 1664.0: 1272, 10092.0: 1273, 1337.0: 1274, 8844.0: 1275, 2270.0: 1276, 1488.0: 1277, 4977.0: 1278, 1452.0: 1279, 1161.0: 1280, 930.0: 1281, 3177.0: 1282, 1470.0: 1283, 1036.0: 1284, 6000.0: 1285, 1890.0: 1286, 1928.0: 1287, 2596.0: 1288, 931.0: 1289, 5762.0: 1290, 741.0: 1291, 6528.0: 1292, 5026.0: 1293, 1792.0: 1294, 4752.0: 1295, 25935.0: 1296, 5726.0: 1297, 2096.0: 1298, 1623.0: 1299, 3069.0: 1300, 265.0: 1301, 206.0: 1302, 20867.0: 1303, 12275.0: 1304, 1551.0: 1305, 3311.0: 1306, 2852.0: 1307, 7868.0: 1308, 3825.0: 1309, 2400.0: 1310, 513.0: 1311, 3927.0: 1312, 25619.0: 1313, 721.0: 1314, 1606.0: 1315, 5733.0: 1316, 2420.0: 1317, 2800.0: 1318, 2511.0: 1319, 1712.0: 1320, 3861.0: 1321, 1270.0: 1322, 1744.0: 1323, 5016.0: 1324, 9999.0: 1325, 2529.0: 1326, 5100.0: 1327, 4576.0: 1328, 9828.0: 1329, 1204.0: 1330, 2472.0: 1331, 2277.0: 1332, 3888.0: 1333, 2380.0: 1334, 2950.0: 1335, 4872.0: 1336, 1870.0: 1337, 5768.0: 1338, 1705.0: 1339, 5040.0: 1340, 2925.0: 1341, 752.0: 1342, 3969.0: 1343, 3784.0: 1344, 5070.0: 1345, 498.0: 1346, 26386.0: 1347, 4675.0: 1348, 73.0: 1349, 214.0: 1350, 8160.0: 1351, 4625.0: 1352, 2997.0: 1353, 2337.0: 1354, 3601.0: 1355, 6672.0: 1356, 3038.0: 1357, 1503.0: 1358, 249.0: 1359, 3170.0: 1360, 5180.0: 1361, 2664.0: 1362, 7644.0: 1363, 1232.0: 1364, 45760.0: 1365, 3768.0: 1366, 578.0: 1367, 3624.0: 1368, 1730.0: 1369, 4480.0: 1370, 1233.0: 1371, 2050.0: 1372, 1276.0: 1373, 2940.0: 1374, 2444.0: 1375, 5720.0: 1376, 4158.0: 1377, 1281.0: 1378, 1206.0: 1379, 892.0: 1380, 979.0: 1381, 1112.0: 1382, 3408.0: 1383, 247.0: 1384, 1150.0: 1385, 12561.0: 1386, 3084.0: 1387, 379.0: 1388, 1057.0: 1389, 1974.0: 1390, 4224.0: 1391, 39006.0: 1392, 2714.0: 1393, 5090.0: 1394}]
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



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    o=[0]*num_output_logits
    h_0 = max((((0.15467753 * float(x[0]))+ (-1.4744223 * float(x[1]))+ (-7.9502707 * float(x[2]))+ (-0.44148737 * float(x[3]))+ (0.2852267 * float(x[4]))+ (-0.15419835 * float(x[5]))+ (0.72514087 * float(x[6]))+ (-0.71546507 * float(x[7]))+ (-8.401621 * float(x[8]))+ (-0.70075005 * float(x[9]))) + 0.057044823), 0)
    h_1 = max((((-0.31360233 * float(x[0]))+ (0.02267368 * float(x[1]))+ (0.00453256 * float(x[2]))+ (-0.31376815 * float(x[3]))+ (-9.9400215 * float(x[4]))+ (10.686762 * float(x[5]))+ (0.0011635651 * float(x[6]))+ (0.0034265476 * float(x[7]))+ (-0.005138299 * float(x[8]))+ (0.042471375 * float(x[9]))) + 3.7862406), 0)
    o[0] = (-0.5735456 * h_0)+ (-0.8125336 * h_1) + 1.6847848

    if num_output_logits==1:
        return o[0]>=0
    else:
        return argmax(o)

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
        if not args.cleanfile: # File is not preprocessed
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            test_tensor = np.loadtxt(cleanfile,delimiter=',',dtype='float64')
            os.remove(cleanfile)
        else: # File is already preprocessed
            test_tensor = np.loadtxt(args.File,delimiter = ',',dtype = 'float64')               
        test_tensor = Normalize(test_tensor)
        if transform_true:
            test_tensor = transform(test_tensor)
        with open(args.csvfile,'r') as csvinput:
            writer = csv.writer(sys.stdout, lineterminator='\n')
            reader = csv.reader(csvinput)
            if (not args.headerless):
                writer.writerow((next(reader, None)+['Prediction']))
            i=0
            for row in reader:
                if (classify(test_tensor[i])):
                    pred="1"
                else:
                    pred="0"
                row.append(pred)
                writer.writerow(row)
                i=i+1
    elif args.validate: # Then validate this predictor, always clean first.
        if n_classes==2:
            tempdir=tempfile.gettempdir()
            temp_name = next(tempfile._get_candidate_names())
            cleanfile=tempdir+os.sep+temp_name
            clean(args.csvfile,cleanfile, -1, args.headerless)
            val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
            os.remove(cleanfile)
            val_tensor = Normalize(val_tensor)
            if transform_true:
                trans = transform(val_tensor[:,:-1])
                val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
            count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0 = 0,0,0,0,0,0,0,0
            for i,row in enumerate(val_tensor):
                if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                    correct_count+=1
                    if int(float(row[-1]))==1:
                        num_class_1+=1
                        num_TP+=1
                    else:
                        num_class_0+=1
                        num_TN+=1
                else:
                    if int(float(row[-1]))==1:
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
            val_tensor = np.loadtxt(cleanfile,delimiter = ',',dtype = 'float64')
            os.remove(cleanfile)
            val_tensor = Normalize(val_tensor)
            if transform_true:
                trans = transform(val_tensor[:,:-1])
                val_tensor = np.concatenate((trans,val_tensor[:,-1].reshape(-1,1)),axis = 1)
            numeachclass={}
            count,correct_count = 0,0
            for i,row in enumerate(val_tensor):
                if int(classify(val_tensor[i].tolist())) == int(float(val_tensor[i,-1])):
                    correct_count+=1
                    if int(float(val_tensor[i,-1])) in numeachclass.keys():
                        numeachclass[int(float(val_tensor[i,-1]))]+=1
                    else:
                        numeachclass[int(float(val_tensor[i,-1]))]=0
                count+=1

        model_cap=25

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






