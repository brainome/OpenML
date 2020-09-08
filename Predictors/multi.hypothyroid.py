#!/usr/bin/env python3
#
# This code has been produced by an evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
# Distribution of this code in binary form or commercial use of any kind is forbidden.
# For a detailed license agreement see: http://brainome.ai/license
# Use of predictions results at your own risk.
#
# Output of Brainome Daimensions(tm) 0.98 Table Compiler v0.98.
# Invocation: btc -f NN -target binaryClass hypothyroid.csv -o hypothyroid_NN.py -nsamples 0 --yes -nsamples 0 -e 20
# Total compiler execution time: 0:11:46.85. Finished on: Sep-04-2020 11:09:02.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           70:30%
Best-guess accuracy:                 92.28%
Training accuracy:                   98.56% (1859/1886 correct)
Validation accuracy:                 98.83% (1864/1886 correct)
Overall Model accuracy:              98.70% (3723/3772 correct)
Overall Improvement over best guess: 6.42% (of possible 7.72%)
Model capacity (MEC):                63 bits
Generalization ratio:                59.09 bits/bit
Model efficiency:                    0.10%/parameter
System behavior
True Negatives:                      91.76% (3461/3772)
True Positives:                      6.95% (262/3772)
False Negatives:                     0.77% (29/3772)
False Positives:                     0.53% (20/3772)
True Pos. Rate/Sensitivity/Recall:   0.90
True Neg. Rate/Specificity:          0.99
Precision:                           0.93
F-1 Measure:                         0.91
False Negative Rate/Miss Rate:       0.10
Critical Success Index:              0.84
Confusion Matrix:
 [91.76% 0.53%]
 [0.77% 6.95%]
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
TRAINFILE = "hypothyroid.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29
n_classes = 2

mappings = [{1.0: 0, 2.0: 1, 4.0: 2, 5.0: 3, 7.0: 4, 8.0: 5, 11.0: 6, 12.0: 7, 13.0: 8, 14.0: 9, 15.0: 10, 16.0: 11, 17.0: 12, 18.0: 13, 19.0: 14, 20.0: 15, 21.0: 16, 22.0: 17, 23.0: 18, 24.0: 19, 25.0: 20, 26.0: 21, 27.0: 22, 28.0: 23, 29.0: 24, 30.0: 25, 31.0: 26, 32.0: 27, 33.0: 28, 34.0: 29, 35.0: 30, 36.0: 31, 37.0: 32, 38.0: 33, 39.0: 34, 40.0: 35, 41.0: 36, 42.0: 37, 43.0: 38, 44.0: 39, 45.0: 40, 46.0: 41, 47.0: 42, 48.0: 43, 49.0: 44, 50.0: 45, 51.0: 46, 52.0: 47, 53.0: 48, 54.0: 49, 55.0: 50, 56.0: 51, 57.0: 52, 58.0: 53, 59.0: 54, 60.0: 55, 61.0: 56, 62.0: 57, 63.0: 58, 64.0: 59, 65.0: 60, 66.0: 61, 67.0: 62, 68.0: 63, 69.0: 64, 70.0: 65, 71.0: 66, 72.0: 67, 73.0: 68, 74.0: 69, 75.0: 70, 76.0: 71, 77.0: 72, 78.0: 73, 79.0: 74, 80.0: 75, 81.0: 76, 82.0: 77, 83.0: 78, 84.0: 79, 85.0: 80, 86.0: 81, 87.0: 82, 88.0: 83, 89.0: 84, 90.0: 85, 92.0: 86, 93.0: 87, 94.0: 88, 1684325040.0: 89, 10.0: 90, 455.0: 91, 91.0: 92, 6.0: 93}, {1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.17: 22, 0.18: 23, 0.19: 24, 0.2: 25, 0.21: 26, 0.22: 27, 0.23: 28, 0.25: 29, 0.26: 30, 0.27: 31, 0.28: 32, 0.29: 33, 0.3: 34, 0.31: 35, 0.32: 36, 0.33: 37, 0.34: 38, 0.35: 39, 0.36: 40, 0.37: 41, 0.38: 42, 0.39: 43, 0.4: 44, 0.41: 45, 0.42: 46, 0.43: 47, 0.44: 48, 0.45: 49, 0.46: 50, 0.47: 51, 0.48: 52, 0.49: 53, 0.5: 54, 0.51: 55, 0.52: 56, 0.53: 57, 0.54: 58, 0.55: 59, 0.56: 60, 0.57: 61, 0.58: 62, 0.59: 63, 0.6: 64, 0.61: 65, 0.62: 66, 0.63: 67, 0.64: 68, 0.65: 69, 0.66: 70, 0.67: 71, 0.68: 72, 0.69: 73, 0.7: 74, 0.71: 75, 0.72: 76, 0.73: 77, 0.74: 78, 0.75: 79, 0.76: 80, 0.77: 81, 0.78: 82, 0.79: 83, 0.8: 84, 0.81: 85, 0.82: 86, 0.83: 87, 0.84: 88, 0.85: 89, 0.86: 90, 0.87: 91, 0.88: 92, 0.89: 93, 0.9: 94, 0.91: 95, 0.92: 96, 0.93: 97, 0.94: 98, 0.95: 99, 0.96: 100, 0.97: 101, 0.98: 102, 0.99: 103, 1.0: 104, 1.1: 105, 1.2: 106, 1.3: 107, 1.4: 108, 1.5: 109, 1.6: 110, 1.7: 111, 1.8: 112, 1.9: 113, 2.0: 114, 2.1: 115, 2.2: 116, 2.3: 117, 2.4: 118, 2.5: 119, 2.6: 120, 2.7: 121, 2.8: 122, 2.9: 123, 3.0: 124, 3.1: 125, 3.2: 126, 3.3: 127, 3.4: 128, 3.5: 129, 3.6: 130, 3.7: 131, 3.8: 132, 3.9: 133, 4.0: 134, 4.1: 135, 4.2: 136, 4.3: 137, 4.4: 138, 4.5: 139, 4.6: 140, 4.7: 141, 4.8: 142, 4.9: 143, 5.0: 144, 5.1: 145, 5.2: 146, 5.3: 147, 5.4: 148, 5.5: 149, 5.6: 150, 5.7: 151, 5.73: 152, 5.8: 153, 5.9: 154, 6.0: 155, 6.1: 156, 6.2: 157, 6.3: 158, 6.4: 159, 6.5: 160, 6.6: 161, 6.7: 162, 6.8: 163, 6.9: 164, 7.0: 165, 7.1: 166, 7.2: 167, 7.3: 168, 7.4: 169, 7.5: 170, 7.6: 171, 7.7: 172, 7.8: 173, 7.9: 174, 8.0: 175, 8.1: 176, 8.2: 177, 8.3: 178, 8.4: 179, 8.5: 180, 8.6: 181, 8.8: 182, 8.9: 183, 9.0: 184, 9.1: 185, 9.2: 186, 9.3: 187, 9.4: 188, 9.5: 189, 9.6: 190, 9.7: 191, 9.8: 192, 10.0: 193, 10.3: 194, 11.0: 195, 11.1: 196, 11.4: 197, 12.0: 198, 13.0: 199, 14.0: 200, 14.4: 201, 14.8: 202, 15.0: 203, 16.0: 204, 17.0: 205, 18.0: 206, 18.4: 207, 19.0: 208, 20.0: 209, 21.0: 210, 22.0: 211, 23.0: 212, 24.0: 213, 25.0: 214, 26.0: 215, 26.4: 216, 27.0: 217, 28.0: 218, 30.0: 219, 30.5: 220, 31.0: 221, 32.0: 222, 34.0: 223, 35.0: 224, 36.0: 225, 38.0: 226, 39.0: 227, 41.0: 228, 42.0: 229, 43.0: 230, 44.0: 231, 45.0: 232, 46.0: 233, 47.0: 234, 50.0: 235, 51.0: 236, 52.0: 237, 54.0: 238, 55.0: 239, 58.0: 240, 60.0: 241, 61.0: 242, 65.0: 243, 66.0: 244, 70.0: 245, 76.0: 246, 78.0: 247, 80.0: 248, 82.0: 249, 89.0: 250, 100.0: 251, 103.0: 252, 108.0: 253, 116.0: 254, 117.0: 255, 126.0: 256, 143.0: 257, 145.0: 258, 151.0: 259, 160.0: 260, 165.0: 261, 183.0: 262, 440.0: 263, 468.0: 264, 472.0: 265, 478.0: 266, 530.0: 267, 1684325040.0: 268, 9.9: 269, 0.24: 270, 29.0: 271, 109.0: 272, 33.0: 273, 86.0: 274, 199.0: 275, 139.0: 276, 12.1: 277, 178.0: 278, 188.0: 279, 1.01: 280, 98.0: 281, 40.0: 282, 99.0: 283, 236.0: 284, 1.02: 285, 400.0: 286, 230.0: 287}, {1993550816.0: 0, 2238339752.0: 1}, {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 1.1: 10, 1.2: 11, 1.3: 12, 1.4: 13, 1.44: 14, 1.5: 15, 1.6: 16, 1.7: 17, 1.8: 18, 1.9: 19, 2.0: 20, 2.1: 21, 2.2: 22, 2.3: 23, 2.4: 24, 2.5: 25, 2.6: 26, 2.7: 27, 2.8: 28, 2.9: 29, 3.0: 30, 3.1: 31, 3.2: 32, 3.3: 33, 3.4: 34, 3.5: 35, 3.6: 36, 3.7: 37, 3.8: 38, 3.9: 39, 4.0: 40, 4.1: 41, 4.2: 42, 4.3: 43, 4.4: 44, 4.5: 45, 4.6: 46, 4.8: 47, 5.0: 48, 5.1: 49, 5.2: 50, 5.3: 51, 5.4: 52, 5.5: 53, 5.7: 54, 6.2: 55, 6.6: 56, 6.7: 57, 7.0: 58, 7.1: 59, 7.3: 60, 7.6: 61, 8.5: 62, 10.6: 63, 1684325040.0: 64, 4.7: 65, 4.9: 66, 6.0: 67, 0.05: 68, 6.1: 69}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 3.0: 1, 4.0: 2, 4.8: 3, 5.8: 4, 6.0: 5, 9.5: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 17.0: 13, 18.0: 14, 19.0: 15, 21.0: 16, 22.0: 17, 24.0: 18, 25.0: 19, 28.0: 20, 30.0: 21, 32.0: 22, 33.0: 23, 34.0: 24, 35.0: 25, 37.0: 26, 38.0: 27, 39.0: 28, 41.0: 29, 42.0: 30, 43.0: 31, 44.0: 32, 45.0: 33, 46.0: 34, 48.0: 35, 50.0: 36, 51.0: 37, 52.0: 38, 53.0: 39, 54.0: 40, 55.0: 41, 56.0: 42, 57.0: 43, 58.0: 44, 59.0: 45, 60.0: 46, 61.0: 47, 62.0: 48, 63.0: 49, 64.0: 50, 65.0: 51, 66.0: 52, 67.0: 53, 68.0: 54, 69.0: 55, 70.0: 56, 71.0: 57, 72.0: 58, 73.0: 59, 74.0: 60, 75.0: 61, 76.0: 62, 77.0: 63, 78.0: 64, 79.0: 65, 80.0: 66, 81.0: 67, 82.0: 68, 83.0: 69, 84.0: 70, 85.0: 71, 86.0: 72, 87.0: 73, 88.0: 74, 89.0: 75, 90.0: 76, 91.0: 77, 92.0: 78, 93.0: 79, 94.0: 80, 95.0: 81, 96.0: 82, 97.0: 83, 98.0: 84, 99.0: 85, 100.0: 86, 101.0: 87, 102.0: 88, 103.0: 89, 104.0: 90, 105.0: 91, 106.0: 92, 107.0: 93, 108.0: 94, 109.0: 95, 110.0: 96, 111.0: 97, 112.0: 98, 113.0: 99, 114.0: 100, 115.0: 101, 116.0: 102, 117.0: 103, 118.0: 104, 119.0: 105, 120.0: 106, 121.0: 107, 122.0: 108, 123.0: 109, 124.0: 110, 125.0: 111, 126.0: 112, 127.0: 113, 128.0: 114, 129.0: 115, 130.0: 116, 131.0: 117, 132.0: 118, 133.0: 119, 134.0: 120, 135.0: 121, 136.0: 122, 137.0: 123, 138.0: 124, 139.0: 125, 140.0: 126, 141.0: 127, 142.0: 128, 143.0: 129, 144.0: 130, 145.0: 131, 146.0: 132, 147.0: 133, 148.0: 134, 149.0: 135, 150.0: 136, 151.0: 137, 152.0: 138, 153.0: 139, 154.0: 140, 155.0: 141, 156.0: 142, 157.0: 143, 158.0: 144, 159.0: 145, 160.0: 146, 161.0: 147, 162.0: 148, 163.0: 149, 164.0: 150, 165.0: 151, 166.0: 152, 167.0: 153, 168.0: 154, 169.0: 155, 170.0: 156, 171.0: 157, 172.0: 158, 173.0: 159, 174.0: 160, 175.0: 161, 176.0: 162, 177.0: 163, 178.0: 164, 179.0: 165, 180.0: 166, 181.0: 167, 182.0: 168, 183.0: 169, 184.0: 170, 186.0: 171, 187.0: 172, 188.0: 173, 189.0: 174, 191.0: 175, 192.0: 176, 193.0: 177, 194.0: 178, 196.0: 179, 197.0: 180, 198.0: 181, 199.0: 182, 200.0: 183, 203.0: 184, 204.0: 185, 205.0: 186, 206.0: 187, 207.0: 188, 209.0: 189, 210.0: 190, 212.0: 191, 213.0: 192, 214.0: 193, 216.0: 194, 217.0: 195, 219.0: 196, 223.0: 197, 225.0: 198, 226.0: 199, 230.0: 200, 232.0: 201, 235.0: 202, 237.0: 203, 239.0: 204, 240.0: 205, 244.0: 206, 246.0: 207, 248.0: 208, 250.0: 209, 252.0: 210, 255.0: 211, 256.0: 212, 258.0: 213, 261.0: 214, 263.0: 215, 272.0: 216, 273.0: 217, 301.0: 218, 372.0: 219, 430.0: 220, 1684325040.0: 221, 222.0: 222, 2.9: 223, 253.0: 224, 289.0: 225, 49.0: 226, 257.0: 227, 47.0: 228, 23.0: 229, 36.0: 230, 211.0: 231, 27.0: 232, 220.0: 233, 233.0: 234, 201.0: 235, 231.0: 236, 31.0: 237, 40.0: 238, 16.0: 239, 195.0: 240, 29.0: 241}, {1993550816.0: 0, 2238339752.0: 1}, {0.25: 0, 0.31: 1, 0.41: 2, 0.46: 3, 0.48: 4, 0.5: 5, 0.53: 6, 0.54: 7, 0.56: 8, 0.58: 9, 0.59: 10, 0.6: 11, 0.61: 12, 0.62: 13, 0.63: 14, 0.64: 15, 0.65: 16, 0.66: 17, 0.67: 18, 0.68: 19, 0.69: 20, 0.7: 21, 0.71: 22, 0.72: 23, 0.73: 24, 0.74: 25, 0.75: 26, 0.76: 27, 0.77: 28, 0.78: 29, 0.79: 30, 0.8: 31, 0.81: 32, 0.82: 33, 0.83: 34, 0.84: 35, 0.85: 36, 0.86: 37, 0.87: 38, 0.88: 39, 0.89: 40, 0.9: 41, 0.91: 42, 0.92: 43, 0.93: 44, 0.94: 45, 0.95: 46, 0.96: 47, 0.97: 48, 0.98: 49, 0.99: 50, 1.0: 51, 1.01: 52, 1.02: 53, 1.03: 54, 1.04: 55, 1.05: 56, 1.06: 57, 1.07: 58, 1.08: 59, 1.09: 60, 1.1: 61, 1.11: 62, 1.12: 63, 1.13: 64, 1.14: 65, 1.15: 66, 1.16: 67, 1.17: 68, 1.18: 69, 1.19: 70, 1.2: 71, 1.21: 72, 1.22: 73, 1.23: 74, 1.24: 75, 1.25: 76, 1.26: 77, 1.27: 78, 1.28: 79, 1.29: 80, 1.3: 81, 1.31: 82, 1.32: 83, 1.33: 84, 1.34: 85, 1.35: 86, 1.36: 87, 1.37: 88, 1.38: 89, 1.39: 90, 1.4: 91, 1.41: 92, 1.42: 93, 1.43: 94, 1.44: 95, 1.45: 96, 1.46: 97, 1.47: 98, 1.48: 99, 1.49: 100, 1.5: 101, 1.51: 102, 1.52: 103, 1.53: 104, 1.55: 105, 1.57: 106, 1.58: 107, 1.59: 108, 1.61: 109, 1.62: 110, 1.63: 111, 1.65: 112, 1.66: 113, 1.67: 114, 1.68: 115, 1.69: 116, 1.7: 117, 1.71: 118, 1.73: 119, 1.75: 120, 1.76: 121, 1.77: 122, 1.79: 123, 1.8: 124, 1.82: 125, 1.83: 126, 1.84: 127, 1.93: 128, 1.94: 129, 1.97: 130, 2.01: 131, 2.03: 132, 2.12: 133, 2.32: 134, 1684325040.0: 135, 1.88: 136, 0.49: 137, 0.52: 138, 1.54: 139, 1.74: 140, 0.944: 141, 1.56: 142, 0.36: 143, 0.57: 144, 0.47: 145, 0.38: 146}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 3.0: 1, 4.0: 2, 5.4: 3, 7.6: 4, 8.4: 5, 8.5: 6, 8.9: 7, 9.0: 8, 10.0: 9, 11.0: 10, 13.0: 11, 14.0: 12, 15.0: 13, 16.0: 14, 17.0: 15, 18.0: 16, 19.0: 17, 20.0: 18, 21.0: 19, 28.0: 20, 29.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 36.0: 25, 37.0: 26, 39.0: 27, 41.0: 28, 42.0: 29, 43.0: 30, 46.0: 31, 47.0: 32, 49.0: 33, 50.0: 34, 51.0: 35, 52.0: 36, 53.0: 37, 54.0: 38, 55.0: 39, 56.0: 40, 57.0: 41, 58.0: 42, 59.0: 43, 60.0: 44, 61.0: 45, 62.0: 46, 63.0: 47, 64.0: 48, 65.0: 49, 66.0: 50, 67.0: 51, 68.0: 52, 69.0: 53, 70.0: 54, 71.0: 55, 72.0: 56, 73.0: 57, 74.0: 58, 75.0: 59, 76.0: 60, 77.0: 61, 78.0: 62, 79.0: 63, 80.0: 64, 81.0: 65, 82.0: 66, 83.0: 67, 84.0: 68, 85.0: 69, 86.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 93.0: 77, 94.0: 78, 95.0: 79, 96.0: 80, 97.0: 81, 98.0: 82, 99.0: 83, 100.0: 84, 101.0: 85, 102.0: 86, 103.0: 87, 104.0: 88, 105.0: 89, 106.0: 90, 107.0: 91, 108.0: 92, 109.0: 93, 110.0: 94, 111.0: 95, 112.0: 96, 113.0: 97, 114.0: 98, 115.0: 99, 116.0: 100, 117.0: 101, 118.0: 102, 119.0: 103, 120.0: 104, 121.0: 105, 122.0: 106, 123.0: 107, 124.0: 108, 125.0: 109, 126.0: 110, 127.0: 111, 128.0: 112, 129.0: 113, 130.0: 114, 131.0: 115, 132.0: 116, 133.0: 117, 134.0: 118, 135.0: 119, 136.0: 120, 137.0: 121, 138.0: 122, 139.0: 123, 140.0: 124, 141.0: 125, 142.0: 126, 143.0: 127, 144.0: 128, 145.0: 129, 146.0: 130, 147.0: 131, 148.0: 132, 149.0: 133, 150.0: 134, 151.0: 135, 152.0: 136, 153.0: 137, 154.0: 138, 155.0: 139, 156.0: 140, 157.0: 141, 158.0: 142, 159.0: 143, 160.0: 144, 161.0: 145, 162.0: 146, 163.0: 147, 164.0: 148, 165.0: 149, 166.0: 150, 167.0: 151, 168.0: 152, 169.0: 153, 170.0: 154, 171.0: 155, 172.0: 156, 173.0: 157, 174.0: 158, 175.0: 159, 176.0: 160, 177.0: 161, 178.0: 162, 179.0: 163, 180.0: 164, 181.0: 165, 182.0: 166, 183.0: 167, 184.0: 168, 185.0: 169, 186.0: 170, 187.0: 171, 188.0: 172, 190.0: 173, 191.0: 174, 194.0: 175, 195.0: 176, 196.0: 177, 197.0: 178, 198.0: 179, 200.0: 180, 201.0: 181, 203.0: 182, 204.0: 183, 205.0: 184, 206.0: 185, 207.0: 186, 209.0: 187, 210.0: 188, 213.0: 189, 214.0: 190, 215.0: 191, 216.0: 192, 217.0: 193, 218.0: 194, 219.0: 195, 220.0: 196, 221.0: 197, 222.0: 198, 223.0: 199, 227.0: 200, 232.0: 201, 235.0: 202, 244.0: 203, 245.0: 204, 247.0: 205, 249.0: 206, 265.0: 207, 274.0: 208, 280.0: 209, 281.0: 210, 283.0: 211, 291.0: 212, 349.0: 213, 362.0: 214, 395.0: 215, 1684325040.0: 216, 2.8: 217, 48.0: 218, 24.0: 219, 189.0: 220, 32.0: 221, 312.0: 222, 40.0: 223, 228.0: 224, 253.0: 225, 7.0: 226, 251.0: 227, 242.0: 228, 9.1: 229, 199.0: 230, 27.0: 231, 237.0: 232, 224.0: 233, 26.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

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
    h_0 = max((((-11.838188 * float(x[0]))+ (-0.3624735 * float(x[1]))+ (-0.32377002 * float(x[2]))+ (-0.16944534 * float(x[3]))+ (0.11797351 * float(x[4]))+ (-0.008563565 * float(x[5]))+ (0.34882486 * float(x[6]))+ (-0.039042175 * float(x[7]))+ (0.11641401 * float(x[8]))+ (-0.13314697 * float(x[9]))+ (-0.08647725 * float(x[10]))+ (-0.4202973 * float(x[11]))+ (-0.29136527 * float(x[12]))+ (-0.18135399 * float(x[13]))+ (0.016297013 * float(x[14]))+ (0.17392433 * float(x[15]))+ (-0.07135662 * float(x[16]))+ (-40.194317 * float(x[17]))+ (-0.5267186 * float(x[18]))+ (-5.886804 * float(x[19]))+ (0.02999053 * float(x[20]))+ (-27.25169 * float(x[21]))+ (-0.006118203 * float(x[22]))+ (-13.821195 * float(x[23]))+ (-0.28877947 * float(x[24]))+ (-29.778692 * float(x[25]))+ (-0.40812054 * float(x[26]))+ (-0.27695978 * float(x[27]))+ (-1.1175253 * float(x[28]))) + -0.33862522), 0)
    h_1 = max((((-0.015149767 * float(x[0]))+ (-0.4260426 * float(x[1]))+ (-11.649362 * float(x[2]))+ (-1.0031037 * float(x[3]))+ (-1.4514359 * float(x[4]))+ (-0.6316401 * float(x[5]))+ (-6.758179 * float(x[6]))+ (-11.102059 * float(x[7]))+ (0.2816551 * float(x[8]))+ (0.33944556 * float(x[9]))+ (0.55254984 * float(x[10]))+ (-10.614743 * float(x[11]))+ (-9.539527 * float(x[12]))+ (-0.58309895 * float(x[13]))+ (-2.6421897 * float(x[14]))+ (0.1938651 * float(x[15]))+ (17.054482 * float(x[16]))+ (0.10585127 * float(x[17]))+ (-4.905209 * float(x[18]))+ (-0.09578622 * float(x[19]))+ (-2.0251625 * float(x[20]))+ (-0.0022073186 * float(x[21]))+ (-3.6981683 * float(x[22]))+ (-0.008171485 * float(x[23]))+ (-4.0722003 * float(x[24]))+ (-0.048238706 * float(x[25]))+ (-0.34108895 * float(x[26]))+ (-0.30491725 * float(x[27]))+ (-0.20680515 * float(x[28]))) + -4.0313883), 0)
    o[0] = (0.49254394 * h_0)+ (1.1246772 * h_1) + -8.544547

    

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
    w_h = np.array([[-11.838188171386719, -0.3624734878540039, -0.32377001643180847, -0.16944533586502075, 0.11797350645065308, -0.008563565090298653, 0.3488248586654663, -0.03904217481613159, 0.11641401052474976, -0.1331469714641571, -0.08647724986076355, -0.4202972948551178, -0.29136526584625244, -0.18135398626327515, 0.01629701256752014, 0.17392432689666748, -0.07135661691427231, -40.19431686401367, -0.5267186164855957, -5.886804103851318, 0.029990529641509056, -27.251689910888672, -0.006118203047662973, -13.821194648742676, -0.28877946734428406, -29.7786922454834, -0.4081205427646637, -0.27695977687835693, -1.117525339126587], [-0.015149766579270363, -0.4260425865650177, -11.649361610412598, -1.0031037330627441, -1.451435923576355, -0.6316400766372681, -6.758179187774658, -11.102059364318848, 0.28165510296821594, 0.339445561170578, 0.5525498390197754, -10.61474323272705, -9.53952693939209, -0.5830989480018616, -2.6421897411346436, 0.193865105509758, 17.054481506347656, 0.10585127025842667, -4.905209064483643, -0.09578622132539749, -2.0251624584198, -0.002207318553701043, -3.6981682777404785, -0.008171484805643559, -4.072200298309326, -0.04823870584368706, -0.3410889506340027, -0.30491724610328674, -0.20680515468120575]])
    b_h = np.array([-0.3386252224445343, -4.031388282775879])
    w_o = np.array([[0.49254393577575684, 1.1246771812438965]])
    b_o = np.array(-8.544547080993652)

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
        model_cap = 63
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

