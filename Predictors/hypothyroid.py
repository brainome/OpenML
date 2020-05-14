#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is 
# licensed under GNU GPL v2.0 or higher. For details, please see: 
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.93 Table Compiler v0.94.
# Invocation: btc https://www.openml.org/data/get_csv/53534/hypothyroid.arff -o Predictors/hypothyroid_NN.py -target binaryClass -stopat 99.76 -f NN -e 20 --yes
# Total compiler execution time: 0:38:42.04. Finished on: Apr-21-2020 18:48:57.
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                92.28%
Model accuracy:                     99.07% (3737/3772 correct)
Improvement over best guess:        6.79% (of possible 7.72%)
Model capacity (MEC):               94 bits
Generalization ratio:               39.75 bits/bit
Model efficiency:                   0.07%/parameter
System behavior
True Negatives:                     91.97% (3469/3772)
True Positives:                     7.10% (268/3772)
False Negatives:                    0.61% (23/3772)
False Positives:                    0.32% (12/3772)
True Pos. Rate/Sensitivity/Recall:  0.92
True Neg. Rate/Specificity:         1.00
Precision:                          0.96
F-1 Measure:                        0.94
False Negative Rate/Miss Rate:      0.08
Critical Success Index:             0.88

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
TRAINFILE = "hypothyroid.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29
n_classes = 2

mappings = [{1.0: 0, 2.0: 1, 4.0: 2, 5.0: 3, 7.0: 4, 8.0: 5, 10.0: 6, 11.0: 7, 12.0: 8, 13.0: 9, 14.0: 10, 15.0: 11, 16.0: 12, 17.0: 13, 18.0: 14, 19.0: 15, 20.0: 16, 21.0: 17, 22.0: 18, 23.0: 19, 24.0: 20, 25.0: 21, 26.0: 22, 27.0: 23, 28.0: 24, 29.0: 25, 30.0: 26, 31.0: 27, 32.0: 28, 33.0: 29, 34.0: 30, 35.0: 31, 36.0: 32, 37.0: 33, 38.0: 34, 39.0: 35, 40.0: 36, 41.0: 37, 42.0: 38, 43.0: 39, 44.0: 40, 45.0: 41, 46.0: 42, 47.0: 43, 48.0: 44, 49.0: 45, 50.0: 46, 51.0: 47, 52.0: 48, 53.0: 49, 54.0: 50, 55.0: 51, 56.0: 52, 57.0: 53, 58.0: 54, 59.0: 55, 60.0: 56, 61.0: 57, 62.0: 58, 63.0: 59, 64.0: 60, 65.0: 61, 66.0: 62, 67.0: 63, 68.0: 64, 69.0: 65, 70.0: 66, 71.0: 67, 72.0: 68, 73.0: 69, 74.0: 70, 75.0: 71, 76.0: 72, 77.0: 73, 78.0: 74, 79.0: 75, 80.0: 76, 81.0: 77, 82.0: 78, 83.0: 79, 84.0: 80, 85.0: 81, 86.0: 82, 87.0: 83, 88.0: 84, 89.0: 85, 90.0: 86, 91.0: 87, 92.0: 88, 94.0: 89, 455.0: 90, 1684325040.0: 91, 93.0: 92, 6.0: 93}, {1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.17: 22, 0.18: 23, 0.19: 24, 0.2: 25, 0.21: 26, 0.22: 27, 0.23: 28, 0.24: 29, 0.25: 30, 0.26: 31, 0.27: 32, 0.29: 33, 0.3: 34, 0.31: 35, 0.32: 36, 0.33: 37, 0.34: 38, 0.35: 39, 0.36: 40, 0.37: 41, 0.38: 42, 0.39: 43, 0.4: 44, 0.41: 45, 0.42: 46, 0.43: 47, 0.44: 48, 0.45: 49, 0.46: 50, 0.47: 51, 0.49: 52, 0.5: 53, 0.51: 54, 0.52: 55, 0.53: 56, 0.54: 57, 0.55: 58, 0.56: 59, 0.57: 60, 0.58: 61, 0.59: 62, 0.6: 63, 0.61: 64, 0.62: 65, 0.63: 66, 0.64: 67, 0.65: 68, 0.66: 69, 0.67: 70, 0.68: 71, 0.69: 72, 0.7: 73, 0.71: 74, 0.72: 75, 0.73: 76, 0.74: 77, 0.75: 78, 0.76: 79, 0.77: 80, 0.78: 81, 0.79: 82, 0.8: 83, 0.81: 84, 0.82: 85, 0.83: 86, 0.84: 87, 0.85: 88, 0.86: 89, 0.87: 90, 0.88: 91, 0.89: 92, 0.9: 93, 0.91: 94, 0.92: 95, 0.93: 96, 0.94: 97, 0.95: 98, 0.96: 99, 0.97: 100, 0.98: 101, 0.99: 102, 1.0: 103, 1.01: 104, 1.1: 105, 1.2: 106, 1.3: 107, 1.4: 108, 1.5: 109, 1.6: 110, 1.7: 111, 1.8: 112, 1.9: 113, 2.0: 114, 2.1: 115, 2.2: 116, 2.3: 117, 2.4: 118, 2.5: 119, 2.6: 120, 2.7: 121, 2.8: 122, 2.9: 123, 3.0: 124, 3.1: 125, 3.2: 126, 3.3: 127, 3.4: 128, 3.5: 129, 3.6: 130, 3.7: 131, 3.8: 132, 3.9: 133, 4.0: 134, 4.1: 135, 4.2: 136, 4.3: 137, 4.4: 138, 4.5: 139, 4.6: 140, 4.7: 141, 4.8: 142, 4.9: 143, 5.0: 144, 5.1: 145, 5.2: 146, 5.3: 147, 5.4: 148, 5.5: 149, 5.6: 150, 5.7: 151, 5.73: 152, 5.8: 153, 5.9: 154, 6.0: 155, 6.1: 156, 6.2: 157, 6.3: 158, 6.5: 159, 6.6: 160, 6.7: 161, 6.8: 162, 6.9: 163, 7.0: 164, 7.1: 165, 7.2: 166, 7.3: 167, 7.4: 168, 7.5: 169, 7.6: 170, 7.7: 171, 7.8: 172, 7.9: 173, 8.0: 174, 8.1: 175, 8.2: 176, 8.3: 177, 8.4: 178, 8.5: 179, 8.6: 180, 8.8: 181, 8.9: 182, 9.0: 183, 9.1: 184, 9.2: 185, 9.3: 186, 9.4: 187, 9.5: 188, 9.6: 189, 9.7: 190, 9.9: 191, 10.0: 192, 10.3: 193, 11.0: 194, 11.1: 195, 11.4: 196, 12.0: 197, 13.0: 198, 14.0: 199, 14.8: 200, 15.0: 201, 16.0: 202, 17.0: 203, 18.0: 204, 19.0: 205, 20.0: 206, 21.0: 207, 22.0: 208, 24.0: 209, 25.0: 210, 26.0: 211, 26.4: 212, 27.0: 213, 28.0: 214, 30.0: 215, 30.5: 216, 31.0: 217, 34.0: 218, 35.0: 219, 36.0: 220, 39.0: 221, 40.0: 222, 41.0: 223, 42.0: 224, 43.0: 225, 44.0: 226, 45.0: 227, 47.0: 228, 50.0: 229, 54.0: 230, 55.0: 231, 58.0: 232, 60.0: 233, 66.0: 234, 70.0: 235, 78.0: 236, 86.0: 237, 98.0: 238, 99.0: 239, 100.0: 240, 103.0: 241, 108.0: 242, 109.0: 243, 116.0: 244, 117.0: 245, 126.0: 246, 139.0: 247, 145.0: 248, 151.0: 249, 160.0: 250, 165.0: 251, 183.0: 252, 188.0: 253, 230.0: 254, 236.0: 255, 400.0: 256, 440.0: 257, 530.0: 258, 1684325040.0: 259, 23.0: 260, 0.48: 261, 0.28: 262, 14.4: 263, 89.0: 264, 9.8: 265, 80.0: 266, 1.02: 267, 65.0: 268, 38.0: 269, 12.1: 270, 199.0: 271, 6.4: 272, 82.0: 273, 468.0: 274, 178.0: 275, 29.0: 276, 478.0: 277, 52.0: 278, 76.0: 279, 33.0: 280, 46.0: 281, 143.0: 282, 472.0: 283, 61.0: 284, 18.4: 285, 32.0: 286, 51.0: 287}, {1993550816.0: 0, 2238339752.0: 1}, {0.05: 0, 0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9, 1.0: 10, 1.1: 11, 1.2: 12, 1.3: 13, 1.4: 14, 1.44: 15, 1.5: 16, 1.6: 17, 1.7: 18, 1.8: 19, 1.9: 20, 2.0: 21, 2.1: 22, 2.2: 23, 2.3: 24, 2.4: 25, 2.5: 26, 2.6: 27, 2.7: 28, 2.8: 29, 2.9: 30, 3.0: 31, 3.1: 32, 3.2: 33, 3.3: 34, 3.4: 35, 3.5: 36, 3.6: 37, 3.7: 38, 3.8: 39, 3.9: 40, 4.0: 41, 4.1: 42, 4.2: 43, 4.3: 44, 4.4: 45, 4.5: 46, 4.6: 47, 4.7: 48, 4.8: 49, 4.9: 50, 5.0: 51, 5.3: 52, 5.4: 53, 5.5: 54, 6.0: 55, 6.2: 56, 6.6: 57, 6.7: 58, 7.0: 59, 7.1: 60, 7.3: 61, 8.5: 62, 10.6: 63, 1684325040.0: 64, 5.2: 65, 5.7: 66, 5.1: 67, 7.6: 68, 6.1: 69}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.9: 1, 3.0: 2, 4.0: 3, 4.8: 4, 5.8: 5, 6.0: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 16.0: 13, 17.0: 14, 18.0: 15, 22.0: 16, 24.0: 17, 25.0: 18, 27.0: 19, 30.0: 20, 31.0: 21, 32.0: 22, 33.0: 23, 35.0: 24, 36.0: 25, 37.0: 26, 38.0: 27, 39.0: 28, 40.0: 29, 42.0: 30, 43.0: 31, 44.0: 32, 45.0: 33, 46.0: 34, 48.0: 35, 49.0: 36, 50.0: 37, 51.0: 38, 52.0: 39, 53.0: 40, 54.0: 41, 56.0: 42, 57.0: 43, 58.0: 44, 59.0: 45, 60.0: 46, 61.0: 47, 62.0: 48, 63.0: 49, 64.0: 50, 65.0: 51, 66.0: 52, 67.0: 53, 68.0: 54, 69.0: 55, 70.0: 56, 71.0: 57, 72.0: 58, 73.0: 59, 74.0: 60, 75.0: 61, 76.0: 62, 77.0: 63, 78.0: 64, 79.0: 65, 80.0: 66, 81.0: 67, 82.0: 68, 83.0: 69, 84.0: 70, 85.0: 71, 86.0: 72, 87.0: 73, 88.0: 74, 89.0: 75, 90.0: 76, 91.0: 77, 92.0: 78, 93.0: 79, 94.0: 80, 95.0: 81, 96.0: 82, 97.0: 83, 98.0: 84, 99.0: 85, 100.0: 86, 101.0: 87, 102.0: 88, 103.0: 89, 104.0: 90, 105.0: 91, 106.0: 92, 107.0: 93, 108.0: 94, 109.0: 95, 110.0: 96, 111.0: 97, 112.0: 98, 113.0: 99, 114.0: 100, 115.0: 101, 116.0: 102, 117.0: 103, 118.0: 104, 119.0: 105, 120.0: 106, 121.0: 107, 122.0: 108, 123.0: 109, 124.0: 110, 125.0: 111, 126.0: 112, 127.0: 113, 128.0: 114, 129.0: 115, 130.0: 116, 131.0: 117, 132.0: 118, 133.0: 119, 134.0: 120, 135.0: 121, 136.0: 122, 137.0: 123, 138.0: 124, 139.0: 125, 140.0: 126, 141.0: 127, 142.0: 128, 143.0: 129, 144.0: 130, 145.0: 131, 146.0: 132, 147.0: 133, 148.0: 134, 149.0: 135, 150.0: 136, 151.0: 137, 152.0: 138, 153.0: 139, 154.0: 140, 155.0: 141, 156.0: 142, 157.0: 143, 158.0: 144, 159.0: 145, 160.0: 146, 161.0: 147, 162.0: 148, 163.0: 149, 164.0: 150, 165.0: 151, 166.0: 152, 167.0: 153, 168.0: 154, 169.0: 155, 170.0: 156, 171.0: 157, 172.0: 158, 174.0: 159, 175.0: 160, 176.0: 161, 177.0: 162, 178.0: 163, 179.0: 164, 180.0: 165, 181.0: 166, 182.0: 167, 183.0: 168, 184.0: 169, 186.0: 170, 187.0: 171, 188.0: 172, 189.0: 173, 191.0: 174, 192.0: 175, 193.0: 176, 194.0: 177, 195.0: 178, 197.0: 179, 198.0: 180, 199.0: 181, 200.0: 182, 201.0: 183, 204.0: 184, 205.0: 185, 206.0: 186, 209.0: 187, 210.0: 188, 212.0: 189, 213.0: 190, 214.0: 191, 216.0: 192, 217.0: 193, 219.0: 194, 220.0: 195, 223.0: 196, 225.0: 197, 226.0: 198, 230.0: 199, 231.0: 200, 232.0: 201, 233.0: 202, 235.0: 203, 237.0: 204, 239.0: 205, 244.0: 206, 246.0: 207, 248.0: 208, 250.0: 209, 252.0: 210, 253.0: 211, 255.0: 212, 256.0: 213, 258.0: 214, 261.0: 215, 263.0: 216, 272.0: 217, 273.0: 218, 301.0: 219, 430.0: 220, 1684325040.0: 221, 207.0: 222, 257.0: 223, 41.0: 224, 28.0: 225, 372.0: 226, 9.5: 227, 21.0: 228, 47.0: 229, 173.0: 230, 23.0: 231, 222.0: 232, 289.0: 233, 19.0: 234, 29.0: 235, 196.0: 236, 203.0: 237, 55.0: 238, 34.0: 239, 240.0: 240, 211.0: 241}, {1993550816.0: 0, 2238339752.0: 1}, {0.31: 0, 0.38: 1, 0.41: 2, 0.48: 3, 0.5: 4, 0.52: 5, 0.53: 6, 0.54: 7, 0.56: 8, 0.57: 9, 0.58: 10, 0.59: 11, 0.6: 12, 0.61: 13, 0.62: 14, 0.63: 15, 0.64: 16, 0.65: 17, 0.66: 18, 0.67: 19, 0.68: 20, 0.69: 21, 0.7: 22, 0.71: 23, 0.72: 24, 0.73: 25, 0.74: 26, 0.75: 27, 0.76: 28, 0.77: 29, 0.78: 30, 0.79: 31, 0.8: 32, 0.81: 33, 0.82: 34, 0.83: 35, 0.84: 36, 0.85: 37, 0.86: 38, 0.87: 39, 0.88: 40, 0.89: 41, 0.9: 42, 0.91: 43, 0.92: 44, 0.93: 45, 0.94: 46, 0.95: 47, 0.96: 48, 0.97: 49, 0.98: 50, 0.99: 51, 1.0: 52, 1.01: 53, 1.02: 54, 1.03: 55, 1.04: 56, 1.05: 57, 1.06: 58, 1.07: 59, 1.08: 60, 1.09: 61, 1.1: 62, 1.11: 63, 1.12: 64, 1.13: 65, 1.14: 66, 1.15: 67, 1.16: 68, 1.17: 69, 1.18: 70, 1.19: 71, 1.2: 72, 1.21: 73, 1.22: 74, 1.23: 75, 1.24: 76, 1.25: 77, 1.26: 78, 1.27: 79, 1.28: 80, 1.29: 81, 1.3: 82, 1.31: 83, 1.32: 84, 1.33: 85, 1.34: 86, 1.35: 87, 1.36: 88, 1.38: 89, 1.39: 90, 1.4: 91, 1.41: 92, 1.42: 93, 1.43: 94, 1.44: 95, 1.45: 96, 1.46: 97, 1.47: 98, 1.48: 99, 1.49: 100, 1.5: 101, 1.51: 102, 1.52: 103, 1.53: 104, 1.55: 105, 1.56: 106, 1.57: 107, 1.58: 108, 1.59: 109, 1.62: 110, 1.63: 111, 1.65: 112, 1.66: 113, 1.67: 114, 1.68: 115, 1.69: 116, 1.7: 117, 1.71: 118, 1.73: 119, 1.74: 120, 1.75: 121, 1.76: 122, 1.77: 123, 1.8: 124, 1.82: 125, 1.83: 126, 1.93: 127, 1.94: 128, 1.97: 129, 2.03: 130, 2.12: 131, 2.32: 132, 1684325040.0: 133, 1.84: 134, 0.49: 135, 0.46: 136, 2.01: 137, 0.25: 138, 1.37: 139, 1.79: 140, 1.61: 141, 0.36: 142, 1.88: 143, 0.47: 144, 0.944: 145, 1.54: 146}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.8: 1, 3.0: 2, 4.0: 3, 5.4: 4, 7.0: 5, 7.6: 6, 8.4: 7, 8.5: 8, 8.9: 9, 9.1: 10, 10.0: 11, 11.0: 12, 13.0: 13, 14.0: 14, 15.0: 15, 16.0: 16, 17.0: 17, 18.0: 18, 20.0: 19, 21.0: 20, 24.0: 21, 27.0: 22, 32.0: 23, 33.0: 24, 35.0: 25, 37.0: 26, 39.0: 27, 40.0: 28, 41.0: 29, 42.0: 30, 46.0: 31, 47.0: 32, 49.0: 33, 50.0: 34, 51.0: 35, 52.0: 36, 53.0: 37, 54.0: 38, 55.0: 39, 56.0: 40, 57.0: 41, 58.0: 42, 59.0: 43, 60.0: 44, 61.0: 45, 62.0: 46, 63.0: 47, 64.0: 48, 65.0: 49, 66.0: 50, 67.0: 51, 68.0: 52, 69.0: 53, 70.0: 54, 71.0: 55, 72.0: 56, 73.0: 57, 74.0: 58, 75.0: 59, 76.0: 60, 77.0: 61, 78.0: 62, 79.0: 63, 80.0: 64, 81.0: 65, 82.0: 66, 83.0: 67, 84.0: 68, 85.0: 69, 86.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 93.0: 77, 94.0: 78, 95.0: 79, 96.0: 80, 97.0: 81, 98.0: 82, 99.0: 83, 100.0: 84, 101.0: 85, 102.0: 86, 103.0: 87, 104.0: 88, 105.0: 89, 106.0: 90, 107.0: 91, 108.0: 92, 109.0: 93, 110.0: 94, 111.0: 95, 112.0: 96, 113.0: 97, 114.0: 98, 115.0: 99, 116.0: 100, 117.0: 101, 118.0: 102, 119.0: 103, 120.0: 104, 121.0: 105, 122.0: 106, 123.0: 107, 124.0: 108, 125.0: 109, 126.0: 110, 127.0: 111, 128.0: 112, 129.0: 113, 130.0: 114, 131.0: 115, 132.0: 116, 133.0: 117, 134.0: 118, 135.0: 119, 136.0: 120, 137.0: 121, 138.0: 122, 139.0: 123, 140.0: 124, 141.0: 125, 142.0: 126, 143.0: 127, 144.0: 128, 145.0: 129, 146.0: 130, 147.0: 131, 148.0: 132, 149.0: 133, 150.0: 134, 151.0: 135, 152.0: 136, 153.0: 137, 154.0: 138, 155.0: 139, 156.0: 140, 157.0: 141, 158.0: 142, 159.0: 143, 160.0: 144, 161.0: 145, 162.0: 146, 163.0: 147, 164.0: 148, 165.0: 149, 166.0: 150, 167.0: 151, 168.0: 152, 169.0: 153, 170.0: 154, 171.0: 155, 172.0: 156, 173.0: 157, 174.0: 158, 175.0: 159, 176.0: 160, 177.0: 161, 178.0: 162, 179.0: 163, 180.0: 164, 183.0: 165, 184.0: 166, 185.0: 167, 186.0: 168, 187.0: 169, 188.0: 170, 189.0: 171, 190.0: 172, 191.0: 173, 194.0: 174, 195.0: 175, 196.0: 176, 197.0: 177, 198.0: 178, 199.0: 179, 200.0: 180, 203.0: 181, 204.0: 182, 205.0: 183, 206.0: 184, 207.0: 185, 209.0: 186, 210.0: 187, 213.0: 188, 214.0: 189, 215.0: 190, 216.0: 191, 217.0: 192, 218.0: 193, 221.0: 194, 222.0: 195, 227.0: 196, 232.0: 197, 235.0: 198, 237.0: 199, 242.0: 200, 244.0: 201, 247.0: 202, 249.0: 203, 251.0: 204, 253.0: 205, 265.0: 206, 274.0: 207, 280.0: 208, 281.0: 209, 283.0: 210, 312.0: 211, 349.0: 212, 362.0: 213, 395.0: 214, 1684325040.0: 215, 28.0: 216, 220.0: 217, 181.0: 218, 9.0: 219, 36.0: 220, 291.0: 221, 182.0: 222, 19.0: 223, 48.0: 224, 34.0: 225, 43.0: 226, 26.0: 227, 228.0: 228, 29.0: 229, 224.0: 230, 201.0: 231, 219.0: 232, 245.0: 233, 223.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

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
    h_0 = max((((-22.050417 * float(x[0]))+ (0.20552675 * float(x[1]))+ (0.08976637 * float(x[2]))+ (-0.1526904 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.124825574 * float(x[5]))+ (0.78354603 * float(x[6]))+ (0.92732555 * float(x[7]))+ (-0.23346941 * float(x[8]))+ (0.5834501 * float(x[9]))+ (0.05778984 * float(x[10]))+ (0.13608912 * float(x[11]))+ (0.85119325 * float(x[12]))+ (-0.85792786 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (-0.9595632 * float(x[15]))+ (0.36547226 * float(x[16]))+ (-33.614002 * float(x[17]))+ (0.44025686 * float(x[18]))+ (-2.6445544 * float(x[19]))+ (0.29854968 * float(x[20]))+ (-22.262299 * float(x[21]))+ (0.2612909 * float(x[22]))+ (-11.863305 * float(x[23]))+ (-0.019925395 * float(x[24]))+ (-27.086483 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-0.7705634 * float(x[28]))) + -0.20214044), 0)
    h_1 = max((((0.0006039885 * float(x[0]))+ (-0.04277582 * float(x[1]))+ (7.7004356 * float(x[2]))+ (1.7322677 * float(x[3]))+ (3.4036293 * float(x[4]))+ (-2.6684968 * float(x[5]))+ (9.602706 * float(x[6]))+ (6.9966183 * float(x[7]))+ (-2.4073725 * float(x[8]))+ (-0.81525797 * float(x[9]))+ (1.0902576 * float(x[10]))+ (-3.0962996 * float(x[11]))+ (8.519222 * float(x[12]))+ (-0.05819207 * float(x[13]))+ (-0.31029496 * float(x[14]))+ (2.419761 * float(x[15]))+ (-10.760323 * float(x[16]))+ (-0.023496058 * float(x[17]))+ (-1.2372999 * float(x[18]))+ (-0.03909717 * float(x[19]))+ (3.4672256 * float(x[20]))+ (0.028175702 * float(x[21]))+ (1.9848782 * float(x[22]))+ (0.039351057 * float(x[23]))+ (1.3290975 * float(x[24]))+ (0.032713518 * float(x[25]))+ (-0.5008052 * float(x[26]))+ (-0.3295877 * float(x[27]))+ (0.2529845 * float(x[28]))) + -0.24284995), 0)
    h_2 = max((((0.0013867386 * float(x[0]))+ (0.24480917 * float(x[1]))+ (3.2013738 * float(x[2]))+ (1.8518745 * float(x[3]))+ (1.438293 * float(x[4]))+ (1.6154524 * float(x[5]))+ (1.9754236 * float(x[6]))+ (3.0606406 * float(x[7]))+ (0.80660033 * float(x[8]))+ (0.31938177 * float(x[9]))+ (-0.34866592 * float(x[10]))+ (0.57039684 * float(x[11]))+ (3.3555481 * float(x[12]))+ (0.80185306 * float(x[13]))+ (2.0708635 * float(x[14]))+ (0.1110068 * float(x[15]))+ (2.6752727 * float(x[16]))+ (-0.11437927 * float(x[17]))+ (2.973258 * float(x[18]))+ (0.054529443 * float(x[19]))+ (4.020701 * float(x[20]))+ (0.052667134 * float(x[21]))+ (2.9636288 * float(x[22]))+ (-0.05081639 * float(x[23]))+ (3.6002584 * float(x[24]))+ (-0.045832608 * float(x[25]))+ (-0.12777132 * float(x[26]))+ (0.2086929 * float(x[27]))+ (0.017885162 * float(x[28]))) + 2.9557862), 0)
    o[0] = (0.05652739 * h_0)+ (-3.0496848 * h_1)+ (-5.7085133 * h_2) + 4.1781063

    

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
        model_cap=94
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
