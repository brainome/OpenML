#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.91.
# Compile time: Mar-19-2020 23:43:46
# Invocation: btc -server brain.brainome.ai Data/sick.csv -o Models/sick.py -v -v -v -stopat 98.75 -port 8100 -f NN -e 10
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                93.87%
Model accuracy:                     97.00% (3659/3772 correct)
Improvement over best guess:        3.13% (of possible 6.13%)
Model capacity (MEC):               66 bits
Generalization ratio:               55.43 bits/bit
Model efficiency:                   0.04%/parameter
System behavior
True Negatives:                     92.87% (3503/3772)
True Positives:                     4.14% (156/3772)
False Negatives:                    1.99% (75/3772)
False Positives:                    1.01% (38/3772)
True Pos. Rate/Sensitivity/Recall:  0.68
True Neg. Rate/Specificity:         0.99
Precision:                          0.80
F-1 Measure:                        0.73
False Negative Rate/Miss Rate:      0.32
Critical Success Index:             0.58

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
TRAINFILE="sick.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29
n_classes = 2

mappings = [{1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.19: 22, 0.2: 23, 0.21: 24, 0.24: 25, 0.25: 26, 0.26: 27, 0.27: 28, 0.28: 29, 0.29: 30, 0.3: 31, 0.31: 32, 0.32: 33, 0.33: 34, 0.34: 35, 0.35: 36, 0.36: 37, 0.37: 38, 0.38: 39, 0.39: 40, 0.4: 41, 0.41: 42, 0.42: 43, 0.43: 44, 0.44: 45, 0.45: 46, 0.46: 47, 0.47: 48, 0.48: 49, 0.49: 50, 0.5: 51, 0.51: 52, 0.52: 53, 0.53: 54, 0.54: 55, 0.55: 56, 0.56: 57, 0.57: 58, 0.58: 59, 0.59: 60, 0.6: 61, 0.61: 62, 0.62: 63, 0.63: 64, 0.64: 65, 0.65: 66, 0.66: 67, 0.67: 68, 0.68: 69, 0.69: 70, 0.7: 71, 0.71: 72, 0.72: 73, 0.73: 74, 0.74: 75, 0.75: 76, 0.76: 77, 0.77: 78, 0.78: 79, 0.79: 80, 0.8: 81, 0.81: 82, 0.82: 83, 0.83: 84, 0.84: 85, 0.85: 86, 0.86: 87, 0.87: 88, 0.88: 89, 0.89: 90, 0.9: 91, 0.91: 92, 0.92: 93, 0.93: 94, 0.94: 95, 0.95: 96, 0.96: 97, 0.97: 98, 0.98: 99, 0.99: 100, 1.0: 101, 1.01: 102, 1.02: 103, 1.1: 104, 1.2: 105, 1.3: 106, 1.4: 107, 1.5: 108, 1.6: 109, 1.7: 110, 1.8: 111, 1.9: 112, 2.0: 113, 2.1: 114, 2.2: 115, 2.3: 116, 2.4: 117, 2.5: 118, 2.6: 119, 2.7: 120, 2.8: 121, 2.9: 122, 3.0: 123, 3.1: 124, 3.2: 125, 3.3: 126, 3.4: 127, 3.5: 128, 3.6: 129, 3.7: 130, 3.8: 131, 3.9: 132, 4.0: 133, 4.1: 134, 4.2: 135, 4.3: 136, 4.4: 137, 4.5: 138, 4.6: 139, 4.7: 140, 4.8: 141, 4.9: 142, 5.0: 143, 5.1: 144, 5.2: 145, 5.3: 146, 5.4: 147, 5.5: 148, 5.6: 149, 5.7: 150, 5.73: 151, 5.8: 152, 5.9: 153, 6.0: 154, 6.1: 155, 6.2: 156, 6.3: 157, 6.5: 158, 6.6: 159, 6.7: 160, 6.8: 161, 6.9: 162, 7.0: 163, 7.1: 164, 7.2: 165, 7.3: 166, 7.4: 167, 7.5: 168, 7.6: 169, 7.7: 170, 7.8: 171, 7.9: 172, 8.0: 173, 8.1: 174, 8.2: 175, 8.3: 176, 8.5: 177, 8.6: 178, 8.8: 179, 8.9: 180, 9.0: 181, 9.1: 182, 9.2: 183, 9.3: 184, 9.4: 185, 9.6: 186, 9.7: 187, 9.8: 188, 9.9: 189, 10.0: 190, 10.3: 191, 11.0: 192, 11.1: 193, 11.4: 194, 12.0: 195, 13.0: 196, 14.4: 197, 14.8: 198, 15.0: 199, 16.0: 200, 17.0: 201, 18.0: 202, 19.0: 203, 20.0: 204, 21.0: 205, 22.0: 206, 23.0: 207, 24.0: 208, 25.0: 209, 26.0: 210, 26.4: 211, 27.0: 212, 30.0: 213, 31.0: 214, 34.0: 215, 36.0: 216, 38.0: 217, 40.0: 218, 42.0: 219, 43.0: 220, 44.0: 221, 45.0: 222, 47.0: 223, 50.0: 224, 51.0: 225, 52.0: 226, 55.0: 227, 58.0: 228, 60.0: 229, 61.0: 230, 66.0: 231, 70.0: 232, 76.0: 233, 82.0: 234, 89.0: 235, 98.0: 236, 103.0: 237, 108.0: 238, 109.0: 239, 116.0: 240, 117.0: 241, 126.0: 242, 143.0: 243, 145.0: 244, 151.0: 245, 160.0: 246, 178.0: 247, 183.0: 248, 400.0: 249, 440.0: 250, 468.0: 251, 472.0: 252, 478.0: 253, 530.0: 254, 1684325040.0: 255, 32.0: 256, 0.17: 257, 46.0: 258, 236.0: 259, 14.0: 260, 80.0: 261, 9.5: 262, 65.0: 263, 28.0: 264, 0.22: 265, 29.0: 266, 6.4: 267, 139.0: 268, 230.0: 269, 54.0: 270, 30.5: 271, 12.1: 272, 41.0: 273, 8.4: 274, 18.4: 275, 0.18: 276, 78.0: 277, 35.0: 278, 86.0: 279, 39.0: 280, 33.0: 281, 100.0: 282, 0.23: 283, 99.0: 284, 165.0: 285, 199.0: 286, 188.0: 287}, {1993550816.0: 0, 2238339752.0: 1}, {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 1.1: 10, 1.2: 11, 1.3: 12, 1.4: 13, 1.5: 14, 1.6: 15, 1.7: 16, 1.8: 17, 1.9: 18, 2.0: 19, 2.1: 20, 2.2: 21, 2.3: 22, 2.4: 23, 2.5: 24, 2.6: 25, 2.7: 26, 2.8: 27, 2.9: 28, 3.0: 29, 3.1: 30, 3.2: 31, 3.3: 32, 3.4: 33, 3.5: 34, 3.6: 35, 3.7: 36, 3.8: 37, 3.9: 38, 4.0: 39, 4.1: 40, 4.2: 41, 4.3: 42, 4.4: 43, 4.5: 44, 4.6: 45, 4.7: 46, 4.8: 47, 4.9: 48, 5.0: 49, 5.1: 50, 5.2: 51, 5.3: 52, 5.4: 53, 5.5: 54, 5.7: 55, 6.1: 56, 6.2: 57, 6.6: 58, 6.7: 59, 7.0: 60, 7.1: 61, 7.3: 62, 7.6: 63, 8.5: 64, 1684325040.0: 65, 6.0: 66, 1.44: 67, 0.05: 68, 10.6: 69}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.9: 1, 3.0: 2, 4.8: 3, 5.8: 4, 6.0: 5, 9.5: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 17.0: 13, 19.0: 14, 21.0: 15, 22.0: 16, 24.0: 17, 28.0: 18, 29.0: 19, 30.0: 20, 31.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 37.0: 25, 38.0: 26, 39.0: 27, 40.0: 28, 41.0: 29, 42.0: 30, 44.0: 31, 45.0: 32, 46.0: 33, 48.0: 34, 49.0: 35, 50.0: 36, 51.0: 37, 52.0: 38, 53.0: 39, 54.0: 40, 56.0: 41, 57.0: 42, 58.0: 43, 59.0: 44, 60.0: 45, 61.0: 46, 62.0: 47, 63.0: 48, 64.0: 49, 65.0: 50, 66.0: 51, 67.0: 52, 68.0: 53, 69.0: 54, 70.0: 55, 71.0: 56, 72.0: 57, 73.0: 58, 74.0: 59, 75.0: 60, 76.0: 61, 77.0: 62, 78.0: 63, 79.0: 64, 80.0: 65, 81.0: 66, 82.0: 67, 83.0: 68, 84.0: 69, 85.0: 70, 86.0: 71, 87.0: 72, 88.0: 73, 89.0: 74, 90.0: 75, 91.0: 76, 92.0: 77, 93.0: 78, 94.0: 79, 95.0: 80, 96.0: 81, 97.0: 82, 98.0: 83, 99.0: 84, 100.0: 85, 101.0: 86, 102.0: 87, 103.0: 88, 104.0: 89, 105.0: 90, 106.0: 91, 107.0: 92, 108.0: 93, 109.0: 94, 110.0: 95, 111.0: 96, 112.0: 97, 113.0: 98, 114.0: 99, 115.0: 100, 116.0: 101, 117.0: 102, 118.0: 103, 119.0: 104, 120.0: 105, 121.0: 106, 122.0: 107, 123.0: 108, 124.0: 109, 125.0: 110, 126.0: 111, 127.0: 112, 128.0: 113, 129.0: 114, 130.0: 115, 131.0: 116, 132.0: 117, 133.0: 118, 134.0: 119, 135.0: 120, 136.0: 121, 137.0: 122, 138.0: 123, 139.0: 124, 140.0: 125, 141.0: 126, 142.0: 127, 143.0: 128, 144.0: 129, 145.0: 130, 146.0: 131, 147.0: 132, 148.0: 133, 149.0: 134, 150.0: 135, 151.0: 136, 152.0: 137, 153.0: 138, 154.0: 139, 155.0: 140, 156.0: 141, 157.0: 142, 158.0: 143, 159.0: 144, 160.0: 145, 161.0: 146, 162.0: 147, 163.0: 148, 164.0: 149, 165.0: 150, 166.0: 151, 167.0: 152, 168.0: 153, 169.0: 154, 170.0: 155, 171.0: 156, 172.0: 157, 174.0: 158, 175.0: 159, 176.0: 160, 177.0: 161, 178.0: 162, 179.0: 163, 180.0: 164, 181.0: 165, 182.0: 166, 183.0: 167, 184.0: 168, 186.0: 169, 187.0: 170, 189.0: 171, 191.0: 172, 192.0: 173, 193.0: 174, 194.0: 175, 195.0: 176, 196.0: 177, 197.0: 178, 198.0: 179, 199.0: 180, 200.0: 181, 201.0: 182, 203.0: 183, 205.0: 184, 207.0: 185, 209.0: 186, 210.0: 187, 211.0: 188, 212.0: 189, 213.0: 190, 214.0: 191, 220.0: 192, 223.0: 193, 225.0: 194, 230.0: 195, 231.0: 196, 232.0: 197, 237.0: 198, 239.0: 199, 240.0: 200, 244.0: 201, 248.0: 202, 252.0: 203, 253.0: 204, 255.0: 205, 257.0: 206, 258.0: 207, 261.0: 208, 263.0: 209, 289.0: 210, 372.0: 211, 430.0: 212, 1684325040.0: 213, 32.0: 214, 204.0: 215, 16.0: 216, 250.0: 217, 273.0: 218, 18.0: 219, 217.0: 220, 216.0: 221, 4.0: 222, 47.0: 223, 36.0: 224, 222.0: 225, 188.0: 226, 206.0: 227, 23.0: 228, 272.0: 229, 233.0: 230, 27.0: 231, 173.0: 232, 226.0: 233, 301.0: 234, 256.0: 235, 246.0: 236, 25.0: 237, 235.0: 238, 43.0: 239, 55.0: 240, 219.0: 241}, {1993550816.0: 0, 2238339752.0: 1}, {0.31: 0, 0.38: 1, 0.48: 2, 0.5: 3, 0.52: 4, 0.54: 5, 0.56: 6, 0.57: 7, 0.58: 8, 0.59: 9, 0.6: 10, 0.61: 11, 0.62: 12, 0.63: 13, 0.64: 14, 0.65: 15, 0.66: 16, 0.67: 17, 0.68: 18, 0.69: 19, 0.7: 20, 0.71: 21, 0.72: 22, 0.73: 23, 0.74: 24, 0.75: 25, 0.76: 26, 0.77: 27, 0.78: 28, 0.79: 29, 0.8: 30, 0.81: 31, 0.82: 32, 0.83: 33, 0.84: 34, 0.85: 35, 0.86: 36, 0.87: 37, 0.88: 38, 0.89: 39, 0.9: 40, 0.91: 41, 0.92: 42, 0.93: 43, 0.94: 44, 0.95: 45, 0.96: 46, 0.97: 47, 0.98: 48, 0.99: 49, 1.0: 50, 1.01: 51, 1.02: 52, 1.03: 53, 1.04: 54, 1.05: 55, 1.06: 56, 1.07: 57, 1.08: 58, 1.09: 59, 1.1: 60, 1.11: 61, 1.12: 62, 1.13: 63, 1.14: 64, 1.15: 65, 1.16: 66, 1.17: 67, 1.18: 68, 1.19: 69, 1.2: 70, 1.21: 71, 1.22: 72, 1.23: 73, 1.24: 74, 1.25: 75, 1.26: 76, 1.27: 77, 1.28: 78, 1.29: 79, 1.3: 80, 1.31: 81, 1.32: 82, 1.33: 83, 1.34: 84, 1.35: 85, 1.36: 86, 1.37: 87, 1.38: 88, 1.39: 89, 1.4: 90, 1.41: 91, 1.42: 92, 1.43: 93, 1.44: 94, 1.45: 95, 1.46: 96, 1.47: 97, 1.48: 98, 1.49: 99, 1.5: 100, 1.51: 101, 1.52: 102, 1.53: 103, 1.54: 104, 1.55: 105, 1.56: 106, 1.57: 107, 1.58: 108, 1.59: 109, 1.61: 110, 1.62: 111, 1.65: 112, 1.66: 113, 1.67: 114, 1.68: 115, 1.69: 116, 1.7: 117, 1.71: 118, 1.73: 119, 1.75: 120, 1.76: 121, 1.77: 122, 1.8: 123, 1.82: 124, 1.83: 125, 1.84: 126, 1.88: 127, 1.93: 128, 1.97: 129, 2.01: 130, 2.12: 131, 2.32: 132, 1684325040.0: 133, 0.53: 134, 2.03: 135, 0.46: 136, 0.47: 137, 0.944: 138, 1.79: 139, 1.63: 140, 0.36: 141, 0.25: 142, 1.94: 143, 1.74: 144, 0.41: 145, 0.49: 146}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.8: 1, 3.0: 2, 5.4: 3, 7.0: 4, 7.6: 5, 8.4: 6, 8.5: 7, 8.9: 8, 9.0: 9, 9.1: 10, 10.0: 11, 11.0: 12, 13.0: 13, 14.0: 14, 15.0: 15, 17.0: 16, 18.0: 17, 19.0: 18, 21.0: 19, 26.0: 20, 29.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 37.0: 25, 39.0: 26, 42.0: 27, 46.0: 28, 47.0: 29, 49.0: 30, 50.0: 31, 51.0: 32, 52.0: 33, 54.0: 34, 55.0: 35, 56.0: 36, 57.0: 37, 58.0: 38, 59.0: 39, 60.0: 40, 61.0: 41, 62.0: 42, 63.0: 43, 64.0: 44, 65.0: 45, 66.0: 46, 67.0: 47, 68.0: 48, 69.0: 49, 70.0: 50, 71.0: 51, 72.0: 52, 73.0: 53, 74.0: 54, 75.0: 55, 76.0: 56, 77.0: 57, 78.0: 58, 79.0: 59, 80.0: 60, 81.0: 61, 82.0: 62, 83.0: 63, 84.0: 64, 85.0: 65, 86.0: 66, 87.0: 67, 88.0: 68, 89.0: 69, 90.0: 70, 91.0: 71, 92.0: 72, 93.0: 73, 94.0: 74, 95.0: 75, 96.0: 76, 97.0: 77, 98.0: 78, 99.0: 79, 100.0: 80, 101.0: 81, 102.0: 82, 103.0: 83, 104.0: 84, 105.0: 85, 106.0: 86, 107.0: 87, 108.0: 88, 109.0: 89, 110.0: 90, 111.0: 91, 112.0: 92, 113.0: 93, 114.0: 94, 115.0: 95, 116.0: 96, 117.0: 97, 118.0: 98, 119.0: 99, 120.0: 100, 121.0: 101, 122.0: 102, 123.0: 103, 124.0: 104, 125.0: 105, 126.0: 106, 127.0: 107, 128.0: 108, 129.0: 109, 130.0: 110, 131.0: 111, 132.0: 112, 133.0: 113, 134.0: 114, 135.0: 115, 136.0: 116, 137.0: 117, 138.0: 118, 139.0: 119, 140.0: 120, 141.0: 121, 142.0: 122, 143.0: 123, 144.0: 124, 145.0: 125, 146.0: 126, 147.0: 127, 148.0: 128, 149.0: 129, 150.0: 130, 151.0: 131, 152.0: 132, 153.0: 133, 154.0: 134, 155.0: 135, 156.0: 136, 157.0: 137, 158.0: 138, 159.0: 139, 160.0: 140, 161.0: 141, 162.0: 142, 163.0: 143, 164.0: 144, 165.0: 145, 166.0: 146, 167.0: 147, 168.0: 148, 169.0: 149, 170.0: 150, 171.0: 151, 172.0: 152, 173.0: 153, 174.0: 154, 175.0: 155, 176.0: 156, 177.0: 157, 178.0: 158, 179.0: 159, 180.0: 160, 183.0: 161, 184.0: 162, 185.0: 163, 186.0: 164, 188.0: 165, 189.0: 166, 190.0: 167, 191.0: 168, 194.0: 169, 195.0: 170, 196.0: 171, 197.0: 172, 198.0: 173, 200.0: 174, 201.0: 175, 203.0: 176, 204.0: 177, 206.0: 178, 207.0: 179, 209.0: 180, 213.0: 181, 214.0: 182, 215.0: 183, 216.0: 184, 217.0: 185, 219.0: 186, 220.0: 187, 222.0: 188, 223.0: 189, 224.0: 190, 227.0: 191, 228.0: 192, 235.0: 193, 237.0: 194, 244.0: 195, 249.0: 196, 251.0: 197, 265.0: 198, 280.0: 199, 291.0: 200, 362.0: 201, 395.0: 202, 1684325040.0: 203, 199.0: 204, 312.0: 205, 43.0: 206, 16.0: 207, 221.0: 208, 36.0: 209, 28.0: 210, 242.0: 211, 182.0: 212, 4.0: 213, 48.0: 214, 40.0: 215, 205.0: 216, 27.0: 217, 247.0: 218, 218.0: 219, 53.0: 220, 24.0: 221, 187.0: 222, 245.0: 223, 281.0: 224, 181.0: 225, 253.0: 226, 32.0: 227, 349.0: 228, 41.0: 229, 283.0: 230, 232.0: 231, 20.0: 232, 210.0: 233, 274.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
list_of_cols_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

transform_true = True

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
        mean = np.array([51.51568714096332, 0.6420680512593901, 0.12019443216968625, 0.012814847547503314, 0.013256738842244807, 0.0410958904109589, 0.015908086610693768, 0.011931064958020326, 0.015024304021210782, 0.059213433495360145, 0.059213433495360145, 0.005302695536897923, 0.009279717189571365, 0.02695536897923111, 0.0, 0.0481661511268228, 0.901458241272647, 109.39770216526735, 0.79584622182943, 28.498453380468405, 0.9385771100309324, 100.39681838267786, 0.8996906760936809, 57.788775961113565, 0.9001325673884224, 101.57136544410075, 0.0, 0.0, 2.427308882015024])
        components = np.array([array([-3.27648351e-02, -9.06339570e-04, -3.98421273e-04,  8.91337385e-05,
       -7.48839802e-05,  2.89532758e-07, -7.78466532e-05, -1.36648438e-05,
       -9.16471056e-05,  4.44619254e-05,  9.00297480e-06, -5.31074934e-06,
        2.72282643e-05,  3.62630878e-05,  0.00000000e+00, -1.05011617e-04,
       -3.07878701e-03,  9.26413548e-01, -2.03848105e-03,  8.72279804e-02,
       -2.08484348e-03,  2.03810845e-01, -2.38505013e-03,  2.29161944e-01,
       -2.39145647e-03,  1.97454882e-01,  0.00000000e+00,  0.00000000e+00,
        1.66808612e-03]), array([-2.04256842e-02, -2.19812429e-03,  1.04635261e-03, -1.00192260e-04,
        9.63541624e-05, -1.31190536e-04,  1.90995189e-04, -5.81483904e-05,
       -2.14176788e-05, -1.34150948e-04,  4.74286964e-04, -3.99226431e-05,
       -3.02200969e-05,  2.35079353e-04, -0.00000000e+00, -1.17346671e-04,
       -8.73043797e-04, -3.51336601e-01, -1.33799021e-03,  1.06908340e-01,
       -1.94191459e-03,  6.18912728e-01, -2.70370031e-03,  2.78388630e-01,
       -2.70421805e-03,  6.35715299e-01, -0.00000000e+00, -0.00000000e+00,
        1.26149397e-03]), array([ 3.83446550e-01,  5.97912732e-03, -6.88174046e-04, -6.96940479e-05,
       -3.77244961e-04,  3.58701034e-04, -1.64228487e-03, -1.16422182e-04,
       -4.93207633e-05,  5.95373875e-05, -4.40560990e-04, -7.33142386e-05,
       -2.99952775e-04, -6.80589564e-04, -0.00000000e+00,  9.66309524e-05,
        4.51401978e-04,  7.86576566e-02,  3.98082491e-03, -3.11788192e-01,
        8.55800281e-04, -4.21983363e-01, -2.63277861e-03, -3.47581003e-01,
       -2.64732927e-03,  6.71270590e-01, -0.00000000e+00, -0.00000000e+00,
        3.52450652e-03]), array([ 3.66943857e-01, -2.21145726e-04,  1.99724075e-03,  7.18266401e-05,
       -3.54295842e-04,  2.91049592e-04, -6.57073016e-04, -2.57033994e-04,
        1.56489383e-04,  7.19923190e-05,  2.10953516e-06,  6.77476604e-06,
       -1.12052721e-04, -8.93533485e-05, -0.00000000e+00, -2.21480056e-04,
       -2.20027570e-03,  8.94690992e-02, -5.22036498e-03,  2.51669220e-01,
       -5.53174088e-04,  5.04180737e-01,  6.18541272e-03, -7.17509903e-01,
        6.08919668e-03, -1.57706718e-01, -0.00000000e+00, -0.00000000e+00,
        2.72316120e-03]), array([ 8.14214384e-01, -4.99683511e-03, -5.17702693e-04, -1.92143597e-04,
       -5.96406037e-05,  7.42135257e-04,  5.73845215e-04,  5.69162331e-05,
        3.95498594e-04,  2.93626363e-04, -1.77219683e-04, -8.27505137e-05,
       -1.50129881e-04,  4.39510404e-04,  0.00000000e+00, -6.94984569e-04,
        1.48068707e-03, -3.32632498e-02,  4.10034862e-03, -2.11441195e-01,
       -5.17146784e-04,  9.92840643e-02, -1.06651554e-03,  4.64198496e-01,
       -1.04274462e-03, -2.56616313e-01,  0.00000000e+00,  0.00000000e+00,
       -1.72496904e-03]), array([ 2.32095580e-01,  1.68598243e-03,  1.40742927e-03, -2.99042647e-05,
        3.65696831e-05, -2.12220535e-05, -3.57057024e-04,  1.16351504e-04,
        4.33031661e-04,  5.99659381e-04, -6.82936497e-04,  9.62350974e-05,
        1.20515342e-04,  6.70132243e-05,  0.00000000e+00, -1.11482018e-03,
       -1.54813104e-03, -5.48638630e-02, -1.80637739e-02,  8.80464218e-01,
       -7.52755345e-05, -3.64986402e-01, -2.82506245e-03,  1.37017660e-01,
       -2.79925228e-03,  1.24324964e-01,  0.00000000e+00,  0.00000000e+00,
        1.03431250e-02]), array([ 2.69512465e-03,  9.05821183e-01, -5.44326366e-02,  1.07250119e-03,
       -5.43391253e-03, -4.52437463e-03,  3.20635359e-03, -5.95493085e-03,
       -4.83151691e-03, -1.34980723e-02, -1.24039952e-02,  2.34176144e-03,
       -1.34201933e-03, -9.64422433e-03,  5.16987883e-26,  4.41707319e-02,
       -1.57535819e-03, -3.80037390e-06,  1.02706102e-02,  3.96666160e-03,
       -6.72045619e-03,  3.64413222e-03, -6.52065068e-03,  3.49497748e-03,
       -5.80100023e-03, -1.64899332e-03,  0.00000000e+00,  0.00000000e+00,
       -4.16817608e-01]), array([ 2.47526018e-03, -4.21522276e-01, -5.65509707e-02, -1.00677920e-02,
       -5.93862046e-03, -2.07874774e-02,  1.73044417e-02, -2.21904111e-03,
       -1.33311006e-02,  2.19723792e-03, -8.72982091e-03,  7.66956542e-03,
       -3.79956108e-03,  1.03168802e-02, -2.06795153e-25,  8.08320313e-02,
        2.85611967e-02,  1.27753756e-03,  1.10374396e-02,  9.04455707e-03,
        1.31759190e-02, -4.82193366e-03,  1.12909331e-02, -3.62880164e-03,
        1.08623087e-02,  6.17386479e-03, -0.00000000e+00, -0.00000000e+00,
       -8.99863783e-01]), array([-2.40925059e-04,  2.97723512e-02,  9.56204658e-01,  6.06279189e-03,
       -2.46850410e-02, -4.06652664e-02,  2.10508413e-03,  8.78209115e-03,
        1.53922993e-02,  1.88557416e-01, -1.03483894e-01,  1.67100927e-03,
       -5.23771854e-03, -4.16056620e-02,  1.08420217e-19,  6.73635900e-03,
        1.00073797e-01,  1.11839107e-03, -4.50126513e-02, -2.05465402e-03,
        9.75389314e-02, -1.05513075e-03,  6.38436556e-02,  1.82396300e-03,
        6.40680047e-02,  8.59569259e-04, -0.00000000e+00, -0.00000000e+00,
       -6.59725608e-02]), array([ 7.57395803e-04,  1.22268038e-02,  9.41380372e-02,  2.02250951e-02,
        7.22415446e-02, -1.55440163e-01,  6.90707066e-02,  1.32148869e-02,
        4.76227454e-02,  3.70508142e-02,  9.48930952e-01,  4.13811655e-03,
       -9.16051778e-03,  4.20149381e-02,  5.55111512e-17, -1.31670681e-01,
       -1.08407068e-01,  2.17282731e-04,  1.11947661e-01,  2.20097967e-03,
        6.26292046e-02, -1.32827713e-03,  4.42960184e-02,  3.61988544e-04,
        4.41121712e-02,  7.09117882e-04, -0.00000000e+00, -0.00000000e+00,
       -2.85801417e-02]), array([-4.80419015e-04,  8.88354700e-03, -2.00969238e-01, -3.14822103e-02,
       -1.30729994e-02,  1.43755428e-01, -1.25637242e-02, -9.54001385e-03,
        3.26641628e-02,  9.48778425e-01,  3.44256871e-02, -6.55955203e-03,
       -1.69759253e-02, -2.91231694e-02,  9.71445147e-17,  9.34353992e-02,
        1.49315378e-01,  2.51407092e-04,  1.98034476e-03,  4.28328266e-04,
        3.61305524e-02,  8.01245882e-04,  2.36648847e-02, -6.93358797e-06,
        2.26984520e-02,  3.43470282e-04, -0.00000000e+00, -0.00000000e+00,
        2.08065184e-02])])
        whiten = False
        explained_variance = np.array([5390.031037031869, 3555.7736740937808, 574.6212541965924, 485.0233884153657, 392.0269290078801, 270.2568302716658, 0.8194897940979599, 0.5910666746277631, 0.09853950718441447, 0.05637866464661387, 0.054984397100819815])
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
    h_0 = max((((17.721788 * float(x[0]))+ (66.47728 * float(x[1]))+ (10.622732 * float(x[2]))+ (-15.173367 * float(x[3]))+ (-8.999497 * float(x[4]))+ (-19.459984 * float(x[5]))+ (-3.5168393 * float(x[6]))+ (9.960969 * float(x[7]))+ (-5.3347564 * float(x[8]))+ (0.54263306 * float(x[9]))+ (0.5378378 * float(x[10]))) + -1.7530222), 0)
    h_1 = max((((-183.11325 * float(x[0]))+ (-85.646164 * float(x[1]))+ (-25.784431 * float(x[2]))+ (21.615057 * float(x[3]))+ (-35.999786 * float(x[4]))+ (34.620125 * float(x[5]))+ (-16.844698 * float(x[6]))+ (-9.031094 * float(x[7]))+ (-3.800823 * float(x[8]))+ (11.539448 * float(x[9]))+ (-1.4857177 * float(x[10]))) + -14.398463), 0)
    h_2 = max((((3.1773438 * float(x[0]))+ (-14.502895 * float(x[1]))+ (-2.5046947 * float(x[2]))+ (-13.897065 * float(x[3]))+ (13.611604 * float(x[4]))+ (2.4288588 * float(x[5]))+ (20.889816 * float(x[6]))+ (-8.346707 * float(x[7]))+ (-3.9890573 * float(x[8]))+ (0.36581787 * float(x[9]))+ (-7.1549435 * float(x[10]))) + 0.3764775), 0)
    h_3 = max((((-0.028609566 * float(x[0]))+ (-0.37468028 * float(x[1]))+ (0.46671697 * float(x[2]))+ (-0.65778714 * float(x[3]))+ (0.5745514 * float(x[4]))+ (-1.17572 * float(x[5]))+ (1.6716272 * float(x[6]))+ (1.6366723 * float(x[7]))+ (-0.12534769 * float(x[8]))+ (0.09236806 * float(x[9]))+ (0.8923404 * float(x[10]))) + -7.1948905), 0)
    h_4 = max((((0.07476912 * float(x[0]))+ (0.5920285 * float(x[1]))+ (0.19978362 * float(x[2]))+ (-0.102526 * float(x[3]))+ (-0.06755987 * float(x[4]))+ (-0.50841457 * float(x[5]))+ (-0.12266678 * float(x[6]))+ (0.6964732 * float(x[7]))+ (-10.085347 * float(x[8]))+ (-1.8563622 * float(x[9]))+ (1.7091376 * float(x[10]))) + -6.529579), 0)
    o[0] = (-0.006232861 * h_0)+ (0.00013860757 * h_1)+ (-0.010084625 * h_2)+ (0.49404907 * h_3)+ (0.7046941 * h_4) + -9.636838

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

        model_cap=66

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






