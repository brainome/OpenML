#!/usr/bin/env python3
#
# This code is licensed under GNU GPL v2.0 or higher. Please see LICENSE for details.
#
#
# Output of Brainome Daimensions(tm) Table Compiler v0.8.
# Compile time: Feb-28-2020 18:38:41
# Invocation: btc Data/hypothyroid.csv -o Models/hypothyroid.py -v -v -v -stopat 99.76 -port 8090 -e 9
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                92.28%
Model accuracy:                     98.64% (3721/3772 correct)
Improvement over best guess:        6.36% (of possible 7.72%)
Model capacity (MEC):               94 bits
Generalization ratio:               39.58 bits/bit
Model efficiency:                   0.06%/parameter
System behavior
True Negatives:                     91.76% (3461/3772)
True Positives:                     6.89% (260/3772)
False Negatives:                    0.82% (31/3772)
False Positives:                    0.53% (20/3772)
True Pos. Rate/Sensitivity/Recall:  0.89
True Neg. Rate/Specificity:         0.99
Precision:                          0.93
F-1 Measure:                        0.91
False Negative Rate/Miss Rate:      0.11
Critical Success Index:             0.84
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
TRAINFILE="hypothyroid.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29

mappings = [{1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.19: 22, 0.2: 23, 0.24: 24, 0.25: 25, 0.26: 26, 0.27: 27, 0.28: 28, 0.29: 29, 0.3: 30, 0.31: 31, 0.32: 32, 0.33: 33, 0.34: 34, 0.35: 35, 0.36: 36, 0.38: 37, 0.39: 38, 0.4: 39, 0.41: 40, 0.42: 41, 0.43: 42, 0.44: 43, 0.45: 44, 0.46: 45, 0.47: 46, 0.48: 47, 0.49: 48, 0.5: 49, 0.51: 50, 0.52: 51, 0.53: 52, 0.54: 53, 0.55: 54, 0.56: 55, 0.57: 56, 0.58: 57, 0.59: 58, 0.6: 59, 0.61: 60, 0.62: 61, 0.63: 62, 0.64: 63, 0.65: 64, 0.66: 65, 0.67: 66, 0.68: 67, 0.69: 68, 0.7: 69, 0.71: 70, 0.72: 71, 0.73: 72, 0.74: 73, 0.75: 74, 0.76: 75, 0.77: 76, 0.78: 77, 0.79: 78, 0.8: 79, 0.81: 80, 0.82: 81, 0.83: 82, 0.84: 83, 0.85: 84, 0.86: 85, 0.87: 86, 0.88: 87, 0.89: 88, 0.9: 89, 0.91: 90, 0.92: 91, 0.93: 92, 0.94: 93, 0.95: 94, 0.96: 95, 0.97: 96, 0.98: 97, 0.99: 98, 1.0: 99, 1.02: 100, 1.1: 101, 1.2: 102, 1.3: 103, 1.4: 104, 1.5: 105, 1.6: 106, 1.7: 107, 1.8: 108, 1.9: 109, 2.0: 110, 2.1: 111, 2.2: 112, 2.3: 113, 2.4: 114, 2.5: 115, 2.6: 116, 2.7: 117, 2.8: 118, 2.9: 119, 3.0: 120, 3.1: 121, 3.2: 122, 3.3: 123, 3.4: 124, 3.5: 125, 3.6: 126, 3.7: 127, 3.8: 128, 3.9: 129, 4.0: 130, 4.1: 131, 4.2: 132, 4.3: 133, 4.4: 134, 4.5: 135, 4.6: 136, 4.7: 137, 4.8: 138, 4.9: 139, 5.0: 140, 5.1: 141, 5.2: 142, 5.3: 143, 5.4: 144, 5.5: 145, 5.6: 146, 5.7: 147, 5.8: 148, 5.9: 149, 6.0: 150, 6.1: 151, 6.2: 152, 6.3: 153, 6.5: 154, 6.6: 155, 6.7: 156, 6.8: 157, 6.9: 158, 7.0: 159, 7.1: 160, 7.2: 161, 7.3: 162, 7.4: 163, 7.5: 164, 7.6: 165, 7.7: 166, 7.8: 167, 7.9: 168, 8.0: 169, 8.1: 170, 8.2: 171, 8.3: 172, 8.5: 173, 8.6: 174, 8.9: 175, 9.0: 176, 9.1: 177, 9.2: 178, 9.4: 179, 9.6: 180, 9.7: 181, 9.8: 182, 9.9: 183, 10.0: 184, 11.0: 185, 11.1: 186, 11.4: 187, 12.0: 188, 13.0: 189, 14.4: 190, 14.8: 191, 15.0: 192, 16.0: 193, 17.0: 194, 18.0: 195, 19.0: 196, 20.0: 197, 21.0: 198, 22.0: 199, 23.0: 200, 24.0: 201, 25.0: 202, 26.0: 203, 27.0: 204, 30.0: 205, 31.0: 206, 34.0: 207, 36.0: 208, 38.0: 209, 40.0: 210, 42.0: 211, 43.0: 212, 44.0: 213, 45.0: 214, 47.0: 215, 51.0: 216, 52.0: 217, 55.0: 218, 58.0: 219, 60.0: 220, 70.0: 221, 76.0: 222, 82.0: 223, 89.0: 224, 98.0: 225, 103.0: 226, 108.0: 227, 109.0: 228, 117.0: 229, 126.0: 230, 143.0: 231, 151.0: 232, 160.0: 233, 178.0: 234, 183.0: 235, 400.0: 236, 440.0: 237, 468.0: 238, 472.0: 239, 530.0: 240, 1684325040.0: 241, 1.01: 242, 5.73: 243, 478.0: 244, 145.0: 245, 26.4: 246, 66.0: 247, 0.37: 248, 61.0: 249, 9.3: 250, 10.3: 251, 50.0: 252, 8.8: 253, 0.21: 254, 116.0: 255, 32.0: 256, 0.17: 257, 46.0: 258, 236.0: 259, 14.0: 260, 80.0: 261, 9.5: 262, 65.0: 263, 28.0: 264, 0.22: 265, 29.0: 266, 6.4: 267, 139.0: 268, 230.0: 269, 54.0: 270, 30.5: 271, 12.1: 272, 41.0: 273, 8.4: 274, 18.4: 275, 0.18: 276, 78.0: 277, 35.0: 278, 86.0: 279, 39.0: 280, 33.0: 281, 100.0: 282, 0.23: 283, 99.0: 284, 165.0: 285, 199.0: 286, 188.0: 287}, {1993550816.0: 0, 2238339752.0: 1}, {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 1.1: 10, 1.2: 11, 1.3: 12, 1.4: 13, 1.5: 14, 1.6: 15, 1.7: 16, 1.8: 17, 1.9: 18, 2.0: 19, 2.1: 20, 2.2: 21, 2.3: 22, 2.4: 23, 2.5: 24, 2.6: 25, 2.7: 26, 2.8: 27, 2.9: 28, 3.0: 29, 3.1: 30, 3.2: 31, 3.3: 32, 3.4: 33, 3.5: 34, 3.6: 35, 3.7: 36, 3.8: 37, 3.9: 38, 4.0: 39, 4.1: 40, 4.2: 41, 4.3: 42, 4.5: 43, 4.6: 44, 4.7: 45, 4.8: 46, 4.9: 47, 5.0: 48, 5.1: 49, 5.2: 50, 5.3: 51, 5.4: 52, 5.5: 53, 5.7: 54, 6.6: 55, 6.7: 56, 7.0: 57, 7.1: 58, 7.3: 59, 7.6: 60, 8.5: 61, 1684325040.0: 62, 6.2: 63, 6.1: 64, 4.4: 65, 6.0: 66, 1.44: 67, 0.05: 68, 10.6: 69}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.9: 1, 3.0: 2, 4.8: 3, 5.8: 4, 6.0: 5, 9.5: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 17.0: 13, 19.0: 14, 21.0: 15, 22.0: 16, 24.0: 17, 28.0: 18, 29.0: 19, 30.0: 20, 31.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 37.0: 25, 38.0: 26, 39.0: 27, 40.0: 28, 41.0: 29, 42.0: 30, 44.0: 31, 45.0: 32, 46.0: 33, 48.0: 34, 50.0: 35, 51.0: 36, 52.0: 37, 53.0: 38, 54.0: 39, 56.0: 40, 57.0: 41, 58.0: 42, 59.0: 43, 60.0: 44, 61.0: 45, 62.0: 46, 63.0: 47, 64.0: 48, 65.0: 49, 66.0: 50, 67.0: 51, 68.0: 52, 69.0: 53, 70.0: 54, 71.0: 55, 72.0: 56, 73.0: 57, 74.0: 58, 75.0: 59, 76.0: 60, 77.0: 61, 78.0: 62, 79.0: 63, 80.0: 64, 81.0: 65, 82.0: 66, 83.0: 67, 84.0: 68, 85.0: 69, 86.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 93.0: 77, 94.0: 78, 95.0: 79, 96.0: 80, 97.0: 81, 98.0: 82, 99.0: 83, 100.0: 84, 101.0: 85, 102.0: 86, 103.0: 87, 104.0: 88, 105.0: 89, 106.0: 90, 107.0: 91, 108.0: 92, 109.0: 93, 110.0: 94, 111.0: 95, 112.0: 96, 113.0: 97, 114.0: 98, 115.0: 99, 116.0: 100, 117.0: 101, 118.0: 102, 119.0: 103, 120.0: 104, 121.0: 105, 122.0: 106, 123.0: 107, 124.0: 108, 125.0: 109, 126.0: 110, 127.0: 111, 128.0: 112, 129.0: 113, 130.0: 114, 131.0: 115, 132.0: 116, 133.0: 117, 134.0: 118, 135.0: 119, 136.0: 120, 137.0: 121, 138.0: 122, 139.0: 123, 140.0: 124, 141.0: 125, 142.0: 126, 143.0: 127, 144.0: 128, 145.0: 129, 146.0: 130, 147.0: 131, 148.0: 132, 149.0: 133, 150.0: 134, 151.0: 135, 152.0: 136, 153.0: 137, 154.0: 138, 155.0: 139, 156.0: 140, 157.0: 141, 158.0: 142, 159.0: 143, 160.0: 144, 161.0: 145, 162.0: 146, 163.0: 147, 164.0: 148, 165.0: 149, 166.0: 150, 167.0: 151, 168.0: 152, 169.0: 153, 170.0: 154, 171.0: 155, 172.0: 156, 174.0: 157, 175.0: 158, 176.0: 159, 177.0: 160, 178.0: 161, 179.0: 162, 180.0: 163, 181.0: 164, 182.0: 165, 183.0: 166, 184.0: 167, 187.0: 168, 189.0: 169, 191.0: 170, 192.0: 171, 193.0: 172, 194.0: 173, 196.0: 174, 197.0: 175, 198.0: 176, 199.0: 177, 200.0: 178, 201.0: 179, 203.0: 180, 205.0: 181, 207.0: 182, 209.0: 183, 210.0: 184, 211.0: 185, 212.0: 186, 213.0: 187, 220.0: 188, 223.0: 189, 225.0: 190, 230.0: 191, 231.0: 192, 232.0: 193, 237.0: 194, 239.0: 195, 244.0: 196, 248.0: 197, 252.0: 198, 253.0: 199, 257.0: 200, 261.0: 201, 263.0: 202, 289.0: 203, 372.0: 204, 430.0: 205, 1684325040.0: 206, 255.0: 207, 214.0: 208, 186.0: 209, 49.0: 210, 195.0: 211, 240.0: 212, 258.0: 213, 32.0: 214, 204.0: 215, 16.0: 216, 250.0: 217, 273.0: 218, 18.0: 219, 217.0: 220, 216.0: 221, 4.0: 222, 47.0: 223, 36.0: 224, 222.0: 225, 188.0: 226, 206.0: 227, 23.0: 228, 272.0: 229, 233.0: 230, 27.0: 231, 173.0: 232, 226.0: 233, 301.0: 234, 256.0: 235, 246.0: 236, 25.0: 237, 235.0: 238, 43.0: 239, 55.0: 240, 219.0: 241}, {1993550816.0: 0, 2238339752.0: 1}, {0.31: 0, 0.38: 1, 0.48: 2, 0.5: 3, 0.52: 4, 0.54: 5, 0.56: 6, 0.57: 7, 0.58: 8, 0.59: 9, 0.6: 10, 0.61: 11, 0.62: 12, 0.63: 13, 0.64: 14, 0.65: 15, 0.66: 16, 0.67: 17, 0.68: 18, 0.69: 19, 0.7: 20, 0.71: 21, 0.72: 22, 0.73: 23, 0.74: 24, 0.75: 25, 0.76: 26, 0.77: 27, 0.78: 28, 0.79: 29, 0.8: 30, 0.81: 31, 0.82: 32, 0.83: 33, 0.84: 34, 0.85: 35, 0.86: 36, 0.87: 37, 0.88: 38, 0.89: 39, 0.9: 40, 0.91: 41, 0.92: 42, 0.93: 43, 0.94: 44, 0.95: 45, 0.96: 46, 0.97: 47, 0.98: 48, 0.99: 49, 1.0: 50, 1.01: 51, 1.02: 52, 1.03: 53, 1.04: 54, 1.05: 55, 1.06: 56, 1.07: 57, 1.08: 58, 1.09: 59, 1.1: 60, 1.11: 61, 1.12: 62, 1.13: 63, 1.14: 64, 1.15: 65, 1.16: 66, 1.17: 67, 1.18: 68, 1.19: 69, 1.2: 70, 1.21: 71, 1.22: 72, 1.23: 73, 1.24: 74, 1.25: 75, 1.26: 76, 1.27: 77, 1.28: 78, 1.29: 79, 1.3: 80, 1.31: 81, 1.32: 82, 1.34: 83, 1.35: 84, 1.36: 85, 1.37: 86, 1.38: 87, 1.39: 88, 1.4: 89, 1.42: 90, 1.43: 91, 1.44: 92, 1.45: 93, 1.46: 94, 1.47: 95, 1.48: 96, 1.49: 97, 1.5: 98, 1.51: 99, 1.52: 100, 1.53: 101, 1.54: 102, 1.55: 103, 1.56: 104, 1.57: 105, 1.58: 106, 1.59: 107, 1.61: 108, 1.62: 109, 1.65: 110, 1.66: 111, 1.67: 112, 1.68: 113, 1.69: 114, 1.7: 115, 1.71: 116, 1.73: 117, 1.75: 118, 1.76: 119, 1.77: 120, 1.8: 121, 1.82: 122, 1.83: 123, 1.84: 124, 1.88: 125, 1.93: 126, 1.97: 127, 2.12: 128, 2.32: 129, 1684325040.0: 130, 1.33: 131, 1.41: 132, 2.01: 133, 0.53: 134, 2.03: 135, 0.46: 136, 0.47: 137, 0.944: 138, 1.79: 139, 1.63: 140, 0.36: 141, 0.25: 142, 1.94: 143, 1.74: 144, 0.41: 145, 0.49: 146}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.8: 1, 3.0: 2, 5.4: 3, 7.0: 4, 7.6: 5, 8.4: 6, 8.5: 7, 8.9: 8, 9.0: 9, 9.1: 10, 10.0: 11, 11.0: 12, 13.0: 13, 14.0: 14, 15.0: 15, 17.0: 16, 18.0: 17, 19.0: 18, 21.0: 19, 26.0: 20, 29.0: 21, 33.0: 22, 35.0: 23, 37.0: 24, 39.0: 25, 42.0: 26, 46.0: 27, 47.0: 28, 49.0: 29, 50.0: 30, 51.0: 31, 54.0: 32, 55.0: 33, 56.0: 34, 58.0: 35, 59.0: 36, 60.0: 37, 61.0: 38, 62.0: 39, 63.0: 40, 64.0: 41, 65.0: 42, 66.0: 43, 67.0: 44, 68.0: 45, 69.0: 46, 70.0: 47, 71.0: 48, 72.0: 49, 73.0: 50, 74.0: 51, 75.0: 52, 76.0: 53, 77.0: 54, 78.0: 55, 79.0: 56, 80.0: 57, 81.0: 58, 82.0: 59, 83.0: 60, 84.0: 61, 85.0: 62, 86.0: 63, 87.0: 64, 88.0: 65, 89.0: 66, 90.0: 67, 91.0: 68, 92.0: 69, 93.0: 70, 94.0: 71, 95.0: 72, 96.0: 73, 97.0: 74, 98.0: 75, 99.0: 76, 100.0: 77, 101.0: 78, 102.0: 79, 103.0: 80, 104.0: 81, 105.0: 82, 106.0: 83, 107.0: 84, 108.0: 85, 109.0: 86, 110.0: 87, 111.0: 88, 112.0: 89, 113.0: 90, 114.0: 91, 115.0: 92, 116.0: 93, 117.0: 94, 118.0: 95, 119.0: 96, 120.0: 97, 121.0: 98, 122.0: 99, 123.0: 100, 124.0: 101, 125.0: 102, 126.0: 103, 127.0: 104, 128.0: 105, 129.0: 106, 130.0: 107, 131.0: 108, 132.0: 109, 133.0: 110, 134.0: 111, 135.0: 112, 136.0: 113, 137.0: 114, 138.0: 115, 139.0: 116, 140.0: 117, 141.0: 118, 142.0: 119, 143.0: 120, 144.0: 121, 145.0: 122, 146.0: 123, 147.0: 124, 148.0: 125, 149.0: 126, 150.0: 127, 151.0: 128, 152.0: 129, 153.0: 130, 154.0: 131, 155.0: 132, 156.0: 133, 157.0: 134, 158.0: 135, 159.0: 136, 160.0: 137, 161.0: 138, 162.0: 139, 163.0: 140, 164.0: 141, 165.0: 142, 166.0: 143, 167.0: 144, 168.0: 145, 169.0: 146, 170.0: 147, 171.0: 148, 172.0: 149, 173.0: 150, 174.0: 151, 175.0: 152, 176.0: 153, 177.0: 154, 178.0: 155, 180.0: 156, 183.0: 157, 184.0: 158, 185.0: 159, 186.0: 160, 188.0: 161, 189.0: 162, 190.0: 163, 191.0: 164, 194.0: 165, 195.0: 166, 196.0: 167, 197.0: 168, 198.0: 169, 200.0: 170, 201.0: 171, 203.0: 172, 204.0: 173, 206.0: 174, 207.0: 175, 209.0: 176, 213.0: 177, 214.0: 178, 215.0: 179, 216.0: 180, 217.0: 181, 219.0: 182, 220.0: 183, 222.0: 184, 224.0: 185, 235.0: 186, 249.0: 187, 251.0: 188, 265.0: 189, 280.0: 190, 291.0: 191, 362.0: 192, 395.0: 193, 1684325040.0: 194, 244.0: 195, 34.0: 196, 228.0: 197, 57.0: 198, 52.0: 199, 237.0: 200, 223.0: 201, 179.0: 202, 227.0: 203, 199.0: 204, 312.0: 205, 43.0: 206, 16.0: 207, 221.0: 208, 36.0: 209, 28.0: 210, 242.0: 211, 182.0: 212, 4.0: 213, 48.0: 214, 40.0: 215, 205.0: 216, 27.0: 217, 247.0: 218, 218.0: 219, 53.0: 220, 24.0: 221, 187.0: 222, 245.0: 223, 281.0: 224, 181.0: 225, 253.0: 226, 32.0: 227, 349.0: 228, 41.0: 229, 283.0: 230, 232.0: 231, 20.0: 232, 210.0: 233, 274.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
list_of_cols_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

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


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

# Classifier
def classify(row):
    x=row
    h_0 = max((((-10.442831 * float(x[0]))+ (0.20552675 * float(x[1]))+ (0.08957078 * float(x[2]))+ (-0.1526904 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.124825574 * float(x[5]))+ (0.78354603 * float(x[6]))+ (0.92732555 * float(x[7]))+ (-0.23311697 * float(x[8]))+ (0.5832545 * float(x[9]))+ (0.05778984 * float(x[10]))+ (0.13608912 * float(x[11]))+ (0.85119325 * float(x[12]))+ (-0.85792786 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (-0.9595632 * float(x[15]))+ (-0.0136391455 * float(x[16]))+ (-70.72812 * float(x[17]))+ (0.06114546 * float(x[18]))+ (-11.938136 * float(x[19]))+ (-0.080561705 * float(x[20]))+ (-40.811924 * float(x[21]))+ (-0.11782049 * float(x[22]))+ (-36.742466 * float(x[23]))+ (-0.3990368 * float(x[24]))+ (-34.660755 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-2.207117 * float(x[28]))) + -0.5812518), 0)
    h_1 = max((((0.04119016 * float(x[0]))+ (0.14476857 * float(x[1]))+ (3.6970792 * float(x[2]))+ (0.8573665 * float(x[3]))+ (0.9307542 * float(x[4]))+ (-1.3309727 * float(x[5]))+ (1.6575176 * float(x[6]))+ (5.7396994 * float(x[7]))+ (-11.703717 * float(x[8]))+ (-1.4395212 * float(x[9]))+ (-0.90357363 * float(x[10]))+ (1.5568918 * float(x[11]))+ (2.4615865 * float(x[12]))+ (-1.836634 * float(x[13]))+ (-0.2358069 * float(x[14]))+ (-4.6481466 * float(x[15]))+ (-9.509123 * float(x[16]))+ (0.049410738 * float(x[17]))+ (2.1802382 * float(x[18]))+ (0.04406391 * float(x[19]))+ (2.3614798 * float(x[20]))+ (0.014205158 * float(x[21]))+ (-0.3326514 * float(x[22]))+ (-0.0012889225 * float(x[23]))+ (-1.0099335 * float(x[24]))+ (0.0018420003 * float(x[25]))+ (-0.5008052 * float(x[26]))+ (-0.3295877 * float(x[27]))+ (-0.46504778 * float(x[28]))) + -5.7751546), 0)
    h_2 = max((((0.0012142176 * float(x[0]))+ (0.07047378 * float(x[1]))+ (1.0582912 * float(x[2]))+ (0.21169016 * float(x[3]))+ (0.24149388 * float(x[4]))+ (0.030349204 * float(x[5]))+ (2.2921453 * float(x[6]))+ (0.8304212 * float(x[7]))+ (0.098762535 * float(x[8]))+ (-0.16998877 * float(x[9]))+ (-0.035226777 * float(x[10]))+ (3.5223582 * float(x[11]))+ (3.743319 * float(x[12]))+ (-0.0053190626 * float(x[13]))+ (0.6009352 * float(x[14]))+ (-0.28321558 * float(x[15]))+ (-2.4463334 * float(x[16]))+ (-0.028401347 * float(x[17]))+ (0.59519345 * float(x[18]))+ (0.011114295 * float(x[19]))+ (0.7069817 * float(x[20]))+ (0.0055879434 * float(x[21]))+ (0.11217856 * float(x[22]))+ (-0.004336063 * float(x[23]))+ (-0.29390275 * float(x[24]))+ (0.00017140231 * float(x[25]))+ (0.4822089 * float(x[26]))+ (-0.10955198 * float(x[27]))+ (-0.05326002 * float(x[28]))) + 6.247742), 0)
    o_0 = (-0.21587601 * h_0)+ (-1.200237 * h_1)+ (-6.4980135 * h_2) + 5.715536

    if num_output_logits==1:
        return o_0>=0
    else:
        return argmax([eval('o'+str(i)) for i in range(num_output_logits)])

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

        model_cap=94

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


