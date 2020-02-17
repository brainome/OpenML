#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 16:27:21
# Invocation: btc -v -v sick-1.csv -o sick-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                93.87%
Model accuracy:                     93.87% (3541/3772 correct)
Improvement over best guess:        0.00% (of possible 6.13%)
Model capacity (MEC):               63 bits
Generalization ratio:               56.20 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     93.88% (3541/3772)
True Positives:                     0.00% (0/3772)
False Negatives:                    6.12% (231/3772)
False Positives:                    0.00% (0/3772)
True Pos. Rate/Sensitivity/Recall:  0.00
True Neg. Rate/Specificity:         1.00
F-1 Measure:                        0.00
False Negative Rate/Miss Rate:      1.00
Critical Success Index:             0.00
Model bias:                         100.00% higher chance to pick class 0
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="sick-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29

mappings = [{1304234792.0: 0, 3664761504.0: 1, 1684325040.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {2238339752.0: 0, 1993550816.0: 1}, {0.25: 0, 1.2: 1, 2.9: 2, 1.1: 3, 2.3: 4, 5.4: 5, 2.6: 6, 6.3: 7, 8.1: 8, 4.2: 9, 8.5: 10, 2.7: 11, 9.7: 12, 11.0: 13, 1.0: 14, 8.2: 15, 3.0: 16, 2.0: 17, 13.0: 18, 14.4: 19, 2.5: 20, 530.0: 21, 22.0: 22, 4.0: 23, 24.0: 24, 25.0: 25, 4.5: 26, 27.0: 27, 26.0: 28, 5.0: 29, 30.0: 30, 0.75: 31, 0.5: 32, 1.5: 33, 6.0: 34, 6.5: 35, 7.5: 36, 34.0: 37, 31.0: 38, 38.0: 39, 40.0: 40, 8.0: 41, 42.0: 42, 43.0: 43, 44.0: 44, 45.0: 45, 9.0: 46, 47.0: 47, 50.0: 48, 10.0: 49, 51.0: 50, 52.0: 51, 55.0: 52, 58.0: 53, 60.0: 54, 12.0: 55, 61.0: 56, 0.21: 57, 0.79: 58, 0.54: 59, 66.0: 60, 70.0: 61, 0.13: 62, 0.63: 63, 76.0: 64, 15.0: 65, 0.47: 66, 0.72: 67, 3.5: 68, 16.0: 69, 0.97: 70, 17.0: 71, 82.0: 72, 89.0: 73, 18.0: 74, 19.0: 75, 98.0: 76, 20.0: 77, 1.4: 78, 1.9: 79, 103.0: 80, 21.0: 81, 2.4: 82, 108.0: 83, 109.0: 84, 3.9: 85, 3.4: 86, 23.0: 87, 117.0: 88, 116.0: 89, 126.0: 90, 0.43: 91, 0.2: 92, 0.41: 93, 0.42: 94, 0.065: 95, 0.44: 96, 0.45: 97, 0.46: 98, 5.5: 99, 0.92: 100, 26.4: 101, 143.0: 102, 0.055: 103, 0.76: 104, 0.51: 105, 0.26: 106, 145.0: 107, 4.9: 108, 4.4: 109, 151.0: 110, 1.01: 111, 0.85: 112, 5.9: 113, 0.35: 114, 6.9: 115, 160.0: 116, 0.045: 117, 0.94: 118, 0.69: 119, 7.4: 120, 7.9: 121, 8.9: 122, 9.9: 123, 9.4: 124, 1684325040.0: 125, 178.0: 126, 7.0: 127, 36.0: 128, 183.0: 129, 11.4: 130, 0.15: 131, 0.035: 132, 0.16: 133, 2.8: 134, 3.8: 135, 0.8: 136, 3.3: 137, 0.55: 138, 0.025: 139, 4.8: 140, 1.8: 141, 1.3: 142, 4.3: 143, 0.14: 144, 5.8: 145, 5.3: 146, 0.64: 147, 0.39: 148, 0.89: 149, 6.8: 150, 0.015: 151, 0.98: 152, 0.73: 153, 0.48: 154, 7.3: 155, 7.8: 156, 0.59: 157, 0.57: 158, 0.005: 159, 0.6: 160, 0.93: 161, 0.67: 162, 0.62: 163, 0.88: 164, 0.38: 165, 0.4: 166, 0.82: 167, 0.9: 168, 0.84: 169, 0.68: 170, 5.73: 171, 0.52: 172, 0.77: 173, 0.27: 174, 0.19: 175, 1.02: 176, 0.66: 177, 0.86: 178, 0.36: 179, 0.61: 180, 8.3: 181, 8.8: 182, 9.2: 183, 9.8: 184, 9.3: 185, 0.95: 186, 0.7: 187, 10.3: 188, 1.7: 189, 0.1: 190, 0.09: 191, 14.8: 192, 2.2: 193, 3.2: 194, 3.7: 195, 4.7: 196, 5.2: 197, 5.7: 198, 6.2: 199, 6.7: 200, 0.56: 201, 0.81: 202, 0.08: 203, 7.2: 204, 7.7: 205, 0.65: 206, 0.74: 207, 0.99: 208, 0.24: 209, 0.49: 210, 0.58: 211, 0.83: 212, 0.31: 213, 0.07: 214, 0.3: 215, 0.33: 216, 0.06: 217, 0.32: 218, 0.34: 219, 0.29: 220, 400.0: 221, 4.6: 222, 4.1: 223, 5.6: 224, 5.1: 225, 6.1: 226, 0.05: 227, 6.6: 228, 7.1: 229, 7.6: 230, 1.6: 231, 8.6: 232, 2.1: 233, 9.6: 234, 9.1: 235, 0.04: 236, 3.1: 237, 0.78: 238, 0.28: 239, 3.6: 240, 0.53: 241, 440.0: 242, 0.12: 243, 11.1: 244, 0.87: 245, 0.37: 246, 0.03: 247, 0.71: 248, 0.96: 249, 468.0: 250, 472.0: 251, 0.02: 252, 478.0: 253, 0.01: 254, 0.91: 255, 32.0: 256, 0.17: 257, 46.0: 258, 236.0: 259, 14.0: 260, 80.0: 261, 9.5: 262, 65.0: 263, 28.0: 264, 0.22: 265, 29.0: 266, 6.4: 267, 139.0: 268, 230.0: 269, 54.0: 270, 30.5: 271, 12.1: 272, 41.0: 273, 8.4: 274, 18.4: 275, 0.18: 276, 78.0: 277, 35.0: 278, 86.0: 279, 39.0: 280, 33.0: 281, 100.0: 282, 0.23: 283, 99.0: 284, 165.0: 285, 199.0: 286, 188.0: 287}, {2238339752.0: 0, 1993550816.0: 1}, {0.7: 0, 1.4: 1, 2.0: 2, 2.2: 3, 2.4: 4, 1.6: 5, 1.1: 6, 0.8: 7, 2.3: 8, 1.5: 9, 2.1: 10, 3.8: 11, 1.0: 12, 2.5: 13, 5.4: 14, 0.5: 15, 3.5: 16, 3.0: 17, 4.0: 18, 4.1: 19, 4.5: 20, 4.6: 21, 0.6: 22, 5.1: 23, 5.0: 24, 4.9: 25, 5.5: 26, 4.4: 27, 6.6: 28, 6.1: 29, 7.6: 30, 7.0: 31, 7.1: 32, 8.5: 33, 2.6: 34, 1684325040.0: 35, 3.6: 36, 3.1: 37, 0.2: 38, 1.2: 39, 1.7: 40, 0.1: 41, 2.7: 42, 2.8: 43, 0.3: 44, 3.7: 45, 3.2: 46, 3.3: 47, 4.3: 48, 1.8: 49, 1.3: 50, 4.7: 51, 4.8: 52, 5.3: 53, 5.7: 54, 4.2: 55, 5.2: 56, 6.7: 57, 6.2: 58, 7.3: 59, 1.9: 60, 0.9: 61, 0.4: 62, 2.9: 63, 3.4: 64, 3.9: 65, 6.0: 66, 1.44: 67, 0.05: 68, 10.6: 69}, {2238339752.0: 0, 1993550816.0: 1}, {2.9: 0, 3.0: 1, 2.0: 2, 5.8: 3, 6.0: 4, 4.8: 5, 9.5: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 17.0: 13, 19.0: 14, 21.0: 15, 22.0: 16, 24.0: 17, 28.0: 18, 29.0: 19, 30.0: 20, 31.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 37.0: 25, 38.0: 26, 39.0: 27, 40.0: 28, 41.0: 29, 42.0: 30, 44.0: 31, 45.0: 32, 46.0: 33, 48.0: 34, 49.0: 35, 50.0: 36, 51.0: 37, 52.0: 38, 53.0: 39, 54.0: 40, 56.0: 41, 57.0: 42, 58.0: 43, 59.0: 44, 60.0: 45, 61.0: 46, 62.0: 47, 63.0: 48, 64.0: 49, 65.0: 50, 66.0: 51, 67.0: 52, 68.0: 53, 69.0: 54, 70.0: 55, 71.0: 56, 72.0: 57, 73.0: 58, 74.0: 59, 75.0: 60, 76.0: 61, 77.0: 62, 78.0: 63, 79.0: 64, 80.0: 65, 81.0: 66, 82.0: 67, 83.0: 68, 84.0: 69, 85.0: 70, 86.0: 71, 87.0: 72, 88.0: 73, 89.0: 74, 90.0: 75, 91.0: 76, 92.0: 77, 93.0: 78, 94.0: 79, 95.0: 80, 96.0: 81, 97.0: 82, 98.0: 83, 99.0: 84, 100.0: 85, 101.0: 86, 102.0: 87, 103.0: 88, 104.0: 89, 105.0: 90, 106.0: 91, 107.0: 92, 108.0: 93, 109.0: 94, 110.0: 95, 111.0: 96, 112.0: 97, 113.0: 98, 114.0: 99, 115.0: 100, 116.0: 101, 117.0: 102, 118.0: 103, 119.0: 104, 120.0: 105, 121.0: 106, 122.0: 107, 123.0: 108, 124.0: 109, 125.0: 110, 126.0: 111, 127.0: 112, 128.0: 113, 129.0: 114, 130.0: 115, 131.0: 116, 132.0: 117, 133.0: 118, 134.0: 119, 135.0: 120, 136.0: 121, 137.0: 122, 138.0: 123, 139.0: 124, 140.0: 125, 141.0: 126, 142.0: 127, 143.0: 128, 144.0: 129, 145.0: 130, 146.0: 131, 147.0: 132, 148.0: 133, 149.0: 134, 150.0: 135, 151.0: 136, 152.0: 137, 153.0: 138, 154.0: 139, 155.0: 140, 156.0: 141, 157.0: 142, 158.0: 143, 159.0: 144, 160.0: 145, 161.0: 146, 162.0: 147, 163.0: 148, 164.0: 149, 165.0: 150, 166.0: 151, 167.0: 152, 168.0: 153, 169.0: 154, 170.0: 155, 171.0: 156, 172.0: 157, 174.0: 158, 175.0: 159, 1684325040.0: 160, 176.0: 161, 177.0: 162, 179.0: 163, 180.0: 164, 181.0: 165, 182.0: 166, 183.0: 167, 184.0: 168, 178.0: 169, 186.0: 170, 187.0: 171, 189.0: 172, 191.0: 173, 192.0: 174, 193.0: 175, 194.0: 176, 195.0: 177, 196.0: 178, 197.0: 179, 198.0: 180, 199.0: 181, 200.0: 182, 201.0: 183, 203.0: 184, 205.0: 185, 207.0: 186, 209.0: 187, 210.0: 188, 211.0: 189, 212.0: 190, 213.0: 191, 214.0: 192, 220.0: 193, 223.0: 194, 225.0: 195, 230.0: 196, 231.0: 197, 232.0: 198, 237.0: 199, 239.0: 200, 240.0: 201, 244.0: 202, 248.0: 203, 252.0: 204, 253.0: 205, 255.0: 206, 257.0: 207, 258.0: 208, 261.0: 209, 263.0: 210, 289.0: 211, 372.0: 212, 430.0: 213, 32.0: 214, 204.0: 215, 16.0: 216, 250.0: 217, 273.0: 218, 18.0: 219, 217.0: 220, 216.0: 221, 4.0: 222, 47.0: 223, 36.0: 224, 222.0: 225, 188.0: 226, 206.0: 227, 23.0: 228, 272.0: 229, 233.0: 230, 27.0: 231, 173.0: 232, 226.0: 233, 301.0: 234, 256.0: 235, 246.0: 236, 25.0: 237, 235.0: 238, 43.0: 239, 55.0: 240, 219.0: 241}, {2238339752.0: 0, 1993550816.0: 1}, {0.96: 0, 1.15: 1, 1.09: 2, 1.28: 3, 1.54: 4, 1.16: 5, 0.75: 6, 1.17: 7, 1.08: 8, 0.76: 9, 1.01: 10, 1.25: 11, 1.0: 12, 1.75: 13, 0.5: 14, 1.5: 15, 0.59: 16, 1.18: 17, 1.68: 18, 1.43: 19, 1.93: 20, 1.11: 21, 1.36: 22, 1.61: 23, 0.79: 24, 0.54: 25, 1.04: 26, 1.29: 27, 0.88: 28, 0.63: 29, 0.38: 30, 0.97: 31, 0.72: 32, 1.22: 33, 1.97: 34, 1.47: 35, 1.65: 36, 1.4: 37, 1.83: 38, 1.58: 39, 1.33: 40, 0.67: 41, 2.01: 42, 1.51: 43, 1.76: 44, 1.26: 45, 0.6: 46, 0.94: 47, 0.69: 48, 1.19: 49, 1.69: 50, 1.44: 51, 1684325040.0: 52, 1.12: 53, 1.37: 54, 1.62: 55, 2.12: 56, 0.8: 57, 1.05: 58, 1.3: 59, 1.55: 60, 0.64: 61, 1.8: 62, 0.98: 63, 0.73: 64, 0.48: 65, 1.23: 66, 1.48: 67, 1.73: 68, 0.57: 69, 1.66: 70, 1.41: 71, 0.83: 72, 0.87: 73, 0.82: 74, 0.89: 75, 0.86: 76, 0.92: 77, 0.84: 78, 0.68: 79, 0.93: 80, 0.85: 81, 1.34: 82, 1.59: 83, 1.84: 84, 0.77: 85, 0.52: 86, 0.66: 87, 1.02: 88, 1.52: 89, 1.27: 90, 0.61: 91, 1.77: 92, 0.7: 93, 0.95: 94, 1.2: 95, 1.7: 96, 1.45: 97, 1.13: 98, 1.88: 99, 1.38: 100, 0.81: 101, 0.56: 102, 1.06: 103, 1.31: 104, 1.56: 105, 0.9: 106, 0.65: 107, 0.74: 108, 0.99: 109, 1.24: 110, 1.49: 111, 0.58: 112, 0.31: 113, 1.67: 114, 1.42: 115, 1.1: 116, 1.35: 117, 0.78: 118, 1.53: 119, 1.03: 120, 0.62: 121, 0.71: 122, 1.21: 123, 1.46: 124, 1.71: 125, 1.14: 126, 1.39: 127, 2.32: 128, 1.07: 129, 1.57: 130, 1.32: 131, 0.91: 132, 1.82: 133, 0.53: 134, 2.03: 135, 0.46: 136, 0.47: 137, 0.944: 138, 1.79: 139, 1.63: 140, 0.36: 141, 0.25: 142, 1.94: 143, 1.74: 144, 0.41: 145, 0.49: 146}, {2238339752.0: 0, 1993550816.0: 1}, {2.8: 0, 3.0: 1, 2.0: 2, 5.4: 3, 7.6: 4, 8.9: 5, 8.5: 6, 8.4: 7, 9.0: 8, 11.0: 9, 10.0: 10, 14.0: 11, 15.0: 12, 9.1: 13, 17.0: 14, 18.0: 15, 19.0: 16, 13.0: 17, 21.0: 18, 26.0: 19, 29.0: 20, 33.0: 21, 34.0: 22, 35.0: 23, 7.0: 24, 37.0: 25, 39.0: 26, 42.0: 27, 46.0: 28, 47.0: 29, 49.0: 30, 50.0: 31, 51.0: 32, 52.0: 33, 54.0: 34, 55.0: 35, 56.0: 36, 57.0: 37, 58.0: 38, 59.0: 39, 60.0: 40, 61.0: 41, 62.0: 42, 63.0: 43, 64.0: 44, 65.0: 45, 66.0: 46, 67.0: 47, 68.0: 48, 69.0: 49, 70.0: 50, 71.0: 51, 72.0: 52, 73.0: 53, 74.0: 54, 75.0: 55, 76.0: 56, 77.0: 57, 78.0: 58, 79.0: 59, 80.0: 60, 81.0: 61, 82.0: 62, 83.0: 63, 84.0: 64, 85.0: 65, 86.0: 66, 87.0: 67, 88.0: 68, 89.0: 69, 90.0: 70, 91.0: 71, 92.0: 72, 93.0: 73, 94.0: 74, 95.0: 75, 96.0: 76, 97.0: 77, 98.0: 78, 99.0: 79, 100.0: 80, 101.0: 81, 102.0: 82, 103.0: 83, 104.0: 84, 105.0: 85, 106.0: 86, 107.0: 87, 108.0: 88, 109.0: 89, 110.0: 90, 111.0: 91, 112.0: 92, 113.0: 93, 114.0: 94, 115.0: 95, 116.0: 96, 117.0: 97, 118.0: 98, 119.0: 99, 120.0: 100, 121.0: 101, 122.0: 102, 123.0: 103, 124.0: 104, 125.0: 105, 126.0: 106, 127.0: 107, 128.0: 108, 129.0: 109, 130.0: 110, 131.0: 111, 132.0: 112, 133.0: 113, 134.0: 114, 135.0: 115, 136.0: 116, 137.0: 117, 138.0: 118, 139.0: 119, 140.0: 120, 141.0: 121, 142.0: 122, 143.0: 123, 144.0: 124, 145.0: 125, 146.0: 126, 147.0: 127, 148.0: 128, 149.0: 129, 150.0: 130, 151.0: 131, 152.0: 132, 153.0: 133, 154.0: 134, 155.0: 135, 156.0: 136, 157.0: 137, 158.0: 138, 159.0: 139, 160.0: 140, 161.0: 141, 162.0: 142, 163.0: 143, 164.0: 144, 165.0: 145, 166.0: 146, 167.0: 147, 168.0: 148, 169.0: 149, 170.0: 150, 171.0: 151, 172.0: 152, 173.0: 153, 174.0: 154, 175.0: 155, 1684325040.0: 156, 177.0: 157, 178.0: 158, 176.0: 159, 180.0: 160, 179.0: 161, 183.0: 162, 184.0: 163, 185.0: 164, 186.0: 165, 188.0: 166, 189.0: 167, 190.0: 168, 191.0: 169, 194.0: 170, 195.0: 171, 196.0: 172, 197.0: 173, 198.0: 174, 200.0: 175, 201.0: 176, 203.0: 177, 204.0: 178, 206.0: 179, 207.0: 180, 209.0: 181, 213.0: 182, 214.0: 183, 215.0: 184, 216.0: 185, 217.0: 186, 219.0: 187, 220.0: 188, 222.0: 189, 223.0: 190, 224.0: 191, 227.0: 192, 228.0: 193, 235.0: 194, 237.0: 195, 244.0: 196, 249.0: 197, 251.0: 198, 265.0: 199, 280.0: 200, 291.0: 201, 362.0: 202, 395.0: 203, 199.0: 204, 312.0: 205, 43.0: 206, 16.0: 207, 221.0: 208, 36.0: 209, 28.0: 210, 242.0: 211, 182.0: 212, 4.0: 213, 48.0: 214, 40.0: 215, 205.0: 216, 27.0: 217, 247.0: 218, 218.0: 219, 53.0: 220, 24.0: 221, 187.0: 222, 245.0: 223, 281.0: 224, 181.0: 225, 253.0: 226, 32.0: 227, 349.0: 228, 41.0: 229, 283.0: 230, 232.0: 231, 20.0: 232, 210.0: 233, 274.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {3646436640.0: 0, 596708387.0: 1, 1203304565.0: 2, 3655101910.0: 3, 1918519837.0: 4}]
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
            raise ValueError("All cells in the target column need to contain a class label.")
        try:
            result=int(value)
            if (not (result==0 or result==1)):
                raise ValueError("Integer class labels need to be 0 or 1.")
            if (not str(result) in clean.classlist):
                clean.classlist=clean.classlist+[str(result)]
            return result
        except:
            try:
                result=float(value)
                if (rounding!=-1):
                    result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
                if (not (result==0 or result==1)):
                    raise ValueError("Numeric class labels need to be 0 or 1.")
                if (not str(result) in clean.classlist):
                    clean.classlist=clean.classlist+[str(result)]
                return result
            except:
                result=(binascii.crc32(value.encode('utf8')) % (1<<32))
                if (result in clean.classlist):
                    result=clean.classlist.index(result)
                else:
                    clean.classlist=clean.classlist+[result]
                    result=clean.classlist.index(result)
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
    h_0 = max((((-5.878482 * float(x[0]))+ (0.20552675 * float(x[1]))+ (0.08976637 * float(x[2]))+ (-0.1526904 * float(x[3]))+ (0.29178822 * float(x[4]))+ (-0.22796209 * float(x[5]))+ (0.78354603 * float(x[6]))+ (0.92732555 * float(x[7]))+ (-0.23311697 * float(x[8]))+ (0.48031357 * float(x[9]))+ (0.05778984 * float(x[10]))+ (0.13608912 * float(x[11]))+ (0.85119325 * float(x[12]))+ (-0.85792786 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (-0.9595632 * float(x[15]))+ (0.6652397 * float(x[16]))+ (-1.2312768 * float(x[17]))+ (0.6368878 * float(x[18]))+ (-2.6545336 * float(x[19]))+ (0.59831715 * float(x[20]))+ (-7.333257 * float(x[21]))+ (0.56105834 * float(x[22]))+ (-2.0014877 * float(x[23]))+ (0.27984205 * float(x[24]))+ (-7.4462557 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-0.17067613 * float(x[28]))) + -0.0059079854), 0)
    h_1 = max((((4.90012 * float(x[0]))+ (-0.23559436 * float(x[1]))+ (-0.6721813 * float(x[2]))+ (0.29873067 * float(x[3]))+ (0.17053959 * float(x[4]))+ (-0.033111602 * float(x[5]))+ (0.07418153 * float(x[6]))+ (0.083964445 * float(x[7]))+ (0.5733836 * float(x[8]))+ (-0.39105523 * float(x[9]))+ (0.27819368 * float(x[10]))+ (0.045081966 * float(x[11]))+ (-0.68495506 * float(x[12]))+ (-0.08251589 * float(x[13]))+ (0.52216923 * float(x[14]))+ (-0.3197378 * float(x[15]))+ (-0.48435396 * float(x[16]))+ (21.74786 * float(x[17]))+ (0.3669301 * float(x[18]))+ (2.0460424 * float(x[19]))+ (0.030432517 * float(x[20]))+ (4.252876 * float(x[21]))+ (0.19302508 * float(x[22]))+ (3.0717301 * float(x[23]))+ (0.29609114 * float(x[24]))+ (4.5404553 * float(x[25]))+ (-0.54167783 * float(x[26]))+ (0.3081239 * float(x[27]))+ (0.14664751 * float(x[28]))) + 0.4888187), 0)
    o_0 = (-0.8188079 * h_0)+ (-0.4357815 * h_1) + -0.85125875

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

        model_cap=63

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


