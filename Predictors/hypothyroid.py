#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 16:25:20
# Invocation: btc -v -v hypothyroid-1.csv -o hypothyroid-1.py
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                92.28%
Model accuracy:                     93.50% (3527/3772 correct)
Improvement over best guess:        1.22% (of possible 7.72%)
Model capacity (MEC):               63 bits
Generalization ratio:               55.98 bits/bit
Model efficiency:                   0.01%/parameter
System behavior
True Negatives:                     91.54% (3453/3772)
True Positives:                     1.96% (74/3772)
False Negatives:                    5.75% (217/3772)
False Positives:                    0.74% (28/3772)
True Pos. Rate/Sensitivity/Recall:  0.25
True Neg. Rate/Specificity:         0.99
Precision:                          0.73
F-1 Measure:                        0.38
False Negative Rate/Miss Rate:      0.75
Critical Success Index:             0.23
Model bias:                         100.00% higher chance to pick class 1
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
TRAINFILE="hypothyroid-1.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 29

mappings = [{1304234792.0: 0, 3664761504.0: 1, 1684325040.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {2238339752.0: 0, 1993550816.0: 1}, {0.25: 0, 1.2: 1, 2.9: 2, 1.1: 3, 2.3: 4, 5.4: 5, 2.6: 6, 6.3: 7, 8.1: 8, 4.2: 9, 8.5: 10, 2.7: 11, 9.7: 12, 11.0: 13, 1.0: 14, 8.2: 15, 3.0: 16, 2.0: 17, 13.0: 18, 14.4: 19, 2.5: 20, 530.0: 21, 22.0: 22, 4.0: 23, 24.0: 24, 25.0: 25, 4.5: 26, 27.0: 27, 26.0: 28, 5.0: 29, 30.0: 30, 0.75: 31, 0.5: 32, 1.5: 33, 6.0: 34, 6.5: 35, 7.5: 36, 34.0: 37, 31.0: 38, 38.0: 39, 40.0: 40, 8.0: 41, 42.0: 42, 43.0: 43, 44.0: 44, 45.0: 45, 9.0: 46, 47.0: 47, 10.0: 48, 51.0: 49, 52.0: 50, 55.0: 51, 58.0: 52, 60.0: 53, 12.0: 54, 0.79: 55, 0.54: 56, 70.0: 57, 0.13: 58, 0.63: 59, 76.0: 60, 15.0: 61, 0.47: 62, 0.72: 63, 3.5: 64, 16.0: 65, 0.97: 66, 17.0: 67, 82.0: 68, 89.0: 69, 18.0: 70, 19.0: 71, 98.0: 72, 20.0: 73, 1.4: 74, 1.9: 75, 103.0: 76, 21.0: 77, 2.4: 78, 108.0: 79, 109.0: 80, 3.9: 81, 3.4: 82, 23.0: 83, 117.0: 84, 126.0: 85, 0.43: 86, 0.2: 87, 0.41: 88, 0.42: 89, 0.065: 90, 0.44: 91, 0.45: 92, 0.46: 93, 5.5: 94, 0.92: 95, 143.0: 96, 0.055: 97, 0.76: 98, 0.51: 99, 0.26: 100, 4.9: 101, 4.4: 102, 151.0: 103, 0.85: 104, 5.9: 105, 0.35: 106, 6.9: 107, 160.0: 108, 0.045: 109, 0.94: 110, 0.69: 111, 7.4: 112, 7.9: 113, 8.9: 114, 9.9: 115, 9.4: 116, 1684325040.0: 117, 178.0: 118, 7.0: 119, 36.0: 120, 183.0: 121, 11.4: 122, 0.15: 123, 0.035: 124, 0.16: 125, 2.8: 126, 3.8: 127, 0.8: 128, 3.3: 129, 0.55: 130, 0.025: 131, 4.8: 132, 1.8: 133, 1.3: 134, 4.3: 135, 0.14: 136, 5.8: 137, 5.3: 138, 0.64: 139, 0.39: 140, 0.89: 141, 6.8: 142, 0.015: 143, 0.98: 144, 0.73: 145, 0.48: 146, 7.3: 147, 7.8: 148, 0.59: 149, 0.57: 150, 0.005: 151, 0.6: 152, 0.93: 153, 0.67: 154, 0.62: 155, 0.88: 156, 0.38: 157, 0.4: 158, 0.82: 159, 0.9: 160, 0.84: 161, 0.68: 162, 0.52: 163, 0.77: 164, 0.27: 165, 0.19: 166, 1.02: 167, 0.66: 168, 0.86: 169, 0.36: 170, 0.61: 171, 8.3: 172, 9.2: 173, 9.8: 174, 0.95: 175, 0.7: 176, 1.7: 177, 0.1: 178, 0.09: 179, 14.8: 180, 2.2: 181, 3.2: 182, 3.7: 183, 4.7: 184, 5.2: 185, 5.7: 186, 6.2: 187, 6.7: 188, 0.56: 189, 0.81: 190, 0.08: 191, 7.2: 192, 7.7: 193, 0.65: 194, 0.74: 195, 0.99: 196, 0.24: 197, 0.49: 198, 0.58: 199, 0.83: 200, 0.31: 201, 0.07: 202, 0.3: 203, 0.33: 204, 0.06: 205, 0.32: 206, 0.34: 207, 0.29: 208, 400.0: 209, 4.6: 210, 4.1: 211, 5.6: 212, 5.1: 213, 6.1: 214, 0.05: 215, 6.6: 216, 7.1: 217, 7.6: 218, 1.6: 219, 8.6: 220, 2.1: 221, 9.6: 222, 9.1: 223, 0.04: 224, 3.1: 225, 0.78: 226, 0.28: 227, 3.6: 228, 0.53: 229, 440.0: 230, 0.12: 231, 11.1: 232, 0.87: 233, 0.03: 234, 0.71: 235, 0.96: 236, 468.0: 237, 472.0: 238, 0.02: 239, 0.01: 240, 0.91: 241, 1.01: 242, 5.73: 243, 478.0: 244, 145.0: 245, 26.4: 246, 66.0: 247, 0.37: 248, 61.0: 249, 9.3: 250, 10.3: 251, 50.0: 252, 8.8: 253, 0.21: 254, 116.0: 255, 32.0: 256, 0.17: 257, 46.0: 258, 236.0: 259, 14.0: 260, 80.0: 261, 9.5: 262, 65.0: 263, 28.0: 264, 0.22: 265, 29.0: 266, 6.4: 267, 139.0: 268, 230.0: 269, 54.0: 270, 30.5: 271, 12.1: 272, 41.0: 273, 8.4: 274, 18.4: 275, 0.18: 276, 78.0: 277, 35.0: 278, 86.0: 279, 39.0: 280, 33.0: 281, 100.0: 282, 0.23: 283, 99.0: 284, 165.0: 285, 199.0: 286, 188.0: 287}, {2238339752.0: 0, 1993550816.0: 1}, {0.7: 0, 1.4: 1, 2.0: 2, 2.2: 3, 2.4: 4, 1.6: 5, 1.1: 6, 0.8: 7, 2.3: 8, 1.5: 9, 2.1: 10, 3.8: 11, 1.0: 12, 2.5: 13, 5.4: 14, 0.5: 15, 3.5: 16, 3.0: 17, 4.0: 18, 4.1: 19, 4.5: 20, 4.6: 21, 0.6: 22, 5.1: 23, 5.0: 24, 4.9: 25, 5.5: 26, 6.6: 27, 7.6: 28, 7.0: 29, 7.1: 30, 8.5: 31, 2.6: 32, 1684325040.0: 33, 3.6: 34, 3.1: 35, 0.2: 36, 1.2: 37, 1.7: 38, 0.1: 39, 2.7: 40, 2.8: 41, 0.3: 42, 3.7: 43, 3.2: 44, 3.3: 45, 4.3: 46, 1.8: 47, 1.3: 48, 4.7: 49, 4.8: 50, 5.3: 51, 5.7: 52, 4.2: 53, 5.2: 54, 6.7: 55, 7.3: 56, 1.9: 57, 0.9: 58, 0.4: 59, 2.9: 60, 3.4: 61, 3.9: 62, 6.2: 63, 6.1: 64, 4.4: 65, 6.0: 66, 1.44: 67, 0.05: 68, 10.6: 69}, {2238339752.0: 0, 1993550816.0: 1}, {2.9: 0, 3.0: 1, 2.0: 2, 5.8: 3, 6.0: 4, 4.8: 5, 9.5: 6, 10.0: 7, 11.0: 8, 12.0: 9, 13.0: 10, 14.0: 11, 15.0: 12, 17.0: 13, 19.0: 14, 21.0: 15, 22.0: 16, 24.0: 17, 28.0: 18, 29.0: 19, 30.0: 20, 31.0: 21, 33.0: 22, 34.0: 23, 35.0: 24, 37.0: 25, 38.0: 26, 39.0: 27, 40.0: 28, 41.0: 29, 42.0: 30, 44.0: 31, 45.0: 32, 46.0: 33, 48.0: 34, 50.0: 35, 51.0: 36, 52.0: 37, 53.0: 38, 54.0: 39, 56.0: 40, 57.0: 41, 58.0: 42, 59.0: 43, 60.0: 44, 61.0: 45, 62.0: 46, 63.0: 47, 64.0: 48, 65.0: 49, 66.0: 50, 67.0: 51, 68.0: 52, 69.0: 53, 70.0: 54, 71.0: 55, 72.0: 56, 73.0: 57, 74.0: 58, 75.0: 59, 76.0: 60, 77.0: 61, 78.0: 62, 79.0: 63, 80.0: 64, 81.0: 65, 82.0: 66, 83.0: 67, 84.0: 68, 85.0: 69, 86.0: 70, 87.0: 71, 88.0: 72, 89.0: 73, 90.0: 74, 91.0: 75, 92.0: 76, 93.0: 77, 94.0: 78, 95.0: 79, 96.0: 80, 97.0: 81, 98.0: 82, 99.0: 83, 100.0: 84, 101.0: 85, 102.0: 86, 103.0: 87, 104.0: 88, 105.0: 89, 106.0: 90, 107.0: 91, 108.0: 92, 109.0: 93, 110.0: 94, 111.0: 95, 112.0: 96, 113.0: 97, 114.0: 98, 115.0: 99, 116.0: 100, 117.0: 101, 118.0: 102, 119.0: 103, 120.0: 104, 121.0: 105, 122.0: 106, 123.0: 107, 124.0: 108, 125.0: 109, 126.0: 110, 127.0: 111, 128.0: 112, 129.0: 113, 130.0: 114, 131.0: 115, 132.0: 116, 133.0: 117, 134.0: 118, 135.0: 119, 136.0: 120, 137.0: 121, 138.0: 122, 139.0: 123, 140.0: 124, 141.0: 125, 142.0: 126, 143.0: 127, 144.0: 128, 145.0: 129, 146.0: 130, 147.0: 131, 148.0: 132, 149.0: 133, 150.0: 134, 151.0: 135, 152.0: 136, 153.0: 137, 154.0: 138, 155.0: 139, 156.0: 140, 157.0: 141, 158.0: 142, 159.0: 143, 160.0: 144, 161.0: 145, 162.0: 146, 163.0: 147, 164.0: 148, 165.0: 149, 166.0: 150, 167.0: 151, 168.0: 152, 169.0: 153, 170.0: 154, 171.0: 155, 172.0: 156, 174.0: 157, 175.0: 158, 1684325040.0: 159, 176.0: 160, 177.0: 161, 179.0: 162, 180.0: 163, 181.0: 164, 182.0: 165, 183.0: 166, 184.0: 167, 178.0: 168, 187.0: 169, 189.0: 170, 191.0: 171, 192.0: 172, 193.0: 173, 194.0: 174, 196.0: 175, 197.0: 176, 198.0: 177, 199.0: 178, 200.0: 179, 201.0: 180, 203.0: 181, 205.0: 182, 207.0: 183, 209.0: 184, 210.0: 185, 211.0: 186, 212.0: 187, 213.0: 188, 220.0: 189, 223.0: 190, 225.0: 191, 230.0: 192, 231.0: 193, 232.0: 194, 237.0: 195, 239.0: 196, 244.0: 197, 248.0: 198, 252.0: 199, 253.0: 200, 257.0: 201, 261.0: 202, 263.0: 203, 289.0: 204, 372.0: 205, 430.0: 206, 255.0: 207, 214.0: 208, 186.0: 209, 49.0: 210, 195.0: 211, 240.0: 212, 258.0: 213, 32.0: 214, 204.0: 215, 16.0: 216, 250.0: 217, 273.0: 218, 18.0: 219, 217.0: 220, 216.0: 221, 4.0: 222, 47.0: 223, 36.0: 224, 222.0: 225, 188.0: 226, 206.0: 227, 23.0: 228, 272.0: 229, 233.0: 230, 27.0: 231, 173.0: 232, 226.0: 233, 301.0: 234, 256.0: 235, 246.0: 236, 25.0: 237, 235.0: 238, 43.0: 239, 55.0: 240, 219.0: 241}, {2238339752.0: 0, 1993550816.0: 1}, {0.96: 0, 1.15: 1, 1.09: 2, 1.28: 3, 1.54: 4, 1.16: 5, 0.75: 6, 1.17: 7, 1.08: 8, 0.76: 9, 1.01: 10, 1.25: 11, 1.0: 12, 1.75: 13, 0.5: 14, 1.5: 15, 0.59: 16, 1.18: 17, 1.68: 18, 1.43: 19, 1.93: 20, 1.11: 21, 1.36: 22, 1.61: 23, 0.79: 24, 0.54: 25, 1.04: 26, 1.29: 27, 0.88: 28, 0.63: 29, 0.38: 30, 0.97: 31, 0.72: 32, 1.22: 33, 1.97: 34, 1.47: 35, 1.65: 36, 1.4: 37, 1.83: 38, 1.58: 39, 0.67: 40, 1.51: 41, 1.76: 42, 1.26: 43, 0.6: 44, 0.94: 45, 0.69: 46, 1.19: 47, 1.69: 48, 1.44: 49, 1684325040.0: 50, 1.12: 51, 1.37: 52, 1.62: 53, 2.12: 54, 0.8: 55, 1.05: 56, 1.3: 57, 1.55: 58, 0.64: 59, 1.8: 60, 0.98: 61, 0.73: 62, 0.48: 63, 1.23: 64, 1.48: 65, 1.73: 66, 0.57: 67, 1.66: 68, 0.83: 69, 0.87: 70, 0.82: 71, 0.89: 72, 0.86: 73, 0.92: 74, 0.84: 75, 0.68: 76, 0.93: 77, 0.85: 78, 1.34: 79, 1.59: 80, 1.84: 81, 0.77: 82, 0.52: 83, 0.66: 84, 1.02: 85, 1.52: 86, 1.27: 87, 0.61: 88, 1.77: 89, 0.7: 90, 0.95: 91, 1.2: 92, 1.7: 93, 1.45: 94, 1.13: 95, 1.88: 96, 1.38: 97, 0.81: 98, 0.56: 99, 1.06: 100, 1.31: 101, 1.56: 102, 0.9: 103, 0.65: 104, 0.74: 105, 0.99: 106, 1.24: 107, 1.49: 108, 0.58: 109, 0.31: 110, 1.67: 111, 1.42: 112, 1.1: 113, 1.35: 114, 0.78: 115, 1.53: 116, 1.03: 117, 0.62: 118, 0.71: 119, 1.21: 120, 1.46: 121, 1.71: 122, 1.14: 123, 1.39: 124, 2.32: 125, 1.07: 126, 1.57: 127, 1.32: 128, 0.91: 129, 1.82: 130, 1.33: 131, 1.41: 132, 2.01: 133, 0.53: 134, 2.03: 135, 0.46: 136, 0.47: 137, 0.944: 138, 1.79: 139, 1.63: 140, 0.36: 141, 0.25: 142, 1.94: 143, 1.74: 144, 0.41: 145, 0.49: 146}, {2238339752.0: 0, 1993550816.0: 1}, {2.8: 0, 3.0: 1, 2.0: 2, 5.4: 3, 7.6: 4, 8.9: 5, 8.5: 6, 8.4: 7, 9.0: 8, 11.0: 9, 10.0: 10, 14.0: 11, 15.0: 12, 9.1: 13, 17.0: 14, 18.0: 15, 19.0: 16, 13.0: 17, 21.0: 18, 26.0: 19, 29.0: 20, 33.0: 21, 35.0: 22, 7.0: 23, 37.0: 24, 39.0: 25, 42.0: 26, 46.0: 27, 47.0: 28, 49.0: 29, 50.0: 30, 51.0: 31, 54.0: 32, 55.0: 33, 56.0: 34, 58.0: 35, 59.0: 36, 60.0: 37, 61.0: 38, 62.0: 39, 63.0: 40, 64.0: 41, 65.0: 42, 66.0: 43, 67.0: 44, 68.0: 45, 69.0: 46, 70.0: 47, 71.0: 48, 72.0: 49, 73.0: 50, 74.0: 51, 75.0: 52, 76.0: 53, 77.0: 54, 78.0: 55, 79.0: 56, 80.0: 57, 81.0: 58, 82.0: 59, 83.0: 60, 84.0: 61, 85.0: 62, 86.0: 63, 87.0: 64, 88.0: 65, 89.0: 66, 90.0: 67, 91.0: 68, 92.0: 69, 93.0: 70, 94.0: 71, 95.0: 72, 96.0: 73, 97.0: 74, 98.0: 75, 99.0: 76, 100.0: 77, 101.0: 78, 102.0: 79, 103.0: 80, 104.0: 81, 105.0: 82, 106.0: 83, 107.0: 84, 108.0: 85, 109.0: 86, 110.0: 87, 111.0: 88, 112.0: 89, 113.0: 90, 114.0: 91, 115.0: 92, 116.0: 93, 117.0: 94, 118.0: 95, 119.0: 96, 120.0: 97, 121.0: 98, 122.0: 99, 123.0: 100, 124.0: 101, 125.0: 102, 126.0: 103, 127.0: 104, 128.0: 105, 129.0: 106, 130.0: 107, 131.0: 108, 132.0: 109, 133.0: 110, 134.0: 111, 135.0: 112, 136.0: 113, 137.0: 114, 138.0: 115, 139.0: 116, 140.0: 117, 141.0: 118, 142.0: 119, 143.0: 120, 144.0: 121, 145.0: 122, 146.0: 123, 147.0: 124, 148.0: 125, 149.0: 126, 150.0: 127, 151.0: 128, 152.0: 129, 153.0: 130, 154.0: 131, 155.0: 132, 156.0: 133, 157.0: 134, 158.0: 135, 159.0: 136, 160.0: 137, 161.0: 138, 162.0: 139, 163.0: 140, 164.0: 141, 165.0: 142, 166.0: 143, 167.0: 144, 168.0: 145, 169.0: 146, 170.0: 147, 171.0: 148, 172.0: 149, 173.0: 150, 174.0: 151, 175.0: 152, 1684325040.0: 153, 177.0: 154, 178.0: 155, 176.0: 156, 180.0: 157, 183.0: 158, 184.0: 159, 185.0: 160, 186.0: 161, 188.0: 162, 189.0: 163, 190.0: 164, 191.0: 165, 194.0: 166, 195.0: 167, 196.0: 168, 197.0: 169, 198.0: 170, 200.0: 171, 201.0: 172, 203.0: 173, 204.0: 174, 206.0: 175, 207.0: 176, 209.0: 177, 213.0: 178, 214.0: 179, 215.0: 180, 216.0: 181, 217.0: 182, 219.0: 183, 220.0: 184, 222.0: 185, 224.0: 186, 235.0: 187, 249.0: 188, 251.0: 189, 265.0: 190, 280.0: 191, 291.0: 192, 362.0: 193, 395.0: 194, 244.0: 195, 34.0: 196, 228.0: 197, 57.0: 198, 52.0: 199, 237.0: 200, 223.0: 201, 179.0: 202, 227.0: 203, 199.0: 204, 312.0: 205, 43.0: 206, 16.0: 207, 221.0: 208, 36.0: 209, 28.0: 210, 242.0: 211, 182.0: 212, 4.0: 213, 48.0: 214, 40.0: 215, 205.0: 216, 27.0: 217, 247.0: 218, 218.0: 219, 53.0: 220, 24.0: 221, 187.0: 222, 245.0: 223, 281.0: 224, 181.0: 225, 253.0: 226, 32.0: 227, 349.0: 228, 41.0: 229, 283.0: 230, 232.0: 231, 20.0: 232, 210.0: 233, 274.0: 234}, {1993550816.0: 0}, {1684325040.0: 0}, {3646436640.0: 0, 596708387.0: 1, 1203304565.0: 2, 3655101910.0: 3, 1918519837.0: 4}]
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
    h_0 = max((((-0.029882696 * float(x[0]))+ (29.855007 * float(x[1]))+ (35.40287 * float(x[2]))+ (-8.706339 * float(x[3]))+ (50.117935 * float(x[4]))+ (-11.964555 * float(x[5]))+ (47.399662 * float(x[6]))+ (40.30522 * float(x[7]))+ (-6.4618683 * float(x[8]))+ (-32.300465 * float(x[9]))+ (9.941693 * float(x[10]))+ (48.79053 * float(x[11]))+ (49.16536 * float(x[12]))+ (12.510127 * float(x[13]))+ (-0.8257414 * float(x[14]))+ (15.294056 * float(x[15]))+ (52.630726 * float(x[16]))+ (0.07938144 * float(x[17]))+ (-13.773563 * float(x[18]))+ (-0.05699416 * float(x[19]))+ (-35.38818 * float(x[20]))+ (0.75866413 * float(x[21]))+ (-40.612907 * float(x[22]))+ (0.080227256 * float(x[23]))+ (-40.894176 * float(x[24]))+ (0.81509036 * float(x[25]))+ (0.88933784 * float(x[26]))+ (0.043696642 * float(x[27]))+ (-2.9598029 * float(x[28]))) + -2.698026), 0)
    h_1 = max((((-1.2238773 * float(x[0]))+ (0.867157 * float(x[1]))+ (-0.9124103 * float(x[2]))+ (0.8218965 * float(x[3]))+ (0.17270157 * float(x[4]))+ (0.65931153 * float(x[5]))+ (0.16249523 * float(x[6]))+ (-0.91809106 * float(x[7]))+ (-0.68984854 * float(x[8]))+ (0.81960297 * float(x[9]))+ (0.66274256 * float(x[10]))+ (0.95624906 * float(x[11]))+ (-0.40394607 * float(x[12]))+ (-0.8456451 * float(x[13]))+ (0.91711956 * float(x[14]))+ (-0.5875069 * float(x[15]))+ (-0.5738755 * float(x[16]))+ (-0.6268158 * float(x[17]))+ (-0.29765055 * float(x[18]))+ (-1.223453 * float(x[19]))+ (0.031409893 * float(x[20]))+ (-2.14212 * float(x[21]))+ (-0.5167477 * float(x[22]))+ (-0.65025014 * float(x[23]))+ (0.32645535 * float(x[24]))+ (-0.7440124 * float(x[25]))+ (0.49383533 * float(x[26]))+ (0.3623441 * float(x[27]))+ (-0.83040154 * float(x[28]))) + 0.735787), 0)
    o_0 = (-0.041979466 * h_0)+ (0.1770837 * h_1) + 2.7347364

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


