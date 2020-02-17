#!/usr/bin/env python3
#
# Output of Brainome Daimensions(tm) Table Compiler v0.5.
# Compile time: Feb-16-2020 09:29:23
# Invocation: btc -target class -v -v file7b5365fa741c-10.csv -o file7b5365fa741c-10.py -f ME
# This source code requires Python 3.
#
"""
System Type:                        Binary classifier
Best-guess accuracy:                75.18%
Model accuracy:                     78.20% (3243/4147 correct)
Improvement over best guess:        3.02% (of possible 24.82%)
Model capacity (MEC):               607 bits
Generalization ratio:               5.34 bits/bit
Model efficiency:                   0.00%/parameter
System behavior
True Negatives:                     63.32% (2626/4147)
True Positives:                     14.88% (617/4147)
False Negatives:                    9.93% (412/4147)
False Positives:                    11.86% (492/4147)
True Pos. Rate/Sensitivity/Recall:  0.60
True Neg. Rate/Specificity:         0.84
Precision:                          0.56
F-1 Measure:                        0.58
False Negative Rate/Miss Rate:      0.40
Critical Success Index:             0.41
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF=100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE="file7b5365fa741c-10.csv"


#Number of attributes
num_attr = 48

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


# Calculate equilibrium energy ($_i=1)
def eqenergy(row):
    result=0
    for elem in row:
        result = result + float(elem)
    return result

# Classifier 
def classify(row):
    energy=eqenergy(row)
    if (energy>2731.0):
        return 1.0
    if (energy>2698.5):
        return 0.0
    if (energy>2666.0):
        return 1.0
    if (energy>2621.5):
        return 0.0
    if (energy>2444.0):
        return 1.0
    if (energy>2432.0):
        return 0.0
    if (energy>2393.0):
        return 1.0
    if (energy>2388.5):
        return 0.0
    if (energy>2380.0):
        return 1.0
    if (energy>2371.5):
        return 0.0
    if (energy>2355.0):
        return 1.0
    if (energy>2343.5):
        return 0.0
    if (energy>2329.5):
        return 1.0
    if (energy>2324.5):
        return 0.0
    if (energy>2279.0):
        return 1.0
    if (energy>2274.0):
        return 0.0
    if (energy>2265.5):
        return 1.0
    if (energy>2264.0):
        return 0.0
    if (energy>2263.5):
        return 1.0
    if (energy>2260.5):
        return 0.0
    if (energy>2242.5):
        return 1.0
    if (energy>2238.0):
        return 0.0
    if (energy>2236.0):
        return 1.0
    if (energy>2233.5):
        return 0.0
    if (energy>2229.5):
        return 1.0
    if (energy>2226.5):
        return 0.0
    if (energy>2224.0):
        return 1.0
    if (energy>2222.5):
        return 0.0
    if (energy>2219.0):
        return 1.0
    if (energy>2209.0):
        return 0.0
    if (energy>2206.0):
        return 1.0
    if (energy>2205.0):
        return 0.0
    if (energy>2196.5):
        return 1.0
    if (energy>2196.0):
        return 0.0
    if (energy>2167.5):
        return 1.0
    if (energy>2164.0):
        return 0.0
    if (energy>2162.5):
        return 1.0
    if (energy>2160.5):
        return 0.0
    if (energy>2155.0):
        return 1.0
    if (energy>2153.0):
        return 0.0
    if (energy>2151.0):
        return 1.0
    if (energy>2151.0):
        return 0.0
    if (energy>2149.5):
        return 1.0
    if (energy>2145.0):
        return 0.0
    if (energy>2141.5):
        return 1.0
    if (energy>2140.5):
        return 0.0
    if (energy>2137.5):
        return 1.0
    if (energy>2135.5):
        return 0.0
    if (energy>2128.5):
        return 1.0
    if (energy>2128.0):
        return 0.0
    if (energy>2127.5):
        return 1.0
    if (energy>2125.5):
        return 0.0
    if (energy>2124.5):
        return 1.0
    if (energy>2122.5):
        return 0.0
    if (energy>2121.0):
        return 1.0
    if (energy>2118.5):
        return 0.0
    if (energy>2092.0):
        return 1.0
    if (energy>2089.5):
        return 0.0
    if (energy>2088.5):
        return 1.0
    if (energy>2087.5):
        return 0.0
    if (energy>2084.5):
        return 1.0
    if (energy>2082.5):
        return 0.0
    if (energy>2080.0):
        return 1.0
    if (energy>2079.5):
        return 0.0
    if (energy>2069.5):
        return 1.0
    if (energy>2068.5):
        return 0.0
    if (energy>2067.5):
        return 1.0
    if (energy>2066.5):
        return 0.0
    if (energy>2063.5):
        return 1.0
    if (energy>2063.0):
        return 0.0
    if (energy>2060.5):
        return 1.0
    if (energy>2060.0):
        return 0.0
    if (energy>2056.5):
        return 1.0
    if (energy>2055.5):
        return 0.0
    if (energy>2054.0):
        return 1.0
    if (energy>2050.5):
        return 0.0
    if (energy>2035.5):
        return 1.0
    if (energy>2033.5):
        return 0.0
    if (energy>2031.0):
        return 1.0
    if (energy>2031.0):
        return 0.0
    if (energy>2030.5):
        return 1.0
    if (energy>2029.5):
        return 0.0
    if (energy>2029.0):
        return 1.0
    if (energy>2029.0):
        return 0.0
    if (energy>2028.5):
        return 1.0
    if (energy>2026.0):
        return 0.0
    if (energy>2025.0):
        return 1.0
    if (energy>2025.0):
        return 0.0
    if (energy>2021.5):
        return 1.0
    if (energy>2020.5):
        return 0.0
    if (energy>2020.0):
        return 1.0
    if (energy>2019.5):
        return 0.0
    if (energy>2017.0):
        return 1.0
    if (energy>2015.0):
        return 0.0
    if (energy>2013.5):
        return 1.0
    if (energy>2011.0):
        return 0.0
    if (energy>2011.0):
        return 1.0
    if (energy>2011.0):
        return 0.0
    if (energy>2009.5):
        return 1.0
    if (energy>2008.5):
        return 0.0
    if (energy>2007.0):
        return 1.0
    if (energy>2007.0):
        return 0.0
    if (energy>2004.0):
        return 1.0
    if (energy>2000.5):
        return 0.0
    if (energy>1999.5):
        return 1.0
    if (energy>1998.5):
        return 0.0
    if (energy>1997.0):
        return 1.0
    if (energy>1996.0):
        return 0.0
    if (energy>1995.0):
        return 1.0
    if (energy>1994.5):
        return 0.0
    if (energy>1993.0):
        return 1.0
    if (energy>1991.5):
        return 0.0
    if (energy>1991.0):
        return 1.0
    if (energy>1991.0):
        return 0.0
    if (energy>1990.5):
        return 1.0
    if (energy>1988.0):
        return 0.0
    if (energy>1986.0):
        return 1.0
    if (energy>1984.5):
        return 0.0
    if (energy>1979.0):
        return 1.0
    if (energy>1978.0):
        return 0.0
    if (energy>1977.5):
        return 1.0
    if (energy>1975.5):
        return 0.0
    if (energy>1973.5):
        return 1.0
    if (energy>1972.0):
        return 0.0
    if (energy>1965.5):
        return 1.0
    if (energy>1965.0):
        return 0.0
    if (energy>1964.5):
        return 1.0
    if (energy>1963.0):
        return 0.0
    if (energy>1961.5):
        return 1.0
    if (energy>1960.0):
        return 0.0
    if (energy>1959.0):
        return 1.0
    if (energy>1958.0):
        return 0.0
    if (energy>1957.0):
        return 1.0
    if (energy>1956.5):
        return 0.0
    if (energy>1955.5):
        return 1.0
    if (energy>1954.0):
        return 0.0
    if (energy>1954.0):
        return 1.0
    if (energy>1953.0):
        return 0.0
    if (energy>1952.0):
        return 1.0
    if (energy>1951.0):
        return 0.0
    if (energy>1949.0):
        return 1.0
    if (energy>1946.5):
        return 0.0
    if (energy>1943.5):
        return 1.0
    if (energy>1941.0):
        return 0.0
    if (energy>1940.0):
        return 1.0
    if (energy>1939.5):
        return 0.0
    if (energy>1937.5):
        return 1.0
    if (energy>1936.5):
        return 0.0
    if (energy>1931.5):
        return 1.0
    if (energy>1930.5):
        return 0.0
    if (energy>1930.0):
        return 1.0
    if (energy>1928.0):
        return 0.0
    if (energy>1926.0):
        return 1.0
    if (energy>1923.5):
        return 0.0
    if (energy>1922.5):
        return 1.0
    if (energy>1921.0):
        return 0.0
    if (energy>1920.0):
        return 1.0
    if (energy>1914.5):
        return 0.0
    if (energy>1913.0):
        return 1.0
    if (energy>1912.0):
        return 0.0
    if (energy>1909.5):
        return 1.0
    if (energy>1905.5):
        return 0.0
    if (energy>1904.0):
        return 1.0
    if (energy>1903.5):
        return 0.0
    if (energy>1901.0):
        return 1.0
    if (energy>1899.5):
        return 0.0
    if (energy>1898.5):
        return 1.0
    if (energy>1898.0):
        return 0.0
    if (energy>1897.0):
        return 1.0
    if (energy>1895.0):
        return 0.0
    if (energy>1893.5):
        return 1.0
    if (energy>1892.5):
        return 0.0
    if (energy>1891.5):
        return 1.0
    if (energy>1891.0):
        return 0.0
    if (energy>1891.0):
        return 1.0
    if (energy>1890.5):
        return 0.0
    if (energy>1890.0):
        return 1.0
    if (energy>1889.0):
        return 0.0
    if (energy>1888.0):
        return 1.0
    if (energy>1886.5):
        return 0.0
    if (energy>1886.0):
        return 1.0
    if (energy>1885.5):
        return 0.0
    if (energy>1884.5):
        return 1.0
    if (energy>1881.0):
        return 0.0
    if (energy>1881.0):
        return 1.0
    if (energy>1877.5):
        return 0.0
    if (energy>1876.5):
        return 1.0
    if (energy>1872.5):
        return 0.0
    if (energy>1871.0):
        return 1.0
    if (energy>1871.0):
        return 0.0
    if (energy>1870.5):
        return 1.0
    if (energy>1869.5):
        return 0.0
    if (energy>1869.0):
        return 1.0
    if (energy>1867.5):
        return 0.0
    if (energy>1866.5):
        return 1.0
    if (energy>1865.0):
        return 0.0
    if (energy>1864.0):
        return 1.0
    if (energy>1861.0):
        return 0.0
    if (energy>1860.5):
        return 1.0
    if (energy>1860.0):
        return 0.0
    if (energy>1860.0):
        return 1.0
    if (energy>1860.0):
        return 0.0
    if (energy>1860.0):
        return 1.0
    if (energy>1860.0):
        return 0.0
    if (energy>1859.0):
        return 1.0
    if (energy>1858.0):
        return 0.0
    if (energy>1857.5):
        return 1.0
    if (energy>1856.0):
        return 0.0
    if (energy>1855.5):
        return 1.0
    if (energy>1855.0):
        return 0.0
    if (energy>1851.5):
        return 1.0
    if (energy>1851.0):
        return 0.0
    if (energy>1850.5):
        return 1.0
    if (energy>1849.5):
        return 0.0
    if (energy>1849.0):
        return 1.0
    if (energy>1847.5):
        return 0.0
    if (energy>1844.5):
        return 1.0
    if (energy>1843.0):
        return 0.0
    if (energy>1838.5):
        return 1.0
    if (energy>1835.0):
        return 0.0
    if (energy>1833.0):
        return 1.0
    if (energy>1832.5):
        return 0.0
    if (energy>1832.0):
        return 1.0
    if (energy>1831.5):
        return 0.0
    if (energy>1831.0):
        return 1.0
    if (energy>1830.0):
        return 0.0
    if (energy>1829.0):
        return 1.0
    if (energy>1828.5):
        return 0.0
    if (energy>1828.0):
        return 1.0
    if (energy>1828.0):
        return 0.0
    if (energy>1827.5):
        return 1.0
    if (energy>1827.0):
        return 0.0
    if (energy>1826.0):
        return 1.0
    if (energy>1826.0):
        return 0.0
    if (energy>1826.0):
        return 1.0
    if (energy>1825.5):
        return 0.0
    if (energy>1825.0):
        return 1.0
    if (energy>1825.0):
        return 0.0
    if (energy>1825.0):
        return 1.0
    if (energy>1824.5):
        return 0.0
    if (energy>1823.5):
        return 1.0
    if (energy>1823.0):
        return 0.0
    if (energy>1823.0):
        return 1.0
    if (energy>1821.5):
        return 0.0
    if (energy>1820.0):
        return 1.0
    if (energy>1819.0):
        return 0.0
    if (energy>1819.0):
        return 1.0
    if (energy>1817.5):
        return 0.0
    if (energy>1816.5):
        return 1.0
    if (energy>1815.0):
        return 0.0
    if (energy>1814.0):
        return 1.0
    if (energy>1814.0):
        return 0.0
    if (energy>1813.5):
        return 1.0
    if (energy>1812.5):
        return 0.0
    if (energy>1812.0):
        return 1.0
    if (energy>1810.0):
        return 0.0
    if (energy>1807.5):
        return 1.0
    if (energy>1807.0):
        return 0.0
    if (energy>1805.5):
        return 1.0
    if (energy>1805.0):
        return 0.0
    if (energy>1804.0):
        return 1.0
    if (energy>1804.0):
        return 0.0
    if (energy>1803.5):
        return 1.0
    if (energy>1803.0):
        return 0.0
    if (energy>1802.0):
        return 1.0
    if (energy>1801.0):
        return 0.0
    if (energy>1800.5):
        return 1.0
    if (energy>1799.5):
        return 0.0
    if (energy>1799.0):
        return 1.0
    if (energy>1797.5):
        return 0.0
    if (energy>1796.0):
        return 1.0
    if (energy>1795.0):
        return 0.0
    if (energy>1794.5):
        return 1.0
    if (energy>1792.0):
        return 0.0
    if (energy>1791.0):
        return 1.0
    if (energy>1788.0):
        return 0.0
    if (energy>1786.0):
        return 1.0
    if (energy>1784.0):
        return 0.0
    if (energy>1784.0):
        return 1.0
    if (energy>1784.0):
        return 0.0
    if (energy>1783.5):
        return 1.0
    if (energy>1782.0):
        return 0.0
    if (energy>1781.5):
        return 1.0
    if (energy>1780.0):
        return 0.0
    if (energy>1778.5):
        return 1.0
    if (energy>1778.0):
        return 0.0
    if (energy>1776.5):
        return 1.0
    if (energy>1776.0):
        return 0.0
    if (energy>1775.5):
        return 1.0
    if (energy>1775.0):
        return 0.0
    if (energy>1774.5):
        return 1.0
    if (energy>1774.0):
        return 0.0
    if (energy>1774.0):
        return 1.0
    if (energy>1773.0):
        return 0.0
    if (energy>1771.0):
        return 1.0
    if (energy>1770.0):
        return 0.0
    if (energy>1770.0):
        return 1.0
    if (energy>1769.0):
        return 0.0
    if (energy>1769.0):
        return 1.0
    if (energy>1769.0):
        return 0.0
    if (energy>1769.0):
        return 1.0
    if (energy>1767.0):
        return 0.0
    if (energy>1767.0):
        return 1.0
    if (energy>1765.5):
        return 0.0
    if (energy>1764.5):
        return 1.0
    if (energy>1764.0):
        return 0.0
    if (energy>1764.0):
        return 1.0
    if (energy>1762.0):
        return 0.0
    if (energy>1761.5):
        return 1.0
    if (energy>1759.0):
        return 0.0
    if (energy>1758.5):
        return 1.0
    if (energy>1758.0):
        return 0.0
    if (energy>1758.0):
        return 1.0
    if (energy>1757.0):
        return 0.0
    if (energy>1757.0):
        return 1.0
    if (energy>1756.0):
        return 0.0
    if (energy>1755.0):
        return 1.0
    if (energy>1750.0):
        return 0.0
    if (energy>1749.5):
        return 1.0
    if (energy>1749.0):
        return 0.0
    if (energy>1748.0):
        return 1.0
    if (energy>1747.0):
        return 0.0
    if (energy>1747.0):
        return 1.0
    if (energy>1747.0):
        return 0.0
    if (energy>1746.0):
        return 1.0
    if (energy>1745.0):
        return 0.0
    if (energy>1745.0):
        return 1.0
    if (energy>1743.5):
        return 0.0
    if (energy>1743.0):
        return 1.0
    if (energy>1742.0):
        return 0.0
    if (energy>1742.0):
        return 1.0
    if (energy>1742.0):
        return 0.0
    if (energy>1742.0):
        return 1.0
    if (energy>1741.5):
        return 0.0
    if (energy>1741.0):
        return 1.0
    if (energy>1739.0):
        return 0.0
    if (energy>1739.0):
        return 1.0
    if (energy>1738.5):
        return 0.0
    if (energy>1737.5):
        return 1.0
    if (energy>1730.0):
        return 0.0
    if (energy>1730.0):
        return 1.0
    if (energy>1729.0):
        return 0.0
    if (energy>1729.0):
        return 1.0
    if (energy>1728.0):
        return 0.0
    if (energy>1727.0):
        return 1.0
    if (energy>1724.5):
        return 0.0
    if (energy>1724.0):
        return 1.0
    if (energy>1721.0):
        return 0.0
    if (energy>1721.0):
        return 1.0
    if (energy>1719.5):
        return 0.0
    if (energy>1719.0):
        return 1.0
    if (energy>1719.0):
        return 0.0
    if (energy>1719.0):
        return 1.0
    if (energy>1718.0):
        return 0.0
    if (energy>1717.5):
        return 1.0
    if (energy>1716.0):
        return 0.0
    if (energy>1715.5):
        return 1.0
    if (energy>1713.5):
        return 0.0
    if (energy>1713.0):
        return 1.0
    if (energy>1709.0):
        return 0.0
    if (energy>1709.0):
        return 1.0
    if (energy>1707.0):
        return 0.0
    if (energy>1706.5):
        return 1.0
    if (energy>1705.0):
        return 0.0
    if (energy>1704.5):
        return 1.0
    if (energy>1703.0):
        return 0.0
    if (energy>1703.0):
        return 1.0
    if (energy>1702.5):
        return 0.0
    if (energy>1701.5):
        return 1.0
    if (energy>1701.0):
        return 0.0
    if (energy>1701.0):
        return 1.0
    if (energy>1700.5):
        return 0.0
    if (energy>1700.0):
        return 1.0
    if (energy>1700.0):
        return 0.0
    if (energy>1699.5):
        return 1.0
    if (energy>1699.0):
        return 0.0
    if (energy>1699.0):
        return 1.0
    if (energy>1698.5):
        return 0.0
    if (energy>1698.0):
        return 1.0
    if (energy>1697.5):
        return 0.0
    if (energy>1694.5):
        return 1.0
    if (energy>1693.0):
        return 0.0
    if (energy>1692.5):
        return 1.0
    if (energy>1692.0):
        return 0.0
    if (energy>1691.5):
        return 1.0
    if (energy>1686.5):
        return 0.0
    if (energy>1685.5):
        return 1.0
    if (energy>1680.0):
        return 0.0
    if (energy>1679.5):
        return 1.0
    if (energy>1679.0):
        return 0.0
    if (energy>1679.0):
        return 1.0
    if (energy>1676.0):
        return 0.0
    if (energy>1675.0):
        return 1.0
    if (energy>1674.0):
        return 0.0
    if (energy>1674.0):
        return 1.0
    if (energy>1672.5):
        return 0.0
    if (energy>1672.0):
        return 1.0
    if (energy>1672.0):
        return 0.0
    if (energy>1671.0):
        return 1.0
    if (energy>1671.0):
        return 0.0
    if (energy>1671.0):
        return 1.0
    if (energy>1670.0):
        return 0.0
    if (energy>1670.0):
        return 1.0
    if (energy>1669.5):
        return 0.0
    if (energy>1668.5):
        return 1.0
    if (energy>1668.0):
        return 0.0
    if (energy>1668.0):
        return 1.0
    if (energy>1665.5):
        return 0.0
    if (energy>1665.0):
        return 1.0
    if (energy>1664.5):
        return 0.0
    if (energy>1663.0):
        return 1.0
    if (energy>1661.0):
        return 0.0
    if (energy>1661.0):
        return 1.0
    if (energy>1660.0):
        return 0.0
    if (energy>1660.0):
        return 1.0
    if (energy>1659.0):
        return 0.0
    if (energy>1659.0):
        return 1.0
    if (energy>1658.5):
        return 0.0
    if (energy>1658.0):
        return 1.0
    if (energy>1654.5):
        return 0.0
    if (energy>1654.0):
        return 1.0
    if (energy>1653.5):
        return 0.0
    if (energy>1653.0):
        return 1.0
    if (energy>1652.5):
        return 0.0
    if (energy>1652.0):
        return 1.0
    if (energy>1651.5):
        return 0.0
    if (energy>1651.0):
        return 1.0
    if (energy>1650.0):
        return 0.0
    if (energy>1650.0):
        return 1.0
    if (energy>1648.5):
        return 0.0
    if (energy>1647.5):
        return 1.0
    if (energy>1647.0):
        return 0.0
    if (energy>1647.0):
        return 1.0
    if (energy>1647.0):
        return 0.0
    if (energy>1647.0):
        return 1.0
    if (energy>1644.5):
        return 0.0
    if (energy>1643.5):
        return 1.0
    if (energy>1643.0):
        return 0.0
    if (energy>1643.0):
        return 1.0
    if (energy>1642.5):
        return 0.0
    if (energy>1641.5):
        return 1.0
    if (energy>1640.5):
        return 0.0
    if (energy>1640.0):
        return 1.0
    if (energy>1640.0):
        return 0.0
    if (energy>1640.0):
        return 1.0
    if (energy>1637.0):
        return 0.0
    if (energy>1637.0):
        return 1.0
    if (energy>1636.0):
        return 0.0
    if (energy>1636.0):
        return 1.0
    if (energy>1635.5):
        return 0.0
    if (energy>1635.0):
        return 1.0
    if (energy>1635.0):
        return 0.0
    if (energy>1634.5):
        return 1.0
    if (energy>1634.0):
        return 0.0
    if (energy>1633.5):
        return 1.0
    if (energy>1633.0):
        return 0.0
    if (energy>1633.0):
        return 1.0
    if (energy>1632.0):
        return 0.0
    if (energy>1632.0):
        return 1.0
    if (energy>1630.5):
        return 0.0
    if (energy>1630.0):
        return 1.0
    if (energy>1629.0):
        return 0.0
    if (energy>1628.0):
        return 1.0
    if (energy>1627.5):
        return 0.0
    if (energy>1627.0):
        return 1.0
    if (energy>1621.5):
        return 0.0
    if (energy>1621.0):
        return 1.0
    if (energy>1618.0):
        return 0.0
    if (energy>1617.5):
        return 1.0
    if (energy>1612.0):
        return 0.0
    if (energy>1611.5):
        return 1.0
    if (energy>1608.0):
        return 0.0
    if (energy>1607.5):
        return 1.0
    if (energy>1606.5):
        return 0.0
    if (energy>1606.0):
        return 1.0
    if (energy>1601.5):
        return 0.0
    if (energy>1600.5):
        return 1.0
    if (energy>1599.0):
        return 0.0
    if (energy>1597.5):
        return 1.0
    if (energy>1597.0):
        return 0.0
    if (energy>1597.0):
        return 1.0
    if (energy>1596.5):
        return 0.0
    if (energy>1596.0):
        return 1.0
    if (energy>1592.5):
        return 0.0
    if (energy>1592.0):
        return 1.0
    if (energy>1591.5):
        return 0.0
    if (energy>1591.0):
        return 1.0
    if (energy>1591.0):
        return 0.0
    if (energy>1591.0):
        return 1.0
    if (energy>1589.0):
        return 0.0
    if (energy>1588.5):
        return 1.0
    if (energy>1588.0):
        return 0.0
    if (energy>1587.0):
        return 1.0
    if (energy>1580.0):
        return 0.0
    if (energy>1580.0):
        return 1.0
    if (energy>1580.0):
        return 0.0
    if (energy>1579.5):
        return 1.0
    if (energy>1579.0):
        return 0.0
    if (energy>1579.0):
        return 1.0
    if (energy>1578.0):
        return 0.0
    if (energy>1578.0):
        return 1.0
    if (energy>1576.5):
        return 0.0
    if (energy>1576.0):
        return 1.0
    if (energy>1572.0):
        return 0.0
    if (energy>1571.5):
        return 1.0
    if (energy>1571.0):
        return 0.0
    if (energy>1571.0):
        return 1.0
    if (energy>1565.0):
        return 0.0
    if (energy>1564.5):
        return 1.0
    if (energy>1563.0):
        return 0.0
    if (energy>1563.0):
        return 1.0
    if (energy>1561.0):
        return 0.0
    if (energy>1560.5):
        return 1.0
    if (energy>1559.5):
        return 0.0
    if (energy>1558.5):
        return 1.0
    if (energy>1556.5):
        return 0.0
    if (energy>1555.5):
        return 1.0
    if (energy>1554.0):
        return 0.0
    if (energy>1553.5):
        return 1.0
    if (energy>1549.5):
        return 0.0
    if (energy>1549.0):
        return 1.0
    if (energy>1546.5):
        return 0.0
    if (energy>1546.0):
        return 1.0
    if (energy>1543.0):
        return 0.0
    if (energy>1543.0):
        return 1.0
    if (energy>1538.5):
        return 0.0
    if (energy>1538.0):
        return 1.0
    if (energy>1538.0):
        return 0.0
    if (energy>1538.0):
        return 1.0
    if (energy>1538.0):
        return 0.0
    if (energy>1538.0):
        return 1.0
    if (energy>1537.0):
        return 0.0
    if (energy>1536.5):
        return 1.0
    if (energy>1534.0):
        return 0.0
    if (energy>1534.0):
        return 1.0
    if (energy>1533.0):
        return 0.0
    if (energy>1532.0):
        return 1.0
    if (energy>1527.5):
        return 0.0
    if (energy>1527.0):
        return 1.0
    if (energy>1520.5):
        return 0.0
    if (energy>1519.5):
        return 1.0
    if (energy>1519.0):
        return 0.0
    if (energy>1519.0):
        return 1.0
    if (energy>1519.0):
        return 0.0
    if (energy>1518.5):
        return 1.0
    if (energy>1517.0):
        return 0.0
    if (energy>1517.0):
        return 1.0
    if (energy>1515.0):
        return 0.0
    if (energy>1515.0):
        return 1.0
    if (energy>1514.0):
        return 0.0
    if (energy>1514.0):
        return 1.0
    if (energy>1512.5):
        return 0.0
    if (energy>1511.5):
        return 1.0
    if (energy>1508.0):
        return 0.0
    if (energy>1508.0):
        return 1.0
    if (energy>1508.0):
        return 0.0
    if (energy>1507.5):
        return 1.0
    if (energy>1507.0):
        return 0.0
    if (energy>1506.0):
        return 1.0
    if (energy>1505.0):
        return 0.0
    if (energy>1505.0):
        return 1.0
    if (energy>1503.5):
        return 0.0
    if (energy>1503.0):
        return 1.0
    if (energy>1499.0):
        return 0.0
    if (energy>1499.0):
        return 1.0
    if (energy>1497.0):
        return 0.0
    if (energy>1496.5):
        return 1.0
    if (energy>1494.5):
        return 0.0
    if (energy>1493.5):
        return 1.0
    if (energy>1483.0):
        return 0.0
    if (energy>1482.5):
        return 1.0
    if (energy>1477.0):
        return 0.0
    if (energy>1477.0):
        return 1.0
    if (energy>1473.0):
        return 0.0
    if (energy>1473.0):
        return 1.0
    if (energy>1464.0):
        return 0.0
    if (energy>1464.0):
        return 1.0
    if (energy>1454.0):
        return 0.0
    if (energy>1453.5):
        return 1.0
    if (energy>1434.0):
        return 0.0
    if (energy>1433.5):
        return 1.0
    if (energy>1433.0):
        return 0.0
    if (energy>1432.5):
        return 1.0
    if (energy>1428.0):
        return 0.0
    if (energy>1427.5):
        return 1.0
    if (energy>1412.0):
        return 0.0
    if (energy>1412.0):
        return 1.0
    if (energy>1406.0):
        return 0.0
    if (energy>1406.0):
        return 1.0
    if (energy>1399.0):
        return 0.0
    if (energy>1398.5):
        return 1.0
    if (energy>1390.0):
        return 0.0
    if (energy>1388.5):
        return 1.0
    if (energy>1354.5):
        return 0.0
    if (energy>1354.0):
        return 1.0
    if (energy>1330.5):
        return 0.0
    if (energy>1329.5):
        return 1.0
    if (energy>1303.5):
        return 0.0
    if (energy>1303.0):
        return 1.0
    if (energy>1298.0):
        return 0.0
    if (energy>1297.5):
        return 1.0
    if (energy>1266.5):
        return 0.0
    if (energy>1266.0):
        return 1.0
    if (energy>1261.5):
        return 0.0
    if (energy>1261.0):
        return 1.0
    if (energy>1247.5):
        return 0.0
    if (energy>1246.0):
        return 1.0
    return 0.0

numthresholds=607


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()

    if not args.validate: # Then predict
        if args.cleanfile:
            with open(args.csvfile,'r') as cleancsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                for cleanrow in cleancsvreader:
                    if len(cleanrow)==0:
                        continue
                print(str(','.join(str(j) for j in ([i for i in cleanrow])))+','+str(int(classify(cleanrow))))
        else:
            tempdir=tempfile.gettempdir()
            cleanfile=tempdir+os.sep+"clean.csv"
            clean(args.csvfile,cleanfile, -1, args.headerless, True)
            with open(cleanfile,'r') as cleancsvfile, open(args.csvfile,'r') as dirtycsvfile:
                cleancsvreader = csv.reader(cleancsvfile)
                dirtycsvreader = csv.reader(dirtycsvfile)
                if (not args.headerless):
                        print(','.join(next(dirtycsvreader, None)+['Prediction']))
                for cleanrow,dirtyrow in zip(cleancsvreader,dirtycsvreader):
                    if len(cleanrow)==0:
                        continue
                    print(str(','.join(str(j) for j in ([i for i in dirtyrow])))+','+str(int(classify(cleanrow))))
            os.remove(cleanfile)
            
    else: # Then validate this predictor
        tempdir=tempfile.gettempdir()
        temp_name = next(tempfile._get_candidate_names())
        cleanvalfile=tempdir+os.sep+temp_name
        clean(args.csvfile,cleanvalfile, -1, args.headerless)
        with open(cleanvalfile,'r') as valcsvfile:
            count,correct_count,num_TP,num_TN,num_FP,num_FN,num_class_1,num_class_0=0,0,0,0,0,0,0,0
            valcsvreader = csv.reader(valcsvfile)
            for valrow in valcsvreader:
                if len(valrow)==0:
                    continue
                if int(classify(valrow[:-1]))==int(float(valrow[-1])):
                    correct_count+=1
                    if int(float(valrow[-1]))==1:
                        num_class_1+=1
                        num_TP+=1
                    else:
                        num_class_0+=1
                        num_TN+=1
                else:
                    if int(float(valrow[-1]))==1:
                        num_class_1+=1
                        num_FN+=1
                    else:
                        num_class_0+=1
                        num_FP+=1
                count+=1

        model_cap=numthresholds

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


        os.remove(cleanvalfile)

