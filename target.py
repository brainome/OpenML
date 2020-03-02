#! /usr/bin/env python3
#
# Quick and dirty scrip to change target column to the outmost right.
#
# Copyright (c) 2019 - 2020 Brainome Incorporated. 
#
# This file is distributed under GNU General Public License v2.0 or higher.
# See LICENSE for details
#

import sys
import csv
from collections import defaultdict

def eprint(string):
	sys.stderr.write(string+"\n")

if not (len(sys.argv)==3):
	print("Usage:")
	print(sys.argv[0]+" <filename.csv> <targetcolumn>") 
	print("Puts the target column onto outmost right.")
	print ("<filename.csv> -- comma separated value file")
	print ("<targetcolumn> -- target column name or index") 
	print 
	sys.exit()

target=sys.argv[2]

with open(sys.argv[1]) as csvfile:
	sniffer = csv.Sniffer()
	hasheader=sniffer.has_header(csvfile.read(10000000))
	csvfile.seek(0)
	reader = csv.reader(csvfile)
	if (hasheader==True):
		header=next(reader, None)
		try:
			hc=header.index(target)
		except:
			eprint("Target '"+target+"' not found!")
			eprint("Header is: "+str(header))
			sys.exit()
		for i in range(0,len(header)):		
			if (i==hc):
				continue
			print(header[i]+",", end = ''),
		print(header[hc])

		for row in csv.DictReader(open(sys.argv[1])):
			for name in header:
				if (name==target):
					continue
				if (',' in row[name]):
					print ('"'+row[name]+'"'+",",end = '')
				else:
					print (row[name]+",",end = '')
				
			print (row[target])
	else:
		try:
			hc=int(target)
		except:
			eprint("No header found but column name given as target. Specify index instead.")
			sys.exit()
		
		for row in reader:
			for i in range(0,len(row)):
				if (i==hc):
					continue
				if (',' in row[i]):
					print ('"'+row[i]+'"'+",",end = '')
				else:
					print(row[i]+",",end = '')
			print (row[hc])
