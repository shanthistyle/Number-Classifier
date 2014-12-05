__author__ = 'ShanthiS'

import numpy
import csv

from numpy import *	
from scipy import stats

# Load data from csv
import csv, sys
filename = 'digitsDataset/trainFeatures.csv'
num = 5 # change num to 6000
k = 2
training_features = numpy.zeros((num,784))
training_labels = numpy.zeros(num)
val_features = numpy.zeros((num,784))
with open(filename, 'rb') as f:
	reader = csv.reader(f)
	try:
		for row in reader:
			row_num = 0
			if row_num < (num + 1):
				for i in range(0,784):
					training_features[row_num][i] = row[i]
			else:
				break
			row_num += 1
	except csv.Error as e:
		sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

filename = 'digitsDataset/trainLabels.csv'
with open(filename, 'rb') as f:
	reader = csv.reader(f)
	try:
		row_num = 0
		for row in reader:
			if row_num < num:
				#print row
				#training_labels.append(row)
				training_labels[row_num] = row[0]
			else:
				break
			row_num += 1
	except csv.Error as e:
		sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))
print(training_labels)

filename = 'digitsDataset/valFeatures.csv'
with open(filename, 'rb') as f:
	reader = csv.reader(f)
	try:
		for row in reader:
			row_num = 0
			if row_num < (num + 1):
				for i in range(0,784):
					val_features[row_num][i] = row[i]
			else:
				break
			row_num += 1
	except csv.Error as e:
		sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

#for each feature vector in val_features, compute distance between each feature vector in training_features -- 
temp = []
temp2 = []
sorted_indeces = []
k_index = []
#for item1 in val_features:
item1 = val_features[0] #iterate and classify all items in validation set
for i in range(0, num):
	item2 = training_features[i] 
	c = item2 - item1
	dist = numpy.sqrt(numpy.sum(c**2))
	#dist = numpy.linalg.norm(item2 - item1, axis=1)
	#dist = numpy.apply_along_axis(numpy.linalg.norm, 1, c)
	temp.append(dist)

# find index of k smalled values -- index into training_labels
sorted_indeces = numpy.argsort(temp)
k_index = sorted_indeces[:k]

# find corresponding training_labels
candidates = []
for index in k_index:
	candidates.append(training_labels[index])

# choose mode (breaking ties)
frequency = stats.itemfreq(candidates)
# frequency takes form of [[val1, count], [val2, count]..]
counts = []
for item in frequency:
	counts.append(item[1])

final_choice = candidates[argmax(counts)]

#use argmax to find value that has highest count, if multiple options with same highest count, return first occurence (arbitrary tie break)



print(temp)
print(k_index)
print(candidates)
print(counts)
print("the final choice!")
print(final_choice)


# Calculate error rate
# for each classifiation, set to 1 if right --- [actual (1000) - ours (sum of all 1)]/ actual (1000)






