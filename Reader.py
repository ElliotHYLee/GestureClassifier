import numpy as np
import random
import csv

class Reader():
	def __init__(self):
		pass

	def get_data(self, csvFile):
		reader = csv.reader(csvFile, delimiter = ' ', quotechar='|')
		data = list(reader)
		result = np.array(data).astype("float32")
		numCol = result.shape[1]
		numRow = result.shape[0]
		y = result[:,0:8]
		labels = np.zeros((numRow, 1))
		#for i in range(0, numRow-1):
		#	labels[i] = np.argmax(y[i,:])
		labels = y
		x = result[:,8:numCol]
		return [x, labels]

	def get_next_batch(self):
		with open('Train.csv' ) as csvFile:
			[x, labels] = self.get_data(csvFile)
		return [x, labels]

	def get_test_data(self):
		with open('Test.csv' ) as csvFile:
			[x, labels] = self.get_data(csvFile)
		return [x, labels]
