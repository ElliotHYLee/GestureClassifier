import numpy as np
import random
import csv


class Reader():
	def __init__(self):
		pass

	def get_next_batch(self): #gives out the next clean batch in matrix form, this is for validation and train data
		#features_here = np.random.rand(1000000,5) * 10 - 5;
		#labels_here = [ np.sum(features_here,axis=1) , np.mean(features_here,axis=1)]
		#return [np.float32(features_here) , np.float32(np.transpose(labels_here))]
		with open('Train.csv', newline='' ) as csvFile:
			reader = csv.reader(csvFile, delimiter = ' ', quotechar='|')
			data = list(reader)
			print(data[0][1])
			result = np.array(data).astype("float32")
			numCol = result.shape[1]
			numRow = result.shape[0]
			#print(numCol)
			#print(result)
			y = result[:,0:8]
			#print(y)
			label = np.transpose(y)
			print(np.shape(label))
			x = result[:,8:numCol]
			return [x, label.reshape(numRow,8)]

	def get_test_data(self):
		with open('Test.csv', newline='' ) as csvFile:
			reader = csv.reader(csvFile, delimiter = ' ', quotechar='|')
			data = list(reader)
			result = np.array(data).astype("float32")
			numCol = result.shape[1]
			numRow = result.shape[0]
			#print(numCol)
			#print(result)
			y = result[:,0:8]
			#print(y)
			label = np.transpose(y)
			#print(label)
			x = result[:,8:numCol]
		return [x , label.reshape(numRow,8)]
