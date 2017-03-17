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
			x = list(reader)
			result = np.array(x).astype("float32")
			numCol = result.shape[1]
			#print(numCol)
			#print(result)
			y = result[:,0]
			x = result[:,1:numCol]
			return [x, y]

	def get_test_data(self):
		labels_here = []
		features_here = []

		features_here.append([ -1.5, 2.5, -4.5, 3.5, -2.5 ])
		labels_here.append( [-2.5 , -0.5] )
		features_here.append([ +1.5, -2.5, -4.5, 2.5, 4.5 ])
		labels_here.append( [ 1.5, 0.3] )
		return [np.float32(features_here) , np.float32(labels_here)]
