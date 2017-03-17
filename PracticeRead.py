import csv
import numpy

def get_next_batch(): #gives out the next clean batch in matrix form, this is for validation and train data
	#numpy.loadtxt(open("Train.csv", "rb"), delimiter=",", skiprows=1)

	with open('Train.csv', newline='' ) as csvFile:
		reader = csv.reader(csvFile, delimiter = ' ', quotechar='|')
		x = list(reader)
		result = numpy.array(x).astype("float32")
		numCol = result.shape[1]
		#print(numCol)
		#print(result)
		y = result[:,0]
		x = result[:,1:numCol]
		return [x, y]

#if __name__ == '__main__':
#	get_next_batch()
