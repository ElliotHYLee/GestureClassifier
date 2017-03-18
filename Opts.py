class Opts():
	def __init__(self):
		self.max_epochs = 500# 100000 #epochs for training
		self.output_size = 1 #num of units in output layer
		self.input_size = 56 #num of units in input layer
		self.num_hidden_units = 20 #in each hidden layer
		self.init_learning_rate = 0.01 #.0001
		self.decay_rate = 1.0 #for the learning rate ( less than 1 -> simulated annealing)
		self.save_path = './model/'  #in which folder to save the model in

		#self.num_layers= 2	# hidden layers count - not supported in this code
