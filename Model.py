import tensorflow as tf
import numpy as np
import random

class Model():
	def __init__(self , opts):
		self.opts = opts
		self.learning_rate = tf.Variable(opts.init_learning_rate, trainable=False)
		self.epoch_number = tf.Variable(0.0, trainable=False)

		self.input = tf.placeholder(tf.float32, [ None, opts.input_size]) #None is batch size
		self.desired_output = tf.placeholder(tf.float32, [ None, opts.output_size])

		self.w = [None,None,None,None];
		self.b = [None,None,None,None];
		self.w[0] =  tf.Variable(tf.random_normal([opts.input_size, opts.num_hidden_units]))
		self.b[0] =  tf.Variable(tf.random_normal([opts.num_hidden_units]))

		self.w[1] =  tf.Variable(tf.random_normal([opts.num_hidden_units, opts.num_hidden_units]))
		self.b[1] =  tf.Variable(tf.random_normal([opts.num_hidden_units]))

		self.w[2] =  tf.Variable(tf.random_normal([opts.num_hidden_units, opts.num_hidden_units]))
		self.b[2] =  tf.Variable(tf.random_normal([opts.num_hidden_units]))


		self.w[3] =  tf.Variable(tf.random_normal([opts.num_hidden_units, opts.output_size]))
		self.b[3] =  tf.Variable(tf.random_normal([opts.output_size]))

		self.pred = self.predict(self.input)
		self.cost = tf.reduce_mean(tf.pow(self.pred-self.desired_output, 2))/2
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

	def predict(self , x):
		ans_here_1 = tf.nn.relu( tf.matmul(x, self.w[0]) + self.b[0])
		ans_here_2 = tf.nn.relu( tf.matmul(ans_here_1, self.w[1]) + self.b[1])
		ans_here_3 = tf.nn.sigmoid( tf.matmul(ans_here_2, self.w[2]) + self.b[2])
		ans_here_4 =  tf.matmul(ans_here_3, self.w[3]) + self.b[3]
		return ans_here_4

	def train(self, sess, x, y):
		cost , _   = sess.run([self.cost,  self.optimizer ], {self.input: x, self.desired_output: y   })
		return cost

	def test(self, sess, x ):
		return sess.run(self.pred, {self.input: x})
