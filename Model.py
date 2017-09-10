import tensorflow as tf
import numpy as np
import random
from sklearn import preprocessing

class Model():
	def __init__(self , opts):
		self.opts = opts
		self.epoch_number = tf.Variable(0.0, trainable=False)

		self.input = tf.placeholder(tf.float32, [ None, opts.input_size]) #None is batch size
		self.desired_output = tf.placeholder(tf.float32, [ None, opts.output_size])

		self.w = [None,None,None]
		self.b = [None,None,None]

		self.w[0] =  tf.Variable(tf.random_normal([opts.input_size, opts.num_hidden_units]))
		self.b[0] =  tf.Variable(tf.random_normal([opts.num_hidden_units]))

		self.w[1] =  tf.Variable(tf.random_normal([opts.num_hidden_units, opts.num_hidden_units]))
		self.b[1] =  tf.Variable(tf.random_normal([opts.num_hidden_units]))

		self.w[2] =  tf.Variable(tf.random_normal([opts.num_hidden_units, opts.output_size]))
		self.b[2] =  tf.Variable(tf.random_normal([opts.output_size]))

		self.pred ,logits = self.predict(self.input)
		self.bool_eq = tf.equal(tf.argmax(self.pred,axis=1),tf.argmax(self.desired_output,axis=1))
		self.pp = self.pred
		self.predict_acc_op = tf.reduce_mean(tf.cast(self.bool_eq,tf.float32))
		self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits,  labels=self.desired_output)
		self.cost = tf.reduce_mean(self.cost)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self.cost)

	def predict(self , x):
		ans_here_1 = tf.nn.softmax(tf.matmul(x, self.w[0]) + self.b[0])
		# ans_here_2 = tf.nn.tanh(tf.matmul(ans_here_1, self.w[1]) + self.b[1])
		logits = tf.matmul(ans_here_1, self.w[2]) + self.b[2]
		ans_here_3 =  tf.nn.softmax(logits)
		return ans_here_3 , logits

	def train(self, sess, x, y):
		x = self.mapMinMax(x)
		cost , _ ,acc,p_np,be,inp  = sess.run([self.cost, self.optimizer,self.predict_acc_op,self.pp,self.bool_eq,self.input], {self.input: x, self.desired_output: y})
		return cost, acc, p_np,be,inp

	def test(self, sess, x ):
		x = self.mapMinMax(x)
		return sess.run(self.pred, {self.input: x})

	def mapMinMax(self, x):
		return preprocessing.scale(x)
		# maxNum = np.max(x)
		# minNum = np.min(x)
		# newX = 2*(x - minNum)/(maxNum-minNum) -1
		# return newX
		#print(newX)
