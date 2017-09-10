from Model import Model
from Reader import Reader
from Opts import Opts
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

def train(opts):
	m = Model(opts)
	saver = tf.train.Saver()
	[x,y] = Reader().get_next_batch()     #-----test
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		isFirst = True
		for epoch in range(opts.max_epochs) and isFirst:
			if Path(opts.save_path+'checkpoint').is_file():
				saver.restore(sess,opts.save_path)
				isFirst = False
			epoch = sess.run(m.epoch_number)
			#sess.run(tf.assign(m.learning_rate, opts.init_learning_rate) )
			#print(x.shape)
			#for i in range(x.shape[0]):
			sess.run(tf.assign(m.learning_rate, opts.init_learning_rate* (opts.decay_rate ** m.epoch_number ) ) )
			#print((x[i].reshape((1,112))).shape)
			# each_row_input = (x[i].reshape((1,112)))
			# each_row_label = y[i].reshape((1,8))
			#c = m.train(sess, each_row_input, each_row_label)
			c = m.train(sess, x, y)
			if epoch%1==0:
				print('epoch number: '+ str(sess.run(m.epoch_number))+ ', learning rate: '+ str(sess.run(m.learning_rate)))
				#c = sess.run(tf.reduce_sum(tf.multiply(c, c)))
				print('training mse: '+ str(c)+'\n')
			sess.run(tf.assign(m.epoch_number,m.epoch_number+1))
			saver.save(sess,opts.save_path)


def test(opts):
	m = Model(opts)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,opts.save_path)
		[x,y] = Reader().get_test_data()
		est_y = m.test(sess, x)
		mySum = 0
		numRow = x.shape[0]
		for i in range (0, numRow-1):
			label = np.argmax(y[i])
			est_label = np.argmax(est_y[i])
		#	print(est_y[i])
			if (label == est_label):
				mySum = mySum + 1
		accuracy = mySum*1.0/numRow
		print("accuracy: ", np.round(accuracy,2))

if __name__ == '__main__':
	if sys.argv[1] == 'Train':
		train(Opts())
	elif sys.argv[1] == 'Test':
		test(Opts())
