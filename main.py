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
	[x,y] = Reader().get_next_batch()
	 with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(opts.max_epochs):
			if Path(opts.save_path+'checkpoint').is_file():
				saver.restore(sess,opts.save_path)
			epoch = sess.run(m.epoch_number)
			sess.run(tf.assign(m.learning_rate, opts.init_learning_rate* (opts.decay_rate ** m.epoch_number ) ) )
			print('epoch number: '+ str(sess.run(m.epoch_number))+ ', learning rate: '+ str(sess.run(m.learning_rate)))
			print('training mse: '+ str(m.train(sess, x, y))+'\n')

			sess.run(tf.assign(m.epoch_number,m.epoch_number+1))
			saver.save(sess,opts.save_path)

def test(opts):
	m = Model(opts)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,opts.save_path)
		[x,y] = Reader().get_test_data()
		estimated_y = m.test(sess, x)
		print(y)
		print(estimated_y)

if __name__ == '__main__':
	if sys.argv[1] == 'Train':
		train(Opts())
	elif sys.argv[1] == 'Test':
		test(Opts())