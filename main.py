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
			c, acc, p_n,be,inp = m.train(sess, x, y)
			if epoch%10==0:
				print('epoch number: '+ str(sess.run(m.epoch_number))+ ', acc: ' + str(acc))
				print('p_n '+ str(p_n)+'\n')
				print('y '+str(y))
				print('be '+str(be))
				print('input norm: '+str(inp))
				print('input: '+str(x[:20,:]))
				print('training mse: ' + str(c) + '\n')
				print('Acc ' + str(acc))
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
			if (label == est_label):
				mySum = mySum + 1
		accuracy = mySum*1.0/numRow
		print("accuracy: ", np.round(accuracy,2))

if __name__ == '__main__':
	if sys.argv[1] == 'Train':
		train(Opts())
	elif sys.argv[1] == 'Test':
		test(Opts())
