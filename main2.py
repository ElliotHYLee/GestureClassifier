from Reader import Reader
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import time
from pathlib import Path
ops.reset_default_graph()
# Create graph session
sess = tf.Session()

[x_vals, y_vals] = Reader().get_next_batch()
y_vals = y_vals.reshape(15984)
print(x_vals.shape)
print(y_vals.shape)


# make results reproducible
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(len(x_vals)*0.8), replace=False)
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 112], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for both NN layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[112,hidden_layer_nodes])) # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output


# Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.006)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
start = time.time()
saver = tf.train.Saver()
for i in range(500000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    #test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    #test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%500==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

    if temp_loss < 0.01:
         break


end = time.time()
print('elapsed: ' + str(end - start))
print(temp_loss)

[input_test, label_test] = Reader().get_test_data()
dim = label_test.shape
print(dim)
label_test = label_test.reshape(dim[0], 1)
pred = sess.run(final_output, feed_dict={x_data: input_test, y_target: label_test})
pred = (np.round(pred)).reshape(dim[0],1)

print(pred.shape)

diff = pred - label_test


cnt = 0
for i in range(dim[0]):
    if diff[i]==0:
        cnt = cnt +1

print(cnt*1.0/dim[0])


# Plot class
plt.plot(pred, 'b.', label='pred')
plt.plot(label_test, 'ro', label='labels')
plt.ylim((-1,9))
plt.show()


# # Plot loss (MSE) over time
# plt.plot(loss_vec, 'k-', label='Train Loss')
# plt.plot(test_loss, 'r--', label='Test Loss')
# plt.title('Loss (MSE) per Generation')
# plt.legend(loc='upper right')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()
