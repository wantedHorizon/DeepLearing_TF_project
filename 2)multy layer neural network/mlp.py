
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from random import randint
from random import shuffle
from random import choice
import csv

trainig_size = 10000
test_size = trainig_size*0.2
pixels = 32*32*3
x_train = np.empty((trainig_size,32,32,3))
path = 'D:/pyCharm Projects/DL/cifar32/'
train_steps = 500
train_r = []
test_r = []
for x in range(int(trainig_size*0.8)):
  train_r.append(x)
for x in range(int(trainig_size*0.2)):
  test_r.append(x)
shuffle(train_r)
shuffle(test_r)


with open(path + 'trainLabels.csv') as csv_file:
  csv_reader = list(csv.reader(csv_file, delimiter=','))


def load_imgs(print_bool = True):
  # global x_train
  # global y_train
  global c
  for k in range(100):
    if print_bool:
        print('loading images - ' + str(k) + '%')
    for i in range(int(trainig_size / 100)):
        index = int(k * (trainig_size / 100) + i)
        e = str(index + 1)
        x = np.asarray(Image.open(path +'train/'+ e + '.png'))
        x_train[index] = x
  if print_bool:
    print('loading images - 100%')

def get_label(index):
  label = csv_reader[index][1]
  num_label = 100
  if label == 'cat':
    num_label = 0
  elif label == 'dog':
    num_label = 1
  elif label == 'horse':
    num_label = 2
  elif label == 'ship':
    num_label = 3
  return num_label

def next_batch(train=True, batch_size=100):
  set = np.empty((batch_size, pixels))
  labels = np.empty((batch_size, 4))

  for i in range(batch_size):
    if train:
      r = choice(train_r)
    else:
      r = choice(test_r)

    while get_label(r) == 100:
      if train:
        r = choice(train_r)
      else:
        r = choice(test_r)

    set[i] = x_train[r-1]
    lbl = np.zeros(4)
    lbl[get_label(r)] = 1
    labels[i] = lbl
    idx = np.arange(0, batch_size)
    np.random.shuffle(idx)
    data_shuffle = [set[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


load_imgs(True)
x_train /= 255
x_train = x_train.reshape(trainig_size,pixels)
x = tf.placeholder(tf.float32, [None, pixels])

y_ = tf.placeholder(tf.float32, [None, 4])

W1 = tf.Variable(tf.zeros([pixels, 2048]))

b1 = tf.Variable(tf.zeros([2048]))

y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.zeros([2048, 1024]))

b2 = tf.Variable(tf.zeros([1024]))

y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(tf.zeros([1024, 512]))

b3 = tf.Variable(tf.zeros([512]))

y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)

W4 = tf.Variable(tf.zeros([512, 4]))

b4 = tf.Variable(tf.zeros([4]))

y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y4), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for ep in range(1,4):
  for i in range(train_steps):
    if i % (train_steps/100) == 0:
      print ("epoch - ",ep,",    ",i/(train_steps/100))
    batch_xs, batch_ys = next_batch(train=True,batch_size=200)
    batch_xs = batch_xs.reshape((batch_xs.shape[0],pixels))
    batch_ys = batch_ys.reshape((batch_ys.shape[0], 4))

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y4,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_xs, batch_ys = next_batch(train=False,batch_size=500)
batch_xs = batch_xs.reshape((batch_xs.shape[0],pixels))
batch_ys = batch_ys.reshape((batch_ys.shape[0], 4))


print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))