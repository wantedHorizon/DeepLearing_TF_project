import sys
import time
from PIL import Image
from random import randint
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

trainig_size = 10000
test_size = trainig_size*0.2
x_train = np.empty((trainig_size,32,32,3))

def load_imgs(print_bool = True):
  # global x_train
  # global y_train
  global c
  for k in range(100):
    if print_bool:
        print('loading images - ' + str(k) + '%')
    for i in range(trainig_size / 100):
        index = k * (trainig_size / 100) + i
        e = str(index + 1)
        x = np.asarray(Image.open('../../cifar32/test/' + e + '.png'))
        x_train[index] = x
  if print_bool:
    print('loading images - 100%')

def get_label(index):
    with open('../../cifar32/trainLabels.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        label = csv_reader[index][1]
        num_label=0
        if label == 'cat':
            num_label = 0
        elif label == 'dog':
            num_label = 1
        elif label == 'horse':
            num_label = 2
        elif label == 'ship':
            num_label = 3
        ret = np.zeros(4)
        ret[num_label] = 1
        return ret

def randnum(train):
    if train:
        return randint(1, trainig_size * 0.8 - 1)
    else:
        return randint(trainig_size * 0.8, trainig_size - 1)
def next_batch(train = True, batch_size = 100):
    set = np.empty((batch_size,32,32,3))
    labels = np.empty((batch_size,4))
    for i in range(batch_size):
        r = randint(train)
        if i < batch_size/4:
            while get_label(r) == 0:
                r = randint(train)
        elif i < batch_size/2:
            while get_label(r) == 1:
                r = randint(train)
        elif i < 3*batch_size/4:
            while get_label(r) == 2:
                r = randint(train)
        else:
            while get_label(r) == 3:
                r = randint(train)
        set[i] = x_train[r]
        labels[i] = get_label(r)
    return set,labels

def next_batch(train = True, batch_size = 100):
    set = np.empty((batch_size,32,32,3))
    labels = np.empty((batch_size,4))
    for i in range(batch_size):
        if train:
            r = randint(1,trainig_size*0.8-1)
        else:
            r = randint(trainig_size * 0.8, trainig_size-1)
        set[i] = x_train[r]
        labels[i] = get_label(r)
    return set,labels

load_imgs()
learning_rate = 0.01
batch_size = 128
n_epochs = 1

X = tf.placeholder(tf.float32, [batch_size, 32*32*3], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 4], name="label")

w = tf.Variable(tf.random_normal(shape=[32*32*3, 4], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1,4]), name='bias')

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels= Y)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

loss_history = []
acc_history = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    n_batches = int(trainig_size / batch_size)
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch, Y_batch = next_batch(batch_size=batch_size)
            print('epoch ' + str(i+1) + ' : ' + str(float(j) / float(n_batches) * 100) + '%')
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: X_batch.reshape((128, 3072)), Y: Y_batch.reshape((128, 4))})
        loss_history.append(loss_value)

        n_v_batches = int(test_size / batch_size)
        total_correct_preds = 0
        for j in range(n_v_batches):
            X_batch, Y_batch = next_batch(train=False,batch_size=batch_size)
            _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch.reshape((128, 3072)), Y: Y_batch.reshape((128, 4))})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
        validation_accuracy = total_correct_preds / test_size
        acc_history.append(validation_accuracy)
        print('epoch ' + str(i+1) + ' loss:' + str(loss_value) + ' accuracy:' + str(validation_accuracy))

    n_batches = int(test_size/ batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        print('calculating test accuracy - ' + str(float(i) / float(n_batches) * 100) + '%')
        X_batch, Y_batch = next_batch(train=False,batch_size=batch_size)
        logits_batch = sess.run(logits, feed_dict={X: X_batch.reshape((128, 3072)), Y: Y_batch.reshape((128, 4))})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print "Test accuracy is {0}".format(total_correct_preds / test_size)

