import tensorflow as tf
import numpy as np
import math
from six.moves import cPickle as pickle
import os

def load_CIFAR_batch(filename):
  # load single batch of cifar
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
  # load all of cifar
  xs = []
  ys = []
  for b in range(1, 6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b,))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=45000, num_validation=5000, num_test=1000):
  # Load the raw CIFAR-10 data
  cifar10_dir = '../input'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


class CifarNet():
  def __init__(self):

    self.Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32])
    self.bconv1 = tf.get_variable("bconv1", shape=[32])

    self.Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 64])
    self.bconv2 = tf.get_variable("bconv2", shape=[64])
    # affine layer with 1024
    self.W1 = tf.get_variable("W1", shape=[3136, 1024])
    self.b1 = tf.get_variable("b1", shape=[1024])

    self.W2 = tf.get_variable("W2", shape=[1024, 512])
    self.b2 = tf.get_variable("b2", shape=[512])
    # affine layer with 10
    self.W3 = tf.get_variable("W3", shape=[512, 10])
    self.b3 = tf.get_variable("b3", shape=[10])

  def forward(self, X, y, is_training):
      conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1

      relu1 = tf.nn.relu(conv1)

      drop1 = tf.layers.dropout(inputs=relu1,rate=0.25, training=is_training)

      conv2 = tf.nn.conv2d(drop1, self.Wconv2, strides=[1, 2, 2, 1], padding='VALID') + self.bconv2

      relu2 = tf.nn.relu(conv2)

      drop2 = tf.layers.dropout(inputs=relu2,rate=0.25, training=is_training)

      maxpool = tf.layers.max_pooling2d(drop2, pool_size=(2, 2), strides=2)

      maxpool_flat = tf.reshape(maxpool, [-1, 3136])

      affine1 = tf.matmul(maxpool_flat, self.W1) + self.b1

      relu3 = tf.nn.relu(affine1)

      drop3 = tf.layers.dropout(inputs=relu3,rate=0.25, training=is_training)

      affine2 = tf.matmul(drop3, self.W2) + self.b2

      relu4 = tf.nn.relu(affine2)

      drop4 = tf.layers.dropout(inputs=relu4,rate=0.25, training=is_training)

      affine3 = tf.matmul(drop4, self.W3) + self.b3

      self.predict = tf.layers.batch_normalization(inputs=affine3, center=True, scale=True, training=is_training)
      print(self.predict.shape)
      return self.predict

  def run(self, session, Xd, yd, batch_size=32, training=None):
    correct_prediction = tf.equal(tf.argmax(self.predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
      variables[-1] = training

    correct = 0
    losses = []

    for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
      # generate indicies for the batch
      start_idx = (i * batch_size) % Xd.shape[0]
      idx = train_indicies[start_idx:start_idx + batch_size]

      # create a feed dictionary for this batch
      feed_dict = {X: Xd[idx, :],y: yd[idx],is_training: training_now}
      # get batch size
      actual_batch_size = yd[idx].shape[0]

      loss, corr, _ = session.run(variables, feed_dict=feed_dict)

      losses.append(loss * actual_batch_size)
      correct += np.sum(corr)

    total_correct = float(correct) / float(Xd.shape[0])
    total_loss = np.sum(losses) / Xd.shape[0]
    return total_loss, total_correct

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

net = CifarNet()
net.forward(X,y,is_training)

# Annealing the learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
end_learning_rate = 0.005
decay_steps = 10000

learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)

exp_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)

cross_entr_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=net.predict)
mean_loss = tf.reduce_mean(cross_entr_loss)

optimizer = tf.train.AdamOptimizer(exp_learning_rate)


train_step = optimizer.minimize(mean_loss, global_step=global_step)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
print '\n\n\n'
for i in range(30):
  train_loss, train_correct = net.run(sess, X_train, y_train, 128, train_step)
  val_loss, val_correct = net.run(sess, X_val, y_val, 128)
  print 'epoch: '+str(i+1)+'   train loss: ' + str(train_loss)[:5] + '   validation accuracy: ' + str(val_correct*100) + '%'
