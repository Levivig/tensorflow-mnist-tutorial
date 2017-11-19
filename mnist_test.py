# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
import matplotlib.pyplot
import cv2
import sys

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0.0)

path = "."
img_path = "{path}/kepek/".format(path=path)
model_path = "{path}/models/model.ckpt".format(path=path)

def readimg(kep):
    img = cv2.imread("{path}/input/{img}".format(path=img_path, img=kep),0)
    if img.shape != [28,28]:
        img2 = cv2.resize(img,(28,28))
        img = img2.reshape(28,28,-1);
    else:
        img = img.reshape(28,28,-1);
    #revert the image,and normalize it to 0-1 range
    img = 1.0 - img/255.0

    return img

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets(
    "data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(
        variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * \
        tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

# 6x6 patch, 1 input channel, K output channels
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 28x28
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 14x14
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]),
                        tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]),
                       tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
conv_activations = tf.concat([tf.reshape(tf.reduce_max(Y1r, [0]), [-1]), tf.reshape(
    tf.reduce_max(Y2r, [0]), [-1]), tf.reshape(tf.reduce_max(Y3r, [0]), [-1])], 0)
dense_activations = tf.reduce_max(Y4r, [0])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
#datavis = tensorflowvisu.MnistDataVis(title4="batch-max conv activation",title5="batch-max dense activations", histogram4colornum=2, histogram5colornum=2)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    learning_rate = min_learning_rate + \
        (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, ca, da = sess.run([accuracy, cross_entropy, I, conv_activations, dense_activations], {
                                    X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " +
              str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, ca, da)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {
                            X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i) + ": ********* epoch " + str(i * 100 //
                                                  mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y,
                          lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y,
                          tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

#datavis.animate(training_step, 10001, train_data_update_freq=20, test_data_update_freq=100, save_movie=True)


# Save model weights to disk
#save_path = saver.save(sess, model_path)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

#print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# Restore trained model
saver.restore(sess, model_path)


print("-" * 40)
# Test trained model
print("-- A halozat tesztelese")
saver.restore(sess, model_path)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("-- Pontossag: ", sess.run(accuracy, feed_dict={
                    X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0}))
print("-" * 40)


print("-- A MNIST 42. tesztkepenek felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

img = mnist.test.images[42]
image = img
matplotlib.pyplot.imshow(image.reshape(
    28, 28), cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.savefig("{path}/output/4.png".format(path=img_path))
matplotlib.pyplot.show()

classification = sess.run(tf.argmax(Y, 1), feed_dict={
                          X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
print("-" * 40)


print("-- A sajat kezi 1-esem felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

img = readimg("sajat1.png")
image = img
image = image.reshape(28, 28,1)

matplotlib.pyplot.imshow(image.reshape(28, 28), cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.savefig("{path}/output/1.png".format(path=img_path))
matplotlib.pyplot.show()

classification = sess.run(tf.argmax(Y, 1), feed_dict={
                          X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
print("-" * 40)


print("-- Sajat kezi 5-ösöm felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

img = readimg("sajat5.png")
image = img
#image = Image.reshape(28 * 28)

matplotlib.pyplot.imshow(image.reshape(
    28, 28), cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.savefig("{path}/output/5.png".format(path=img_path))
matplotlib.pyplot.show()

classification = sess.run(tf.argmax(Y, 1), feed_dict={
                          X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
print("-" * 40)


print("-- Sajat kezi 6-osom felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

img = readimg("sajat6.png")
image = img
#image = Image.reshape(28 * 28)

matplotlib.pyplot.imshow(image.reshape(
    28, 28), cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.savefig("{path}/output/6.png".format(path=img_path))
matplotlib.pyplot.show()

classification = sess.run(tf.argmax(Y, 1), feed_dict={
                          X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
print("-" * 40)


print("-- Tina 2-ese felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

img = readimg("tina.png")
image = img
#image = Image.reshape(28 * 28)

matplotlib.pyplot.imshow(image.reshape(
    28, 28), cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.savefig("{path}/output/tina_2.png".format(path=img_path))
matplotlib.pyplot.show()

classification = sess.run(tf.argmax(Y, 1), feed_dict={
                          X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
print("-" * 40)

if len(sys.argv) > 1:
    print("-" * 40)
    print("-- Input kep felismerese, mutatom a szamot, a tovabblepeshez csukd be az ablakat")

    img = readimg(sys.argv[1])
    image = img
    image = image.reshape(28, 28,1)

    matplotlib.pyplot.imshow(image.reshape(28, 28), cmap=matplotlib.pyplot.cm.binary)
    matplotlib.pyplot.savefig("{path}/output/out.png".format(path=img_path))
    matplotlib.pyplot.show()

    classification = sess.run(tf.argmax(Y, 1), feed_dict={
                              X: [image], tst: True, pkeep: 1.0, pkeep_conv: 1.0})

    print("-- Ezt a halozat ennek ismeri fel: ", classification[0])
    print("-" * 40)
