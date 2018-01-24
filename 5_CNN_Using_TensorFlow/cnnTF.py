import os
import time
import math
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test (label shape:     {0}'.format(yTest.shape))

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Select device
deviceType = "/gpu:0"

# Simple Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def simpleModel():
    with tf.device(deviceType):
        wConv = tf.get_variable("wConv", shape=[7, 7, 3, 32])
        bConv = tf.get_variable("bConv", shape=[32])
        w = tf.get_variable("w", shape=[5408, 10]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
        b = tf.get_variable("b", shape=[10])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(x, wConv, strides=[1, 2, 2, 1], padding='VALID') + bConv # Stride [batch, height, width, channels]
        h = tf.nn.relu(a)
        hFlat = tf.reshape(h, [-1, 5408]) # Flat the output to be size 5408 each row
        yOut = tf.matmul(hFlat, w) + b

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

def train(Model, xT, yT, xV, yV, xTe, yTe, batchSize=1000, epochs=100, printEvery=10):
    # Train Model
    trainIndex = np.arange(xTrain.shape[0])
    np.random.shuffle(trainIndex)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Mini-batch
            losses = []
            accs = []
            # For each batch in training data
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                currentBatchSize = yTrain[idX].shape[0]

                # Train
                loss, acc, _ = sess.run(Model, feed_dict={x: xT[idX, :], y: yT[idX]})

                # Collect all mini-batch loss and accuracy
                losses.append(loss * currentBatchSize)
                accs.append(acc * currentBatchSize)

            totalAcc = np.sum(accs) / float(xTrain.shape[0])
            totalLoss = np.sum(losses) / xTrain.shape[0]
            if e % printEvery == 0:
                print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100), end='')
                loss, acc = sess.run(Model[:-1], feed_dict={x: xV, y: yV})
                print(' Validate loss = {0:.3f} and validate accuracy = {1:.2f}%'.format(loss, acc * 100))

        loss, acc = sess.run(Model[:-1], feed_dict={x: xTe, y: yTe})
        print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))

# Start training simple model
print("\n################ Simple Model #########################")
train(simpleModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Complex Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def complexModel():
    with tf.device(deviceType):
        #############################################################################
        # TODO: 40 points                                                           #
        # - Construct model follow below architecture                               #
        #       7x7 Convolution with stride = 2                                     #
        #       Relu Activation                                                     #
        #       2x2 Max Pooling                                                     #
        #       Fully connected layer with 1024 hidden neurons                      #
        #       Relu Activation                                                     #
        #       Fully connected layer to map to 10 outputs                          #
        # - Store last layer output in yOut                                         #
        #############################################################################

        # weights for 1st convolution layer with 2 strides
        wc1 = tf.get_variable("wc1", shape=[7, 7, 3, 64])

        # Biases  for 1st convolution layer
        bc1 = tf.get_variable("bc1", shape=[64])

        # weights for 1st fully connected layer
        FullyCW1 = tf.get_variable("FullyCW1", shape=[2304, 1024])

        # Biases for 1st fully connected layer
        FullyCB1 = tf.get_variable("FullyCB1", shape=[1024])

        # weights for 2st fully connected layer
        FullyCW2 = tf.get_variable("FullyCW2", shape=[1024, 10])

        # Biases for 2st fully connected layer
        FullyCB2 = tf.get_variable("FullyCB2", shape=[10])

        # 2d convolution layer
        conv2dlayer = tf.nn.conv2d(x, wc1, strides=[1, 2, 2, 1], padding='VALID') + bc1

        # relu activation for 1st layer
        reluactivation = tf.nn.relu(conv2dlayer)

        # 2 X 2 max pooling layer
        pooling_layer = tf.nn.max_pool(reluactivation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        make_vector = tf.reshape(pooling_layer, [-1, 2304])

        # fully connected layer
        fully_connected_lyr = tf.matmul(make_vector, FullyCW1) + FullyCB1

        # relu activation layer
        relu_after_pooling = tf.nn.relu(fully_connected_lyr)

        # multiply weights and add bias for fully connected second layer
        yOut = tf.matmul(relu_after_pooling, FullyCW2) + FullyCB2

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

# Start training complex model
print("\n################ Complex Model #########################")
train(complexModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Your Own Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def yourOwnModel():
    with tf.device(deviceType):
        #############################################################################
        # TODO: 60 points                                                           #
        # - Construct your own model to get validation accuracy > 70%               #
        # - Store last layer output in yOut                                         #
        #############################################################################

        # we will apply 2d convolution and max pooling 2 times and then attach fully connected layer at the end with
        # batch normalization

        # layer : 2d convolution with relu as activation function
        layer_2d_conv1 = tf.layers.conv2d(inputs=x, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)

        # layer : max pooling
        max_pooling_1 = tf.layers.max_pooling2d(inputs=layer_2d_conv1, pool_size=[2, 2], strides=2)

        # layer : 2d convolution with activation function as relu
        layer_2d_conv2 = tf.layers.conv2d(inputs=max_pooling_1, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)

        # layer : max pooling
        max_pooling_2 = tf.layers.max_pooling2d(inputs=layer_2d_conv2, pool_size=[2, 2], strides=2)

        # flatten to provide as input to fully connected layer
        reshape_vector = tf.reshape(max_pooling_2, [-1, 4096])
        fully_connected_layer1 = tf.layers.dense(inputs=reshape_vector, units=1024, activation=tf.nn.relu)

        # batch normalization
        batch_normalization = tf.layers.batch_normalization(inputs=fully_connected_layer1, training=True)

        # fully connected layer
        yOut = tf.layers.dense(inputs=batch_normalization, units=10, activation=None)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

# Start your own Model model
print("\n################ Your Own Model #########################")
#########################################################################
# TODO: 0 points                                                        #
# - You can set your own batchSize and epochs                           #
#########################################################################
train(yourOwnModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)
#########################################################################
#                       END OF YOUR CODE                                #
#########################################################################

