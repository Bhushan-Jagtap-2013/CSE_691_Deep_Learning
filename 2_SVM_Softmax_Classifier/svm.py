import numpy as np


class Svm (object):
    """" Svm classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random svm weight matrix to compute loss                 #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################

        self.W = np.random.normal(0.01, 0.01, (inputDim, outputDim))

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        Svm loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the svm loss and store to loss variable.                        #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################

        # calculate dot product of x and w
        s = np.dot(x, self.W)

        # Find correct score
        s_yi = s[np.arange(x.shape[0]), y]

        # find loss
        delta = s - s_yi[:, np.newaxis] + 1
        loss_i = np.maximum(0, delta)

        loss_i[np.arange(x.shape[0]),y] = 0
        loss = np.sum(loss_i) / x.shape[0]

        L2_loss_weight_squre = np.sum(self.W * self.W)
        multiply_lambda = reg * L2_loss_weight_squre
        loss = loss + multiply_lambda

        # find gradient decent
        ds = s - s_yi[:, np.newaxis] + 1
        ds[ds < 0] = 0
        ds[ds > 0] = 1

        ds[np.arange(s.shape[0]), y] = 0
        ds[np.arange(s.shape[0]), y] = -1 * np.sum(ds, axis=1)
        xT = np.transpose(x)
        dWh = xT.dot(ds)
        dWh = dWh / x.shape[0]
        dW = dWh + (2 * reg * self.W)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Svm classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            random_index_vector = np.random.choice(x.shape[0], batchSize) #, replace=True)
            xBatch = x[random_index_vector]
            yBatch = y[random_index_vector]

            # calculate loss and gradient decent
            # modify weight accordingly

            loss, dw = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.W = self.W - (lr * dw)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        # find score using dot product of x and weights. Max score will give us correct class

        s = np.dot(x, self.W)
        yPred = s.argmax(axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################

        # find number of correct predictions and divide by total number of examples

        s = np.dot(x, self.W)
        yPred = s.argmax(axis=1)
        correct_output = np.sum(yPred == y)
        acc = correct_output / x.shape[0]

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc