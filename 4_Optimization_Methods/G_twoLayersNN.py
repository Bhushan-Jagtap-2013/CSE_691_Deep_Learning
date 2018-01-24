import numpy as np

class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim, update=0):
        self.params = dict()
        self.update = update
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)

    def calLoss (self, x, y, reg):
        grads = dict()

        # Forward pass to calculate loss
        tmp = x.dot(self.params['w1']) + self.params['b1']
        hOutput = np.maximum(0.01 * tmp, tmp)
        scores = hOutput.dot(self.params['w2']) + self.params['b2']
        scores = np.maximum(0.01 * scores, scores)
        scores -= np.max(scores, axis=1, keepdims=True)
        scores = np.exp(scores)
        scoresProbs = scores/np.sum(scores, axis=1, keepdims=True)
        logProbs = -np.log(scoresProbs[np.arange(x.shape[0]), y])
        loss = np.sum(logProbs) / x.shape[0]
        loss += 0.5 * reg * np.sum(self.params['w1'] * self.params['w1']) + 0.5 * reg * np.sum(self.params['w2'] * self.params['w2'])

        # Backward pass to calculate each gradient
        dScoresProbs = scoresProbs
        dScoresProbs[range(x.shape[0]), list(y)] -= 1
        dScoresProbs /= x.shape[0]
        grads['w2'] = hOutput.T.dot(dScoresProbs) + reg * self.params['w2']
        grads['b2'] = np.sum(dScoresProbs, axis=0)

        dhOutput = dScoresProbs.dot(self.params['w2'].T)
        dhOutputAct = (hOutput >= 0) * dhOutput + (hOutput < 0) * dhOutput * 0.01
        grads['w1'] = x.T.dot(dhOutputAct) + reg * self.params['w1']
        grads['b1'] = np.sum(dhOutputAct, axis=0)

        return loss, grads

    def train (self, x, y, lr=1e-3, reg=1e-5, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
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
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []

        # Initialize value for each update optimizer
        self.params['VW2'] = 0
        self.params['VW1'] = 0
        self.params['cacheW2'] = 0
        self.params['cacheW1'] = 0

        for i in range(iterations):
            batchID = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[batchID]
            yBatch = y[batchID]
            loss, grads = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            if self.update == 0:
                #########################################################################
                # TODO: 10 points                                                       #
                # - Use Naive Update to update weight parameter                         #
                #########################################################################

                # we just update weights by multiplying with learning rate
                self.params['w1'] += - lr * grads['w1']
                self.params['w2'] += - lr * grads['w2']


            elif self.update == 1:
                #########################################################################
                # TODO: 10 points                                                       #
                # - Use Momentum Update to update weight parameter                      #
                # - Momentum = 0.9                                                      #
                #########################################################################

                # actual speed is combination of gradient and previous speed

                mom = 0.9
                self.params['VW1'] = mom * self.params['VW1'] - lr * grads['w1']
                self.params['w1'] += self.params['VW1']
                self.params['VW2'] = mom * self.params['VW2'] - lr * grads['w2']
                self.params['w2'] += self.params['VW2']

            elif self.update == 2:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use Nesterov Update to update weight parameter                      #
                # - Momentum = 0.9                                                      #
                # - Hint                                                                #
                #   v_prev = v                                                          #
                #   v = mu * v - lr * dw                                                #
                #   w += -mu * v_prev + (1 + mu) * v                                    #
                #########################################################################

                # instead of using gradient calculated at current rate use
                # gradient for new weight that has considered the momentum
                # it uses same learning rate for all weights

                VPrevW1 = self.params['VW1']
                VPrevW2 = self.params['VW2']
                mom = 0.9
                self.params['VW1'] = mom * self.params['VW1'] - lr * grads['w1']
                self.params['w1'] += - mom * VPrevW1 + (1 + mom) * self.params['VW1']
                self.params['VW2'] = mom * self.params['VW2'] - lr * grads['w2']
                self.params['w2'] += - mom * VPrevW2 + (1 + mom) * self.params['VW2']

            elif self.update == 3:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use AdaGrad Update to update weight parameter                       #
                #########################################################################

                # uses small learning rate for large gradients and large for smaller one

                self.params['cacheW1'] += grads['w1'] ** 2
                self.params['w1'] += - lr * grads['w1'] / (np.sqrt(self.params['cacheW1']) + lr)
                self.params['cacheW2'] += grads['w2'] ** 2
                self.params['w2'] += - lr * grads['w2'] / (np.sqrt(self.params['cacheW2']) + lr)

            elif self.update == 4:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use RMSProp Update to update weight parameter                       #
                #########################################################################

                # Cache is an exponential moving average of gradient
                # and learning rate does not approach to zero

                self.params['cacheW1'] = decay * self.params['cacheW1'] + (1 - decay) * grads['w1'] ** 2
                self.params['w1'] += - lr * grads['w1'] / (np.sqrt(self.params['cacheW1']) + lr)
                self.params['cacheW2'] = decay * self.params['cacheW2'] + (1 - decay) * grads['w2'] ** 2
                self.params['w2'] += - lr * grads['w2'] / (np.sqrt(self.params['cacheW2']) + lr)
            else:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use Adam Update to update weight parameter                          #
                # - B1 = 0.9, B2 = 0.999                                                #
                #########################################################################

                # VW1 and VW2 estimates to the first mo which is the mean and cache estimates second row Momentum

                B1 = 0.9
                B2 = 0.999
                iter = i+1

                self.params['VW1'] = B1 * self.params['VW1'] + (1 - B1) * grads['w1']
                self.params['cacheW1'] = B2 * self.params['cacheW1'] + (1 - B2) * (grads['w1'] ** 2)
                vb = self.params['VW1'] / (1 - (B1 ** iter))
                cacheb = self.params['cacheW1'] / (1 - (B2 ** iter))
                self.params['w1'] -= lr * vb / (np.sqrt(cacheb) + lr)

                self.params['VW2'] = B1 * self.params['VW2'] + (1 - B1) * grads['w2']
                self.params['cacheW2'] = B2 * self.params['cacheW2'] + (1 - B2) * (grads['w2'] ** 2)
                vb2 = self.params['VW2'] / (1 - (B1 ** iter))
                cacheb2= self.params['cacheW2'] / (1 - (B2 ** iter))
                self.params['w2'] -= lr * vb2 / (np.sqrt(cacheb2) + lr)

            self.params['b2'] += -lr * grads['b2']
            self.params['b1'] += -lr * grads['b1']
            lr *= decay
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        tmp = x.dot(self.params['w1']) + self.params['b1']
        hOutput = np.maximum(0.01 * tmp, tmp)
        scores = hOutput.dot(self.params['w2']) + self.params['b2']
        yPred = np.argmax(scores, axis=1)
        return yPred

    def calAccuracy (self, x, y):
        acc = 100.0 * (np.sum(self.predict(x) == y) / float(x.shape[0]))
        return acc