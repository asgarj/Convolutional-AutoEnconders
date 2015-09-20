import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.sandbox.neighbours import images2neibs
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from HiddenLayer import HiddenLayer
from preprocess_input import Preprocess_Input
from ConvAE import main
from Plotter import plot


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, k, W=None, k_Top=6):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type k: an integer
        :param k: the downsampling (pooling) factor
        
        :type k_Top: integer
        :param k_Top: Top layer pooling factor
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        self.shape = image_shape

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        
        # initialize weights with random weights
        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
        else:
            self.W = W

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, border_mode='full',
                filter_shape=filter_shape, image_shape=image_shape)

        # k-max pooling.
        pooled_out = self.kmaxPool(conv_out, conv_out.shape, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        res = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = res*(res>0)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def Fold(conv_out, orig, ds=(2,1), ignore_border=False):
        '''Fold into two. (Sum up vertical neighbours)'''
        imgs = images2neibs(conv_out, T.as_tensor_variable(ds), mode='ignore_borders')  # Correct 'mode' if there's a typo!
        res = T.reshape(T.sum(imgs, axis=-1), orig)
        return res

    def Pool(self, conv_out):
        '''Average Pooling. Consequently, converts matrix into scalar.'''
        res = T.mean(T.sum(conv_out, axis=-1), axis=-1)
        return res
    
    def kmaxPool(self, conv_out, pool_shape, k):
        '''
        Perform k-max Pooling.
        '''
        n0, n1, d, size = pool_shape
        imgs = images2neibs(conv_out, T.as_tensor_variable((1, size)))

        indices = T.argsort(T.mul(imgs, -1))
        k_max_indices = T.sort(indices[:, :k])
        
        S = T.arange(d*n1*n0).reshape((d*n1*n0, 1))
        return imgs[S, k_max_indices].reshape((n0, n1, d, k))


def evaluate_lenet5(n_epochs=200,
                    nkerns=[6, 3], batch_size=10):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    #theano.config.compute_test_value = 'warn'
    rng = numpy.random.RandomState(23455)

    datasets = Preprocess_Input().load_data()

    train_set_x, train_set_y, _ = datasets[0]
    test_set_x, test_set_y, _ = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    print 'Train\tTest\tbatch_size\n', n_train_batches, '\t', n_test_batches, '\t', batch_size
    n_train_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized sentences. Each row is an instance of sentence.
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = T.dscalar('rate')

    k_Top = 6
    s_shape = (25, 37)  # this is the size of sentence matrix.

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized sentence of shape (batch_size,25*37)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 25, 37))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (25,37+3-1)=(25,39)
    # maxpooling reduces this further to (25,19)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],25,19)
    
    # _W = main(dataset=datasets)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, W=None,
            image_shape=(batch_size, 1, 25, 37),
            filter_shape=(nkerns[0], 1, 1, 5), k=k_Top)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (25,19+3-1)=(25,21)
    # maxpooling reduces this further to (25,12)
    # 4D output tensor is thus of shape (nkerns[1], nkerns[0], 1, 2)
    
    #k = max(k_Top, numpy.ceil((2.-2.)/2. * 37.))
    
    #layer1 = LeNetConvPoolLayer(rng, input=layer0.output, #W=_W[1],
    #        image_shape=(batch_size, nkerns[0], 25, 19),
    #        filter_shape=(nkerns[1], nkerns[0], 1, 3), k=12)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size,3*25*12) = (100,900)
    layer2_input = layer0.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[0] * 25 * k_Top,
                         n_out=200, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=200, n_out=6)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    #validate_model = theano.function([index], layer3.errors(y),
    #        givens={
    #            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #            y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index, learning_rate], cost, updates=updates, #g mode='DebugMode',
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    

    '''f = theano.function([index], T.shape(layer1.pooled_out), givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
    print 'pool_out.shape', f(0)
    '''

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    ls = []
    rate = 1e-1

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index, rate)


        # test it on the test set
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        test_score = numpy.mean(test_losses)
        ls.append(test_score)
        print(('     epoch %i, minibatch %i/%i, test error of best '
               'model %f %%') %
              (epoch, minibatch_index + 1, n_train_batches,
               test_score * 100.))
        
        # save best validation score and iteration number
        if test_score < best_validation_loss:
            best_validation_loss = test_score
            best_iter = iter

        if epoch <= 40:
            rate *= .9


    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    plot(numpy.asarray(ls)*100, rate)

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)