import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.sandbox.neighbours import images2neibs

from logistic_sgd import LogisticRegression, load_data
from HiddenLayer import HiddenLayer
from ConvAE import CAE
from Autoencoder import AE
from SCAE_Movie import SCAE
from Preprocess import Preprocess
from Plotter import plot
import matplotlib.pyplot as plt


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, factor, s, W=None, b=None, k_Top=5, do_Fold=False):
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

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if not W:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
            # initialize weights with random weights
            W_bound = numpy.sqrt(1. / (fan_in + fan_out))
            W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
        self.W = W

        if not b:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, border_mode='full', # Change border_mode to 'full'
                filter_shape=None, image_shape=None)

        #   Folding into two.
        if do_Fold:
            fold = self.Fold(conv_out)
    
        # k-max pooling.
        k = T.cast(T.max((k_Top, T.ceil(factor * s))), 'int32')
        if do_Fold:
            pooled_out = self.kmaxPool(fold, fold.shape, k)
        else:
            pooled_out = self.kmaxPool(conv_out, conv_out.shape, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

    def Fold(self, conv_out, ds=(2,1)):
        '''Fold into two. (Sum up vertical neighbours)'''
        imgs = images2neibs(conv_out, T.as_tensor_variable(ds), mode='ignore_borders')  # Correct 'mode' if there's a typo!
        orig = conv_out.shape
        shp = (orig[0], orig[1], T.cast(orig[2]/2, 'int32'), orig[3])
        res = T.reshape(T.sum(imgs, axis=-1), shp)
        return res

    def Pool(self, conv_out):
        '''Average Pooling. Consequently, converts matrix into scalar.'''
        res = T.mean(T.sum(conv_out, axis=-1), axis=-1)
        return res

    def kmaxPool(self, fold, pool_shape, k):
        '''
        Perform k-max Pooling.
            '''
        n0, n1, d, size = pool_shape
        imgs = images2neibs(fold, T.as_tensor_variable((1, size)))

        indices = T.argsort(T.mul(imgs, -1))
        k_max_indices = T.sort(indices[:, :k])
    
        S = T.arange(d*n1).reshape((d*n1, 1))
        return imgs[S, k_max_indices].reshape((n0, n1, d, k))
        
def Layer_Wise_preTrain(batch_size = 1, nkerns = [3, 4], dataset=None, n_epochs=6, k_Top=5):
    """
        Layer-wise Convolutional auto-encoders.
    """

    if dataset is None:
        dataset = Preprocess_Input().load_data()
    train_set_x = dataset[0][0]
    train_set_z = dataset[0][2]
    n_train_batch = train_set_x.get_value(borrow=True).shape[0]
    n_train_batch /= batch_size

    print '... Building AutoEncoders'
    rng = numpy.random.RandomState(96813)
    index = T.lscalar('index')
    learning_rate = T.dscalar('rate')
    x = T.matrix('x')
    z = T.iscalar('z')
    #index.tag.test_value = 0
    #learning_rate.tag.test_value = .3
    em = 50
    layer0_input = x[:, :z*50].reshape((batch_size, 1, 50, -1))

    #   Auto-Encoder for Conv. LAYER 1
    layer0 = CAE(rng, input=layer0_input, image_shape=(batch_size, 1, em, None), \
        filter_shape=(nkerns[0], 1, 1, 7), factor=.5, s=z, k_Top=k_Top, do_fold=True)
    
    #zz = layer0.get_cost_updates(learning_rate)
    #print 'hidden:', theano.function([index, learning_rate], [zz], on_unused_input='ignore', \
    #                                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size], \
    #                                        z: train_set_z[index]})(0, .3)
    cost, updates = layer0.get_cost_updates(learning_rate)
    #print 'DECODE RESULT:\n', theano.function([index], [layer0.output.shape, layer0_input.shape, z.type, layer0.zz.shape], \
    #                                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size], \
    #                                        z: train_set_z[index]})(0)
    
    train0 = theano.function([index, learning_rate], cost, updates=updates, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                z: train_set_z[index]})
    em /= 2
    #   Auto-Encoder for Conv. LAYER 2.
    layer1_input = layer0.Output()
    layer1 = CAE(rng, input=layer1_input, image_shape=(batch_size, nkerns[0], em, None), \
                        filter_shape=(nkerns[1], nkerns[0], 1, 3), factor=.0, s=z, k_Top=k_Top, do_fold=True)

    cost1, updates1 = layer1.get_cost_updates(learning_rate)

    train1 = theano.function([index, learning_rate], cost1, updates=updates1, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                z: train_set_z[index]})
    em /= 2
    #   Auto-Encoder for Hidden Layer.
    hidden_input = layer1.Output().flatten(2)
    hidden_layer = AE(rng, input=hidden_input, n_visible=nkerns[1]*em*k_Top, n_hidden=100)
    cost_h, updates_h = hidden_layer.get_cost_updates(learning_rate)
    train_h = theano.function([index, learning_rate], cost_h, updates=updates_h, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                z: train_set_z[index]})
    print '... Pretraining model'

    ls_1 = []
    rate = 1e-2
    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch in xrange(n_train_batch):
            cost_ij = train0(minibatch, rate)
        ls_1.append(cost_ij)
        rate *= .95
        print '\tepoch %i : cost: %f' % (epoch, cost_ij)
        # print layer0.W.get_value(borrow=True)

    ls_2 = []
    rate = 1e-2
    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch in xrange(n_train_batch):
            cost_ij = train1(minibatch, rate)
        ls_2.append(cost_ij)
        rate *= .95
        print '\tepoch %i : cost: %f' % (epoch, cost_ij)
        # print layer1.W.get_value(borrow=True)

    ls_3 = []
    rate=4e-2
    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch in xrange(n_train_batch):
            cost_ij = train_h(minibatch, rate)
        ls_3.append(cost_ij)
        rate *= .95
        print '\tepoch %i : cost: %f' % (epoch, cost_ij)
        
    
    #  PLOT AutoEncoder Cost Function
    plt.subplot(3, 1, 1)
    plt.plot(numpy.arange(len(ls_1)) + 1, numpy.asarray(ls_1), 'r.-')
    plt.title('AutoEncoder Cost function Results')
    plt.xlabel('Epochs')
    plt.ylabel('Convolutional Layer 1')
    
    plt.subplot(3, 1, 2)
    plt.plot(numpy.arange(len(ls_2)) + 1, numpy.asarray(ls_2), 'r.-')
    #plt.title('AutoEncoder Cost function Results')
    plt.xlabel('Epochs')
    plt.ylabel('Convolutional Layer 2')
    
    plt.subplot(3, 1, 3)
    plt.plot(numpy.arange(len(ls_3)) + 1, numpy.asarray(ls_3), 'r.-')
    plt.xlabel('Epochs')
    plt.ylabel('Hidden Layer values')
    
    plt.show()
    
    return [layer0.params, layer1.params, hidden_layer.params]

def SCAE_preTrain(batch_size=1, nkerns= [3, 4], dataset=None, n_epochs=70, k_Top=5, learning_rate=1e-1, binary=True):
    """
        Stacked Convolutional AutoEncoders.
    """

    with open('SCAE_MR_1e-1K34', 'r') as f:
        rval = cPickle.load(f)
    return rval

    if dataset is None:
        dataset = Preprocess().load_data(binary)
    train_set_x = dataset[0][0]
    train_set_z = dataset[0][2]
    n_train_batch = train_set_x.get_value(borrow=True).shape[0]
    n_train_batch /= batch_size

    print '... Building Stacked Conv. AutoEncoders'

    rng = numpy.random.RandomState(96813)
    index = T.lscalar('index')
    
    x = T.dmatrix('Input Sentence')
    z = T.iscalar('Sentence length')
    layer0_input = x[:, :z*50].reshape((batch_size, 1, 50, -1))
    
    layer0 = SCAE(rng, input=layer0_input, image_shape=None, filter_shape=(nkerns[0], 1, 1, 8), \
                                        factor=.5, s=z, k_Top=k_Top, do_fold=False)

    layer1_input = layer0.get_hidden_values(layer0_input)
    layer1 = SCAE(rng, input=layer1_input, filter_shape=(nkerns[1], nkerns[0], 1, 5), \
                        image_shape=None, factor = .0, s=z, k_Top=k_Top, do_fold=False)

    layer1_output = layer1.get_hidden_values(layer1_input)
    
    hidden_input = layer1_output.flatten(2)
    layer2 = AE(rng, input=hidden_input, n_visible=layer1.kshp[0]*50*k_Top, n_hidden=100)
    
    Y = layer2.get_hidden_values(hidden_input)
    
    ################
    #   DECODING   #
    ################

    decode_hidden_layer = layer2.get_reconstructed_input(Y)
    decode_input = decode_hidden_layer.reshape(layer1.shape)
    
    decode_layer1 = layer1.get_reconstructed_input(decode_input)
    Z = layer0.get_reconstructed_input(decode_layer1)

    params = layer2.params + layer1.params + layer0.params
    
    def get_cost_updates(X, Z, params, learning_rate):
        ''' Update the Stacked Convolutional Auto-Encoders. '''
        
        L = T.sum((X-Z) ** 2, axis=(1,2,3))
        cost = T.mean(L)
        
        gparams = T.grad(cost, params)
        
        rho = 1e-7
        G = [(theano.shared(value=numpy.zeros_like(param.get_value()), name="AdaGrad_" + param.name, borrow=True)) for param in params]
        G_update = [T.add(g_adag, T.sqr(grad_i)) for g_adag, grad_i in zip(G, gparams)]
        
        updates = []
        for param_i, g_update, grad_i, g in zip(params, G_update, gparams, G):
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(g_update) ))
            updates.append((g, g_update))
        
        return (cost, updates)
    
    cost, updates = get_cost_updates(layer0_input, Z, params, learning_rate)
    
    train_model = theano.function([index], cost, updates=updates, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                z: train_set_z[index]})

    print '... Pretraining model'

    plot_SCAE = []
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibatch in xrange(n_train_batch):
            cost_ij = train_model(minibatch)
        print '\tepoch %i,\tcost  %f' % (epoch, cost_ij)
        plot_SCAE.append(cost_ij)
    plot('SCAE_Movie Results.', numpy.asarray(plot_SCAE), 74e-2)

    # Serialise the learned parameters
    with open('SCAE_MR_1e-1K%i%i'%(nkerns[0], nkerns[1]), 'wb') as f:
        cPickle.dump(params, f)
    return params

def evaluate_lenet5(n_epochs=130, binary=True, preTrain=True, learning_rate=3e-3, \
                    nkerns=[3, 4], batch_size=1):
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
    rng = numpy.random.RandomState(23455)

    datasets = Preprocess().load_data(binary)

    train_set_x, train_set_y, train_set_z = datasets[0]
    valid_set_x, valid_set_y, valid_set_z = datasets[1]
    test_set_x, test_set_y, test_set_z = datasets[2]
    
    if preTrain:
        pre_W = SCAE_preTrain(nkerns=nkerns, dataset=datasets)
        W_h, b_h,_, W2, b2,_, W1, b1,_ = pre_W
    else:
        W_h, b_h, W2, b2, W1, b1 = [None] * 6


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    print 'Train\tVal\tTest\tbatch_size\n', n_train_batches, '\t', n_valid_batches, '\t', n_test_batches, '\t', batch_size
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized sentences. Each row is an instance of sentence.
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    z = T.iscalar('z')  # the sentence lengths are presented as 1D vector of [int] lengths.
    
    k_Top = 5
    em=50
    s_shape = (50, 56)  # this is the size of (padded) sentence matrix.

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized sentence of shape (batch_size,25*37)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x[:, :z*em].reshape((batch_size, 1, em, -1))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (25,37+3-1)=(25,39)
    # maxpooling reduces this further to (25,19)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],25,19)
    
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, em, None),
            filter_shape=(nkerns[0], 1, 1, 5), factor=.5, W = W1, b=b1, k_Top=k_Top, s=z)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (25,19+3-1)=(25,21)
    # maxpooling reduces this further to (25,12)
    # 4D output tensor is thus of shape (nkerns[1], nkerns[0], 1, 2)

    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], em, None),
            filter_shape=(nkerns[1], nkerns[0], 1, 3), factor=0., W=W2, b=b2, k_Top=k_Top, s=z)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size,3*25*12) = (100,900)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * em * k_Top,
                         n_out=100, W=W_h, b=b_h, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                z: test_set_z[index]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                z: valid_set_z[index]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    
    rho = 1e-7
    G = [(theano.shared(value=numpy.zeros_like(param.get_value())+rho, name="AdaGrad_" + param.name, borrow=True)) for param in params]
    G_update = [T.add(g_adag, T.sqr(grad_i)) for g_adag, grad_i in zip(G, grads)]
    
    updates = []
    for param_i, g_update, grad_i, g in zip(params, G_update, grads, G):
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(g_update) ))
        updates.append((g, g_update))

    train_model = theano.function([index], cost, updates=updates, allow_input_downcast=True,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            z: train_set_z[index]})


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 500 # min(n_train_batches/4, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    res = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

            #    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
            #          (epoch, minibatch_index + 1, n_train_batches, \
            #           this_validation_loss * 100.))
            #    res.append(this_validation_loss)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            #if patience <= iter:
            #    done_looping = True
            #    break
        print('epoch %i, validation error %f %%' % \
              (epoch, this_validation_loss * 100.))
        res.append(this_validation_loss)
    
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    final_test_score = numpy.mean(test_losses)
    
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print 'Final test score:', final_test_score
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    plot('Movie Reviews- Binary. Validation Loss.', numpy.asarray(res) * 100., test_score * 100.)
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)