import theano
import numpy

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.sandbox.neighbours import images2neibs
from theano.sandbox.neighbours import neibs2images

from ConvAE import CAE
from Autoencoder import AE

class SCAE(object):
    def __init__(self, rng, input, filter_shape, image_shape, factor, s, k_Top=5, do_fold=True):
        
        #   Input will be image_shape, filter_shape, input, rng
        self.kshp = filter_shape
        self.imshp = None
        self.i_kshp = (self.kshp[1], self.kshp[0], self.kshp[2], self.kshp[3])
        self.i_imshp = None
        self.do_fold = do_fold
        self.k_Top = k_Top
        self.factor = factor
        self.s = s
        self.rng = rng

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), #   2, 1, 1, 3
            dtype=theano.config.floatX), name='conv_W',
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='conv_b', borrow=True)

        self.c = theano.shared(value=0.0, name='deconv_c')

        self.W_tilde = self.W[:, :, ::-1, ::-1].dimshuffle(1, 0, 2, 3)

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.c]

    def Fold(self, conv_out, ds=(2,1)):
        '''Fold into two. (Sum up vertical neighbours)'''
        imgs = images2neibs(conv_out, T.as_tensor_variable(ds), mode='ignore_borders')  # Correct 'mode' if there's a typo!
        orig = conv_out.shape
        shp = (orig[0], orig[1], T.cast(orig[2]/2, 'int32'), orig[3])
        res = T.reshape(T.sum(imgs, axis=-1), shp)
        return res

    def kmaxPool(self, conv_out, pool_shape, k):
        '''
        Perform k-max Pooling.
        '''
        n0, n1, d, size = pool_shape
        imgs = images2neibs(conv_out, T.as_tensor_variable((1, size)))

        indices = T.argsort(T.mul(imgs, -1))
        self.k_max_indices = T.sort(indices[:, :k])
    
        S = T.arange(d*n1*n0).reshape((d*n1*n0, 1))
        return imgs[S, self.k_max_indices].reshape((n0, n1, d, k))

    def unpooling(self, Y_4D, Z, X_4D):
        """ This method reverses pooling operation.
            """
        Y = images2neibs(Y_4D, T.as_tensor_variable((1, Y_4D.shape[3])))
        X = images2neibs(X_4D, T.as_tensor_variable((1, X_4D.shape[3])))
        X_z = T.zeros_like(X)
        X_ = T.set_subtensor( X_z[T.arange(X.shape[0]).reshape((X.shape[0], 1)), Z], Y )
        
        return X_.reshape(X_4D.shape)
    
    def Output(self):

        #  Convolve input with trained parameters.
        conv_out = conv.conv2d(input=self.x, filters=self.W, border_mode='full',
                filter_shape=self.kshp, image_shape=self.imshp)
        # Fold conv result into two.
        if self.do_fold:
            fold = self.Fold(conv_out)
        
        # k-max pooling.
        k = T.cast(T.max((self.k_Top, T.ceil(self.factor * self.s))), 'int32')
        if self.do_fold:
            pool_shape = fold.shape
            pooled_out = self.kmaxPool(fold, pool_shape, k)
        else:
            pool_shape = conv_out.shape
            pooled_out = self.kmaxPool(conv_out, pool_shape, k)
        
        return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_hidden_values(self, input):
        
        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W, border_mode='full',
                filter_shape=self.kshp, image_shape=self.imshp)
        
        # k-max pooling.
        k = T.cast(T.max((self.k_Top, T.ceil(self.factor * self.s))), 'int32')
        pool_shape = self.conv_out.shape
        pool = self.kmaxPool(self.conv_out, pool_shape, k)
        
        output = T.tanh(pool + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.shape = output.shape
        
        hidden_input = output.flatten(2)
        self.fully_connected = AE((self.rng), input=hidden_input, n_visible=self.kshp[0]*25*self.k_Top, n_hidden=60)  # nkerns[0] replaced with 8
        self.params.extend(self.fully_connected.params)
        
        return self.fully_connected.get_hidden_values(hidden_input)
    
    def get_reconstructed_input(self, hidden, start):
    
        reconstruct_AE = self.fully_connected.get_reconstructed_input(hidden)
        hidden_NN = reconstruct_AE.reshape(self.shape)
        
        unpool = self.unpooling(hidden_NN, self.k_max_indices, start)
        deconv = conv.conv2d(input=unpool, filters=self.W_tilde, filter_shape = self.i_kshp, image_shape=None)
        
        return T.tanh(deconv + self.c.dimshuffle('x', 'x', 'x', 'x'))
        # return val*(val>0)
    
    def get_cost_updates(self, learning_rate):

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y, self.conv_out)

        L = T.sum((self.x-z) ** 2, axis=(1,2,3))
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        
        rho = 1e-7
        G = [(theano.shared(value=numpy.zeros_like(param.get_value()), name="AdaGrad_" + param.name, borrow=True)) for param in self.params]
        G_update = [T.add(g_adag, T.sqr(grad_i)) for g_adag, grad_i in zip(G, gparams)]
        
        updates = []
        for param_i, g_update, grad_i, g in zip(self.params, G_update, gparams, G):
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(g_update) ))
            updates.append((g, g_update))
        
        return (cost, updates)
