import numpy
import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano import function
from theano.sandbox.neighbours import images2neibs

from preprocess_input import Preprocess_Input
from DCNN import LeNetConvPoolLayer


class CAE(object):
    def __init__(self, rng, input, filter_shape, image_shape, factor, s, k_Top=5, do_fold=True):
        
        #   Input will be image_shape, filter_shape, input, rng
        self.kshp = filter_shape
        self.imshp = None # image_shape
        self.i_kshp = (self.kshp[1], self.kshp[0], self.kshp[2], self.kshp[3])
        self.i_imshp = None # (self.imshp[0], 1, self.imshp[2], (self.imshp[3]+self.kshp[3]-1))
        self.do_fold = do_fold
        self.k_Top = k_Top
        self.factor = factor
        self.s = s

        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
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

        self.W_tilde = self.W[:,:, ::-1, ::-1].dimshuffle(1, 0, 2, 3)

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
        k_max_indices = T.sort(indices[:, :k])
    
        S = T.arange(d*n1*n0).reshape((d*n1*n0, 1))
        return imgs[S, k_max_indices].reshape((n0, n1, d, k))


    
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
        conv_out = conv.conv2d(input=input, filters=self.W, border_mode='full',
                filter_shape=self.kshp, image_shape=self.imshp)

        # val = conv_out+self.b.dimshuffle('x', 0, 'x', 'x');   val*(val>0)
        return T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    
    def get_reconstructed_input(self, hidden):
    
        '''lst = []
        for j in xrange(self.i_kshp[0]):
            conv_out, _ = theano.scan(fn=lambda i, hidden: conv.conv2d(input=hidden[:,i,:,:].dimshuffle(0, 'x', 1, 2), \
                                                                        filters=self.W_tilde[j,i,:,:].dimshuffle('x', 'x', 0, 1), \
                                                                        filter_shape=self.i_kshp, image_shape=self.i_imshp), \
                                                            outputs_info=None, \
                                                            sequences=T.arange(hidden.shape[1]),\
                                                            non_sequences=hidden)'''
        conv_out = conv.conv2d(input=hidden, filters=self.W_tilde, filter_shape = self.i_kshp, image_shape=None)
        #    lst.append(T.tanh(T.sum(conv_out, axis=0) + self.c.dimshuffle('x', 'x', 'x', 'x')))
        return T.tanh(conv_out + self.c.dimshuffle('x', 'x', 'x', 'x'))
        #return T.concatenate(lst, axis=1)
        #self.conv_res = conv_out
        #self.test = conv.conv2d(input=hidden, filters=self.W_tilde, filter_shape=(1, 2, 1, 3), image_shape=(50,2,25,39))
        #return T.tanh(T.sum(conv_out, axis=0) + self.c.dimshuffle('x', 'x', 'x', 'x'))
        # return val*(val>0)
    
    def get_cost_updates(self, learning_rate):

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)

        L = T.sum((self.x-z) ** 2, axis=(1,2,3))
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)
    

def main(batch_size = 50, nkerns = [2, 3], dataset=None):

    if dataset is None:
        dataset = Preprocess_Input().load_data()
    train_set_x = dataset[0][0]
    n_train_batch = train_set_x.get_value(borrow=True).shape[0]
    n_train_batch /= batch_size

    print '... building model'
    rng = numpy.random.RandomState(2345)
    index = T.lscalar('index')
    x = T.matrix('x')
    learning_rate = 0.001
    layer0_input = x.reshape((batch_size, 1, 25, -1))

    #   LAYER 1.
    layer0 = CAE(rng, input=layer0_input,
        image_shape=(batch_size, 1, 25, 37),
        filter_shape=(nkerns[0], 1, 1, 3))
    
    cost, updates = layer0.get_cost_updates(learning_rate)
    '''print 'DECODE RESULT:\n', function([index], [layer0.test, layer0.conv_res], givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})(0)'''
    
    train0 = theano.function([index], cost, updates=updates, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
    
    #   LAYER 2.
    layer1 = CAE(rng, input=layer0.output, 
        image_shape=(batch_size, nkerns[0], 25, 19),
        filter_shape=(nkerns[1], nkerns[0], 1, 3), k=12)
    print 'Layer 1, W parameters:\n', layer1.W.get_value(borrow=True)
    
    cost1, updates1 = layer1.get_cost_updates(learning_rate)

    train1 = theano.function([index], cost1, updates=updates1, \
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

#    print 'Initial weights:\n', layer0.W.get_value()

    for epoch in xrange(20):
        for minibatch in xrange(n_train_batch):
            cost_ij = train0(minibatch)
        print 'iter %i..' % (epoch+1)
        #print layer0.W.get_value(borrow=True)
    
    for epoch in xrange(1):
        for minibatch in xrange(20):
            cost_ij = train1(minibatch)
            print 'iter %i..' % (minibatch+1)
            #print layer1.W.get_value(borrow=True)
    
    return [layer0.W, layer1.W]
    
    
    

def test_method(batch_size=1, nkerns=[2]):
    print '... building model'
    rng = numpy.random.RandomState(2345)
    index = T.lscalar('index')
    x = T.matrix('x')
    learning_rate = 0.1
    
    layer0_input = x.reshape((batch_size, 1, 1, 5))
    
    layer0 = AutoEncoder(rng, input=layer0_input,
        image_shape=(batch_size, 1, 1, 5),
        filter_shape=(nkerns[0], 1, 1, 3))
    arr = numpy.asarray([
                       [
                        [[0., 5., 1., 28, -31],
                        [10, -8, -7, 9, 3]],

                        [[3., -2., 1., 23, 40],
                        [17, -11, -12, -89, 32]]
                       ]
                        ])
    
    train_set_x = theano.shared(arr.reshape((4, 5)))
    
    cost, updates = layer0.get_cost_updates(learning_rate)
    sh1, sh2, w, w_t, r1, r2, r3 = function([index], [layer0.test.shape, layer0.conv_res.shape, layer0.W, layer0.W_tilde, layer0.hid, layer0.conv_res, layer0.test], \
                            givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})(0)
    print 'DECODE RESULT:\n', sh1, sh2, '\nW:\n', w, '\nW_Tilde:\n', w_t, '\nHidden:\n', r1, '\nConv_Res:\n', r2, '\nTest:\nc', r3











#if __name__ == "__main__":   main() #   test_method()   #