import theano.tensor as T
import theano
from theano import function
import numpy

class AE(object):
    def __init__(self, numpy_rng, input=None, n_visible = 900, n_hidden=500, W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(low = -4 * numpy.sqrt(6. / (n_hidden + n_visible)), \
                                                        high = 4 * numpy.sqrt(6. / (n_hidden + n_visible)), size=(n_visible, n_hidden)), \
                                                        dtype = theano.config.floatX)
            W = theano.shared(value=initial_W, name='Top_Layer_W')

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), name='bvis')

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='Top_Layer_bias')

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.tanh(T.dot(input, self.W) + self.b)        #   T.nnet.sigmoid(..)
    
    def get_reconstructed_input(self, hidden):
        return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, learning_rate):

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)

        L = T.sum((self.x-z) ** 2, axis=1)

        cost = T.mean(L)
        gparams = T.grad(cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)
    
def main():
    
    learning_rate=1e-1
    
    autoencoder = AE(numpy.random.RandomState(140), n_visible=784, n_hidden=500)
    cost, updates = autoencoder.get_cost_updates()
    train = theano.function([autoencoder.x], cost, updates=updates, mode='DebugMode')
    x = T.dmatrix(name='x')
    print x.type
    x = numpy.asarray( numpy.random.RandomState(5432).uniform(low = -0.2, high=.3, size=(50, 784)), dtype=theano.config.floatX)
    res = file('output.txt', 'w')
    finish1 = 9999;   finish2 = 0
    epoch = 1

    while((epoch < 20)):
        finish1 = finish2;  finish2 = train(x)
        print finish2
        b = autoencoder.params[1]
        res.write('epoch: '+str(epoch)+'\n'+str(b.get_value())+'\n\n')
        epoch += 1
    print 'Number of total epochs is "%d" with learning_rate="%f"' % (epoch, autoencoder.learning_rate.get_value())
    
    #print 'length:', len(b.get_value()), 'Value is as below', b.get_value(), '\nDONE!'

#if __name__ == "__main__":  main()

#    H = theano.gradient.hessian(cost, self.params, consider_constant=None, disconnected_inputs='warn') 
#    H, updates2 = theano.scan(lambda i, gparams: T.grad(gparams[i], self.params), sequences=T.arange(gparams.shape[0]), non_sequences=[gparams, self.params])
#    H_inv = theano.sandbox.linalg.ops.matrix_inverse(H)