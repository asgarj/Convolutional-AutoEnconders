import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
class Preprocess(object):
    
    def __init__(self):
        pref = 'data/stanfordSentimentTreebank/'
        self.path = {'dataset': pref + 'sentences.csv'}
        
        '''labels= np.genfromtxt(pref+'labels.csv', dtype=float)
        print 'labels[14487]:', labels[14487]
        
        self.maxLen=56
        self.dic = {}
        self.embed = np.genfromtxt('data/words50.txt', delimiter="\n", dtype=str, comments='>>XX<<')
        indices = np.arange(self.embed.shape[0])
        self.dic = dict(zip(self.embed, indices))
        self.vector = np.genfromtxt('data/vectors50.csv', dtype=float, delimiter=' ')
        
        self.embedding_size = self.vector.shape[1]
        assert self.embedding_size == 50
        print 'EMBED == WORD:', self.embed.shape[0] == self.vector.shape[0]
        self.sent_len = self.embedding_size * self.maxLen
        
        print '... Reading phrases'
        phraseID = np.genfromtxt(pref+'phrasesIDs.csv', dtype=int, comments='>>XX<<')
        phrases = np.genfromtxt(pref+'phrases.csv', dtype=str, delimiter='\n', comments='>>XX<<')
        print '... making dictionary'
        phrase_dict = dict(zip(phrases, phraseID))
        print 'dict complete! First element:', phrase_dict.keys()[0], phrase_dict.values()[0]

        self.dataset = np.genfromtxt(self.path['dataset'], dtype=str, delimiter='\n', comments='>>XX<<')
        self.y = np.zeros(self.dataset.shape[0], dtype=int)
        it=-1
        inList = []
        found=0
        for line in self.dataset:
            it += 1
            if line in phrases:
                found+=1
                ID = phrase_dict[line]
                l = self.label(labels[ID])
                self.y[it] = l
                inList.append(it)
        print 'Number of iterations:', (it+1), self.y.shape, 'Found:', found

        self.splitData = np.genfromtxt(pref+'datasetSplit.csv', delimiter=',', dtype=int, skip_header=1)
        
        self.dataset = self.dataset[self.y != 0]
        self.splitData = self.splitData[self.y != 0]
        self.y = self.y[self.y != 0]
        print 'Y.shape', self.y.shape
        
        self.train_x = self.dataset[self.splitData[:,1]==1]
        self.train_y = self.y[self.splitData[:,1]==1] - 1
        
        self.test_x = self.dataset[self.splitData[:,1]==2]
        self.test_y = self.y[self.splitData[:,1]==2] - 1
        
        self.valid_x = self.dataset[self.splitData[:,1]==3]
        self.valid_y = self.y[self.splitData[:,1]==3] - 1
        
        print 'Shapes:', self.dataset.shape, self.train_x.shape, self.valid_x.shape, self.test_x.shape, self.test_y.shape
        print 'Y.shape, zeroes:', self.y.shape, self.y[self.y!=0].shape'''
        
    def label(self, l):
        if l <= .5: return 1
        else:       return 2
        '''if l <= .2: return 1
        elif l <= .4:   return 2
        elif l <= .6:   return 3
        elif l <= .8:   return 4
        else:           return 5'''

    def sentence_matrix(self, sentence):
        '''
            Given list of tokens, return corresponding numpy matrix of the sentence. Right-pad shorter sentences to have length of maxLen.

            type sentence  :  String (as sentence)
            param sentence :  Sentence
        '''
        S = sentence.split()
        length = len(S)
        ls = []
        for token in S:
            if not token in self.dic:
                token = '*UNKNOWN*'
            index = self.dic[token]
            ls.append(index)
        
        res = self.vector[ls, :].reshape((-1))
        
        assert res.shape[0] == length * self.embedding_size
        res = np.hstack((res, np.zeros(self.embedding_size * (self.maxLen-length))))
        res = np.hstack((res, np.array([length])))
        return res

    def get_data(self, arg):
        print '... Get_data for', arg
        if arg == 'train':
            data_x = self.train_x
            data_y = self.train_y
        elif arg == 'valid':
            data_x = self.valid_x
            data_y = self.valid_y
        elif arg == 'test':
            data_x = self.test_x
            data_y = self.test_y
        
        assert data_x.shape[0] == data_y.shape[0]
        y = data_y
    
        x = np.zeros(self.sent_len+1)       # +1 because of z
        for s in data_x:
            x = np.vstack((x, self.sentence_matrix(s)))
        x = x[1:]
        z = x[:, -1]
        x = x[:, :-1]
        print 'z.shape', z.shape
        print 'AFTER: shape of x y z:', x.shape, y.shape, z.shape
        print arg, z[0], y[0], x[0]
        return (x, y, z)
        
    def process(self, binary):
        '''print '... Processing'

        valid = self.get_data('valid')
        test = self.get_data('test')
        train = self.get_data('train')
        
        rval = [train, valid, test]
        
        with open('data/stanfordSentimentTreebank/INPUT_Binary', 'wb') as f:
            pickle.dump(rval, f)'''
    
        if binary:
            print '... Binary dataset loading'
            with open('data/stanfordSentimentTreebank/INPUT_Binary', 'r') as f:
                rval = pickle.load(f)
        if not binary:
            print '... Multi-class dataset loading'
            with open('data/stanfordSentimentTreebank/INPUT', 'r') as f:
                rval = pickle.load(f)
        
        # pref = 'data/stanfordSentimentTreebank/'
        # np.savetxt(pref+'train_x.csv', rval[0][0], delimiter=',')
        # np.savetxt(pref+'train_y.csv', rval[0][1], delimiter=',')
        # np.savetxt(pref+'train_z.csv', rval[0][2], delimiter=',')
        
        # np.savetxt(pref+'valid_x.csv', rval[1][0], delimiter=',')
        # np.savetxt(pref+'valid_y.csv', rval[1][1], delimiter=',')
        # np.savetxt(pref+'valid_z.csv', rval[1][2], delimiter=',')
        
        # np.savetxt(pref+'test_x.csv', rval[2][0], delimiter=',')
        # np.savetxt(pref+'test_y.csv', rval[2][1], delimiter=',')
        # np.savetxt(pref+'test_z.csv', rval[2][2], delimiter=',')'''
        
        return rval
        
    def load_data(self, binary=True):
        ''' Loads the dataset
        '''

        #############
        # LOAD DATA #
        #############

        print '... loading datasets'
        train_set, valid_set, test_set = self.process(binary)
        #print 'Train_set_y: Check if it is binary!\n', train_set[1]


        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.

        def shared_dataset(data_xyz, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y, data_z = data_xyz
            shared_x = theano.shared(np.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_z = theano.shared(np.asarray(data_z,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_z, 'int32')
        test_set_x, test_set_y, test_set_z = shared_dataset(test_set)
        valid_set_x, valid_set_y, valid_set_z = shared_dataset(valid_set)
        train_set_x, train_set_y, train_set_z = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y, train_set_z), (valid_set_x, valid_set_y, valid_set_z),
                (test_set_x, test_set_y, test_set_z)]
        return rval
