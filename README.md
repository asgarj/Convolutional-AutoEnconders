# Convolutional-AutoEnconders

This repo consists of some code snapshots from my MSc Final Project.

It's intended to share the implementations for autoencoders for various architectures such as autoencoders for hidden layer, 
layer-wise convolutional autoencoder, and stacked convolutional autoencoder.
All implementations are using theano framework which is for deep learning implementations in Python.

Please note that the convolution (& deconv) and pooling (& unpooling) operations are not like the usual ones and are specific to
the model in my case, as the target tasks are NLP tasks, rather than vision problems (i.e. input wasn't images).
Thus, you'll need to change these parts. The MovieReview.py is the application of the model on Stanford Sentiment TreeBank dataset.
