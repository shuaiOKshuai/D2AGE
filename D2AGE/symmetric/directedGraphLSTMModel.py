#encoding=utf-8
'''
@author: Liu Zemin
Functions and Application : 
DAG LSTM model
'''

import numpy
import theano
from theano import tensor

def directedGraphLSTMModel(options, tparams, x, mask, wemb, buffer_tensor, nodesLen):
    
    """
    """
    
    length=x.shape[0] 
    dimension=wemb.shape[1] 
    
    proj=wemb[x]
    discount_vector=discountModel(options['discount_beta'], nodesLen)
    
    def _step(index,hArr,cArr):
        
        
        hi_sum=None

        discount=mask[index]*discount_vector # shape=maxlen*0
        hi_sum=(discount[:,None] * hArr).max(axis=0)
        
        # input gate, vector, shape= lstm_dimension * 0
        i=tensor.nnet.sigmoid(tensor.dot(tparams['Wi'], proj[index]) + tensor.dot(tparams['Ui'], hi_sum) + tparams['bi'])
        # forget gate, vector, shape= maxlen*lstm_dimension
        f=tensor.nnet.sigmoid(tensor.dot(tparams['Wf'], proj[index]) + tensor.dot((mask[index])[:,None]*hArr, tparams['Uf']) + tparams['bf'])
        # output gate, vector, shape= lstm_dimension * 0
        o=tensor.nnet.sigmoid(tensor.dot(tparams['Wo'], proj[index]) + tensor.dot(tparams['Uo'], hi_sum) + tparams['bo'])
        # new temp cell, vector, shape= lstm_dimension * 0
        c_=tensor.tanh(tensor.dot(tparams['Wc'], proj[index]) + tensor.dot(tparams['Uc'], hi_sum) + tparams['bc'])
        
        c=None
        c=i*c_ + (discount[:,None] * (f * ((mask[index])[:,None]*cArr))).max(axis=0)
        
        h=o*tensor.tanh(c)
        
        hArr=tensor.set_subtensor(hArr[index, :], h)
        cArr=tensor.set_subtensor(cArr[index, :], c)
        
        return hArr, cArr
    
    rval, update=theano.scan(
                             _step,
                             sequences=tensor.arange(x.shape[0]),
                             outputs_info=[tensor.alloc(numpy_floatX(0.), length, options['dimension']),# @UndefinedVariable h
                                           tensor.alloc(numpy_floatX(0.), length, options['dimension'])],# @UndefinedVariable c
                             )
    if options['h_output_method']=='h':
        return rval[0][-1][-1] 
    elif options['h_output_method']=='mean-pooling': 
        return rval[0][-1].mean(axis=0)
    elif options['h_output_method']=='max-pooling':
        return rval[0][-1].max(axis=0)
    else: 
        return rval[0][-1][-1] 

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable

def discountModel(beta,length):
    """
    """
    return tensor.exp(beta*length*(-1))