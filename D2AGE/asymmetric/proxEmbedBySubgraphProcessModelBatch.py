#encoding=utf-8
'''
the processing model
'''

import numpy
import theano
from theano import tensor
import directedGraphLSTMModel
from theano.ifelse import ifelse

def proxEmbedBySubgraphProcessModel(options, tparams):
    xs=tensor.matrix('xs', dtype='int64') # shape=nsamples*maxlen
    masks=tensor.tensor3('masks', dtype=theano.config.floatX)  # @UndefinedVariable # shape=nsamples * maxlen * maxlen
    lengths=tensor.vector('lengths',dtype='int64') # shape=#(xs) * 0
    subgraph_lens=tensor.vector('subgraph_lens', dtype='int64') # shape=nsamples*0
    wordsEmbeddings=tensor.matrix('wordsEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable # shape=#(words) * wordsDimension
    buffer_tensor=tensor.tensor3('buffer_tensor', dtype=theano.config.floatX)  # @UndefinedVariable # shape=maxlen*maxlen*dimension
    nodesLens=tensor.matrix('xs', dtype='int64') # shape=nsamples*maxlen
        
    def _processSubgraph(i):
        length=lengths[i]
        x=xs[i,:length] 
        mask=masks[i,:length,:length] 
        nodesLen=nodesLens[i,:length] 
        emb=directedGraphLSTMModel.directedGraphLSTMModel(options, tparams, x, mask, wordsEmbeddings, buffer_tensor, nodesLen) 
        return emb 
    
    embx=None
    rval,update=theano.scan(
                        _processSubgraph,
                        sequences=tensor.arange(lengths.shape[0]), 
                        )
    rval=discountModel(options['discount_alpha'], subgraph_lens)[:,None]*rval
    embx=rval.max(axis=0)
    
    score=tensor.dot(embx,tparams['w'])
    
    return xs, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, score

def discountModel(alpha,length):
    """
    discount
    """
    return tensor.exp(alpha*length*(-1))