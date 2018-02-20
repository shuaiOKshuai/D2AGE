#encoding=utf-8
'''
@author: Liu Zemin
Functions and Application : 
Training Model
'''

import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse
import directedGraphLSTMModel

def proxEmbedBySubgraphModel(options, tparams):
    """
    """
    trainingPairs=tensor.tensor3('trainingPairs',dtype='int64') # 3D tensor,shape=#(triples)*4*2
    xs=tensor.matrix('xs', dtype='int64') # shape=nsamples*maxlen
    masks=tensor.tensor3('masks', dtype=theano.config.floatX)  # @UndefinedVariable # shape=nsamples*maxlen*maxlen
    subgraph_lens=tensor.vector('subgraph_lens', dtype='int64') # shape=nsamples*0
    lengths=tensor.vector('lengths',dtype='int64') # shape=#(xs) * 0
    wordsEmbeddings=tensor.matrix('wordsEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable # shape=#(words) * wordsDimension
    
    buffer_tensor=tensor.tensor3('buffer_tensor', dtype=theano.config.floatX)  # @UndefinedVariable # shape=maxlen*maxlen*dimension
    nodesLens=tensor.matrix('nodesLens', dtype='int64') # shape=nsamples*maxlen
    
    def _processTuple(index , lossSum):
        tuple=trainingPairs[index] 
        
        def _processSubgraph(i):
            length=lengths[i]
            x=xs[i,:length] 
            mask=masks[i,:length,:length] 
            nodesLen=nodesLens[i,:length] 
            emb=directedGraphLSTMModel.directedGraphLSTMModel(options, tparams, x, mask, wordsEmbeddings, buffer_tensor, nodesLen) 
            return emb 
        
        def iftFunc(): 
            embx=tensor.zeros(options['dimension'],).astype(theano.config.floatX)  # @UndefinedVariable 
            return embx
        
        def iffFunc(start, end):
            embx=None
            rval,update=theano.scan(
                                _processSubgraph,
                                sequences=tensor.arange(start,end), 
                                )
            subgraph_len=subgraph_lens[start:end] 
            
            rval=discountModel(options['discount_alpha'], subgraph_len)[:,None]*rval
            embx=rval.max(axis=0)
            
            return embx
        
        start=tuple[0][0] 
        end=tuple[0][1] 
        emb1=None 
        emb1=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) 
        
        start=tuple[2][0] 
        end=tuple[2][1]
        emb2=None 
        emb2=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) 
        
        loss=0
        param=options['objective_function_param'] 
        if options['objective_function_method']=='sigmoid': 
            loss=-tensor.log(tensor.nnet.sigmoid(param*(tensor.dot(emb1,tparams['w'])-tensor.dot(emb2,tparams['w'])))) # sigmoid
        else: # hinge-loss
            value=param + tensor.dot(emb2,tparams['w']) - tensor.dot(emb1,tparams['w'])
            loss=value*(value>0)
        
        return loss+lossSum
    
    rval, update=theano.scan(
                                 _processTuple,
                                 sequences=tensor.arange(trainingPairs.shape[0]), 
                                 outputs_info=tensor.constant(0., dtype=theano.config.floatX), # @UndefinedVariable
                                 )
    
    cost=rval[-1]
    cost+=options['decay']*(tparams['Wi'] ** 2).sum()
    cost+=options['decay']*(tparams['Wf'] ** 2).sum()
    cost+=options['decay']*(tparams['Wo'] ** 2).sum()
    cost+=options['decay']*(tparams['Wc'] ** 2).sum()
    cost+=options['decay']*(tparams['Ui'] ** 2).sum()
    cost+=options['decay']*(tparams['Uf'] ** 2).sum()
    cost+=options['decay']*(tparams['Uo'] ** 2).sum()
    cost+=options['decay']*(tparams['Uc'] ** 2).sum()
    cost+=options['decay']*(tparams['bi'] ** 2).sum()
    cost+=options['decay']*(tparams['bf'] ** 2).sum()
    cost+=options['decay']*(tparams['bo'] ** 2).sum()
    cost+=options['decay']*(tparams['bc'] ** 2).sum()
    
    return trainingPairs, xs, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, cost


def discountModel(alpha,length):
    return tensor.exp(alpha*length*(-1))