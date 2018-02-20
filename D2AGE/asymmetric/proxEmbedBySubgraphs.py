#encoding=utf-8
'''
@author: Liu Zemin
Functions and Application : 
training 
'''

import numpy
import theano
from theano import tensor
from theano import config
from collections import OrderedDict
import dataProcessTools
import time
import proxEmbedBySubgraphModel
import gc
import six.moves.cPickle as pickle  # @UnresolvedImport

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable

def gradientDescentGroup(learning_rate,tparams,grads,trainingPairs, sequences, masks, lengths, wordsEmbeddings, cost):
    """
    """
    update=[(shared,shared-learning_rate*g) for g,shared in zip(grads,tparams.values())]
    func=theano.function([trainingPairs, sequences, masks, lengths, wordsEmbeddings],cost,updates=update,on_unused_input='ignore',mode='FAST_RUN')
    return func

def adadelta(lr, tparams, grads, trainingPairs, sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, cost):
    """
    An adaptive learning rate optimizer
    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([trainingPairs, sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens], cost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) 
             for ru2, ud in zip(running_up2, updir)] 
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] 
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def ortho_weight(ndim):
    """
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)  # @UndefinedVariable

def init_params_weight(row,column):
    """
    """
    W = numpy.random.rand(row, column) 
    W = W*2.0-1.0 
    return W.astype(theano.config.floatX)  # @UndefinedVariable

def init_sharedVariables(options):
    """
    """
    print 'init shared Variables......'
    params = OrderedDict()
    Wi=init_params_weight(options['dimension'],options['word_dimension'])
    Wf=init_params_weight(options['dimension'],options['word_dimension'])
    Wo=init_params_weight(options['dimension'],options['word_dimension'])
    Wc=init_params_weight(options['dimension'],options['word_dimension'])
    
    Ui=ortho_weight(options['dimension'])
    Uf=ortho_weight(options['dimension'])
    Uo=ortho_weight(options['dimension'])
    Uc=ortho_weight(options['dimension'])
    
    bi=numpy.zeros((options['dimension'],)).astype(config.floatX)  # @UndefinedVariable
    bf=numpy.zeros((options['dimension'],)).astype(config.floatX)  # @UndefinedVariable
    bo=numpy.zeros((options['dimension'],)).astype(config.floatX)  # @UndefinedVariable
    bc=numpy.zeros((options['dimension'],)).astype(config.floatX)  # @UndefinedVariable
    
    w = numpy.random.rand(options['dimension'], ).astype(config.floatX)  # @UndefinedVariable # 将w初始化为(0,1)之间的随机数
    
    params['Wi']=Wi
    params['Wf']=Wf
    params['Wo']=Wo
    params['Wc']=Wc
    params['Ui']=Ui
    params['Uf']=Uf
    params['Uo']=Uo
    params['Uc']=Uc
    params['bi']=bi
    params['bf']=bf
    params['bo']=bo
    params['bc']=bc
    
    params['w']=w
    
    return params
    
    
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

main_dir='D:/dataset/test/'
def proxEmbedBySubgraphs(
                     trainingDataFile=main_dir+'train_classmate_1', 
                     wordsEmbeddings_data=None, 
                     wordsEmbeddings_path=main_dir+'facebook/nodesFeatures', 
                      subpaths_map=None, 
                     subpaths_file=main_dir+'facebook/subpathsSaveFile',
                     subgraphSaveFile='', 
                     maxlen_subpaths=1000, 
                     wordsSize=1000000, 
                     
                     maxlen=100,  # Sequence longer then this get ignored 
                     batch_size=1, 
                     is_shuffle_for_batch=False, 
                     dispFreq=5, 
                     saveFreq=5, 
                     saveto=main_dir+'facebook/path2vec-modelParams.npz',
                     
                     lrate=0.0001, 
                     word_dimension=22, 
                     dimension=64, 
                     discount_alpha=0.3, 
                     discount_beta=0.3, 
                     h_output_method='max-pooling', 
                     objective_function_method='hinge-loss', 
                     objective_function_param=0, 
                     max_epochs=10, 
                     
                     decay=0.01, 
                         ):
    model_options = locals().copy()
    
    if wordsEmbeddings_data is None: 
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings_data,word_dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            exit(0) 
    trainingData,trainingPairs_data=dataProcessTools.getTrainingData(trainingDataFile)
    allBatches=dataProcessTools.get_minibatches_idx(len(trainingData), batch_size, is_shuffle_for_batch)
    
    subgraphs=dataProcessTools.readAllSubgraphDependencyAndSequencesWithLengths(subgraphSaveFile)
    
    params=init_sharedVariables(model_options) 
    tparams=init_tparams(params) 
    print 'Generate models ......'
    
    trainingPairs, sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, cost=proxEmbedBySubgraphModel.proxEmbedBySubgraphModel(model_options, tparams)
    
    print 'Generate gradients ......'
    grads=tensor.grad(cost,wrt=list(tparams.values()))
    print 'Using Adadelta to generate functions ......'
    this_time = time.time() 
    print 'Start to compile and optimize, time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(this_time))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update=adadelta(lr, tparams, grads, trainingPairs, sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, cost)
    
    print 'Start training models ......'
    best_p = None 
    history_cost=[] 
    
    start_time = time.time() 
    print 'start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    uidx=0 
    for eidx in range(max_epochs):
        for _, batch in allBatches: 
            uidx += 1
            trainingDataForBatch=[trainingData[i] for i in batch] 
            trainingPairsForBatch=[trainingPairs_data[i] for i in batch] 
            tuples3DMatrix_data, x_data, mask_data, lens_data, subgraph_lens_data, buffer_tensor_data, nodesLens_data=dataProcessTools.generateSequenceAndMasksForSingleSequenceWithLengthAsymmetric(trainingDataForBatch, trainingPairsForBatch, subgraphs, dimension)
            cost=f_grad_shared(tuples3DMatrix_data, x_data, mask_data, lens_data, subgraph_lens_data, wordsEmbeddings_data, buffer_tensor_data, nodesLens_data)
            f_update(lrate)
            
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                return 
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch =', eidx, ',  Update =', uidx, ',  Cost =', cost
                this_time = time.time() 
                print 'Time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(this_time))
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                if best_p is not None: 
                    params = best_p
                else: 
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_cost, **params)
                pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print('Done')
        gc.collect()
        
    end_time = time.time() 
    print 'end time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
    print 'Training finished! Cost time == ', end_time-start_time,' s'
    
    
    
    
    
