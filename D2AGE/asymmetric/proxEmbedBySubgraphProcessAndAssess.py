#encoding=utf-8
'''
Created on 2017年2月3日
@author: Liu Zemin
Functions and Application : 

'''

import numpy
import theano
from collections import OrderedDict
import proxEmbedBySubgraphProcessModelBatch
import dataProcessTools
import toolsFunction
import evaluateTools



def load_params(path, params):
    """
    load parameters from file
    """
    pp = numpy.load(path) 
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def get_proxEmbedBySubgraphModel(
                      
                   model_params_path='', 
                     word_dimension=0, 
                     dimension=0, 
                     discount_alpha=0.3, 
                     discount_beta=0.3, 
                     h_output_method='max-pooling', 
                      ):
    """
    the processing model
    """
    model_options = locals().copy()
    
    tparams = OrderedDict()
    tparams['Wi']=None
    tparams['Wf']=None
    tparams['Wo']=None
    tparams['Wc']=None
    tparams['Ui']=None
    tparams['Uf']=None
    tparams['Uo']=None
    tparams['Uc']=None
    tparams['bi']=None
    tparams['bf']=None
    tparams['bo']=None
    tparams['bc']=None
    tparams['w']=None
    tparams=load_params(model_params_path, tparams) 
    
    sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens, score=proxEmbedBySubgraphProcessModelBatch.proxEmbedBySubgraphProcessModel(model_options, tparams)
    func=theano.function([sequences, masks, lengths, subgraph_lens, wordsEmbeddings, buffer_tensor, nodesLens], score, on_unused_input='ignore') 
    return func 


def compute_proxEmbedBySubgraph(
                     wordsEmbeddings=None, 
                     wordsEmbeddings_path=None, 
                     word_dimension=0, 
                     dimension=0,
                     wordsSize=0, 
                     subpaths_map=None, 
                     subpaths_file=None,
                     subgraphs_file='', 
                     maxlen_subpaths=1000, 
                     maxlen=100,  # Sequence longer then this get ignored 
                     
                     test_data_file='', 
                     top_num=10, 
                     ideal_data_file='',
                     func=None, 
                   ):
    model_options = locals().copy()
    
    if wordsEmbeddings is None: 
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings,word_dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            exit(0) 

    subgraphs_map=dataProcessTools.readAllSubgraphDependencyAndSequencesWithLengths(subgraphs_file)
    
    line_count=0 
    test_map={} 
    print 'Compute MAP and nDCG for file ',test_data_file
    with open(test_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            query=int(arr[0]) 
            map={} 
            for i in range(1,len(arr)): 
                candidate=int(arr[i]) 
                sequences_data, mask_data, lens_data, subgraph_lens_data, buffer_tensor_data,nodesLens_data=dataProcessTools.prepareDataForTestForSubgraphSingleSequenceWithLengthsAsymmetric(query, candidate, subgraphs_map, dimension)
                if sequences_data is None and mask_data is None and lens_data is None: 
                    map[candidate]=-1000. 
                else: 
                    value=func(sequences_data, mask_data, lens_data, subgraph_lens_data, wordsEmbeddings, buffer_tensor_data, nodesLens_data) 
                    map[candidate]=value
            
            tops_in_line=toolsFunction.mapSortByValueDESC(map, top_num)
            test_map[line_count]=tops_in_line 
            line_count+=1 
                
    line_count=0 
    ideal_map={}
    with open(ideal_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            arr=[int(x) for x in arr] 
            ideal_map[line_count]=arr[1:] 
            line_count+=1 
    
    MAP=evaluateTools.get_MAP(top_num, ideal_map, test_map)
    MnDCG=evaluateTools.get_MnDCG(top_num, ideal_map, test_map)
    
    return MAP,MnDCG
    
    
    
    
    