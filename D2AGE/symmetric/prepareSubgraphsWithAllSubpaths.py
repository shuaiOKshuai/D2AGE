#encoding=utf-8
'''
@author: Liu Zemin
Functions and Application : 
generate DAGs with subpaths
'''

import numpy
import random
import dataProcessTools
import ConfigParser
import string, os, sys
import time
import math

SEED = 123
random.seed(SEED)

cf = ConfigParser.SafeConfigParser()
cf.read("/usr/pythonParamsConfig")
    
rootdir=cf.get("param", "root_dir") 
datasetName=cf.get("param", "dataset_name") 
relationName=cf.get("param", "class_name") 
subgraphNum=cf.getint("param", "subgraphNum")
DAGSaveFile=cf.get("param", "subgraphSaveFile") 
subpaths_file=cf.get("param", "subpaths_file")
maxlen_subpaths=cf.getint("param", "maxlen_subpaths") 
proportion=cf.getfloat("param", "proportion")
upperLimit=cf.getint("param", "upperLimit") 

def getAlltuples(rootdir, datasetName, relationName):
    """
        get all tuples from training data
    """
    folder=rootdir+'/'+datasetName+'.splits/'
    tuples=set()
    folder_train10=folder+'train.10/'
    for i in range(1,11):
        path=folder_train10+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 100
    folder_train100=folder+'train.100/'
    for i in range(1,11):
        path=folder_train100+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 1000
    folder_train1000=folder+'train.1000/'
    for i in range(1,11):
        path=folder_train1000+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # test data
    folder_test=folder+'test/'
    for i in range(1,11):
        path=folder_test+'test_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                for j in range(1,len(tmp)):
                    tuples.add(tmp[0]+'-'+tmp[j])
                    tuples.add(tmp[j]+'-'+tmp[0])
        f.close()
        f=None
    return tuples

def getAlltuplesForSingleDirection(rootdir, datasetName, relationName):
    """
       get all tuples for asymmetric relation
    """
    folder=rootdir+'/'+datasetName+'.splits/'
    tuples=set()
    folder_train10=folder+'train.10/'
    for i in range(1,11):
        path=folder_train10+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[0]+'-'+tmp[2])
        f.close()
        f=None
    # training data 100
    folder_train100=folder+'train.100/'
    for i in range(1,11):
        path=folder_train100+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[0]+'-'+tmp[2])
        f.close()
        f=None
    # training data 1000
    folder_train1000=folder+'train.1000/'
    for i in range(1,11):
        path=folder_train1000+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
#                 tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
#                 tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # test data
    folder_test=folder+'test/'
    for i in range(1,11):
        path=folder_test+'test_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                for j in range(1,len(tmp)):
                    tuples.add(tmp[0]+'-'+tmp[j])
#                     tuples.add(tmp[j]+'-'+tmp[0])
        f.close()
        f=None
    return tuples


def generateSubgraphsByAllSubpathsDirectlyAndSave(tuples, subpathsMap, subgraphNum, proportion, DAGSaveFile, upperLimit):
    """
        generate DAGs by all subpaths, then save to file
    """
    output = open(DAGSaveFile, 'w') 
    for tuple in tuples: 
        arr=tuple.strip().split('-') 
        start=int(arr[0]) 
        end=int(arr[1]) 
        if tuple not in subpathsMap: 
            continue
        subpaths=subpathsMap[tuple] 
        indexes=range(len(subpaths)) 
        number=0 
        if subgraphNum>0: 
            number=subgraphNum
        else: 
            number=int(math.ceil(len(subpaths)*proportion))
            if upperLimit>0: 
                number=min(number, upperLimit) 
        for i in range(number): 
            map={} 
            mapCheck={} 
            random.shuffle(indexes) 
            for j in indexes:
                subpath=subpaths[j] 
                for x in range(len(subpath)-1): 
                    if subpath[x] in map: 
                        if subpath[x+1] not in mapCheck[subpath[x]]: 
                            map[subpath[x]].append(subpath[x+1]) 
                            mapCheck[subpath[x]].add(subpath[x+1]) 
                    else: 
                        map[subpath[x]]=[subpath[x+1]]
                        mapCheck[subpath[x]]=set([subpath[x+1]])
            dependency, sequence, nodesLevel=dataProcessTools.subgraphToOrderedSequence(map, start, end)
            str=bytes(start)+'-'+bytes(end)+'#'
            for depend in dependency: 
                str+=bytes(depend[0])+'-'+bytes(depend[1])+'\t'
            str+='#' 
            for id in sequence:
                str+=bytes(id)+'\t'
            str+='#'
            for id in sequence:
                str+=bytes(id)+'-'+bytes(nodesLevel[id])+'\t'
            str+='\n'
            output.write(str)
            output.flush()
    output.close()
    output=None

                
if __name__=='__main__':
    print 'Read all tuples from files..........'
    start_time = time.time() 
    print 'This time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    tuples=getAlltuples(rootdir, datasetName, relationName) # symmetric
#     tuples=getAlltuplesForSingleDirection(rootdir, datasetName, relationName) # asymmetric
     
    print '-------------------------------------------------------------------------------'
    print 'Read all subpaths from files..........'
    start_time = time.time() 
    print 'This time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    subpathsMap=dataProcessTools.loadAllSubPaths(subpaths_file, maxlen_subpaths)
     
    print '-------------------------------------------------------------------------------'
    print 'Generate subgraphs and save them to file..........'
    start_time = time.time() 
    print 'This time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    
    generateSubgraphsByAllSubpathsDirectlyAndSave(tuples, subpathsMap, subgraphNum, proportion, DAGSaveFile, upperLimit)
    
    print '-------------------------------------------------------------------------------'
    print 'Finished!!!'
    start_time = time.time() 
    print 'End time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))

