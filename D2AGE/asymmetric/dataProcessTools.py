#encoding=utf-8
'''
data process tools
'''

import numpy
import theano

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def getTrainingData(trainingDataFile):
    '''
    get training data from file
    '''
    data=[] 
    pairs=[]
    with open(trainingDataFile) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            arr=[]
            arr.append(tmp[0]+'-'+tmp[1])
            arr.append(tmp[1]+'-'+tmp[0])
            arr.append(tmp[0]+'-'+tmp[2])
            arr.append(tmp[2]+'-'+tmp[0])
            pairs.append(arr)
            tmp=[int(x) for x in tmp] 
            data.append(tmp)
            
    return data,pairs

def getWordsEmbeddings(wordsEmbeddings_path):
    """
    get word embeddings
    """
    size=0
    dimension=0
    wemb=[]
    with open(wordsEmbeddings_path) as f:
        for l in f:
            arr=l.strip().split()
            if len(arr)==2: 
                size=int(arr[0])
                dimension=int(arr[1])
                
                wemb=numpy.zeros((size,dimension)).astype(theano.config.floatX)  # @UndefinedVariable 
                continue
            id=int(arr[0]) 
            for i in range(0,dimension):
                wemb[id][i]=float(arr[i+1])
    return wemb,dimension,size

def loadAllSubPaths(subpaths_file,maxlen=1000):
    """
    load all subpaths from file
    """
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] 
            sentence=[int(y) for y in splitByTab[2].split()[:]] 
            if len(sentence)>maxlen: 
                continue
            if key in map: 
                map[key].append(sentence)
            else: 
                tmp=[]
                tmp.append(sentence)
                map[key]=tmp
    return map

    
def prepareDataForTestForSubgraphSingleSequenceWithLengthsAsymmetric(query,candidate,subgraphs_map,dimension):
    """
    prepare data for test
    """
    key1=bytes(query)+'-'+bytes(candidate)
    if key1 not in subgraphs_map : 
        return None,None,None,None,None,None
    subgraphs=[] 
    if key1 in subgraphs_map:
        subgraphs.append(subgraphs_map[key1]) 
    maxlen=0 
    nsamples=0 
    for value in subgraphs: 
        for sequence in value[1]:
            nsamples+=1 
            if maxlen<len(sequence):
                maxlen=len(sequence)
    sequences=numpy.zeros((nsamples, maxlen)).astype('int64') 
    mask=numpy.zeros((nsamples, maxlen, maxlen)).astype(theano.config.floatX)  # @UndefinedVariable 
    lens=numpy.zeros((nsamples, )).astype('int64') # shape=nsamples*0
    subgraph_lens=numpy.zeros((nsamples,)).astype('int64') 
    nodesLens=numpy.zeros((nsamples,maxlen)).astype('int64') 
    current_index=0 
    for value in subgraphs:
        for i in range(len(value[1])):
            map={} 
            seq=value[1][i] 
            subgraph_len=value[2][i] 
            for j in range(len(seq)):
                sequences[current_index][j]=seq[j] 
                nodesLens[current_index][j]=subgraph_len[seq[j]]
                map[seq[j]]=j
            depend=value[0][i] 
            for dep in depend: 
                mask[current_index][map[dep[1]]][map[dep[0]]]=1.
            lens[current_index]=len(seq) 
            subgraph_lens[current_index]=subgraph_len[seq[-1]] 
            current_index+=1 
    for i in range(nsamples): 
        for j in range(maxlen):
            if mask[i][j].sum()==0: 
                mask[i][j][j]=1. 
    
    buffer_tensor=numpy.zeros([maxlen, maxlen, dimension])
    for i in range(maxlen):
        for j in range(dimension):
            buffer_tensor[i][i][j]=1.
            
    return sequences,mask,lens,subgraph_lens,buffer_tensor,nodesLens

    
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def pathsToSubgraph(paths):
    """
    """
    subgraph={}
    for path in paths: 
        for i in range(len(path)-1): 
            if path[i] in subgraph: 
                subgraph[path[i]].append(path[i+1]) 
            else: 
                subgraph[path[i]]=[path[i+1]]
    return subgraph
    

def subgraphToOrderedSequence(edges, start, end):
    """
    set DAG to a topology ordered sequence
    """
    nodesLevel={}
    nodesSeq={} 
    for key,values in edges.items():
        if key not in nodesLevel:
            nodesLevel[key]=-1
    queue=[] 
    now=start 
    queue.append(now)
    nodesLevel[now]=0 
    nodesSeq[now]=len(nodesSeq) 
    results=[]
    endNodeLevel=-1 
    while len(queue)>0:
        now=queue.pop(0) 
        children=edges[now] 
        for node in children:
            if node==end: 
                results.append([now,node])
                if endNodeLevel==-1: 
                    endNodeLevel=nodesLevel[now]+1
            elif nodesLevel[node]==-1: 
                queue.append(node) 
                nodesLevel[node]=nodesLevel[now]+1 
                nodesSeq[node]=len(nodesSeq)
                results.append([now,node])
            elif nodesSeq[node]>nodesSeq[now]: 
                results.append([now,node])
    nodesSeq[end]=len(nodesSeq)
    items=nodesSeq.items()
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort() 
    sequence=[ backitems[i][1] for i in range(len(items))]  
    nodesLevel[end]=endNodeLevel
    return results, sequence, nodesLevel


def readAllSubgraphDependencyAndSequencesWithLengths(filepath):
    """
    read all DAGs from file
    """
    map={}
    with open(filepath) as f:
        for l in f:
            tmp=l.strip().split('#') 
            if len(tmp)<=0:
                continue
            depend=tmp[1].strip().split('\t')
            dependint=[]
            for edge in depend:
                arr=edge.strip().split('-')
                dependint.append([int(arr[0]),int(arr[1])])
            sequence=tmp[2].strip().split('\t')
            sequenceint=[int(x) for x in sequence]
            lenArr=tmp[3].strip().split('\t')
            lengths={}
            for l in lenArr:
                lArr=l.strip().split('-')
                lengths[int(lArr[0])]=int(lArr[1])
            if tmp[0] in map: 
                value=map[tmp[0]]
                value[0].append(dependint)
                value[1].append(sequenceint)
                value[2].append(lengths)
            else: 
                map[tmp[0]]=[[dependint],[sequenceint],[lengths]] 
    return map

def generateSequenceAndMasksForSingleSequenceWithLengthAsymmetric(tuples, tupleFourPairs, subgraphs, dimension):
    """
    generate data for training
    """
    maxlen=0 
    graphNum=0 
    for tuple in tupleFourPairs: 
        for pair in tuple: 
            if pair not in subgraphs:
                continue
            value=subgraphs[pair]
            sequences=value[1] 
            graphNum+=len(sequences) 
            for seq in sequences: 
                if len(seq)>maxlen:
                    maxlen=len(seq)
    tuples3DMatrix=numpy.zeros((len(tuples),4,2)).astype('int64') 
    x=numpy.zeros((graphNum,maxlen)).astype('int64') 
    mask=numpy.zeros((graphNum,maxlen,maxlen)).astype(theano.config.floatX)  # @UndefinedVariable 
    lens=numpy.zeros((graphNum,)).astype('int64') # shape=graphNum*0
    subgraph_lens=numpy.zeros((graphNum,)).astype('int64')
    nodesLens=numpy.zeros((graphNum,maxlen)).astype('int64') 
    current_index=0
    for i in range(len(tuples)): 
        tuple=tuples[i] 
        fourPairs=tupleFourPairs[i] 
        for j in range(len(fourPairs)):
            if fourPairs[j] not in subgraphs: 
                tuples3DMatrix[i][j][0]=current_index
                tuples3DMatrix[i][j][1]=current_index
                continue
            value=subgraphs[fourPairs[j]] 
            dependency=value[0] 
            sequences=value[1]
            lengths=value[2] 
            tuples3DMatrix[i][j][0]=current_index 
            for index in range(len(sequences)): 
                map={}
                seq=sequences[index] 
                length=lengths[index] 
                for s in range(len(seq)): 
                    x[current_index][s]=seq[s]
                    nodesLens[current_index][s]=length[seq[s]] 
                    map[seq[s]]=s 
                depend=dependency[index] 
                for d in range(len(depend)):
                    dep=depend[d] 
                    mask[current_index][map[dep[1]]][map[dep[0]]]=1. 
                lens[current_index]=len(seq)
                subgraph_lens[current_index]=length[seq[-1]] 
                current_index+=1 
            tuples3DMatrix[i][j][1]=current_index 
    
    count=0
    for i in range(len(tuples3DMatrix)):
        if tuples3DMatrix[i][0][0]!=tuples3DMatrix[i][0][1] and tuples3DMatrix[i][2][0]!=tuples3DMatrix[i][2][1]:
            count+=1
    tuples3DMatrix_new=numpy.zeros((count,4,2)).astype('int64')
    index=0
    for i in range(len(tuples3DMatrix)):
        if tuples3DMatrix[i][0][0]!=tuples3DMatrix[i][0][1] and tuples3DMatrix[i][2][0]!=tuples3DMatrix[i][2][1]:
            tuples3DMatrix_new[index]=tuples3DMatrix[i]
            index+=1
    tuples3DMatrix=tuples3DMatrix_new 
    
    for i in range(graphNum): 
        for j in range(maxlen):
            if mask[i][j].sum()==0: 
                mask[i][j][j]=1. 
    buffer_tensor=numpy.zeros([maxlen, maxlen, dimension])
    for i in range(maxlen):
        for j in range(dimension):
            buffer_tensor[i][i][j]=1.
            
    return tuples3DMatrix, x, mask, lens, subgraph_lens, buffer_tensor, nodesLens