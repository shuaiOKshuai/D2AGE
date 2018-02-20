#encoding=utf-8
'''
Created on 2016年8月9日

@author: Administrator

本文中主要是一些tools工具方法
'''

import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse

def mapSortByValueDESC(map,top):
    """
    将map按照降序进行排列，并返回top个key
    其中top<=len(map)
    已测试过。
    """
    if top>len(map): # 如果设置的top的数值大于map的长度，则将top进行修改为map的长度
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  # 反转
    backitems.sort(reverse=True) # reverse=True是降序
#     backitems.sort() # 升序
    e=[ backitems[i][1] for i in range(top)]  # 把key按照顺序返回
    return e


def mapSortByValueASC(map,top):
    """
    将map按照升序进行排列，并返回top个key
    其中top<=len(map)
    已测试过。
    """
    if top>len(map): # 如果设置的top的数值大于map的长度，则将top进行修改为map的长度
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  # 反转
#     backitems.sort(reverse=True) # reverse=True是降序
    backitems.sort() # 升序
    e=[ backitems[i][1] for i in range(top)]  # 把key按照顺序返回
    return e


def max_poolingForMatrix(x):
    """
        使用scan函数来实现max-pooling的计算
        其中，x是要计算max-pooling的matrix，这里是按照列来进行绝对值的max-pooling
        已测试过。
    """
    def _funcForRow(row,max_array):
        """
                对于每一行，均计算
        """
        def _funcForElement(element,max_value):
            """
                        对于每个元素
            """
#             return tensor.switch(tensor.gt(tensor.abs_(element), tensor.abs_(max_value)),  element,  max_value)
            return ifelse(tensor.gt(tensor.abs_(element), tensor.abs_(max_value)),  element,  max_value)
    
        r,u=theano.scan(
                    fn=_funcForElement,
                    sequences=[row,max_array],
                    )
        # 这里的r便是经过这一个row的处理后的max_array
        return r

    rval,update=theano.scan(
                        fn=_funcForRow,
                        sequences=x,
                        outputs_info=tensor.alloc(numpy.asarray(0., dtype=theano.config.floatX), # 建立一个内容为0，x.shape[0]*0 维度的矩阵 @UndefinedVariable
                                                           x.shape[1],
                                                           ),
                        )
    # 这里的rval的最后的那个，便是经过处理后的abs max
    return rval[-1]