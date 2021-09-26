import sys
sys.path.append(".")
sys.path.append("..")
from .kddwei import knnd_geograph
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def DataSamplingDSited(X, y, covsknn, nnear,indtrainindex,indata,stratifiedflag="stratified_flag"):
    r"""Sampling functions of training and testing with the weights based on spatial or spatiotemporal distances
        Args:
            X (Tensor): Input data matrix.
            y (int): Input matrix of the measured or labelled dependent variable.
            covsknn (int): The index of the features to be used to calculate spatial or spatiotemporal distances.
            nnear (int): The number (k) of nearest nodes for a target node.
            indtrainindex (int): The index vector of the data samples to be used in sampling splitting.
            indata (Tensor): The matrix of all the data.
            stratifiedflag (String): The name of the stratifying factor.
    """

    stdata = X[:, covsknn]
    knodes=torch.FloatTensor(stdata)
    edge_index,edge_dist = knnd_geograph(knodes, k=nnear, batch=None, loop=True)
    x=torch.FloatTensor(X)
    y= torch.FloatTensor(y)
    n=len(indtrainindex)
    sgrp = indata.iloc[indtrainindex][stratifiedflag].value_counts()
    indata['stratified_flag_cnt']=None
    index=indata.iloc[indtrainindex].index
    indata.loc[index,'stratified_flag_cnt'] = sgrp.loc[indata.loc[index,stratifiedflag]].values
    pos1_index = np.where(indata['stratified_flag_cnt'] < 5)[0]
    posT_index = np.where(indata['stratified_flag_cnt'] >= 5)[0]
    np.random.seed()  ## Very important!!!, otherwise, without this, get the same random numbers for each process!
    train_index, test_index = train_test_split(posT_index, stratify=indata.iloc[posT_index][stratifiedflag],
                                         test_size=0.15)
    train_index = np.concatenate([train_index, pos1_index])
    train_index=torch.LongTensor(train_index)
    test_index=torch.LongTensor(test_index)
    print(train_index.shape,test_index.shape)
    return x,edge_index,edge_dist,y,train_index,test_index
