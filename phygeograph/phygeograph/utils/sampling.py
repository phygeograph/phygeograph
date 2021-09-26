from sklearn.model_selection import train_test_split
import numpy as np

def selectSites(datain,idStr="id",stratified="stratified_flag"):
  r""" Function for index selection of site-based independent test.
      Args:
          datain (Torch): Input data.
          idstr (String):  Id column name.
          stratified (String): Stratified factor name.
  """
  sitesDF = datain.drop_duplicates(idStr).copy()
  sgrp = sitesDF[stratified].value_counts()
  sitesDF['stratified_flag_cnt'] = sgrp.loc[sitesDF[stratified]].values
  pos1_index = np.where(sitesDF['stratified_flag_cnt'] < 5)[0]
  posT_index = np.where(sitesDF['stratified_flag_cnt'] >= 5)[0]
  np.random.seed()
  trainsiteIndex, testsiteIndex = train_test_split(posT_index, stratify=sitesDF.iloc[posT_index][stratified],
                                                   test_size=0.15)
  selsites = sitesDF.iloc[testsiteIndex][idStr]
  trainsitesIndex = np.where(~datain[idStr].isin(selsites))[0]
  trainsitesIndex=np.concatenate((trainsitesIndex,pos1_index))
  indTestsitesIndex = np.where(datain[idStr].isin(selsites))[0]
  return trainsitesIndex,indTestsitesIndex 
