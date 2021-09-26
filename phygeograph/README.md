# Library of Physics-Aware Geograph Hybrid Modeling (phygeograph)

The python library of physics-aware geograph hybrid modeling (phygeograph). 
Current version just supports the PyTorch package of deep geometric learning and 
will extend to the others in the future. This package is for the paper, 
`"Physics-Aware Deep Graph Learning for Air Quality Assessment"` 

Authors: Lianfa Li (phygeograph@gmail.com; lspatial@gmail.com)  
         Jinfeng Wang (wangjf@lreis.ac.cn)


## Major modules

**Model**

* PhyAirGCov: Function of physics-aware graph convolution for air pollutants or other similar geo-features. 
              with the application of the thermodynamic laws in the graph space.  
* knngeo: function to retrieve the nearest neighbors with spatial or spatiotemporal distances.  
* knnd_geograph: function of knn to support the output of spatial or spatiotemporal weights.
* PhyGeoGrapH: Physics-aware graph modules inclduing local graph convolutions to simulate 
             spreading of air pollutants and full residual deep layers with attention layers.
* PhyGeoGrapHPDE: Physics-aware graph modules subject to the physical invariance of 
             the continuity partial differential equation.             
* DataSamplingDSited : Using distance-weighted kdd to sample the data to get the network 
            topology and the corresponding sample data. 
* WNeighborSampler : Function of using distance-weighted kdd to obtain the mini-batch data  
                    to train and test geographic graph hybrid network.

**Helper util functions**
* train: Training function of physics-aware graph hybrid model.
* test: Tesing function of physics-aware graph hybrid model.
* selectSites: site sampling function for site-based independent test

## Installation
You can directly install it using the following command for the latest version:
```
  pip install phygeograph
```

## Note for installation and use 

**Runtime requirements**

phygeograph requires installation of PyTorch with support of  PyG (PyTorch Geometric). 
Also Pandas and Numpy should be installed. 

## Use case 
The package provides one example for use of physics-aware geograph hybrid method to predict PM2.5 in mainland China.
See the following example. 

## License

The phygeograph is provided under a MIT license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Test call

**Load the packages**
```python

import pandas as pd
import numpy as np
import torch
from phygeograph.model.wsampler import WNeighborSampler 
from phygeograph.utils.traintest import  train, test
from phygeograph.model import PhyGeoGrapHPDE
from phygeograph.utils.sampling import selectSites
from phygeograph.model.wdatasampling import DataSamplingDSited
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import urllib
import tarfile
```
**Helper function for site-based independent test and downloading 
```python
def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs)
```
**Download and extract the sample dataset**

You can download the sample data from https://github.com/phygeograph. 
```python
url = 'https://github.com/phygeograph/phygeographdata/raw/master/phyg_sampledata1.pkl.tar.gz'
tarfl='./phyg_sampledata1.tar.gz'
print("Downloading from "+url+' ... ...')
urllib.request.urlretrieve(url, tarfl)
target='.'
untar(tarfl,target)
```

**Read and preprocess the data** 
```python
targetFl=target+'/datasam1.pkl'
datatar=pd.read_pickle(targetFl)
print(datatar.columns,datatar.shape)
covs=['idate','lat', 'lon', 'latlon', 'DOY', 'dem', 'OVP10_TOTEXTTAU', 'OVP14_TOTEXTTAU',
       'TOTEXTTAU', 'glnaswind', 'maiacaod', 'o3', 'pblh', 'prs', 'rhu', 'tem',
       'win', 'GAE', 'NO2_BOT', 'NO_BOT', 'PM25_BOT', 'PM_BOT', 'OVP10_CO',
       'OVP10_GOCART_SO2_VMR', 'OVP10_NO', 'OVP10_NO2', 'OVP10_O3', 'BCSMASS',
       'DMSSMASS', 'DUSMASS25', 'HNO3SMASS', 'NISMASS25', 'OCSMASS', 'PM25',
       'SO2SMASS', 'SSSMASS25', 'sdist_roads', 'sdist_poi', 'parea10km',
       'rlen10km', 'wstag', 'wmix', 'CLOUD', 'MYD13C1.NDVI',
       'MYD13C1.EVI', 'MOD13C1.NDVI', 'MOD13C1.EVI', 'is_workday', 'OMI-NO2']
X = datatar[covs].values
scX = preprocessing.StandardScaler().fit(X)
Xn = scX.transform(X)
y = datatar['pm25_log'].values
y = y[:,None]
ypm25 = datatar['PM2.5_24h'].values
scy = preprocessing.StandardScaler().fit(y)
yn = scy.transform(y)
```

**Sampling**
```python
tarcols=[i for i in range(len(covs))]
trainsitesIndex=[i for i in range(datatar.shape[0])]
trainsitesIndex, indTestsitesIndex=selectSites(datatar)
x, edge_index,edge_dist, y, train_index, test_index = DataSamplingDSited(Xn[:,tarcols], yn, 
        [0,1,2], 12,trainsitesIndex ,datatar)
x = x[:, 1:]
edge_weight=1.0/(edge_dist+0.00001)
neighbors=[12,12,12,12]
train_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=train_index,
                               sizes=neighbors, batch_size=3048, shuffle=True,
                               num_workers=20 )
x_index = torch.LongTensor([i for i in range(Xn.shape[0])])
x_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=x_index,
                           sizes=neighbors, batch_size=3048, shuffle=False,
                           num_workers=20 )
```
**Definition of the model and parameters**
```python
gpu=2
if gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(gpu))
nout=1
resnodes = [512, 320, 256, 128, 96, 64, 32, 16]
# 0: original; 1: concated ; 2: dense; 3: only gcn
gcnnhiddens = [128,64,32]
model = PhyGeoGrapHPDE(x.shape[1], gcnnhiddens, nout, len(neighbors), resnodes, 
         weightedmean=True,gcnout=nout,nattlayer=1)
model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)
init_lr=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
best_indtest_r2 = -9999
best_test_r2=-9999
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
oldlr=newlr=init_lr
```

**Model training**
```python
epoch=0
nepoch=200
trpath="/wkspace/pypackages/phygeographPub_data/test"
while epoch< nepoch  :
    # adjust_lr(optimizer, epoch, init_lr)
    print('Conducting ',epoch, ' of ',nepoch,' for PM  ... ...')
    loss, train_r2, train_rmse = train(model, train_loader, device, optimizer, x, y,scy)
    try:
       train_r2, train_rmse, test_r2, test_rmse,indtest_r2,indtest_rmse=
                test(model, x_loader, device, x, y, scy,train_index,
                         test_index, False,indTestsitesIndex)
    except:
        print("Wrong loop for ecpoch "+str(epoch)+ ", continue ... ...")
        epoch=epoch+1
        continue
    if best_indtest_r2 < indtest_r2:
        best_indtest_r2 = indtest_r2
        best_indtest_rmse=indtest_rmse
    scheduler.step(loss)
    newlr= optimizer.param_groups[0]['lr']
    if newlr!=oldlr:
        print('Learning rate is {} from {} '.format(newlr, oldlr))
        oldlr=newlr
    aperformanceDf=pd.DataFrame({'train_r2':train_r2, 'train_rmse':train_rmse,
                                 'test_r2':test_r2, 'test_rmse': test_rmse,
                                 'indtest_r2':indtest_r2,'indtest_rmse':indtest_rmse},
                                 index=[epoch])
    aperformanceDf['epoch']=epoch
    if epoch==0:
        alltrainHist=aperformanceDf
    else:
        alltrainHist=alltrainHist.append(aperformanceDf)
    print('epoch:',epoch,'  ... ...',aperformanceDf)
    epoch=epoch+1
```

**Save the results**
```python
tfl = trpath + '/trainHist.csv'
alltrainHist.to_csv(tfl, header=True, index_label="row")
del optimizer, x, edge_index, y, train_index, test_index, model, alltrainHist
gc.collect()
```
## Collaboration

Welcome to contact Dr. Lianfa Li (Email: phygeograph@gmail.com).
