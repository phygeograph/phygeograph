#Load the library

import pandas as pd
import numpy as np
import torch

from phygeograph.utils.traintest import  train, test
from phygeograph.model import PhyGeoGrapHPDE
from phygeograph.utils.sampling import selectSites
from phygeograph.model.wdatasampling import DataSamplingDSited
from phygeograph.model.wsampler import WNeighborSampler

import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import urllib
import tarfile

def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs)

#Download and extract the data
url = 'https://github.com/phygeograph/phygeographdata/raw/master/phyg_sampledata1.pkl.tar.gz'
tarfl='./phyg_sampledata1.tar.gz'
print("Downloading from "+url+' ... ...')
urllib.request.urlretrieve(url, tarfl)
target='.'
untar(tarfl,target)

#Read the pkl data and preprocess it
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
tarcols=[i for i in range(len(covs))]

#Sample the data and construct the network topology
trainsitesIndex=[i for i in range(datatar.shape[0])]
trainsitesIndex, indTestsitesIndex=selectSites(datatar)
x, edge_index,edge_dist, y, train_index, test_index = DataSamplingDSited(Xn[:,tarcols], yn, [0,1,2], 12,
                        trainsitesIndex ,datatar)
x = x[:, 1:]
edge_weight=1.0/(edge_dist+0.00001)
neighbors=[12,12,12,12]
train_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=train_index,
                               sizes=neighbors, batch_size=2048, shuffle=True,
                               num_workers=20 )
x_index = torch.LongTensor([i for i in range(Xn.shape[0])])
x_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=x_index,
                           sizes=neighbors, batch_size=2048, shuffle=False,
                           num_workers=20 )

#Define the model
gpu=2
if gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(gpu))
nout=1
resnodes = [1024,512, 320, 256, 128, 96, 64, 32, 16]
# 0: original; 1: concated ; 2: dense; 3: only gcn
gcnnhiddens = [128,64,32]
model = PhyGeoGrapHPDE(x.shape[1], gcnnhiddens, nout, len(neighbors), resnodes, weightedmean=True,gcnout=nout,paraout=5,
                                 nattlayer=4)
model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)
init_lr=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
best_indtest_r2 = -9999
best_test_r2=-9999
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, last_epoch=-1)
oldlr=newlr=init_lr

#Start the training
epoch=0
nepoch=200
trpath="/wkspace/pypackages/phygeographPub_data/test2"
while epoch< nepoch  :
    # adjust_lr(optimizer, epoch, init_lr)
    print('Conducting ',epoch, ' of ',nepoch,' for PM  ... ...')
    loss, train_r2, train_rmse = train(model, train_loader, device, optimizer, x, y,scy)
    try:
       train_r2, train_rmse, test_r2, test_rmse,indtest_r2,indtest_rmse= test(model, x_loader, device, x, y, scy,train_index,
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
                                 'indtest_r2':indtest_r2,'indtest_rmse':indtest_rmse},index=[epoch])
    aperformanceDf['epoch']=epoch
    if epoch==0:
        alltrainHist=aperformanceDf
    else:
        alltrainHist=alltrainHist.append(aperformanceDf)
    print('epoch:',epoch,'  ... ...',aperformanceDf)
    epoch=epoch+1

#Save the training results
tfl = trpath + '/trainHist.csv'
alltrainHist.to_csv(tfl, header=True, index_label="row")
del optimizer, x, edge_index, y, train_index, test_index, model, alltrainHist
gc.collect()


