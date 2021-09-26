import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import r2_score as r2
from .metrics import rmse
import numpy as np
import torch

def train(model,dataloader,device,optimizer,X,y,scy):
    r"""Train function for physics-aware geograph hybrid model
        Args:
            model (torch.nn.Module): Trained model. 
            dataloader (torch.utils.data.DataLoader): Data loader for mini batch training.
            device (torch.device): Torch object of cpu or gpu.
            optimizer (torch.optim): Optimizer such as torch.optim.Adam.   
            X (Tensor): Input data matrix.
            y (Tensor): Input matrix of the measured or labelled dependent variable.
            scy (preprocessing.StandardScaler): Scaler of the dependent variable for inverse transformation.
    """
    model.setpremode(False)
    model.train()
#    pbar = tqdm(total=int(train_index.sum()))
#    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    ypre_b,yobs_b=[],[]
    X.requires_grad=True
    for batch_size, n_id, adjs in tqdm(dataloader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if not isinstance(adjs,list):
            adjs=[adjs]
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out,res = model(X[n_id], adjs, X[n_id[:batch_size]])   # .reshape(-1)
        grd=y[n_id[:batch_size]]    #.reshape(-1)
        loss1= F.mse_loss(out.reshape(-1),grd.reshape(-1))
        loss2= torch.sum(torch.square(res).reshape(-1))
        loss=loss1+loss2
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        ypre_b.append(out.cpu().detach().numpy())
        yobs_b.append(grd.cpu().detach().numpy())
        # pbar.update(batch_size)
  #  pbar.close()
    ypre_b=np.concatenate(ypre_b)
    yobs_b=np.concatenate(yobs_b)
    ypre_b=scy.inverse_transform(ypre_b)
    yobs_b=scy.inverse_transform(yobs_b)
    ypre_b=np.exp(ypre_b)
    yobs_b=np.exp(yobs_b)
    ypre_b=np.array(ypre_b, dtype=np.float64)
    yobs_b = np.array(yobs_b, dtype=np.float64)
    train_r2=r2(yobs_b,ypre_b)
    train_rmse=rmse(yobs_b,ypre_b)
    loss = total_loss / X.shape[0]
    return loss, train_r2,train_rmse

@torch.no_grad()
def test(model, dataloader,device,X,y,scy,train_index,test_index,testout=False,indtest_index=None):
    r"""Test function for physics-aware geograph hybrid model
        Args:
            model (torch.nn.Module): Trained model. 
            dataloader (torch.utils.data.DataLoader): Data loader for mini batch training.
            device (torch.device): Torch object of cpu or gpu.
            X (Tensor): Input data matrix.
            y (Tensor): Input matrix of the measured or labelled dependent variable.
            scy (preprocessing.StandardScaler): Scaler of the dependent variable for inverse transformation.
            train_index (LongTensor): Index of the training samples.
            test_index (LongTensor): Index of the test samples.
            testout (bool，optional): Indicator used to determine whether to output the test data set. (default: :obj:`False`)
            indtest_index (LongTensor,optional): Index of the site-based independent test samples.  (default: :obj:`None`)
            
    """
#   with torch.no_grad():
    model.setpremode(True)
    model.eval()
    total_loss = 0
    ypre_b,yobs_b=[],[]
    for batch_size, n_id, adjs in tqdm(dataloader):
        if not isinstance(adjs,list):
            adjs=[adjs]
        adjs = [adj.to(device) for adj in adjs]
        out = model(X[n_id], adjs,X[n_id[:batch_size]]).reshape(-1)
        grd=y[n_id[:batch_size]].reshape(-1)
        loss = F.mse_loss(out,grd)
        total_loss += float(loss)
        ypre_b.append(out.cpu().detach().numpy())
        yobs_b.append(grd.cpu().detach().numpy())
    ypre=np.concatenate(ypre_b)
    yobs=np.concatenate(yobs_b)
    ypre=scy.inverse_transform(ypre)
    yobs=scy.inverse_transform(yobs)
    ypre=np.exp(ypre)
    yobs=np.exp(yobs)
    ypre = np.array(ypre, dtype=np.float64)
    yobs = np.array(yobs, dtype=np.float64)
    loss = total_loss / X.shape[0]
    train_r2=r2(yobs[train_index],ypre[train_index])
    train_rmse=rmse(yobs[train_index],ypre[train_index])
    test_r2=r2(yobs[test_index],ypre[test_index])
    test_rmse=rmse(yobs[test_index],ypre[test_index])
    if indtest_index is not None:
        indtest_r2 = r2(yobs[indtest_index], ypre[indtest_index])
        indtest_rmse = rmse(yobs[indtest_index], ypre[indtest_index])
    if not testout:
        if indtest_index is not None:
            return train_r2, train_rmse, test_r2, test_rmse,indtest_r2,indtest_rmse
        return train_r2, train_rmse,test_r2,test_rmse
    if indtest_index is not None:
            return train_r2, train_rmse, test_r2, test_rmse,indtest_r2,indtest_rmse, \
                   yobs[test_index],ypre[test_index],yobs[indtest_index],ypre[indtest_index]
    return train_r2, train_rmse, test_r2, test_rmse,yobs[test_index],ypre[test_index]


