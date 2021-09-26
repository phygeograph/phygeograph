from torch_geometric.nn import GCNConv, ChebConv
#from torch_geometric.nn import SAGEConv
import torch
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import os
os.path.abspath(__file__)
import sys
sys.path.append(".")
from .phygeograph import PhyGeoGrapH
from torch.autograd import grad

class PhyGeoGrapHPDE(torch.nn.Module):
    r"""Physics-based Graph Hybrid Neural Network with PDE residual
         from the `"Physics-aware deep graph learning for
         air quality assessment" to be published in PNAS`_ paper
       Args:
           in_channels (int or tuple): Size of each input sample. A tuple
               corresponds to the sizes of source and target dimensionalities.
           ngcnode (int or tuple): The number of features for each local graph convolution layer.
           out_channels (int): The number of output features.
           nnei (int): The number of local graph convolutions.
           autolayersNo (int, optional): The number of hidden layers in full deep network.
           weightedmean (bool, optional): If set to :obj:`True`, the weights will be used in graph convolution operations.
           gcnout (int, optional): The number of the output features of the last graph convolution layer.
           paraout (int, optional): The number of the coefficients for the parameters.  (default: :ints:`5`).
           nattlayer (int, optional): The number of attention layers.   (default: :ints:`4`).
           vm_lim (tuple of float, optional): The lower and upper limits for velocity variable. (default: :float:`(-100000,100000)`).
           kd_lim (tuple of float, optional): The lower and upper limits for difussion coefficient. (default: :float:`(-100000,100000)`).
           pC_lim (tuple of float, optional): The lower and upper limits for difussion coefficient. (default: :float:`(0.0,1.0)`).
    The residual is defined as
   .. math::
        \mathbf{e}_2=  \frac {\partial \tilde{\mathbf{C}}} {\partial \mathit{d}}\
        +\frac {\partial \tilde{\mathbf{C}}} {\partial \mathit{l}_x} \mathit{v}_{\mathit{l}_x} + \
        \frac {\partial \tilde{\mathbf{C}}} {\partial \mathit{l}_y} \mathit{v}_{\mathit{l}_y} - \mathit{pC}} - \
        (\frac {\partial^2 \tilde{\mathbf{C}}} {\partial^2 \mathit{l}_x} +(\frac {\partial \tilde{\mathbf{C}}} {\partial \mathit{l}_x})^2)\mathit{p}_{\mathit{l}_x}  - \
        (\frac {\partial^2 \tilde{\mathbf{C}}} {\partial^2 \mathit{l}_y} +(\frac {\partial \tilde{\mathbf{C}}} {\partial \mathit{l}_y})^2)\mathit{p}_{\mathit{l}_y}
   """
    def __init__(self, in_channels, ngcnode, out_channels, nnei, autolayersNo, weightedmean, gcnout,paraout=5,
                            nattlayer=4,vm_lim=(-100000.0,100000.0),kd_lim=(-100000.0,100000.0),pC_lim=(0.0,1.0)):
        super(PhyGeoGrapHPDE, self).__init__()
        self.autolayers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.atts = torch.nn.ModuleList()
        self.attsbn = torch.nn.ModuleList()
        self.vm_lim=vm_lim
        self.kd_lim=kd_lim
        self.gcnmodel = PhyGeoGrapH(in_channels, ngcnode, out_channels, nnei, autolayersNo, weightedmean=weightedmean, gcnout=gcnout,
                            nattlayer=nattlayer,vm_lim=vm_lim,kd_lim=kd_lim,pC=pC_lim)
        if autolayersNo is not None:
            if nattlayer is not None:
                for i in range(nattlayer):
                    self.atts.append(torch.nn.Linear(in_channels , in_channels))
                    self.attsbn.append(torch.nn.BatchNorm1d(in_channels))
            self.autolayers.append(torch.nn.Linear(in_channels, autolayersNo[0]))
            self.bn.append(torch.nn.BatchNorm1d(autolayersNo[0]))
            for i in range(1, len(autolayersNo)):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i - 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            for i in range(len(autolayersNo) - 2, -1, -1):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i + 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            self.lastLayer2 = torch.nn.Linear(autolayersNo[0], in_channels )
            self.bn.append(torch.nn.BatchNorm1d(in_channels))
            self.lastLayer = torch.nn.Linear(in_channels , paraout)
        self.autolayersNo = autolayersNo
        self.premode=False
        self.para=None
   
    def setpremode(self,premode=False):
        self.premode=premode

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, xnode):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        out = self.gcnmodel(x, adjs, xnode )
        if self.premode:
            return out
        all_g1 = grad(out, xnode, grad_outputs=torch.ones_like(out), create_graph=True,retain_graph=True)[0]
        all_g2 = grad(all_g1, xnode,grad_outputs=torch.ones_like(all_g1),retain_graph=True)[0]
        lat_g1 = all_g1[:,0]
        lat_g2 = all_g2[:,0]
        lon_g1 = all_g1[:,1]
        lon_g2 = all_g2[:,1]
        con_gt = all_g1[:,2]
        res = []
        if len(self.autolayers) > 0:
            xin = xnode
            #if self.nattlayer is not None:
            res.append(xin)
            x = F.relu(self.autolayers[0](xin))
            x = self.bn[0](x)
            for i in range(1, len(self.autolayers)):
                if i <= len(self.autolayersNo) - 1:
                    res.append(x)
                x = F.relu(self.autolayers[i](x))
                x = self.bn[i](x)
                if i >= len(self.autolayersNo):
                    x = x + res.pop()
            x = self.lastLayer2(x)
            x = self.bn[i + 1](F.relu(x))
            x = x + res.pop()
            self.para = self.lastLayer(x) 
        res=con_gt + self.para[:,0]*lat_g1 + self.para[:,1]*lon_g1 - self.para[:,2]*(lat_g2+lat_g1*lat_g1) - \
            self.para[:,3]*(lon_g2+lon_g1*lon_g1) - self.para[:,4]
        return out, res
