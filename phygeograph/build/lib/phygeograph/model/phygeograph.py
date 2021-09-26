import torch
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import os
os.path.abspath(__file__)
import sys
sys.path.append(".")
from .phygeoconv  import PhyAirGCov

class Clipper_vm(object):
    def __init__(self,lim=(-100000,100000),frequency=30):
        self.frequency=frequency
        self.low=lim[0]
        self.high=lim[1]
    def __call__(self, module):
        if hasattr(module,'vm'):
            val=module.vm.data
            val=val.clamp(self.low,self.high)

class Clipper_kd(object):
    def __init__(self,lim=(-100000,100000),frequency=30):
        self.frequency=frequency
        self.low=lim[0]
        self.high=lim[1]
    def __call__(self, module):
        if hasattr(module,'kd'):
            val=module.kd.data
            val=val.clamp(self.low,self.high)
        
class Clipper_edep_ratio(object):
    def __init__(self,lim=(-100000,100000),frequency=30):
        self.frequency=frequency
        self.low=lim[0]
        self.high=lim[1]
    def __call__(self, module):
        if hasattr(module,'edep_ratio'):
            val=module.edep_ratio.data
            val=val.clamp(self.low,self.high)

class PhyGeoGrapH(torch.nn.Module):
    r""" Physics-based Graph Hybrid Neural Network 
       Args:
           in_channels (int or tuple): Size of each input sample. A tuple
               corresponds to the sizes of source and target dimensionalities.
           hidden_channels (int): Number of nodes for each graph convolution layer.
           out_channels (int): Size of each output sample.
           num_layers (int): Number of graph layers.
           autolayersNo (int, optional): The number of hidden layers in full deep network.
           weightedmean (bool, optional): If set to :obj:`True`, the weights will be used in graph convolution operations.
               (default: :obj:`True`)
           gcnout (int, optional): The number of the graph convolutions.
           nattlayer (int, optional): The number of the attention layers.
           vm_lim (tuple of float, optional): the lower and upper limits for velocity variable. (default: :float:`(-100000,100000)`).
           kd_lim (tuple of float, optional): the lower and upper limits for difussion coefficient. (default: :float:`(-100000,100000)`).
           pC (tuple of float, optional): the lower and upper limits for difussion coefficient. (default: :float:`(0,1)`).
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=None, autolayersNo=None,weightedmean=False,gcnout=1,
                 nattlayer=None,vm_lim=(-100000.0,100000.0),kd_lim=(-100000.0,100000.0),pC=(-100000,100000)):
        super(PhyGeoGrapH, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convlinks = torch.nn.ModuleList()
        if isinstance(hidden_channels,int):
            nh=hidden_channels
            hidden_channels=[nh for i in range(num_layers) ]
        if num_layers is not None and num_layers==1:
            agcn=PhyAirGCov(in_channels, gcnout,isInWeight=False)
            agcn.apply(Clipper_vm(vm_lim))
            agcn.apply(Clipper_kd(kd_lim))
            agcn.apply(Clipper_edep_ratio(pC))
            self.convs.append(agcn)
        else:
            agcn = PhyAirGCov(in_channels, hidden_channels[0],isInWeight=False)
            agcn.apply(Clipper_vm(vm_lim))
            agcn.apply(Clipper_kd(kd_lim))
            agcn.apply(Clipper_edep_ratio(pC))
            self.convs.append(agcn)
            self.convlinks.append(torch.nn.Linear(hidden_channels[0], 1))
            for i in range(1,num_layers - 1):
                agcn=PhyAirGCov(hidden_channels[i - 1], hidden_channels[i], isInWeight=False)
                agcn.apply(Clipper_vm(vm_lim))
                agcn.apply(Clipper_kd(kd_lim))
                agcn.apply(Clipper_edep_ratio(pC))
                self.convs.append(agcn)
                self.convlinks.append(torch.nn.Linear(hidden_channels[i], 1))
            agcn=PhyAirGCov(hidden_channels[num_layers-2], gcnout,isInWeight=False)
            agcn.apply(Clipper_vm(vm_lim))
            agcn.apply(Clipper_kd(kd_lim))
            agcn.apply(Clipper_edep_ratio(pC))
            self.convs.append(agcn)
        self.autolayers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.weightedmean=weightedmean
        self.atts = torch.nn.ModuleList()
        self.attsbn = torch.nn.ModuleList()
        self.nattlayer=nattlayer
        nlink=len(self.convlinks)
        if autolayersNo is not None:
            if nattlayer is not None:
                for i in range(nattlayer):
                    self.atts.append(torch.nn.Linear(in_channels + gcnout +nlink , in_channels + gcnout + nlink))
                    self.attsbn.append(torch.nn.BatchNorm1d(in_channels + gcnout + nlink))
            self.autolayers.append(torch.nn.Linear(in_channels + gcnout + nlink, autolayersNo[0]))
            self.bn.append(torch.nn.BatchNorm1d(autolayersNo[0]))
            for i in range(1, len(autolayersNo)):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i - 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            for i in range(len(autolayersNo) - 2, -1, -1):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i + 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            self.lastLayer2 = torch.nn.Linear(autolayersNo[0], in_channels + gcnout + nlink)
            self.bn.append(torch.nn.BatchNorm1d(in_channels + gcnout + nlink))
            self.lastLayer = torch.nn.Linear(in_channels + gcnout + nlink, out_channels)
        self.autolayersNo = autolayersNo

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
        res = []
        xlinkout=[]
        for i, (edge_index, e_id, e_weight, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            if self.weightedmean:
                x = self.convs[i]((x, x_target), edge_index,e_id,e_weight)
            else:
                x = self.convs[i]((x, x_target), edge_index, e_id, None )
            if i != self.num_layers - 1:
                x = F.relu(x)
                xlink=x[:len(xnode)]
                xlinkout.append(self.convlinks[i](xlink))
               # x = F.dropout(x, p=0.2, training=self.training)
        xlinkt=torch.cat(xlinkout,1)
        if len(self.autolayers) > 0:
            xin = torch.cat((xnode, x,xlinkt), 1)
            #if self.nattlayer is not None:
            for i in range(len(self.atts)):
                    prob=self.atts[i](xin)
                    prob = F.softmax(prob,dim=1)
                    xin = torch.mul(xin,prob)+xin
                    xin = self.attsbn[i](xin)
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
            x = self.lastLayer(x)
        return x
