from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from typing import Tuple, Optional, Union
import torch.nn as nn
from torch_scatter import gather_csr, scatter, segment_csr

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


class PhyAirGCov(MessagePassing):
    """The PhyAirGCov operator from the `"Physics-aware deep graph learning for
    air quality assessment" to be published in PNAS`_ paper.
    This code refers to the code of GraphSAGE.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_c \mathbf{x}_i + \mathbf{W}_u \cdot
         \left(    \mathbf{v} \cdot \sum_{\mathit{j} \in \mathcal{N}_k(\mathit{i})}
                  \mathit{w}_{\mathit{d}_{\mathit{ij}}} \cdot (\mathbf{x}_j-\mathbf{x}_i) -
                  \mathbf{p} \cdot \sum_{\mathit{j} \in \mathcal{N}_k(\mathit{i})}
                  (\mathbf{x}_j-\mathbf{x}_i)\r)

    For the change of air pollution concentration from deposition, emission and chemical transformation etc.,
    a parameter, pC (pC in the code) is incorporated into  WcX.

    Args:
        in_channels (int or tuple): Size of each input sample.
              A tuple corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features will be :math:`\ell_2`-normalized (default: :obj:`False`)

        root_weight (bool, optional): If set to :obj:`False`, the layer will
              not add transformed root node features to the output. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        isInWeight (bool, optional): If set to :obj:`True`, spatial or 
            spatiotemporal weights will be used in convolution operations. 
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True,
                 isInWeight: bool = False, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(PhyAirGCov, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.vm = torch.nn.Parameter(torch.as_tensor(0.001,dtype=float), requires_grad=True)
        self.kd = torch.nn.Parameter(torch.as_tensor(0.001,dtype=float), requires_grad=True)
        self.pC = torch.nn.Parameter(torch.as_tensor(0.01, dtype=float), requires_grad=True)
        torch.nn.init.uniform_(self.vm)
        torch.nn.init.uniform_(self.kd)
        self.isInWeight = isInWeight
        self.inWeights = None
#        nn.init.xavier_normal(self.vm)
#        nn.init.xavier_normal(self.kd)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, e_id: Tensor, edge_weight: Tensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # Weighted sum
        if self.isInWeight is False and edge_weight is not None:
            nlabs = edge_index[1]
            unique_nodes, node_count = nlabs.unique(dim=0, return_counts=True)
            sum1 = torch.zeros_like(unique_nodes, dtype=torch.float).scatter_add_(0, nlabs, torch.ones_like(edge_weight,
                                                                                                            dtype=torch.float))
            res = torch.zeros_like(unique_nodes, dtype=torch.float).scatter_add_(0, nlabs, edge_weight)
            summ = res[nlabs]
            normedw = edge_weight / summ
            normedw = normedw * sum1[nlabs] # why sum1
            spread1 = self.propagate(edge_index, x=x, size=size, normedw=normedw)
            spread2 = self.propagate(edge_index, x=x, size=size, normedw=None)
            out = spread1 * self.vm - self.kd * spread2   -  self.pC
        elif self.isInWeight:
            if self.inWeights is None:
                self.inWeights = torch.nn.Parameter(torch.zeros_like(edge_weight, dtype=torch.float))
                torch.nn.init.uniform_(self.inWeights)
            x_j=x[0].index_select(-2,edge_index[0])
            spread1 = x_j*self.inWeights.view(self.inWeights.size(0), 1).expand(-1, x_j.size(1))
            nlabs = edge_index[1]
            _, node_count = nlabs.unique(dim=0, return_counts=True)
            spread2 = self.propagate(edge_index, x=x, size=size, normedw=None)
            dimsize=node_count.shape[0]
            spread1=scatter(spread1, nlabs, -2, dim_size=dimsize,reduce= "sum")
            out=spread1*self.vm - self.kd*spread2  - self.pC
        else:
            out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor, normedw: Tensor = None) -> Tensor:
        if normedw is not None:
            normedw = normedw.view(normedw.size(0), 1).expand(-1, x_j.size(1))
            out = x_j * normedw
            return out  # x_j
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
