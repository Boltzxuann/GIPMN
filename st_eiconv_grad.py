"""Torch modules for spatial-temporal-edge-informed-convolution.--PX"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

# pylint: disable=W0235


    

    
class ST_EIConvSPGRAD(nn.Module):
    r"""

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.

    edge_feats : int
        Edge feature size; i.e, the number of dimensions of e_ji^(l)

    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

    """
    def __init__(self,
                 in_feats,
                 out_feats,#每层输出隐向量的维数
                 edge_feats,#每层输入边特征向量的维数
                 activation=None,#输出层激活函数
                activation_edge=None,
                error_mode=False,
                out_on=False):
        super(ST_EIConvSPGRAD, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self.weight0e = nn.Parameter(th.Tensor(edge_feats, 64))
        self.weight0h = nn.Parameter(th.Tensor(in_feats, 64))
        
        self.transform_h = nn.Linear(in_features= edge_feats+in_feats, out_features = out_feats, bias=False) ###输出维加1，第1维是订正降水量
        self.Rainfall_in= nn.Linear(in_features= out_feats+edge_feats*2, out_features = out_feats*2, bias=False)
        self.Rainfall_out = nn.Linear(in_features= out_feats*2, out_features = 1, bias=False)
        self.reset_parameters()
        self._activation = activation
        self._activation_e = activation_edge
        self._error_mode=error_mode
        self._out_on = out_on

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight0e is not None:
            init.xavier_uniform_(self.weight0e)
            init.xavier_uniform_(self.weight0h)

    def forward(self, graph, h_feat, e_feat, rain0):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        e_feat : torch.Tensor
            It represents the edge feature of shape (N,D_e)

        Returns
        -------
        torch.Tensor
            The output feature

        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(h_feat)
            efeat_src, efeat_dst = expand_as_pair(e_feat)
            graph.srcdata['efeat_src']=efeat_src
            graph.dstdata['efeat_dst']=efeat_dst
            #def div_Es_Ed_x(edges):
            #    return {'Es_Ed_x': ( edges['xx'].data['w']*(edges['xx'].src['efeat_src']-edges['xx'].dst['efeat_dst'] ) ) } ##x梯度
            
            #def div_Es_Ed_y(edges):
            #    return {'Es_Ed_y': ( edges['yy'].data['w']*(edges['yy'].src['efeat_src']-edges['yy'].dst['efeat_dst'] ) ) } ##y梯度           
            def dif_Es_Ed(edges):
                return {'Es_Ed': ( edges.data['w'].view(-1,1)*(edges.src['efeat_src']-edges.dst['efeat_dst'] ) ) } ##x梯度和y梯度              
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            

            
            #e_feat0_trans = th.matmul(e_feat,self.weight0e)  ##transform the dimension of edge features0 and h_feat
            #e_feat0_norm =self._activation_e(e_feat0_trans) 
            #h_feat0_trans = th.matmul(feat_src,self.weight0h)
            #h_feat0_norm = self._activation_e(h_feat0_trans) 
            
            #graph.srcdata['e'] = e_feat_norm ##transformed edge weights
            #graph.srcdata['h'] = feat_src
            E_mul_H_EE = th.cat( (feat_src, e_feat),1 ) 
            E_mul_H_EE_trans = self.transform_h(E_mul_H_EE) ## transform the dimension of merged features
            if self._activation_e is not None:
                E_mul_H_EE_trans = self._activation_e(E_mul_H_EE_trans)
            graph.srcdata['e_mul_h'] = E_mul_H_EE_trans #2 different edge features are merged
            graph.update_all(fn.copy_u('e_mul_h','m'),fn.sum(msg='m',out='h_x'),etype='xx')
            graph.update_all(fn.copy_u('e_mul_h','m'),fn.sum(msg='m',out='h_y'),etype='yy')
            
            h = graph.dstdata['h_x']+graph.dstdata['h_y']
            if self._activation is not None:
                h = self._activation(h)
            if self._out_on:
                graph.apply_edges(dif_Es_Ed,etype='xx')
                graph.apply_edges(dif_Es_Ed,etype='yy')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_x'),etype='xx')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_y'),etype='yy')
                h_cat = th.cat((h,graph.dstdata['grad_x'],graph.dstdata['grad_y']),1)
                rain_h = self.Rainfall_in(h_cat)
                if self._activation_e is not None:
                    rain_h = self._activation_e(rain_h) 
                rainfall_modified = self.Rainfall_out(rain_h)##由h到降水量的映射
                
                if self._error_mode:
                    rainfall_modified = rain0.view(-1,1)  + rainfall_modified #

            else :
                rainfall_modified = None
            
            return rainfall_modified, h


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, e_feat1={_edge_feats}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)    
    
    
    
class ST_EIConvSPGRAD2(nn.Module):
    r"""

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.

    edge_feats : int
        Edge feature size; i.e, the number of dimensions of e_ji^(l)

    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

    """
    def __init__(self,
                 in_feats,
                 out_feats,#每层输出隐向量的维数
                 edge_feats,#每层输入边特征向量的维数
                 activation=None,#输出层激活函数
                activation_edge=None,
                error_mode=False,
                out_on=False):
        super(ST_EIConvSPGRAD2, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self.weight0e = nn.Parameter(th.Tensor(edge_feats, 64))
        self.weight0h = nn.Parameter(th.Tensor(in_feats, 64))
        
        self.transform_h = nn.Linear(in_features= edge_feats+in_feats, out_features = out_feats, bias=False) ###输出维加1，第1维是订正降水量
        self.Rainfall_in= nn.Linear(in_features= out_feats+edge_feats*4, out_features = out_feats*4, bias=False)
        self.Rainfall_out = nn.Linear(in_features= out_feats*4, out_features = 1, bias=False)
        self.reset_parameters()
        self._activation = activation
        self._activation_e = activation_edge
        self._error_mode=error_mode
        self._out_on = out_on

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight0e is not None:
            init.xavier_uniform_(self.weight0e)
            init.xavier_uniform_(self.weight0h)

    def forward(self, graph, h_feat, e_feat, rain0):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        e_feat : torch.Tensor
            It represents the edge feature of shape (N,D_e)

        Returns
        -------
        torch.Tensor
            The output feature

        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(h_feat)
            efeat_src, efeat_dst = expand_as_pair(e_feat)
            graph.srcdata['efeat_src']=efeat_src
            graph.dstdata['efeat_dst']=efeat_dst
     
            def dif_Es_Ed(edges):
                return {'Es_Ed': ( edges.data['w'].view(-1,1)*(edges.src['efeat_src']-edges.dst['efeat_dst'] ) ) } ##x梯度和y梯度              
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            

            
            #e_feat0_trans = th.matmul(e_feat,self.weight0e)  ##transform the dimension of edge features0 and h_feat
            #e_feat0_norm =self._activation_e(e_feat0_trans) 
            #h_feat0_trans = th.matmul(feat_src,self.weight0h)
            #h_feat0_norm = self._activation_e(h_feat0_trans) 
            
            #graph.srcdata['e'] = e_feat_norm ##transformed edge weights
            #graph.srcdata['h'] = feat_src
            E_mul_H_EE = th.cat( (feat_src, e_feat),1 ) 
            E_mul_H_EE_trans = self.transform_h(E_mul_H_EE) ## transform the dimension of merged features
            if self._activation_e is not None:
                E_mul_H_EE_trans = self._activation_e(E_mul_H_EE_trans)
            graph.srcdata['e_mul_h'] = E_mul_H_EE_trans #2 different edge features are merged
            graph.update_all(fn.copy_u('e_mul_h','m'),fn.sum(msg='m',out='h_x'),etype='xx')
            graph.update_all(fn.copy_u('e_mul_h','m'),fn.sum(msg='m',out='h_y'),etype='yy')
            
            h = graph.dstdata['h_x']+graph.dstdata['h_y']
            if self._activation is not None:
                h = self._activation(h)
            if self._out_on:
                graph.apply_edges(dif_Es_Ed,etype='xx')
                graph.apply_edges(dif_Es_Ed,etype='yy')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_x'),etype='xx')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_y'),etype='yy')
                def dif2x_Es_Ed(edges):
                    return {'Es_Ed2x': ( edges.data['w'].view(-1,1)*(edges.src['grad_x']-edges.dst['grad_x'] ) ) } ##x梯度和y梯度
                def dif2x_Es_Ed(edges):
                    return {'Es_Ed2y': ( edges.data['w'].view(-1,1)*(edges.src['grad_y']-edges.dst['grad_y'] ) ) } ##x梯度和y梯度
                grad_x = graph.dstdata['grad_x']
                grad_y = graph.dstdata['grad_y']
                gradx_src,gradx_dst=expand_as_pair(grad_x)
                grday_src,grady_dst=expand_as_pair(grad_y)
                graph.apply_edges(dif2x_Es_Ed,etype='xx')
                graph.apply_edges(dif2y_Es_Ed,etype='yy')
                graph.update_all(fn.copy_e('Es_Ed2x','m'),fn.sum(msg='m',out='grad_2x'),etype='xx')
                graph.update_all(fn.copy_e('Es_Ed2y','m'),fn.sum(msg='m',out='grad_2y'),etype='yy') 
                h_cat=th.cat((h,graph.dstdata['grad_x'],graph.dstdata['grad_y'],graph.dstdata['grad_2x'],
                       graph.dstdata['grad_2y']),1)
                rain_h = self.Rainfall_in(h_cat)
                if self._activation_e is not None:
                    rain_h = self._activation_e(rain_h) 
                rainfall_modified = self.Rainfall_out(rain_h)##由h到降水量的映射
                if self._error_mode:
                    rainfall_modified = rain0.view(-1,1)  + rainfall_modified #
                    
                return  rainfall_modified, h

            else :
                return h


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, e_feat1={_edge_feats}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)  
 
    
class ST_EIConvProp(nn.Module):
      
    
    def __init__(self,
                 in_feats,
                 out_feats,#每层输出隐向量的维数
                 edge_feats,#每层输入边特征向量的维数
                 activation=None,#输出层激活函数
                activation_edge=None,
                error_mode=False,
                out_on=False):
        super(ST_EIConvProp, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        
        self.transform_h = nn.Linear(in_features= in_feats, out_features = edge_feats, bias=False) 
        #self.transform_mix = nn.Linear(in_features= edge_feats, out_features = out_feats, bias=False) 
        self.Rainfall_in= nn.Linear(in_features= edge_feats, out_features = out_feats*4, bias=False)#+out_feats
        self.Rainfall_out = nn.Linear(in_features= out_feats*4, out_features = 1, bias=False)
        self.out =  nn.Linear(in_features= edge_feats, out_features = out_feats, bias=False)#
        self._activation = activation
        self._activation_e = activation_edge
        self._error_mode=error_mode
        self._out_on = out_on


    def forward(self, graph, h_feat, e_feat, rain0):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        e_feat : torch.Tensor
            It represents the edge feature of shape (N,D_e)

        Returns
        -------
        torch.Tensor
            The output feature

        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(h_feat)
            efeat_src, efeat_dst = expand_as_pair(e_feat)
            graph.srcdata['efeat_src']=efeat_src
            graph.dstdata['efeat_dst']=efeat_dst
            #def div_Es_Ed_x(edges):
            #    return {'Es_Ed_x': ( edges['xx'].data['w']*(edges['xx'].src['efeat_src']-edges['xx'].dst['efeat_dst'] ) ) } ##x梯度
            
            #def div_Es_Ed_y(edges):
            #    return {'Es_Ed_y': ( edges['yy'].data['w']*(edges['yy'].src['efeat_src']-edges['yy'].dst['efeat_dst'] ) ) } ##y梯度           
            def dif_Es_Ed(edges):
                return {'Es_Ed': ( edges.data['w'].view(-1,1)*(edges.src['efeat_src']-edges.dst['efeat_dst'] ) ) } ##x梯度和y梯度              
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            
            if self._out_on==False:
                W_h = self.transform_h(h_feat) 
                W_h_dot_e = W_h * e_feat  ##w(h)*e
                #w_h_e_mix = self.transform_mix(W_h_dot_e)
                if self._activation_e is not None:
                    W_h_dot_e = self._activation_e(W_h_dot_e)
                graph.srcdata['W_h_dot_e'] = W_h_dot_e #2 different edge features are merged
                graph.update_all(fn.copy_u('W_h_dot_e','m'),fn.sum(msg='m',out='h_x'),etype='xx')
                graph.update_all(fn.copy_u('W_h_dot_e','m'),fn.sum(msg='m',out='h_y'),etype='yy')

                e_feat_new = graph.dstdata['h_x']+graph.dstdata['h_y']
                if self._activation is not None:
                    e_feat_new = self._activation(e_feat_new)
                return e_feat_new
            
            else:
                
                graph.apply_edges(dif_Es_Ed,etype='xx')
                graph.apply_edges(dif_Es_Ed,etype='yy')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_x'),etype='xx')
                graph.update_all(fn.copy_e('Es_Ed','m'),fn.sum(msg='m',out='grad_y'),etype='yy')
                h_cat = th.cat((h_feat, e_feat,graph.dstdata['grad_x'],graph.dstdata['grad_y']),1)
                rain_h = self.Rainfall_in(h_cat)
                if self._activation_e is not None:
                    rain_h = self._activation_e(rain_h) 
                rainfall_modified = self.Rainfall_out(rain_h)##由h到降水量的映射
                
                if self._error_mode:
                    rainfall_modified = rain0.view(-1,1)  + rainfall_modified #
                h_out = self.out(h_cat)
            
                return rainfall_modified, h_out


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, e_feat1={_edge_feats}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)   