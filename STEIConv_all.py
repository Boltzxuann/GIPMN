import torch
import torch.nn as nn
import dgl.function as fn
from st_eiconv_grad import ST_EIConvSPGRAD  , ST_EIConvProp
from dgl.nn.pytorch import edge_softmax #, ST_EIConvSPGRAD  , ST_EIConvProp
import torch.nn.functional as F
from linear_layer import Linear


    
    

    

    

class STEIConvNetMScaleCompactGRAD(nn.Module): ##紧凑版，不同时间步共享模型
    def __init__(self,
                 g,
                 num_layers,
                 scale,
                 in_dim,
                 num_hidden,
                 out_dim,
                 edge_dim,
                 activation_o,
                 activation_edge,
                 error_mode=False):
        super(STEIConvNetMScaleCompactGRAD, self).__init__()
        self.g = g
        self.num_hidden=num_hidden
        self.scale=scale
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.eiconvOUT_layers = nn.ModuleList()
        self.eiconvIN_layers = nn.ModuleList()
        self.activation_o = activation_o
        self.activation_e = activation_edge
        # input projection (no residual)
        self.embedding = Linear(in_features= in_dim, out_features = num_hidden, bias=False)
        
        self.eiconvOUT_layers.append(ST_EIConvSPGRAD(num_hidden, num_hidden, edge_dim,activation=self.activation_o, activation_edge=self.activation_e, error_mode=error_mode, out_on = True ))#
        for s in range(scale-1):
            self.eiconvIN_layers.append(ST_EIConvSPGRAD(num_hidden, num_hidden, edge_dim,activation=self.activation_o, activation_edge=self.activation_e, error_mode=error_mode, out_on = False ))#        
                         
    def forward(self, inputs, e_feats, rain0):
        h = inputs
        #h[:]= 1 ##全1输入
        h = self.embedding(inputs)##inputs为0时刻的要素场
        for s in range(self.scale-1):
            __ , h = self.eiconvIN_layers[s](self.g,h,e_feats[:,:,0], rain0[:,0]) ##一个时间步内的空间特征提取

        rain_all, h = self.eiconvOUT_layers[0](self.g,h,e_feats[:,:,0], rain0[:,0]) ##前3h降水
        rain_all=rain_all.view(-1,1)
        for l in range(self.num_layers-1):
            for s in range(self.scale-1):
                __ , h = self.eiconvIN_layers[s](self.g,h,e_feats[:,:,l+1], rain0[:,l+1]) ##一个时间步内的空间特征提取
            rain_M, h = self.eiconvOUT_layers[0](self.g, h, e_feats[:,:,l+1],rain0[:,l+1])  
            rain_all=torch.cat((rain_all,rain_M.view(-1,1)),1)

        return  rain_all      
    

class STEIConvNetMScaleCompactPROP(nn.Module): ##紧凑版，不同时间步共享模型
    def __init__(self,
                 g,
                 num_layers,
                 scale,
                 in_dim,
                 num_hidden,
                 out_dim,
                 edge_dim,
                 activation_o,
                 activation_edge,
                 num_filters=1,
                 error_mode=False):
        super(STEIConvNetMScaleCompactPROP, self).__init__()
        self.g = g
        self.num_hidden=num_hidden
        self.scale=scale
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.eiconvOUT_layers = nn.ModuleList()
        self.eiconvIN_layers = nn.ModuleList()
        self.activation_o = activation_o
        self.activation_e = activation_edge
        # input projection (no residual)
        self.embedding = Linear(in_features= in_dim, out_features = num_hidden, bias=False)
        #self.embedding_e = Linear(in_features= edge_dim, out_features = num_hidden, bias=False)
        edge_dim_cat= edge_dim*num_filters
        self.num_filters=num_filters
        self.eiconvOUT_layers.append(ST_EIConvProp(num_hidden, num_hidden, edge_dim*(3+num_filters),activation=self.activation_o, activation_edge=self.activation_e, error_mode=error_mode, out_on = True ))#
        for s in range(scale-1):
            self.eiconvIN_layers.append(ST_EIConvProp(num_hidden, edge_dim_cat, edge_dim_cat,activation=self.activation_o, activation_edge=self.activation_e, error_mode=error_mode, out_on = False ))#        
                         
    def forward(self, inputs, e_feats, rain0):
        h = inputs
        #h[:]= 1 ##全1输入
        h = self.embedding(inputs)##inputs为0时刻的要素场
        #efeat_p = self.embedding_e(e_feats[:,:,0])
        efeat_expanded= e_feats[:,:,0].repeat(1, self.num_filters)
        efeat_p = self.eiconvIN_layers[0](self.g,h,efeat_expanded, rain0[:,0])
        for s in range(1,self.scale-1):
            efeat_p = self.eiconvIN_layers[s](self.g,h,efeat_p, rain0[:,0]) ##一个时间步内的空间特征提取

        rain_all, h = self.eiconvOUT_layers[0](self.g,efeat_p,e_feats[:,:,0], rain0[:,0]) ##最后一步h变成efeatp,efeat变成efeat[l]
        rain_all=rain_all.view(-1,1)
        for l in range(self.num_layers-1):
            #efeat_p = self.embedding_e(e_feats[:,:,l+1])
            efeat_expanded = e_feats[:,:,l+1].repeat(1, self.num_filters)
            efeat_p = self.eiconvIN_layers[0](self.g,h,efeat_expanded, rain0[:,l+1]) ##一个时间步内的空间特征提取            
            for s in range(1,self.scale-1):
                efeat_p = self.eiconvIN_layers[s](self.g,h,efeat_p, rain0[:,l+1]) ##一个时间步内的空间特征提取
            rain_M, h = self.eiconvOUT_layers[0](self.g, efeat_p, e_feats[:,:,l+1],rain0[:,l+1])  
            rain_all=torch.cat((rain_all,rain_M.view(-1,1)),1)

        return  rain_all         
    
    
class STEIConvNetMScaleCompactGRAD_PROP(nn.Module): ##紧凑版，不同时间步共享模型
    def __init__(self,
                 g,
                 num_layers,
                 scale,
                 in_dim,
                 num_hidden,
                 out_dim,
                 edge_dim,
                 activation_o,
                 activation_edge,
                 num_filters=1,
                 error_mode=False):
        super(STEIConvNetMScaleCompactGRAD_PROP, self).__init__()
        self.g = g
        self.num_hidden=num_hidden
        self.scale=scale
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.eiconvOUT_layers = nn.ModuleList()
        self.eiconvIN_layers = nn.ModuleList()
        self.PROP_layers = nn.ModuleList()
        self.activation_o = activation_o
        self.activation_e = activation_edge
        self.num_filters=num_filters
        # input projection (no residual)
        self.embedding = Linear(in_features= in_dim, out_features = num_hidden, bias=False)
        edge_dim_cat= edge_dim*num_filters
        self.eiconvOUT_layers.append(ST_EIConvProp(num_hidden, num_hidden,
                                                   edge_dim*(3+num_filters)+num_hidden,activation=self.activation_o,
                                                   activation_edge=self.activation_e, error_mode=error_mode, out_on = True ))#
        for s in range(scale-1):
            self.eiconvIN_layers.append(ST_EIConvSPGRAD(num_hidden, num_hidden,
                                                        edge_dim,activation=self.activation_o,
                                                        activation_edge=self.activation_e, error_mode=error_mode, out_on = False ))# 

        for s in range(scale-1):
            self.PROP_layers.append(ST_EIConvProp(num_hidden, edge_dim_cat, edge_dim_cat,activation=self.activation_o, activation_edge=self.activation_e, error_mode=error_mode, out_on = False ))#                              
    def forward(self, inputs, e_feats, rain0):
        h = inputs
        #h[:]= 1 ##全1输入
        h = self.embedding(inputs)##inputs为0时刻的要素场
        for s in range(self.scale-1):
            __ , h = self.eiconvIN_layers[s](self.g,h,e_feats[:,:,0], rain0[:,0]) ##一个时间步内的空间特征提取
        efeat_expanded= e_feats[:,:,0].repeat(1, self.num_filters)
        efeat_p = self.PROP_layers[0](self.g,h,efeat_expanded, rain0[:,0])
        for s in range(self.scale-1):
            efeat_p = self.PROP_layers[s](self.g,h,efeat_p, rain0[:,0]) ##一个时间步内的空间特征提取
        #efeat_out = torch.cat((e_feats[:,:,0],efeat_p),1) 
        h_out = torch.cat((h,efeat_p),1)
        rain_all, h = self.eiconvOUT_layers[0](self.g,h_out,e_feats[:,:,0], rain0[:,0]) ##前3h降水
        rain_all=rain_all.view(-1,1)
        
        
        for l in range(self.num_layers-1):
            for s in range(self.scale-1):
                __ , h = self.eiconvIN_layers[s](self.g,h,e_feats[:,:,l+1], rain0[:,l+1]) ##一个时间步内的空间特征提取
                
            efeat_expanded = e_feats[:,:,l+1].repeat(1, self.num_filters)
            efeat_p = self.PROP_layers[0](self.g,h,efeat_expanded, rain0[:,l+1]) ##一个时间步内的空间特征提取            
                
            for s in range(1,self.scale-1):
                efeat_p = self.PROP_layers[s](self.g,h,efeat_p, rain0[:,l+1]) ##一个时间步内的空间特征提取 
            #efeat_out = torch.cat((e_feats[:,:,l+1],efeat_p),1)
            h_out = torch.cat((h,efeat_p),1)
            rain_M, h = self.eiconvOUT_layers[0](self.g, h_out, e_feats[:,:,l+1],rain0[:,l+1]) 
            #rain_M, h = self.eiconvOUT_layers[0](self.g, h, efeat_out,rain0[:,l+1])  
            rain_all=torch.cat((rain_all,rain_M.view(-1,1)),1)

        return  rain_all  
    
    
  
    
    