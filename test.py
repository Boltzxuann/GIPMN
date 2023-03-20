import numpy as np
import time
import torch
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from mymodel import  STEIConvNetMScaleCompact, STEIConvNetMScaleCompactGRAD

import math
import os
import reading_functions as rfunc
ReadFunc=rfunc.ReadingFunctions
N_test=sum([os.path.isdir(listx) for listx in os.listdir('./data_test')]) #获取文件夹数目

train_stats=np.load('./params/train_stats.npz',allow_pickle=True)
#mean=feat_mean,std=feat_std,maxx=feat_max,minn=feat_min
feat_mean=torch.from_numpy(train_stats['mean']).float()
feat_std=torch.from_numpy(train_stats['std']).float()
feat_max=torch.from_numpy(train_stats['maxx']).float()
feat_min=torch.from_numpy(train_stats['minn']).float()

factor_stats=np.load('./params/factor_new_stats.npz',allow_pickle=True)
#mean=feat_mean,std=feat_std,maxx=feat_max,minn=feat_min
factor_mean=torch.from_numpy(factor_stats['mean']).float()
factor_std=torch.from_numpy(factor_stats['std']).float()
factor_max=torch.from_numpy(factor_stats['maxx']).float()
factor_min=torch.from_numpy(train_stats['minn']).float()
def transform_log2(rain):
    out=torch.log2(rain+1)
    return out
def inverse_transform_log2(rain):
    out=2**rain-1
    out[out<0.1]=0
    return out
   
def transform(rain):
    index_01_3=(rain>=0.1)&(rain<3)
    index_3_10=(rain>=3)&(rain<10)
    index_10_20=(rain>=10)&(rain<20)
    index_20 = rain>20
    out=torch.zeros(rain.shape)
    if cuda:
        out=out.cuda()
    out[rain<0.1]=0
    out[index_01_3]=0+(rain[index_01_3])/(3) #0.1/3-1
    out[index_3_10]=1+(rain[index_3_10]-3)/(10-3) #1-2
    out[index_10_20]=2+(rain[index_10_20]-10)/(20-10)#2-3
    out[index_20] = 3+ (rain[index_20]-20)/10 #>3
    return out
    
def inverse_transform(rain):
    index_01=(rain>=0.1/3)&(rain<1)
    index_12=(rain>=1)&(rain<2)
    index_23=(rain>=2)&(rain<3)
    index_3 = rain>=3
    out=torch.zeros(rain.shape)
    if cuda:
        out=out.cuda()
    out[index_01]=(rain[index_01])*3 #0.1-3
    out[index_12]=-4+(rain[index_12])*(10-3) #3-10
    out[index_23]=-10+(rain[index_23])*(20-10)#10-20
    out[index_3] = -10+ (rain[index_3])*10 #>20  
    return out        
    

def lerelu(x):
    y=F.leaky_relu(x,negative_slope=0.1)
    return y


def elu0(x):
    y= F.elu(x)+1
    return y

def neg_elu0(x):
    y= -F.elu(x)-1  
    return y

def pos_2elu(x):
    
    return 2*F.elu(x)


def neg_2elu(x):
    y= -2*F.elu(x)##允许有正的
    return y

def softplus_zero(x):
    a = x-x
    y1=F.softplus(x)-F.softplus(a)
    #y=y1
    #y[y1<0]=0
    return y1

def neg_softplus_zero(x):
    return -softplus_zero(x)

def double_tanh(x):
    return 2*torch.tanh(x)

def tanh_scale(x,scales=[100,5]):
    return 100/scales[0]*torch.tanh(x/scales[1])

c_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,21,22,23,27,28,30,31,32,33,34,40,43,44,46,47,48,49,50]
#hyperparameters
scale=10##降水量缩放因子
Error_mode =False##是否在投影模型最后一层加入输入，True：是，False：否
InNow = False ##特征提取部分是否输入现在的原始场
num_feats=  2#1 ##输入层特征维数 53维
edge_feats1= 36+31+8#预报因子 53维
num_layers = 9##时间步长
num_hidden= 64
out_dim=0
early_stop= True
fastmode= True
gpu=0
if gpu < 0:
    cuda = False
else:
    cuda = True
    torch.cuda.set_device(gpu)

from dgl.data.utils import load_graphs
g = load_graphs("./grid_69_73_pro.bin",[0])
g = g[0][0]
#g = dgl.add_self_loop(g) ##加
if cuda:
    g=g.to(torch.device('cuda:%01d'%(gpu)))
# create model


model1 = STEIConvNetMScaleCompactGRAD(g,
                 num_layers,
                 4,
                 num_feats,
                 num_hidden,
                 out_dim,
                 edge_feats1,
                 activation_o= lerelu,
                 activation_edge=lerelu,
                error_mode=Error_mode)  ##activation function for intermidiate layer

models1=[model1]
if cuda:
    model1.cuda()
    
print(model1)
path1='./models/'
model_name='CompactSTEIConvMSGRAD_OlereluElerelu4_1_9_in2_efeat36+31+4+4new_log2_normal_hidden_64_wMAEnew_classifyS01_adam1e-5.pt'
path=path1+model_name
modelss1 = torch.load(path)
models1[0].load_state_dict(modelss1[0])



Error_mode =False##是否在投影模型最后一层加入输入，True：是，False：否
InNow = False ##特征提取部分是否输入现在的原始场
num_feats=  2#1 ##输入层特征维数 53维
edge_feats1= 36+31+4+4#预报因子 53维
num_layers = 9##时间步长
num_hidden= 64
out_dim=0
early_stop= True
fastmode= True

# create model


model2 = STEIConvNetMScaleCompactGRAD(g,
                 num_layers,
                 4,
                 num_feats,
                 num_hidden,
                 out_dim,
                 edge_feats1,
                 activation_o= torch.tanh,
                 activation_edge=torch.tanh,
                error_mode=Error_mode)  ##activation function for intermidiate layer

models=[model2]
if cuda:
    model2.cuda()
    
print(model2)


model_name='CompactSTEIConvMSGRAD_OtanhEtanh4_1_9_in2_efeat36+31+4+4new_segment_normal_hidden_64_wMAEnew_classifyS05_adam5e-5.pt'
path=path1+model_name
modelss = torch.load(path)
models[0].load_state_dict(modelss[0])


for n in range(N_test):
    ###加载测试样本
    test_features=ReadFunc.read_situation_field_predi_test(n,1)
    test_features=test_feature.reshape(69*73,-1)
    test_features=np.expand_dims(test_feature,axis=2)
    if test_features != 'empty':
        for k in range(2,10):
            print(k)
            values_predi=RF.read_situation_field_predi_test(n,k)
            values_predi=values_predi.reshape(69*73,-1) ##拉平
            if values_predi != 'empty':
                values_predi=np.expand_dims(values_predi,axis=2)
                test_features=np.concatenate((test_features,values_predi),axis=2)    
            else: test_features = 'empty'
    
    
    if test_features != 'empty':
        test_features = torch.from_numpy(test_features).float()
        ###计算当前订正场
        rain00T_0 = test_features[:,50,:]*1000 #N*69*73*50*9 原始预报3h累计降水量
        rainLCT_0 = test_features[:,49,:]*1000 ##大尺度降水     
        if cuda:
            rain00T_0=rain00T_0.cuda()
            rainLCT_0=rainLCT_0.cuda()
        rain00T_model1 = transform_log2(rain00T_0)
        rainLCT_model1 = transform_log2(rainLCT_0)
        #rain0 = rain00[:,1:]-rain00[:,:-1]##原始预报3h降水量            
        rain00T_model2 = transform(rain00T_0)#**power #N*69*73*50*9 原始预报3h累计降水量
        rainLCT_model2 = transform(rainLCT_0)#**power ##大尺度降水   
        efeat1 = test_features[:,c_id,:]#.reshape(69*73,len(c_id),9)##预报要素
        efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/feat_std[c_id].reshape(1,len(c_id),1)##正态化
        efeat1N_model1=torch.zeros(efeat1N.shape)
        efeat1N_model2=torch.zeros(efeat1N.shape)
        efeat1N_model1[:,:-2,:]=efeat1N[:,:-2,:]
        efeat1N_model2[:,:-2,:]=efeat1N[:,:-2,:]

        efeat1N_model1[:,-1,:]=rain00T_model1
        efeat1N_model1[:,-2,:]=rainLCT_model1
        efeat1N_model2[:,-1,:]=rain00T_model2
        efeat1N_model2[:,-2,:]=rainLCT_model2
        d98=test_features[:,[38],:]+test_features[:,[39],:]
        d29=test_features[:,[35],:]-test_features[:,[39],:]
        V87=test_features[:,[12],:]+test_features[:,[13],:]
        pv98=test_features[:,[33],:]+test_features[:,[34],:]
        #wq_all=test_features[n][:,:,[10,11,12,13],:]*train_features[index_n][:,:,[5,6,7,8],:]
        factor_new = torch.cat((d98,d29,V87,pv98),1)#.reshape(69*73,4,9)#wq_all
        factor_newN = (factor_new-factor_mean[:4].reshape(1,4,1))/factor_std[:4].reshape(1,4,1)
        efeat1N_div1=torch.zeros(efeat1N.shape[0],efeat1N.shape[1]-5,9)
        efeat1N_div2=torch.zeros(efeat1N.shape[0],efeat1N.shape[1]-5,9)
        factor_newN_div = torch.zeros(efeat1N.shape[0],4,9)
        if cuda== True:
            #weight = weight.cuda()
            #efeat_smooth=efeat_smooth.cuda()
            efeat1N_div1=efeat1N_div1.cuda()
            efeat1N_div2=efeat1N_div2.cuda()
            efeat1N_model1=efeat1N_model1.cuda()
            efeat1N_model2=efeat1N_model2.cuda()
            factor_newN=factor_newN.cuda()
            factor_newN_div=factor_newN_div.cuda()

        #input_0 = efeat_total[:,31:,:]
        efeat1N_div1[:,:,1:]=efeat1N_model1[:,:-5,1:]-efeat1N_model1[:,:-5,:-1] ##各要素的时间差分（三小时）
        efeat1N_div1[:,:,0]=efeat1N_div1[:,:,1] 
        efeat1N_div2[:,:,1:]=efeat1N_model2[:,:-5,1:]-efeat1N_model2[:,:-5,:-1] ##各要素的时间差分（三小时）
        efeat1N_div2[:,:,0]=efeat1N_div2[:,:,1] 
        factor_newN_div[:,:,1:]=factor_newN[:,:,1:]-factor_newN[:,:,:-1] ##各要素的时间差分（三小时）
        factor_newN_div[:,:,0]=factor_newN_div[:,:,1] 
        efeat_total_model1 = torch.cat((efeat1N_model1,factor_newN,efeat1N_div1,factor_newN_div),1)
        efeat_total_model2 = torch.cat((efeat1N_model2,factor_newN,efeat1N_div2,factor_newN_div),1)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            rain_forecastT_model1 = model1(efeat1N_model1[:,-2:,0],efeat_total_model1[:,:,0:],rain00T_model1) #log2
            rain_forecastT_model2 = model2(efeat1N_model2[:,-2:,0],efeat_total_model2[:,:,0:],rain00T_model2) #power1/3
    ######将两个模型降水量融合###########
        rain1=inverse_transform_log2(rain_forecastT_model1) ###它的晴雨更高
        rain2=inverse_transform(rain_forecastT_model2)#**(1/power)

        mask_sun=rain1<0.1
        rain_forecast_combined = rain1##zhegeyao改一下
        #rain_forecast_combined[rain1>rain2]=rain1[rain1>rain2]
        rain_forecast_combined[rain2>rain1]=rain2[rain2>rain1]
        rain_forecast_combined[mask_sun]=0  ##保证雨区是模型1的  
        for step in range(9):
            maskT=RF.read_station_coords(n+1,step+1)
            if maskT.ndim==1:
                maskT=maskT.reshape(1,2)
            predi=(rain_forecast_combined[:,step].reshape(69,73)[maskT[:,0],maskT[:,1]].cpu().numpy())
            predi[predi<0.1]=0

            path_predi='./predi0124combined_3/example%05d/'%(n+1)
            mkdir(path_predi)
            np.savetxt(path_predi+'pred_%02d.txt'%(step+1),predi, fmt='%.04f')    
    