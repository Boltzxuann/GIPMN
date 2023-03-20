import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from STEIConv_all import  STEIConvNetMScaleCompact, STEIConvNetMScaleCompactGRAD
import math
import reading_functions as rfunc
ReadFunc=rfunc.ReadingFunctions
import meteva.method as mem
from utils import EarlyStopping


class model():
    def __init__(self,train_data, **args):
        super(model, self).__init__()    
        self.train_data = train_data
        self.max_epochs=args['max_epochs']
        self.gpus=args['num_gpus']
        self.show_loss=args['show_trainloss_every_num_iterations_per_epoch']
        self.show_performance=args['show_validperformance_every_num_epochs']
        self.num_examples=args['num_examples']
    
    
    
    
    
    def train(self):
        
        def mkdir(path):
            folder = os.path.exists(path)
            if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path)   #makedirs 创建文件时如果路径不存在会创建这个路径        
        
        def transform_log2(rain):
            out=torch.log2(rain+1)
            return out
        def inverse_transform_log2(rain):
            out=2**rain-1
            #out[out<0.1]=0
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
        

        feature=[]
        label=[]
        for n in range(self.num_examples):
            print(n)
            example= self.train_data+'/example%05d'%(n+1)
            train_feature=ReadFunc.read_situation_field_predi_train(example,1)
            train_feature=np.expand_dims(train_feature,axis=3)
            for k in range(2,10):
                values_predi=ReadFunc.read_situation_field_predi_train(example,k)
                #values_predi=values_predi.reshape(69*73,-1) ##拉平
                values_predi=np.expand_dims(values_predi,axis=3)
                train_feature=np.concatenate((train_feature,values_predi),axis=3)
            feature.append(train_feature)
            
            train_label=ReadFunc.read_obs_grid_train(example,1)
            #train_feature=train_feature.reshape(69*73,-1)
            train_label=np.expand_dims(train_label,axis=3)
            for k in range(2,10):
                values_predi=ReadFunc.read_obs_grid_train(example,k)
                #values_predi=values_predi.reshape(69*73,-1) ##拉平
                values_predi=np.expand_dims(values_predi,axis=3)
                train_label=np.concatenate((train_label,values_predi),axis=3)
            label.append(train_label)
            
        feature=np.array(feature)
        label=np.array(label)
            
        train_features= torch.from_numpy(feature)[:int(self.num_examples*0.8)].float()
        train_labels= torch.from_numpy(label)[:int(self.num_examples*0.8)].float()
        test_features= torch.from_numpy(feature)[int(self.num_examples*0.8):].float()
        test_labels= torch.from_numpy(label)[int(self.num_examples*0.8):].float()
        
        N_val  = test_features.shape[0]
        N_train = train_features.shape[0]
        print('number of training samples:',N_train)
        


        def transform20(rain):
            index_0_3=(rain>=0)&(rain<3)
            index_3_10=(rain>=3)&(rain<10)
            index_10_20=(rain>=10)&(rain<20)
            index_20 = rain>20
            out=torch.zeros(rain.shape)
            if cuda:
                out=out.cuda()
            out[index_0_3]=0+(rain[index_0_3])/(3) #0/3-1
            out[index_3_10]=1+(rain[index_3_10]-3)/(10-3) #1-2
            out[index_10_20]=2+(rain[index_10_20]-10)/(20-10)#2-3
            out[index_20] = 3+ (rain[index_20]-20)/10 #>3
            return out

        def inverse_transform20(rain):
            index_01=(rain>=0/3)&(rain<1)
            index_12=(rain>=1)&(rain<2)
            index_23=(rain>=2)&(rain<3)
            index_3 = rain>=3
            out=torch.zeros(rain.shape)
            if cuda:
                out=out.cuda()
            out[index_01]=(rain[index_01])*3 #0-3
            out[index_12]=3+(rain[index_12]-1)*(10-3) #3-10
            out[index_23]=10+(rain[index_23]-2)*(20-10)#10-20
            out[index_3] = 20+ (rain[index_3]-3)*10 #>20  
            return out



        def mean_square_error(y,labels):
            mean_e= torch.sum((y-labels)**2)
            n = y.size(0)
            return mean_e, n

        def mae_error(y,labels):
            mean_e= torch.sum(abs(y-labels))
            n = y.size(0)
            return mean_e, n

        def bmae_error(y,labels):
            mean_e= torch.sum(abs(y-labels)*(labels+1))
            n = (labels+1).sum()
            return mean_e, n

        def p_sun_rain(labels,y,grade_list=0.1):

            return mem.pc_of_sun_rain(labels,y)

        def pod(labels,y,grade_list=[0.1]):

            return mem.pod(labels,y,grade_list=grade_list)

        def ts(labels,y,grade_list=[0.1]):

            return mem.ts(labels,y,grade_list=grade_list)


        def evaluate_3h_simple(y,labels, mask,criterion ):

            y = y[mask].flatten()
            yy = inverse_transform20(y)
            labels = labels[mask].flatten()
            ll = inverse_transform20(labels)
            return criterion(yy, ll)

        def evaluate_simple(labels, y, mask, thrd, criterion):
            y = y[mask].flatten()
            yy = inverse_transform20(y)
            labels = labels[mask].flatten()
            ll = inverse_transform20(labels) 
            yy[yy<0.1]=0
            return criterion(ll.detach().cpu().numpy(), yy.detach().cpu().numpy() , grade_list=[thrd])  


        gpu= -1 #gpu device
        #hyperparameters
        #c_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,40,41,42,43,44,45,46,47,48,49,50]
        c_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,21,22,23,27,28,30,31,32,33,34,40,43,44,46,47,48,49,50]
        #hyperparameters
        Error_mode =False##是否在投影模型最后一层加入输入，True：是，False：否
        InNow = False ##特征提取部分是否输入现在的原始场
        num_feats=  2#1 ##输入层特征维数 53维
        edge_feats1= 36+31+8#预报因子 53维
        num_layers = 9##时间步长
        num_hidden= 128
        out_dim=0
        early_stop= True
        fastmode= True

        if gpu < 0:
            cuda = False
        else:
            cuda = True
            torch.cuda.set_device(gpu)

        from dgl.data.utils import load_graphs
        g = load_graphs("./params/grid_69_73_pro.bin",[0])
        g = g[0][0]
        #g = dgl.add_self_loop(g) ##加
        if cuda:
            g=g.to(torch.device('cuda:%01d'%(gpu)))
        # create model


        model = STEIConvNetMScaleCompactGRAD(g,
                         num_layers,
                         4,
                         num_feats,
                         num_hidden,
                         out_dim,
                         edge_feats1,
                         activation_o= lerelu,
                         activation_edge= lerelu,
                        error_mode=Error_mode)  ##activation function for intermidiate layer

        models=[model]
        if cuda:
            model.cuda()

        print(model)
        if early_stop:
            stopper = EarlyStopping(patience=60)

        loss_fcnMSE = torch.nn.MSELoss(reduction='none')   
        loss_fcnMAE = torch.nn.L1Loss(reduction='none')  
        lossF_sun_rain = torch.nn.BCELoss(reduction='none')
        #lossF_sun_rain = torch.nn.BCEWithLogitsLoss(reduction='none')
        # use optimizer

        optimizer_featureP = torch.optim.Adam(
            model.parameters(), lr=1e-5)##不知道对不对
        '''

        optimizer_featureP = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
        '''

        dur = []
        best=999
        Val_errorP=torch.tensor(0)
        Val_errorN=torch.tensor(0)


        for epoch in range(self.max_epochs):

            val_errorP=0
            val_errorN=0
            val_error=0
            loss_m = 0
            loss_mp = 0
            loss_mn = 0
            N_total = 0
            if epoch >= 3:
                t0 = time.time()
            rand_n=torch.randperm(N_train) 
            batchsize= 1  
            nbatches = int(N_train/batchsize/3*2) #use 2/3 of the training data every epoch
            N_nodes_train= 0
            for batch in range(nbatches):
                N_ba = 0
                for n in range(batchsize): 
                    model.train()
                    # forward
                    index_n = rand_n[n+batchsize*batch]

                    ###计算当前订正场
                    rain00_0 = train_features[index_n][:,:,50,:].reshape(69*73,-1)*1000 #N*69*73*50*9 原始预报3和降水量###原始预报累计降水量
                    rainLC_0 = train_features[index_n][:,:,49,:].reshape(69*73,-1)*1000 ##大尺度降水
                    labels0 = train_labels[index_n][0][:,:,0:].reshape(69*73,-1)###标签的单位是mm!!!
                    if cuda:
                        rain00_0=rain00_0.cuda()
                        rainLC_0=rainLC_0.cuda()
                        labels0=labels0.cuda()
                    rain00 = transform20(rain00_0)
                    rainLC = transform20(rainLC_0)
                    #rain0 = rain00[:,1:]-rain00[:,:-1]##原始预报3h降水量            
                    labels = transform20(labels0)
                    mask= train_labels[index_n][1][:,:,0:].reshape(69*73,-1).bool()
                    M3h = torch.sigmoid(100*(labels0-0.1))
                    M3hL = torch.sigmoid(100*(labels0-10))
                    efeat1 = train_features[index_n][:,:,c_id,:].reshape(69*73,len(c_id),9)##预报要素
                    efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/feat_std[c_id].reshape(1,len(c_id),1)##正态化
                    #efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/(feat_max[c_id]-feat_min[c_id]).reshape(1,len(c_id),1)##（-1，1）
                    d98=train_features[index_n][:,:,[38],:]+train_features[index_n][:,:,[39],:]
                    d29=train_features[index_n][:,:,[35],:]-train_features[index_n][:,:,[39],:]
                    V87=train_features[index_n][:,:,[12],:]+train_features[index_n][:,:,[13],:]
                    pv98=train_features[index_n][:,:,[33],:]+train_features[index_n][:,:,[34],:]
                    #wq_all=train_features[index_n][:,:,[10,11,12,13],:]*train_features[index_n][:,:,[5,6,7,8],:]            #wq_all=train_features[index_n][:,:,[10,11,12,13],:]*train_features[index_n][:,:,[5,6,7,8],:]
                    factor_new = torch.cat((d98,d29,V87,pv98),2).reshape(69*73,4,9)#,wq_all
                    factor_newN = (factor_new-factor_mean[:4].reshape(1,4,1))/factor_std[:4].reshape(1,4,1)

                    efeat1N[:,-1,:]=rain00
                    efeat1N[:,-2,:]=rainLC

                    efeat1N_div=torch.zeros(efeat1N.shape[0],efeat1N.shape[1]-5,9)
                    factor_newN_div = torch.zeros(efeat1N.shape[0],4,9)
                    #weight = 1/9*torch.ones(1,1,3,3).float()
                    #efeat_smooth=torch.zeros(9, 5, 69, 73)
                    if cuda== True:
                        rain00=rain00.cuda()
                        labels0=labels0.cuda()
                        labels=labels.cuda()
                        mask=mask.cuda()
                        M3h =M3h.cuda()
                        M3hL =M3hL.cuda()
                        #weight=weight.cuda()
                        #efeat_smooth=efeat_smooth.cuda()
                        efeat1N_div=efeat1N_div.cuda()
                        efeat1N=efeat1N.cuda()
                        factor_newN=factor_newN.cuda()
                        factor_newN_div=factor_newN_div.cuda()

                    efeat1N_div[:,:,1:]=efeat1N[:,:-5,1:]-efeat1N[:,:-5,:-1] ##各要素的时间差分（三小时）
                    efeat1N_div[:,:,0]=efeat1N_div[:,:,1]
                    factor_newN_div[:,:,1:]=factor_newN[:,:,1:]-factor_newN[:,:,:-1] ##各要素的时间差分（三小时）
                    factor_newN_div[:,:,0]=factor_newN_div[:,:,1] 
                    efeat_total = torch.cat((efeat1N,factor_newN,efeat1N_div,factor_newN_div),1)
                    N_t = (mask).sum()            
                    rain_forecast = model(efeat1N[:,-2:,0],efeat_total[:,:,0:],rain00) ##输入两个连续的粗订正场，提取特征  
                    rf = inverse_transform20(rain_forecast)
                    RF = torch.sigmoid(100*(rf-0.1)) ##用于晴雨分类
                    RFL = torch.sigmoid(100*(rf-10)) ##用于中雨（10mm）分类
                    W_nodes_3h=labels0+1 ##
                    loss_pos = loss_fcnMAE( rain_forecast[mask].flatten(),
                            labels[mask].flatten() ) * W_nodes_3h[mask].flatten()
                    loss_dif = lossF_sun_rain( RF[mask].flatten(),
                            M3h[mask].flatten() )* W_nodes_3h[mask].flatten() ###晴雨分类损失
                    #loss_difL = lossF_sun_rain( RFL[mask].flatten(),
                    #        M3hL[mask].flatten() ) ###晴雨分类损失
                    loss_w = loss_pos.mean()+0.001*loss_dif.mean()#+0.01*loss_difL.mean()
                    if batch % self.show_loss==0:
                        print('loss of a random example:',loss_w)

                    if torch.isnan(loss_w) == False: 
                        loss_w.backward()
                        loss_m += loss_w#loss_dif.mean().item()              
                        N_ba += 1#N_t


                ##更新参数            
                optimizer_featureP.step()

                ##清空梯度
                optimizer_featureP.zero_grad()

                N_nodes_train += N_ba 

            loss_m = loss_m/(N_nodes_train+1)   
            P_sun_rain=0
            P_sun_rain0=0
            POD = 0
            TS=0
            TS0=0
            TSM=0
            TSM0=0
            TSL=0
            TSL0=0
            Ns=0
            Nm=0
            Nl=0
            labelsALL=torch.zeros(1)
            rain_forecastALL=torch.zeros(1)
            rain00ALL=torch.zeros(1)
            if cuda:
                labelsALL=labelsALL.cuda()
                rain_forecastALL=rain_forecastALL.cuda()
                rain00ALL=rain00ALL.cuda()

            for n in range(N_val):
                model.eval()

                    ###计算当前订正场

                rain00T_0 = test_features[n][:,:,50,:].reshape(69*73,-1)*1000 #N*69*73*50*9 原始预报3h累计降水量
                rainLCT_0 = test_features[n][:,:,49,:].reshape(69*73,-1)*1000 ##大尺度降水
                labelsT0 = test_labels[n][0][:,:,0:].reshape(69*73,-1) #        
                if cuda:
                    rain00T_0=rain00T_0.cuda()
                    rainLCT_0=rainLCT_0.cuda()
                    labelsT0=labelsT0.cuda()
                rain00T = transform20(rain00T_0)
                rainLCT = transform20(rainLCT_0)
                #rain0 = rain00[:,1:]-rain00[:,:-1]##原始预报3h降水量            
                labelsT = transform20(labelsT0)
                maskT = test_labels[n][1][:,:,0:].reshape(69*73,-1).bool()
                efeat1 = test_features[n][:,:,c_id,:].reshape(69*73,len(c_id),9)##预报要素
                efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/feat_std[c_id].reshape(1,len(c_id),1)##正态化
                d98=test_features[n][:,:,[38],:]+test_features[n][:,:,[39],:]
                d29=test_features[n][:,:,[35],:]-test_features[n][:,:,[39],:]
                V87=test_features[n][:,:,[12],:]+test_features[n][:,:,[13],:]
                pv98=test_features[n][:,:,[33],:]+test_features[n][:,:,[34],:]
                #wq_all=test_features[n][:,:,[10,11,12,13],:]*train_features[index_n][:,:,[5,6,7,8],:]
                factor_new = torch.cat((d98,d29,V87,pv98),2).reshape(69*73,4,9)#wq_all
                factor_newN = (factor_new-factor_mean[:4].reshape(1,4,1))/factor_std[:4].reshape(1,4,1)
                #efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/(feat_max[c_id]-feat_min[c_id]).reshape(1,len(c_id),1)##（-1，1）
                efeat1N[:,-1,:]=rain00T
                efeat1N[:,-2,:]=rainLCT
                #weight = 1/9*torch.ones(1,1,3,3).float()
                #efeat_smooth=torch.zeros(9, 5, 69, 73) 
                efeat1N_div=torch.zeros(efeat1N.shape[0],efeat1N.shape[1]-5,9)
                factor_newN_div = torch.zeros(efeat1N.shape[0],4,9)
                if cuda== True:
                    rain00T=rain00T.cuda()
                    labelsT=labelsT.cuda()
                    maskT=maskT.cuda()
                    #weight = weight.cuda()
                    #efeat_smooth=efeat_smooth.cuda()
                    efeat1N_div=efeat1N_div.cuda()
                    efeat1N=efeat1N.cuda()
                    factor_newN=factor_newN.cuda()
                    factor_newN_div=factor_newN_div.cuda()

                efeat1N_div[:,:,1:]=efeat1N[:,:-5,1:]-efeat1N[:,:-5,:-1] ##各要素的时间差分（三小时）
                efeat1N_div[:,:,0]=efeat1N_div[:,:,1] 
                factor_newN_div[:,:,1:]=factor_newN[:,:,1:]-factor_newN[:,:,:-1] ##各要素的时间差分（三小时）
                factor_newN_div[:,:,0]=factor_newN_div[:,:,1] 
                efeat_total = torch.cat((efeat1N,factor_newN,efeat1N_div,factor_newN_div),1)
                model.eval()
                with torch.no_grad():
                    rain_forecastT = model(efeat1N[:,-2:,0],efeat_total[:,:,0:],rain00T) ##输入两个连续的粗订正场，提取特征  
                #labelsT[labelsT0<10]=0 
                if (epoch % self.show_performance) ==0:
                    rain_forecastALL=torch.cat((rain_forecastALL,rain_forecastT[maskT]))
                    rain00ALL=torch.cat((rain00ALL,rain00T[maskT]))
                    labelsALL=torch.cat((labelsALL,labelsT[maskT]))
                    val_error0, N0 = evaluate_3h_simple( rain_forecastT,
                                                          labelsT, 
                                                          maskT, 
                                                          bmae_error)
                    val_error += val_error0

                    N_total += N0
            if (epoch % self.show_performance) ==0:
                path1='./best_model/'
                mkdir(path1) 
                maskT = torch.ones(labelsALL.shape).bool()

                P_sun_rain0 = evaluate_simple( labelsALL, rain00ALL,
                                                 maskT, 0.1,
                                                p_sun_rain)   
                P_sun_rain = evaluate_simple( labelsALL, rain_forecastALL,
                                                maskT, 0.1,
                                                p_sun_rain)                             

                TS = evaluate_simple( labelsALL, rain_forecastALL,
                                         maskT, 0.1,
                                                ts)
                TS0 = evaluate_simple( labelsALL, rain00ALL,
                                         maskT, 0.1,
                                                ts) 
                TSM = evaluate_simple( labelsALL, rain_forecastALL,
                                         maskT, 3,
                                                ts)
                TSM0 = evaluate_simple( labelsALL, rain00ALL,
                                         maskT, 3,
                                                ts)   
                TSL = evaluate_simple( labelsALL, rain_forecastALL,
                                         maskT, 10,
                                                ts)
                TSL0 = evaluate_simple( labelsALL, rain00ALL,
                                         maskT, 10,
                                                ts)  
                TSH = evaluate_simple( labelsALL, rain_forecastALL,
                                         maskT, 20,
                                                ts)
                TSH0 = evaluate_simple( labelsALL, rain00ALL,
                                         maskT, 20,
                                                ts)  
                print ('晴雨准确率：',P_sun_rain)
                print('TS(R> 0.1mm/3h):',TS)
                print('TS(R> 3mm/3h):',TSM)
                print('TS(R> 10mm/3h):',TSL)
                print('TS(R> 20mm/3h):',TSH)
                print('订正前晴雨:',P_sun_rain0)
                print('订正前TS(R> 0.1mm/3h):',TS0)
                print('订正前TS(R> 3mm/3h):',TSM0)
                print('订正前TS(R> 10mm/3h):',TSL0)
                print('订正前TS(R> 20mm/3h):',TSH0)

                Val_error=val_error/(N_total+1)
                Val_errorP=val_errorP/(N_total+1)
                Val_errorN=val_errorN/(N_total+1)  
                path=path1+'CompactSTEIConvMSGRAD_OlereluElerelu4_1_9_in2_efeat36+31+4+4new_segment_normal_hidden_128_wMAEnew_wclassifyS001_adam1e-5.pt'

                if early_stop: 
                    if stopper.step(TS+TSM+TSL+TSH, models, path):  #-Val_error*0.1+TS+TSM 
                        break

                if epoch >= 3:
                    dur.append(time.time() - t0)
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.9f} |val_error{:.4f} ".
                          format(epoch, np.mean(dur), loss_m, Val_error))
        
            
            