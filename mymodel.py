import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from STEIConv_all import  STEIConvNetMScaleCompactGRAD, STEIConvNetMScaleCompactGRAD_PROP
import math
import reading_functions as rfunc
import meteva.method as mem
from utils import EarlyStopping
from torch.optim.lr_scheduler import StepLR
ReadFunc=rfunc.ReadingFunctions
class model():
    def __init__(self,train_data, args):
        super(model, self).__init__()    
        self.train_data = train_data
        self.max_epochs=args.max_epochs#['max_epochs']
        self.gpus=args.num_gpus#['num_gpus']
        self.show_loss=args.show_trainloss_every_num_iterations_per_epoch#['show_trainloss_every_num_iterations_per_epoch']
        self.show_performance=args.show_validperformance_every_num_epochs#['show_validperformance_every_num_epochs']
        self.num_examples=args.num_examples#['num_examples']
        self.threshold=args.level_threshold#['level_threshold']
        self.wide = args.level_width#['level_width']
        self.gpu = args.gpu_id#['gpu_id']
    
       
    
    def train(self):
        print('level:',self.threshold)
        
        def mkdir(path):
            folder = os.path.exists(path)
            if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path)   #makedirs 创建文件时如果路径不存在会创建这个路径        

        def PDF_near(R):  # 降水量的概率密度
            P = R
            # P[(R>=0)&(R<1)]=(R-0)/1*(PDF_list[10]-PDF_list[0])+PDF_list[0]
            P[R < 60] = PDF_list[[(R[R < 60] / 0.1).long()]]
            P[R >= 60] = PDF_list[600]
            if cuda:
                P = P.cuda()
            return P

        def Td(p, q, T):
            td = torch.zeros(q.shape)
            a = torch.zeros(q.shape)
            b = torch.zeros(q.shape)
            if cuda:
                td = td.cuda()
                a = a.cuda()
                b = b.cuda()
            td[q == 0] = -999
            a[(q != 0) * (T >= (263 - 273.15))] = 17.26
            a[(q != 0) * (T < (263 - 273.15))] = 21.87
            b[(q != 0) * (T >= (263 - 273.15))] = 35.86
            b[(q != 0) * (T < (263 - 273.15))] = 7.66
            e = p * q / (0.622 + 0.378 * q)
            td[q != 0] = (b[q != 0] - 273.16) / (1 - a[q != 0] / torch.log(e[q != 0] / 6.11))
            return td

        def K1_Tse(efeat1):  # 10个因子

            T200 = efeat1[:, 15, :] - 273.15
            T500 = efeat1[:, 16, :] - 273.15
            T700 = efeat1[:, 17, :] - 273.15
            T850 = efeat1[:, 18, :] - 273.15
            T925 = efeat1[:, 19, :] - 273.15
            q200 = efeat1[:, 5, :]
            q500 = efeat1[:, 6, :]
            q700 = efeat1[:, 7, :]
            q850 = efeat1[:, 8, :]
            q925 = efeat1[:, 9, :]
            Td200 = Td(200, q200, T200)
            Td500 = Td(500, q500, T500)
            Td700 = Td(700, q700, T700)
            Td850 = Td(850, q850, T850)
            Td925 = Td(925, q925, T925)
            # K850=(T850-T500)+Td850-(T700-Td700)
            # K1.append(K850.numpy())
            # K925=(T925-T500)+Td925-(T700-Td700)
            # K2.append(K925.numpy())
            p = 500
            theataSE_500 = (T500 + 273.15) * torch.exp(
                0.28586 * math.log(1000 / p) + 2500 * q500 / (338.52 - 0.24 * (T500 + 273.15) + 1.24 * Td500))
            p = 925
            theataSE_925 = (T925 + 273.15) * torch.exp(
                0.28586 * math.log(1000 / p) + 2500 * q925 / (338.52 - 0.24 * (T925 + 273.15) + 1.24 * Td925))
            Tse_500_925 = theataSE_500 - theataSE_925
            K850 = (T850 - T500) + Td850 - (T700 - Td700)
            # K1.append(K850.numpy())
            K925 = (T925 - T500) + Td925 - (T700 - Td700)
            '''
            K925_sum=torch.zeros(K925.shape).cuda()
            K925_minus=torch.zeros(K925.shape).cuda()
            K925_sum[:,1:]=K925[:,:-1]+K925[:,1:]
            K925_minus[:,1:]=-K925[:,:-1]+K925[:,1:]
            K925_minus[:,0]=K925_minus[:,1]
            K925_p1= K925[:,0]-K925_minus[:,0]##-1时刻的估计值
            K925_sum[:,0]=K925[:,0]+K925_p1    
            mask_l925=K925_sum>20
            mask_r925=K925_sum<100
            mask_u925=K925_minus<25
            mask_d925=K925_minus>-25 
            '''
            # mask_xiaokong925 = mask_l925*mask_r925*mask_u925*mask_d925 ##用阈值法消去负样本

            K850_sum = torch.zeros(K850.shape)  # .cuda()
            K850_minus = torch.zeros(K850.shape)  # .cuda()
            Tse_sum = torch.zeros(K850.shape)  # .cuda()
            Tse_minus = torch.zeros(K850.shape)  # .cuda()
            if cuda:
                K850_sum = K850_sum.cuda()
                K850_minus = K850_minus.cuda()
                Tse_sum = Tse_sum.cuda()
                Tse_minus = Tse_minus.cuda()
            K850_sum[:, 1:] = K850[:, :-1] + K850[:, 1:]
            K850_minus[:, 1:] = -K850[:, :-1] + K850[:, 1:]
            K850_minus[:, 0] = K850_minus[:, 1]
            K850_p1 = K850[:, 0] - K850_minus[:, 0]  ##-1时刻的估计值
            K850_sum[:, 0] = K850[:, 0] + K850_p1

            Tse_sum[:, 1:] = Tse_500_925[:, :-1] + Tse_500_925[:, 1:]
            Tse_minus[:, 1:] = -Tse_500_925[:, :-1] + Tse_500_925[:, 1:]
            Tse_minus[:, 0] = Tse_minus[:, 1]
            Tse_p1 = Tse_500_925[:, 0] - Tse_minus[:, 0]  ##-1时刻的估计值
            Tse_sum[:, 0] = Tse_500_925[:, 0] + Tse_p1

            # mask_l925=K850_sum>20
            # mask_r925=K850_sum<100
            # mask_u925=K850_minus<25
            # mask_d925=K850_minus>-25
            # mask_xiaokong925 = mask_l925*mask_r925*mask_u925*mask_d925 ##用阈值法消去负样本
            # mask_pos = mask_xiaokong925
            return K850_sum, K850_minus, Tse_sum, Tse_minus

        def transform_sigmoid_linear(rain, th, dR):  # 降水量转换成标签，th 是阈值，dR是宽度
            k = 2 / th * math.log(9)
            # k=torch.tensor([k])
            index_0 = (rain >= 0) & (rain < th - dR)
            index_1 = (rain >= th - dR) & (rain < th + dR)
            index_2 = rain >= th + dR
            out = torch.zeros(rain.shape)
            if cuda:
                out = out.cuda()
            out[index_0] = rain[index_0] * 0.1 / (th - dR)  # 0~0.1
            out[index_1] = torch.sigmoid(k * (rain[index_1] - th))  # 0.1~0.9
            out[index_2] = 0.9 + (rain[index_2] - (th + dR)) * 0.1 / (th - dR)  # >1
            return out

        def inverse_transform_sigmoid_linear(y, th, dR):
            k = 2 / th * math.log(9)
            # k=torch.tensor([k])
            index0 = y < 0.1
            index09 = (y >= 0.1) & (y < 0.9)
            index1 = y >= 0.9
            out = torch.zeros(y.shape)
            if cuda:
                out = out.cuda()
            out[index0] = y[index0] * (th - dR) / 0.1  # 0~th-dR
            out[index09] = th + 1 / k * torch.log(y[index09] / (1 - y[index09]))
            out[index1] = (y[index1] - 0.9) * (th - dR) / 0.1 + th + dR  # th+dR~
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

        factor_stats=np.load('./params/factor_newALL_stats.npz',allow_pickle=True)
        #mean=feat_mean,std=feat_std,maxx=feat_max,minn=feat_min
        factor_mean=torch.from_numpy(factor_stats['mean']).float()
        factor_std=torch.from_numpy(factor_stats['std']).float()
        factor_max=torch.from_numpy(factor_stats['maxx']).float()
        factor_min=torch.from_numpy(train_stats['minn']).float()
        pdf_list = np.load('./params/PDF_0_60.npz')
        PDF_list = torch.from_numpy(pdf_list['p'])

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

        def mean_square_error(y, labels):
            mean_e = torch.sum((y - labels) ** 2)
            n = y.size(0)
            return mean_e, n

        def mae_error(y, labels):
            mean_e = torch.sum(abs(y - labels))
            n = y.size(0)
            return mean_e, n

        def bmae_error(y, labels):
            mean_e = torch.sum(abs(y - labels) * (labels + 1))
            n = (labels + 1).sum()
            return mean_e, n

        def p_sun_rain(labels, y, grade_list=0.1):

            return mem.pc_of_sun_rain(labels, y)

        def pod(labels, y, grade_list=[0.1]):

            return mem.pod(labels, y, grade_list=grade_list)

        def ts(labels, y, grade_list=[0.1]):

            return mem.ts(labels, y, grade_list=grade_list)

        def evaluate_3h_simple(y, labels, mask, criterion, inverse_transform, th, dR):

            y = y[mask].flatten()
            yy = inverse_transform(y, th, dR)
            labels = labels[mask].flatten()
            yy[yy < 0.1] = 0
            return criterion(yy, labels)

        def evaluate_simple(labels, y, mask, thrd, criterion, inverse_transform, th, dR):
            y = y[mask].flatten()
            yy = inverse_transform(y, th, dR)
            labels = labels[mask].flatten()
            yy[yy < 0.1] = 0
            return criterion(labels.detach().cpu().numpy(), yy.detach().cpu().numpy(), grade_list=[thrd])


        gpu = int(self.gpu) #gpu device
        #hyperparameters
        #c_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,40,41,42,43,44,45,46,47,48,49,50]
        c_id = [49, 50]
        if self.threshold == 3:
            c_id_add = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 26, 27, 28, 30, 31,
                        32, 33, 34, 40, 43, 44, 46, 47, 48]
            c_id_minus = [10, 11, 12, 13, 14, 31, 32, 36, 37, 51, 52, 55]
        else:
            c_id_add = [2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 34, 40, 41, 44, 47, 51, 52, 54, 56, 57]
            c_id_minus = [2, 3, 4, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 34, 51, 52, 53, 56, 57]
        #hyperparameters
        Error_mode = False  ##是否在投影模型最后一层加入输入，True：是，False：否
        InNow = False  ##特征提取部分是否输入现在的原始场
        if self.threshold == 3:
            num_feats = 0 + (35 + 2) + (12 + 2) + 7 + 7 + 7 # 1 #
            edge_feats1 = 0 + (35 + 2) + (12 + 2) + 7 + 7 + 7  #
        else:
            num_feats = 0 + (24 + 2) + (22 + 2) + 7 + 7 + 7  # 1 ##输入层特征维数 53维
            edge_feats1 = 0 + (24 + 2) + (22 + 2) + 7 + 7 + 7
        num_layers = 9  ##时间步长
        num_hidden = 256
        out_dim = 0
        early_stop = True
        fastmode = True

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

        model = STEIConvNetMScaleCompactGRAD_PROP(g,
                                                  num_layers,
                                                  3,
                                                  num_feats,
                                                  num_hidden,
                                                  out_dim,
                                                  edge_feats1,
                                                  activation_o=torch.tanh,
                                                  activation_edge=F.relu,  # torch.sigmoid,
                                                  num_filters=8,  ##filter个数
                                                  error_mode=Error_mode)  ##activation function for intermidiate layer

        models=[model]
        if cuda:
            model.cuda()

        print(model)
        if early_stop:
            stopper = EarlyStopping(patience=66)

        loss_fcnMSE = torch.nn.MSELoss(reduction='none')
        loss_fcnMAE = torch.nn.L1Loss(reduction='none')
        lossF_sun_rain = torch.nn.BCELoss(reduction='none')
        # lossF_sun_rain = torch.nn.BCEWithLogitsLoss(reduction='none')
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
        kkk = math.log(9)/self.wide

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
            nbatches = int(N_train/batchsize/1) #use 2/3 of the training data every epoch
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
                    rain00 = transform_sigmoid_linear(rain00_0, self.threshold, self.wide)  # transform_C10(rain00_0)
                    rainLC = transform_sigmoid_linear(rainLC_0, self.threshold, self.wide)  # transform_C10(rainLC_0)
                    # #
                    # rain0 = rain00[:,1:]-rain00[:,:-1]##原始预报3h降水量
                    labels = transform_sigmoid_linear(labels0, self.threshold, self.wide)  ####标签转换
                    mask = train_labels[index_n][1][:, :, 0:].reshape(69 * 73, -1).bool()
                    M3h = torch.sigmoid(100 * (labels0 - 0.1))
                    M3h_th = torch.sigmoid((labels0 - self.threshold) * kkk)  ##
                    efeat1 = train_features[index_n][:, :, :, :].reshape(69 * 73, 58, 9)  ##suoyou预报要素
                    efeat1N = (efeat1 - feat_mean.reshape(1, 58, 1)) / feat_std.reshape(1, 58, 1)  ##正态化
                    # efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/(feat_max[c_id]-feat_min[c_id]).reshape(1,len(c_id),1)##（-1，1）
                    d98 = train_features[index_n][:, :, [38], :] + train_features[index_n][:, :, [39], :]
                    d29 = train_features[index_n][:, :, [35], :] - train_features[index_n][:, :, [39], :]
                    pv98 = train_features[index_n][:, :, [33], :] + train_features[index_n][:, :, [34], :]
                    rh98 = train_features[index_n][:, :, [3], :] + train_features[index_n][:, :, [4], :]
                    rh57 = train_features[index_n][:, :, [1], :] + train_features[index_n][:, :, [2], :]
                    vv87 = train_features[index_n][:, :, [12], :] + train_features[index_n][:, :, [13], :]
                    vv57 = train_features[index_n][:, :, [11], :] + train_features[index_n][:, :, [12], :]
                    # wq_all=train_feats[index_n][:,:,[10,11,12,13],:]*train_feats[index_n][:,:,[5,6,7,8],:]            #wq_all=train_feats[index_n][:,:,[10,11,12,13],:]*train_feats[index_n][:,:,[5,6,7,8],:]
                    factor_new = torch.cat((d98, d29, pv98, rh98, rh57, vv87, vv57), 2).reshape(69 * 73, 7,
                                                                                                9)  # ,wq_all
                    factor_newN = (factor_new - factor_mean.reshape(1, 7, 1)) / factor_std.reshape(1, 7, 1)

                    # efeat1N[:,50,:]=rain00
                    # efeat1N[:,49,:]=rainLC

                    if cuda == True:
                        efeat1 = efeat1.cuda()
                        rain00 = rain00.cuda()
                        labels0 = labels0.cuda()
                        labels = labels.cuda()
                        mask = mask.cuda()
                        M3h = M3h.cuda()
                        M3hL = M3h_th.cuda()
                        # weight=weight.cuda()
                        # efeat_smooth=efeat_smooth.cuda()
                        efeat1N = efeat1N.cuda()
                        factor_newN = factor_newN.cuda()

                    efeat1N00 = efeat1N[:, :, [0]] + efeat1N[:, :, [0]] - efeat1N[:, :, [1]]  # 由于-1时刻没有，所以只能线性的推一下
                    efeat1N_past = torch.cat((efeat1N00, efeat1N[:, :, :-1]), 2)  ##前一时刻要素
                    factor_newN00 = factor_newN[:, :, [0]] + factor_newN[:, :, [0]] - factor_newN[:, :, [1]]
                    factor_newN_past = torch.cat((factor_newN00, factor_newN[:, :, :-1]), 2)
                    efeat1N_mean = efeat1N_past + efeat1N  ##相邻时次相加
                    efeat1N_minus = efeat1N - efeat1N_past  ##相邻时次相减
                    abs_efeat1N_minus = abs(efeat1N_minus)
                    factor_newN_mean = factor_newN_past + factor_newN
                    factor_newN_minus = -factor_newN_past + factor_newN
                    abs_factor_newN_minus = abs(factor_newN_minus)
                    K1_mean, K1_minus, Tse_mean, Tse_minus = K1_Tse(efeat1)
                    K1_mean /= 10
                    K1_minus /= 5
                    Tse_mean = (Tse_mean) / 10
                    Tse_minus /= 5
                    abs_K1_minus = abs(K1_minus)
                    # efeat_total = torch.cat((efeat1N[:,c_id,:],factor_newN,efeat1N_mean[:,c_id_add,:],factor_newN_mean
                    #                         ,efeat1N_minus[:,c_id_minus,:],factor_newN_minus),1)
                    efeat_total = torch.cat((factor_newN, efeat1N_mean[:, c_id_add, :], K1_mean.view(-1, 1, 9),
                                             Tse_mean.view(-1, 1, 9), factor_newN_mean
                                             , efeat1N_minus[:, c_id_minus, :], K1_minus.view(-1, 1, 9),
                                             Tse_minus.view(-1, 1, 9), factor_newN_minus),
                                            1)  # efeat1N[:,c_id,:], efeat1N[:,[49,50],:],
                    N_t = (mask).sum()
                    rain_forecast = model(efeat_total[:, :, 0], efeat_total[:, :, 0:], rain00)
                    rf = inverse_transform_sigmoid_linear(rain_forecast, self.threshold, self.wide)
                    RF_th = torch.sigmoid(( rf - self.threshold ) * kkk)

                    if cuda:
                        PDF_list = PDF_list.cuda()
                        # W_nodes_3h=W_nodes_3h.cuda()
                    W_nodes_3h = (PDF_list.mean() / PDF_near(labels0[mask]).flatten()) ** (
                                1 / 2)  # 反比于概率密度mask_pos*mask
                    W_nodes_3h[labels0[mask] == 0] = 1
                    # W_nodes_3h[labels0[mask]<th-dR] = 1/(1-freq01.item())
                    # W_nodes_3h[(labels0[mask]>=th-dR)] = 1/freq01.item()  # &(labels0[mask]<th+dR)
                    # W_nodes_3h[labels0[mask]>th+dR] = 1/(1-freq0_09.item())

                    # W_nodes_3h[labels0[mask]>10]=100
                    # W_nodes_3h[labels0[mask]>20]=1000

                    loss_pos = loss_fcnMAE(rain_forecast[mask].flatten(),
                                           labels[mask].flatten()) * W_nodes_3h  # [mask_pos*mask].flatten()
                    loss_dif = lossF_sun_rain(RF_th[mask].flatten(),
                                              M3h_th[mask].flatten()) * W_nodes_3h  # # # 晴雨分类损失
                    loss_w = 0.99 * loss_pos.mean() + 0.01 * loss_dif.mean()
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
                rain00T = transform_sigmoid_linear(rain00T_0, self.threshold, self.wide)  # transform_C10(rain00T_0)
                rainLCT = transform_sigmoid_linear(rainLCT_0, self.threshold, self.wide)  # transform_C10(rainLCT_0)
                # rain0 = rain00[:,1:]-rain00[:,:-1]##原始预报3h降水量
                # labelsT = transform_C20(labelsT0)
                maskT = test_labels[n][1][:, :, 0:].reshape(69 * 73, -1).bool()
                efeat1 = test_features[n][:, :, :, :].reshape(69 * 73, 58, 9)  ##预报要素
                efeat1N = (efeat1 - feat_mean.reshape(1, 58, 1)) / feat_std.reshape(1, 58, 1)  ##正态化
                d98 = test_features[n][:, :, [38], :] + test_features[n][:, :, [39], :]
                d29 = test_features[n][:, :, [35], :] - test_features[n][:, :, [39], :]
                pv98 = test_features[n][:, :, [33], :] + test_features[n][:, :, [34], :]
                rh98 = test_features[n][:, :, [3], :] + test_features[n][:, :, [4], :]
                rh57 = test_features[n][:, :, [1], :] + test_features[n][:, :, [2], :]
                vv87 = test_features[n][:, :, [12], :] + test_features[n][:, :, [13], :]
                vv57 = test_features[n][:, :, [11], :] + test_features[n][:, :, [12], :]
                # wq_all=train_feats[index_n][:,:,[10,11,12,13],:]*train_feats[index_n][:,:,[5,6,7,8],:]            #wq_all=train_feats[index_n][:,:,[10,11,12,13],:]*train_feats[index_n][:,:,[5,6,7,8],:]
                factor_new = torch.cat((d98, d29, pv98, rh98, rh57, vv87, vv57), 2).reshape(69 * 73, 7, 9)  # ,wq_all
                factor_newN = (factor_new - factor_mean.reshape(1, 7, 1)) / factor_std.reshape(1, 7, 1)
                # efeat1N = (efeat1-feat_mean[c_id].reshape(1,len(c_id),1))/(feat_max[c_id]-feat_min[c_id]).reshape(1,len(c_id),1)##（-1，1）
                # efeat1N[:,50,:]=rain00T
                # efeat1N[:,49,:]=rainLCT
                if cuda == True:
                    efeat1 = efeat1.cuda()
                    rain00T = rain00T.cuda()
                    # labelsT=labelsT.cuda()
                    maskT = maskT.cuda()
                    efeat1N = efeat1N.cuda()
                    factor_newN = factor_newN.cuda()

                efeat1N00 = efeat1N[:, :, [0]] + efeat1N[:, :, [0]] - efeat1N[:, :, [1]]  # 由于-1时刻没有，所以只能线性的推一下
                efeat1N_past = torch.cat((efeat1N00, efeat1N[:, :, :-1]), 2)  ##前一时刻要素
                factor_newN00 = factor_newN[:, :, [0]] + factor_newN[:, :, [0]] - factor_newN[:, :, [1]]
                factor_newN_past = torch.cat((factor_newN00, factor_newN[:, :, :-1]), 2)
                efeat1N_mean = efeat1N_past + efeat1N  ##相邻时次相加
                efeat1N_minus = efeat1N - efeat1N_past  ##相邻时次相减
                abs_efeat1N_minus = abs(efeat1N_minus)
                factor_newN_mean = factor_newN_past + factor_newN
                factor_newN_minus = -factor_newN_past + factor_newN
                abs_factor_newN_minus = abs(factor_newN_minus)
                K1_mean, K1_minus, Tse_mean, Tse_minus = K1_Tse(efeat1)
                K1_mean /= 10
                K1_minus /= 5
                Tse_mean = (Tse_mean) / 10  ###有问题
                Tse_minus /= 5
                abs_K1_minus = abs(K1_minus)
                # efeat_total = torch.cat((efeat1N[:,c_id,:],factor_newN,efeat1N_mean[:,c_id_add,:],factor_newN_mean
                #                         ,efeat1N_minus[:,c_id_minus,:],factor_newN_minus),1)
                efeat_total = torch.cat((factor_newN, efeat1N_mean[:, c_id_add, :], K1_mean.view(-1, 1, 9),
                                         Tse_mean.view(-1, 1, 9), factor_newN_mean
                                         , efeat1N_minus[:, c_id_minus, :], K1_minus.view(-1, 1, 9),
                                         Tse_minus.view(-1, 1, 9), factor_newN_minus),
                                        1)  # efeat1N[:,c_id,:],efeat1N[:,[49,50],:],

                # mask_xiaokong=xiaokong(efeat1)

                with torch.no_grad():
                    rain_forecastT = model(efeat_total[:, :, 0], efeat_total[:, :, 0:], rain00T)
                    #labelsT[labelsT0<10]=0
                if (epoch % self.show_performance) ==0:
                    rain_forecastALL = torch.cat((rain_forecastALL, rain_forecastT[maskT]))
                    rain00ALL = torch.cat((rain00ALL, rain00T[maskT]))
                    labelsALL = torch.cat((labelsALL, labelsT0[maskT]))
                    val_error0, N0 = evaluate_3h_simple(rain_forecastT,
                                                        labelsT0,
                                                        maskT,
                                                        bmae_error, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                    val_error += val_error0

                    N_total += N0
            if (epoch % self.show_performance) ==0:
                path1='./best_model/'
                mkdir(path1) 
                maskT = torch.ones(labelsALL.shape).bool()
                TSL = evaluate_simple(labelsALL, rain_forecastALL,
                                      maskT, 20,
                                      ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                TSL0 = evaluate_simple(labelsALL, rain00ALL,
                                       maskT, 20,
                                       ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                TSL = evaluate_simple(labelsALL, rain_forecastALL,
                                      maskT, 10,
                                      ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                TSL0 = evaluate_simple(labelsALL, rain00ALL,
                                       maskT, 10,
                                       ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                TSH = evaluate_simple(labelsALL, rain_forecastALL,
                                      maskT, 3,
                                      ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                TSH0 = evaluate_simple(labelsALL, rain00ALL,
                                       maskT, 3,
                                       ts, inverse_transform_sigmoid_linear, self.threshold, self.wide)
                #print ('晴雨准确率：',P_sun_rain)
                #print('TS(R> 0.1mm/3h):',TS)
                print('TS(R> 3mm/3h):',TSM)
                print('TS(R> 10mm/3h):',TSL)
                print('TS(R> 20mm/3h):',TSH)
                #print('订正前晴雨:',P_sun_rain0)
                #print('订正前TS(R> 0.1mm/3h):',TS0)
                print('订正前TS(R> 3mm/3h):',TSM0)
                print('订正前TS(R> 10mm/3h):',TSL0)
                print('订正前TS(R> 20mm/3h):',TSH0)

                Val_error=val_error/(N_total+1)
                Val_errorP=val_errorP/(N_total+1)
                Val_errorN=val_errorN/(N_total+1)  
                path=path1+'BestModel_level{:02d}mm.pt'.format(self.threshold)

                if early_stop:
                    if self.threshold == 3:
                        score = TSM
                    elif self.threshold == 10:
                        score = TSL
                    else:
                        score = TSH
                    if stopper.step(score, models, path):  #-Val_error*0.1+TS+TSM
                        break

                if epoch >= 3:
                    dur.append(time.time() - t0)
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.9f} |val_error{:.4f} ".
                          format(epoch, np.mean(dur), loss_m, Val_error))
        
            
            