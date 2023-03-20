###读取各种数据(模式预报、格点实况)的函数库--PX###
import xarray as xr
import numpy as np
import os

class ReadingFunctions:
    

    def read_situation_field_predi_test(example,step):  ##读取预报物理量要素

        data_dir = example +'/grid_inputs_%02d.nc'%(step)
        if os.path.exists(data_dir):
            print('read',data_dir)
            dset=xr.open_dataset(data_dir)
            vars_name = [name for name in dset]
            
            #print(vars_name)

            value_all = dset.data_vars[vars_name[0]]
            for name in vars_name[1:]:
                value0=dset.data_vars[name].values
                if value0.ndim==2:
                    value0=value0.reshape(value0.shape[0],value0.shape[1],1)
                value_all = np.concatenate((value_all,value0),axis=2) ##把多层h要素拼接成一个向量
                      
            return value_all
        else:
            return 'empty'

    def read_situation_field_predi_train(example,step):  ##读取预报物理量要素

        data_dir = example+'/grid_inputs_%02d.nc'%(step)
        if os.path.exists(data_dir):
            print('read',data_dir)
            dset=xr.open_dataset(data_dir)
            vars_name = [name for name in dset]
            
            #print(vars_name)

            value_all = dset.data_vars[vars_name[0]]
            for name in vars_name[1:]:
                value0=dset.data_vars[name].values
                if value0.ndim==2:
                    value0=value0.reshape(value0.shape[0],value0.shape[1],1)
                value_all = np.concatenate((value_all,value0),axis=2) ##把多层h要素拼接成一个向量
                      
            return value_all
        else:
            return 'empty'
                                           
    def read_obs_grid_train(example,step):  ##读取格点真值

        data_dir = example+'/obs_grid_rain%02d.nc'%(step)
        if os.path.exists(data_dir):
            print('read',data_dir)
            dset=xr.open_dataset(data_dir)
            vars_name = [name for name in dset]
            """看 变 量 名
            print(vars_name)
            """
            value_all = dset.data_vars[vars_name[0]]
            mask=np.ones(value_all.shape)
            mask[value_all<-1000]=0
                      
            return value_all, mask
        else:
            return 'empty'

        
        
    def read_station_coords(example,step): ##读取站点经纬度文件。
        data_dir = example+'/ji_loc_inputs_%02d.txt'%(step)
        if os.path.exists(data_dir):
            print('read',data_dir)
     
            station_coords = np.loadtxt(data_dir,dtype=int)
        else:
            station_coords = "empty"
        return station_coords


        