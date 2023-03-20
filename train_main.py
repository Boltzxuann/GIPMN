import numpy as np
import torch
   
if __name__ == "__main__":
    from mymodel import model
    #必 需 的 参 数 有4个 ：
    #1.最 大 迭 代 周 期 数 ；
    #2.使 用 的GPU数 目(num_gpus<=4), 若 不 用GPU，num_gpus=0;
    #3.每 个 周 期 内 输 出 训 练 损 失 的 迭 代 数 间 隔 ；
    #4.输 出 验 证 集 表 现 的 周 期 间 隔 。
    #可 以 在 参 数 字 典 中 加 入 其 它 参 数 ， 如 学 习 率lr等 。
    args = dict()
    args['max_epochs'] = 1000
    args['num_gpus'] = 1
    args['show_trainloss_every_num_iterations_per_epoch'] = 200
    args['show_validperformance_every_num_epochs'] = 1
    args['num_examples'] = 10#1600
    model_inst = model('train_data', **args)
    #输 出train_data中 作 为 验 证 样 本 的 序 号
    #print(model_inst.valid_example)
    #输 出 训 练 损 失 函 数 和 验 证 评 价 函 数 名 称 ，
    #验 证 评 价 函 数 不 一 定 需 要 和 比 赛 评 价 指 标 一 致 。
    #print(model_inst.train_loss, model_inst.valid_perfomace)
    #开 始 训 练
    model_inst.train()
            
        
  
    