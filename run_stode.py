import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from model import ODEGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

# ##on test data## rmse loss: 32.76171330818068, mae loss: 21.16639387664072, mape loss: 15.385958266342985

# 设置图片存储路径
figure_path = './result/figure/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

# loader 为数据加载器，model 为模型，optimizer 为优化器，criterion 为损失函数，device 为设备
def train(loader, model, optimizer, criterion, device):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item()
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)

        out_unnorm = output.detach().cpu().numpy() * std + mean
        target_unnorm = targets.detach().cpu().numpy() * std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)

def plot_loss(train_loss):
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.savefig(os.path.join(figure_path, 'loss.png'))

def plot_mae_loss(train_mae, valid_mae):
    plt.figure()
    plt.plot(train_mae, label='train mae')
    plt.plot(valid_mae, label='valid mae')
    plt.legend()
    plt.savefig(os.path.join(figure_path, 'mae.png'))

def main(args):
    # random seed
    # 设置随机数种子
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # 检查是否支持GPU，如果支持则使用GPU，否则使用CPU
    device = torch.device('cuda:' + str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    # 如果参数中包含日志选项，则设置logger
    if args.log:
        logger.add('log_{time}.log')
    # 将参数保存到options变量中
    options = vars(args)
    # 如果有日志，则记录options
    if args.log:
        logger.info(options)
    # 如果没有日志，则打印options
    else:
        print(options)

    # 读取数据并计算归一化参数
    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    # 生成训练集、验证集和测试集
    train_loader, valid_loader, test_loader = generate_dataset(data, args)
    # 计算归一化后的邻接矩阵
    A_sp_wave = get_normalized_adj(sp_matrix).to(device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(device)

    # 创建模型
    # num_nodes: 节点数
    net = ODEGCN(num_nodes=data.shape[1],
                 # num_features: 特征数
                 num_features=data.shape[2],
                 # num_timesteps_input: 输入时间步
                 num_timesteps_input=args.his_length,
                 # num_timesteps_output: 输出时间步
                 num_timesteps_output=args.pred_length,
                 # A_sp_hat: 归一化后的空间邻接矩阵
                 A_sp_hat=A_sp_wave,
                 # A_se_hat: 归一化后的时间邻接矩阵
                 A_se_hat=A_se_wave)
    # 将模型放到设备上
    net = net.to(device)
    # 设置学习率
    lr = args.lr
    # 设置优化器为 AdamW
    # net.parameters(): 需要优化的模型参数
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # 设置损失函数为平滑L1损失
    criterion = nn.SmoothL1Loss()

    # 将初始最佳验证集RMSE值设为1000,是为了在第一次训练结束后一定会更新这个值
    best_valid_rmse = 1000
    # 设置学习率调度器为StepLR，每50个epoch将学习率降低为原来的一半
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    train_loss = []
    train_mae_loss = []
    valid_mae_loss = []
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        loss = train(train_loader, net, optimizer, criterion, device)
        scheduler.step()
        train_loss.append(loss)

        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, device)

        train_mae_loss.append(train_mae)
        valid_mae_loss.append(valid_mae)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

        if args.log:
            logger.info(f'\n##on train data## loss: {loss}, \n' +
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        else:
            print(f'\n##on train data## loss: {loss}, \n' +
                  f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                  f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')

        scheduler.step()

    print('Testing...')
    # 加载最佳模型
    net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
    test_rmse, test_mae, test_mape = eval(test_loader, net, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')

    # 画图
    plot_loss(train_loss)
    plot_mae_loss(train_mae_loss, valid_mae_loss)

if __name__ == '__main__':
    main(args)
