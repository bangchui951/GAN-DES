from __future__ import print_function

import gc

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import argparse
import os
import torch
import pandas as pd
import numpy as np
from joblib import dump
from keras.src.utils import np_utils
from joblib import dump, load
from torch.nn.utils.spectral_norm import spectral_norm
from tensorflow.keras.models import model_from_json
from dynamic_layers_1 import MaskedConv2d, MaskedMLP
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, Normalizer
import torchvision.datasets as dset
import torchvision.transforms as transforms
# from rggan_model import RG_ResDiscriminator32, RG_ResGenerator32
from pytorch_fid import fid_score
from pytorch_image_generation_metrics import get_inception_score, get_fid,ImageDataset


from tqdm import tqdm
from utils import copy_params, load_params, sparsity_regularizer, print_layer_keep_ratio, set_training_mode
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

batch_size = 16

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            MaskedConv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            MaskedConv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            MaskedConv2d(in_channels, out_channels, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ViewG(nn.Module):
    def __init__(self, shape):
        super(ViewG, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

#生成器，使用全连接层和多个残差块生成图像。包含一个全连接层MaskedMLP,将噪声输入映射到一个初始的4*4*256的特征图，随后经过多个残差块，使用了MaskedConv2d进行卷积操作，最后
#输出经过激活函数Tanh,并且通过一个全连接层映射到目标维度为5的输出
class RG_ResGenerator32(nn.Module):
    def __init__(self, z_dim, sparse_train_mode=False):
        super().__init__()
        self.z_dim = z_dim
        self.linear = MaskedMLP(z_dim, 4 * 4 * 256)
        self.sparse_train_mode = sparse_train_mode
        self.blocks = nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            MaskedConv2d(256, 3, (3, 3), stride=1, padding=1),
            nn.Tanh(),
            ViewG((-1, 3*32*32)),
            nn.Linear(3 * 32 * 32, 5),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # self.set_training_mode()
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        z = self.output(self.blocks(z))
        return z


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            MaskedConv2d(in_channels, out_channels, (1, 1), 1, 0))
        self.residual = nn.Sequential(
            MaskedConv2d(in_channels, out_channels, (3, 3), 1, 1),
            nn.ReLU(),
            MaskedConv2d(out_channels, out_channels, (3, 3), 1, 1),
            nn.AvgPool2d(2))
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, MaskedConv2d):
                # init.xavier_uniform_(m.weight, math.sqrt(2))
                # init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, MaskedConv2d):
                # init.xavier_uniform_(m.weight)
                # init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        # print(self.residual(x))
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                MaskedConv2d(in_channels, out_channels, (1, 1), 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            MaskedConv2d(in_channels, out_channels, (3, 3), 1, 1),
            #nn.ReLU(),
            #MaskedConv2d(out_channels, out_channels, (3, 3), 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, MaskedConv2d):
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, MaskedConv2d):
                spectral_norm(m)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ViewD(nn.Module):
    def __init__(self, shape):
        super(ViewD, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

#判别器，使用多个优化的残差块和全连接层判断图像的真实性。
#包含多个残差块，使用了MaskedConv2d进行卷积操作，输入数据先经过一个全连接层MaskedMLP映射
#到3*32*32的特征图，然后经过一系列残差块和池化操作，最后通过一个全连接层输出真假的判断
class RG_ResDiscriminator32(nn.Module):
    def __init__(self, sparse_train_mode=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5,3*32*32),
            ViewD((-1, 3, 32, 32)),
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            #ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = MaskedMLP(128, 1, bias=False)
        self.initialize()
        self.sparse_train_mode = sparse_train_mode

    def initialize(self):
        spectral_norm(self.linear)

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = x.sum(dim=[2, 3])
        x = self.linear(x)
        return x
class TabularDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

#从 Excel 文件中加载数据，提取特征和标签，并对特征进行归一化。
def load_and_normalize_dataset(dataset_path):
    _dataset = pd.read_excel(dataset_path, engine='openpyxl')
    dataset = _dataset.values
    samples = dataset[:, [0, 1, 2, 3, 6]]
    labels = dataset[:, -1]  # 标签列

    scaler = MinMaxScaler()

    samples_normalized = scaler.fit_transform(samples)
    dump(scaler, 'scaler_attack.joblib')

    return samples_normalized, labels, scaler

#从数据集中提取一个批次的数据
def get_batch(samples, labels, batch_size, batch_idx):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos], labels[start_pos:end_pos]


def save_generated_data(fake_data, scaler, save_path):
    fake_data = scaler.inverse_transform(fake_data)
    df = pd.DataFrame(fake_data)
    df.to_csv(save_path, index=False)
    print(f'Generated data saved to {save_path}')
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("...........................................")
    # Load and preprocess dataset
    dataroot = 'spectre.xlsx'
    samples, labels, scaler = load_and_normalize_dataset(dataroot)


    # 加载数据并将数据集拆分为训练集和验证集。
    # Split dataset into training and validation sets
    validation_split = 0.1
    split_idx = int(len(samples) * (1 - validation_split))
    train_seqs, vali_seqs = samples[:split_idx], samples[split_idx:]
    train_targets, vali_targets = labels[:split_idx], labels[split_idx:]
    # 创建训练和验证的数据集及数据加载器
    # Create dataset and dataloader
    train_dataset = TabularDataset(train_seqs, train_targets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

    vali_dataset = TabularDataset(vali_seqs, vali_targets)
    vali_dataloader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers)

#初始化生成器和判别器模型，并将其移动到指定设备上
    netD = RG_ResDiscriminator32().to(device)
    netG = RG_ResGenerator32(args.noise_size).to(device)
#复制生成器的参数，用于计算移动平均
    netG_avg_param = copy_params(netG)
    netG.sparse_train_mode = True
    netD.sparse_train_mode = True
#设置生成器和判别器的稀疏训练模式
    set_training_mode(netG, netG.sparse_train_mode)
    set_training_mode(netD, netD.sparse_train_mode)
#为生成器和判别器设置Adam优化器
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))
#生成固定噪声，用于图像对比
#fixed_noise = torch.randn(64, args.noise_size, device=device)

    print("Starting Training Loop...")
    best_fid = 9999
    fid_record = []
#每5个epoch进行一次输出记录
#在每个epoch和每个批次地数据中，首先清零判别器的梯度
    for epoch in range(1, args.epoch + 1):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
#获取 real_cpu 的批次大小，即第一维的大小，通常代表当前输入的样本数量
            b_size = real_cpu.size(0)
#将真实数据输入判别器，计算输出
            output = netD(real_cpu).view(-1)
#计算判别器对真实数据的损失
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
#判别器对真实数据的损失计算，如果判别器处于稀疏训练模式，还会计算稀疏性正则化损失，并加到errD_real上
            if netD.sparse_train_mode:
                sr_loss = sparsity_regularizer(netD, args.lambda_)
                #print("sr_loss", sr_loss)
                errD_real = errD_real + sr_loss
#对真实数据的损失进行反向传播，计算判别器在真实数据上的输出均值
            #print("sr_loss", sr_loss)
            errD_real.backward()
            D_x = output.mean().item()
#对生成数据的损失计算
            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)
#detach()函数的作用是将一个张量从计算图中分离出来，使得在随后的计算中，这个张量的梯度不会被计算或存储，从而防止梯度回传到之前的节点
            # print(fake.shape)
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
#反向传播过程中，对梯度进行稀疏化处理，掩码与梯度相乘
            if netD.sparse_train_mode:
                for layer in netD.modules():
                    if isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedMLP):
                        try:
                            layer.weight_orig.grad.data = layer.weight_orig.grad.data * layer.mask
                        except:
                            layer.weight.grad.data = layer.weight.grad.data * layer.mask
#更新判别器的权重
            optimizerD.step()
#生成器训练步骤
            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(batch_size, args.noise_size, device=device)
               # print('noise shape', noise.shape)
                fake = netG(noise)
               # print("fake shape:", fake.shape)

                # #是否对生成的数据fake进行增强，使用指定的策略policy
                # if diffaug_flag:
                #     fake = DiffAugment(fake, policy=policy)
#将生成的假数据传入判别器，获取判别器对这些假数据的评分
                output = netD(fake).view(-1)
                # print(output.shape)
                # print(output)

 # 加载分类器模型用做预测

                json_file = open(r"model_test.json", "r")

                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)

                loaded_model.load_weights("model_test.h5")

                loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                fake_numpy = None
                # # 输出预测类别
                if isinstance(fake, torch.Tensor):
                    fake_numpy = fake.detach().cpu().numpy()  # 转换为 NumPy 数组
                # 创建一个新的数组，形状为 [batchsize, 7]
                new_array = np.zeros((batch_size, 7))

                # 计算新的第 4 列和第 5 列
                new_array[:, 4] = fake_numpy[:, 1] / fake_numpy[:, 0]  # 新的第 4 列
                new_array[:, 5] = fake_numpy[:, 3] / fake_numpy[:, 2]  # 新的第 5 列

                # 将原数组的前 3 列和最后 1 列复制到新数组中
                new_array[:, :4] = fake_numpy[:, :4]  # 前 3 列
                new_array[:, 6] = fake_numpy[:, 4]  # 保留原数组的最后一列
                predicted = loaded_model.predict(new_array)  # 返回对应概率值
                # predicted_labels = np.argmax(predicted, axis=1)
                aim = torch.zeros(batch_size, 2)
                aim[:, 1] = 0  #希望分类器将其判别为非攻击数据，也就是0的概率为1
                predicted = torch.from_numpy(predicted)
                # print(aim.shape)
                # print(predicted.shape)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(predicted, aim)
                print('loss', loss)
#往非攻击数据生成
                errG = -torch.mean(output) + 5*loss
#生成器的损失是对假数据评分的负值
#如果生成器处于稀疏训练模式下，生成器的损失还包括一个稀疏性正则化损失
                if netG.sparse_train_mode:
                    sr_loss = sparsity_regularizer(netG, args.lambda_)
                    errG = errG + sr_loss
#对生成器损失进行反向传播，并计算判别器的平均输出---》2个作用  1.监控生成器的训练情况，较高的D_G_z2表示生成器生成的数据更像真实数据
                errG.backward()
                D_G_z2 = output.mean().item()
#是否进行稀疏化处理
                if netG.sparse_train_mode:
                    for layer in netG.modules():

                        if isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedMLP):
                            try:
                                layer.weight_orig.grad.data = layer.weight_orig.grad.data * layer.mask
                            except:
                                layer.weight.grad.data = layer.weight.grad.data * layer.mask
#更新权重
                optimizerG.step()
#更新生成器参数的移动平均，以稳定训练
                # moving average weight
                for p, avg_p in zip(netG.parameters(), netG_avg_param):
                    avg_p.mul_(0.999).add_(0.001, p.data)

            # Output training stats
            #每隔50次训练步骤，输出训练状态，包括判别器和生成器的损失及其他指标
            if i % 50 == 0:
                print('[%4d/%4d][%3d/%3d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                torch.save(netG, 'G_attack.pth')

        # Evaluation  模型评估
        if epoch % args.eva_epoch == 0:
            print('Evaluation')

            if netD.sparse_train_mode:
                d_current_keep_ratio, d_layer_keep_ratio = print_layer_keep_ratio(netD)
                print('D keep ratio: %.4f' % d_current_keep_ratio)
                with open('%s/D_keep_ratio_lambda_%s.txt' % (current_model_result_dir, str(args.lambda_)), 'a') as f:
                    f.write('epoch: %d' % epoch + '\n')
                    for x in d_layer_keep_ratio:
                        f.write(x + '\n')
                    f.write('Overall:' + str(d_current_keep_ratio.item()) + '\n')
                    f.write('\n')
            if netG.sparse_train_mode:
                print('--------------')
                g_current_keep_ratio, g_layer_keep_ratio = print_layer_keep_ratio(netG)
                print('G keep ratio: %.4f' % g_current_keep_ratio)
                with open('%s/G_keep_ratio_lambda_%s.txt' % (current_model_result_dir, str(args.lambda_)), 'a') as f:
                    f.write('epoch: %d' % epoch + '\n')
                    for x in g_layer_keep_ratio:
                        f.write(x + '\n')
                    f.write('Overall:' + str(g_current_keep_ratio.item()) + '\n')
                    f.write('\n')
#保存生成器参数，是为了在评估阶段能够恢复到训练前的状态
            backup_param = copy_params(netG)
            load_params(netG, netG_avg_param)
            netG.eval()

            Noisee = torch.randn(batch_size, args.noise_size, device=device)
            temp_fake = netG(Noisee)

            temp = None
            for i, data in enumerate(dataloader, 0):
                netD.zero_grad()
                real_cpu = data[0].to(device)

                # if diffaug_flag:
                #     real_cpu = DiffAugment(real_cpu, policy=policy)
                temp = real_cpu
                break

            def calculate_mse(real_sequences, generated_sequences):
                return F.mse_loss(generated_sequences, real_sequences)
            # print(temp)
            mse = calculate_mse(temp, temp_fake)
            print(f'MSE: {mse.item()}')

            fid_record.append(mse)

            load_params(netG, backup_param)

            # avg_netG = deepcopy(netG)
            # load_params(avg_netG, netG_avg_param)
#通过调整训练策略来提高生成器生成数据的质量
            if len(fid_record) >= 5:
                print(fid_record[-5], fid_record[-4], fid_record[-3], fid_record[-2], fid_record[-1])
                average_fid = 0.1 * fid_record[-5] + 0.1 * fid_record[-4] + 0.2 * fid_record[-3] + \
                              0.2 * fid_record[-2] + 0.4 * fid_record[-1]
                print(average_fid)

                if average_fid >= mse:
                    netG.sparse_train_mode = True
                    netD.sparse_train_mode = True
                else:
                    netG.sparse_train_mode = False
                    netD.sparse_train_mode = False
                set_training_mode(netG, netG.sparse_train_mode)
                set_training_mode(netD, netD.sparse_train_mode)

            netG.train()
        del fake_numpy, new_array, predicted, aim
    gc.collect()

if __name__ == '__main__':
    model_name = 'RG-SNGAN'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--lr', type=float, default=0.001)#256 0.001  92.6%  (目前最好的情况)G(删除了两个网络_256_0.001)
    argparser.add_argument('--workers', type=int, default=0)

    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='./dataset')
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--lambda_', type=float, default=1e-12)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--eva_size', type=int, default=10000)
    argparser.add_argument('--eva_epoch', type=int, default=1)
    argparser.add_argument('--diffaug', action='store_true', help='apply DiffAug')

    args = argparser.parse_args()

    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)

    current_model_result_dir = './%s/result' % model_name

    if not os.path.exists(current_model_result_dir):
        os.makedirs(current_model_result_dir)

    current_model_eva_dir = './%s/eva' % model_name

    if not os.path.exists(current_model_eva_dir):
        os.makedirs(current_model_eva_dir)

    device = "cuda"
    main()
#dataroot = '../data/combined.xlsx'
#samples, labels, scaler = load_and_normalize_dataset(dataroot)