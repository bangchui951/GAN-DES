## This page is based on https://github.com/junjieliu2910/DynamicSparseTraining

import torch
import torch.nn as nn

"""
Function for activation binarization
"""
#输入的值转换为 0 或 1
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2 - 4 * torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input * additional

#reset_parameters() 方法用于初始化权重和偏置。使用 Xavier 均匀分布初始化权重，如果有偏置项则初始化为 0。同时，阈值参数也被初始化为 0。
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size, bias=True, sparse_train=False):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_size))
        else:
            self.register_parameter('bias', None)

        self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.step = BinaryStep.apply
        self.mask = None
        self.sparse_train = sparse_train
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)
        #使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform(self.weight.data, 1.)
        #如果有偏置项，则将其初始化为 0
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        #将阈值初始化为 0
        with torch.no_grad():
            # std = self.weight.std()
            self.threshold.data.fill_(0)
#定义前向传播方法
    def forward(self, input):
        if self.sparse_train:
            # print('sparse training')
            abs_weight = torch.abs(self.weight)
            #调整阈值的形状与权重形状相同
            threshold = self.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight - threshold
            mask = self.step(abs_weight)
            #计算掩码中保留的权重比例
            ratio = torch.sum(mask) / mask.numel()
            # print("keep ratio {:.2f}".format(ratio))
            #如果保留的权重比例过低，则重置阈值并重新计算掩码
            if ratio <= 0.01:
                with torch.no_grad():
                    # std = self.weight.std()
                    #确保掩码计算时不再有任何额外的阈值限制
                    #防止模型变得过于稀疏
                    self.threshold.data.fill_(0)
                abs_weight = torch.abs(self.weight)
                threshold = self.threshold.view(abs_weight.shape[0], -1)
                abs_weight = abs_weight - threshold
                mask = self.step(abs_weight)
            self.mask = mask.bool()
#如果未启用稀疏训练，则掩码 mask 设为全 1，表示所有权重都有效。
        else:
            # print('dense training')
            self.mask = torch.ones_like(self.weight).bool()
        # masked_weight = self.weight * mask
        output = torch.nn.functional.linear(input, self.weight * self.mask, self.bias)
        return output


class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, sparse_train=False):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ## define weight
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_c))
        self.step = BinaryStep.apply
        self.mask = None
        self.sparse_train = sparse_train
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)
        nn.init.xavier_uniform(self.weight.data, 1.)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        # print(self.sparse_train)
        if self.sparse_train:
            # print('sparse training')
            weight_shape = self.weight.shape
            threshold = self.threshold.view(weight_shape[0], -1)
            weight = torch.abs(self.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = self.step(weight)
            mask = mask.view(weight_shape)
            ratio = torch.sum(mask) / mask.numel()
            # print("threshold {:3f}".format(self.threshold[0]))
            # print("keep ratio {:.2f}".format(ratio))
            if ratio <= 0.01:
                with torch.no_grad():
                    self.threshold.data.fill_(0.)
                threshold = self.threshold.view(weight_shape[0], -1)
                weight = torch.abs(self.weight)
                weight = weight.view(weight_shape[0], -1)
                weight = weight - threshold
                mask = self.step(weight)
                mask = mask.view(weight_shape)
            self.mask = mask.bool()
        # masked_weight = self.weight * mask
        else:
            # print('dense training')
            self.mask = torch.ones_like(self.weight).bool()
        # self.weight.retain_grad()
        # print(self.mask.shape)
        conv_out = torch.nn.functional.conv2d(x, self.weight * self.mask, bias=self.bias, stride=self.stride,
                                              padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out
