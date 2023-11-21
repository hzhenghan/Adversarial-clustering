import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import math
import warnings
import scipy.io as io
from utils.return_dataset_res import return_dataset_res
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=700, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=1600, help="size of the batches")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--variable_num", type=int, default=64, help="train data second dimension")
parser.add_argument('--manualSeed', type=int, default=36, help='manual seed')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False


class Domain_Generator(nn.Module):
    def __init__(self):
        super(Domain_Generator, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('e_conv1', nn.Linear(64, 128))
        self.encoder.add_module('e_relu1', nn.ReLU(True))
        self.encoder.add_module('e_conv2', nn.Linear(128, 256))
        self.encoder.add_module('e_relu2', nn.ReLU(True))
        self.encoder.add_module('e_conv3', nn.Linear(256, 128))
        self.encoder.add_module('e_relu3', nn.ReLU(True))
        self.encoder.add_module('e_conv4', nn.Linear(128, 64))
        self.encoder.add_module('e_relu4', nn.Sigmoid())

    def forward(self, input_data):
        output = self.encoder(input_data)
        return output

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(0.5)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

generator = Domain_Generator().cuda()
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
loss_MSE = torch.nn.MSELoss().cuda()
generator.apply(weights_init)

def train(train_data_normal_loader, fault_data_fault_loader):
    generator.train()
    for epoch in range(opt.n_epochs):
        for i, (inputs, labels) in enumerate(train_data_normal_loader, 0):
            optimizer.zero_grad()
            input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
            class_label = torch.LongTensor(opt.batch_size)
            print(input_sample.shape)
            # input_sample = inputs
            # class_label = labels
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                input_sample = input_sample.cuda()
                class_label = class_label.cuda()

            input_sample.resize_as_(inputs).copy_(inputs)
            class_label.resize_as_(labels).copy_(labels)

            out = generator(input_sample)
            loss = loss_MSE(out, input_sample)

            loss.backward()
            optimizer.step()

            print(loss)
            generator.zero_grad()
    # torch.save(generator.state_dict(), "model/checkpath/generator.pth.tar")

    generator.eval()

    for i, (samples, label) in enumerate(fault_data_fault_loader):
        input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
        class_label = torch.LongTensor(opt.batch_size)
        if cuda:
            samples = samples.cuda()
            label = label.cuda()
            input_sample = input_sample.cuda()
            class_label = class_label.cuda()

        input_sample.resize_as_(samples).copy_(samples)
        class_label.resize_as_(label).copy_(label)
        out = generator(input_sample)

    # torch.save(generator.state_dict(), "path.pkl")
    residual = input_sample - out
    class_label = class_label.view(-1, 1)
    out = torch.cat((residual, class_label.float()), 1)
    out = np.array(out.cpu().detach())
    # print(out[0])
    # np.save('out1_data.npy', out)
    return out

if __name__ == '__main__':

    train_data_normal_i,  fault_data_fault_i, train_data_normal = return_dataset_res()

    train_data_normal_loader = DataLoader(train_data_normal_i, batch_size=64, shuffle=False,
                                          num_workers=0)
    # test_data_normal_loader = DataLoader(test_data_normal, batch_size=30, shuffle=False, num_workers=0)
    fault_data_fault_loader = DataLoader(fault_data_fault_i, batch_size=len(fault_data_fault_i), shuffle=False,
                                         num_workers=0)
    fault_fault_data = train(train_data_normal_loader, fault_data_fault_loader)
    data_f = pd.DataFrame(np.concatenate((fault_fault_data, train_data_normal.values)), columns=train_data_normal.columns)

    data_f.to_csv('data_chiller/data_res_feature/fault_data_45.csv')

