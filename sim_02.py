# -- coding: utf-8 --
"""
Created on Wed Dec  8 16:57:18 2021

@author: tzvis
"""

import torch
from PIL import Image
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F


class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


net = models.resnet50(pretrained=True)
net = net.cuda()
im = Image.open("C:/Users/rache/Desktop/09 these/dlwpt-code-master/data/p1ch2/bobby.jpg")


def main(im, net):
    tran = transforms.Compose([
        transforms.ToTensor()
    ])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )])
    sign = LBSign.apply
    Img   = tran(im)
    r_img = torch.unsqueeze(preprocess(Img), 0).cuda()
    score_r = torch.nn.functional.softmax(net(r_img), dim=1)[0]
    # camera_spec
    im_s = im.size
    N_r = im_s[0] * 0 + 720
    t_read = 33 * 10 ** -6  # sec
    t_exp = 37 * t_read  # sec
    #
    mat = scipy.io.loadmat('C:/Users/rache/Desktop/09 these/matlab code/simulation/laser_pulse.mat')
    laser_puls = list(mat.items())
    laser_puls = laser_puls[3]
    laser_puls = laser_puls[1]
    p_laser_puls = Image.fromarray(laser_puls.astype('uint8'), 'RGB')
    p_laser_puls = tran(p_laser_puls)
    red_laser = torch.nn.Parameter(torch.randn(int(N_r + (t_exp / t_read)))-2) #.to(torch.device('cuda'))
    red_laser.requires_grad = True
    optimizer = torch.optim.SGD([red_laser],lr=0.1)
    l_t = np.arange(1, 1000)
    for t in l_t:
        red_laser_eff = (sign(red_laser) + 1) / 2
        l = shutter(red_laser_eff, int(torch.round(N_r*torch.rand(1))))  # (t_exp/t_read)
        lp = p_laser_puls * l.view(N_r, 1)
        p_image = torch.unsqueeze(preprocess(Img + lp), 0).cuda
        net.eval()
        score = torch.nn.functional.softmax(net(p_image), dim=1)[0]
        loss_f = torch.nn.CrossEntropyLoss()
        loss = 10000 * (loss_f(score.view(1, 1000), torch.argmax(score_r).view(-1)) - 6.9)
        obj = sum(l) - loss
        obj.backward()
        optimizer.step()
        optimizer.zero_grad()

        # percentage_np = score.detach().numpy()
        # plt.plot(percentage_np.reshape(1000,1))
        #plt.imshow((lp).detach().permute(1, 2, 0))


def shutter(f, ran):
    t_read = 33 * 10 ** -6  # sec
    t_exp = 37 * t_read  # sec
    N_r = 720  # Line
    h_N_r = int(130)
    read_R = torch.arange(1., N_r + 1)
    read_R = read_R - 1
    read_R = read_R * t_read
    exp_R = read_R + t_read + t_exp
    read_R = torch.cat((torch.cat((torch.zeros(h_N_r) ,read_R),0),torch.zeros(h_N_r)),0)
    exp_R = torch.cat((torch.cat((torch.zeros(h_N_r) ,exp_R),0),torch.zeros(h_N_r)),0)
    l = torch.zeros(h_N_r+N_r+h_N_r)
    f_phase = torch.zeros(len(f))
    f_phase[0:ran] = f[(len(f) - ran):len(f)]
    f_phase[ran + 1:len(f)] = f[1:len(f) - ran]
    l_t = np.arange(1, (N_r + (t_exp / t_read)))
    for t in l_t:
        on = (exp_R > (t * t_read)) * (read_R < (t * t_read))
        l = l + (on * f_phase[int(t)])

    l[l > 1] = 1
    sig = 7
    h = (torch.exp(-torch.arange(-18,18)*2) / (2 * sig*2 )) / (np.sqrt(2 * torch.pi) * sig)
    conv = torch.nn.Conv1d(1, 1, 36)
    conv.weight = torch.nn.Parameter(h.view(1, 1, 36))
    l = conv(l.view(1,1,980)).view(945)
    m = 1/(max(l)-min(l))
    n = -m*min(l)
    l = m*l+n
    return l[94:814]


main(im, net)