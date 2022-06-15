import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv2d as Conv2d
#
class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.model = nn.Sequential(
            # conv1
            Conv2d(opt.nc+1,opt.nef,5,1,1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.ReLU(),
            # conv2
            Conv2d(opt.nef,opt.nef*2,3,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*2),
            nn.ReLU(),
            Conv2d(opt.nef*2, opt.nef * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 2),
            nn.ReLU(),
            # conv3-dilate-conv
            Conv2d(opt.nef*2, opt.nef * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),

            Conv2d(opt.nef * 4, opt.nef * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, kernel_size=3, dilation=2,  stride= 1, padding=2),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, kernel_size=3, dilation=4, stride=1, padding=4),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, kernel_size=3, dilation=8, stride=1, padding=8),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, kernel_size=3, dilation=16, stride=1, padding=16),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),

            Conv2d(opt.nef * 4, opt.nef * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),
            Conv2d(opt.nef * 4, opt.nef * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(),

            # deconv
            nn.ConvTranspose2d(opt.nef * 4, opt.nef * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(opt.nef * 2),
            nn.ReLU(),
            Conv2d(opt.nef * 2, opt.nef * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(opt.nef * 2, opt.nef, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(opt.nef),
            nn.ReLU(),
            Conv2d(opt.nef, opt.nef // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.nef //2),
            nn.ReLU(),

            Conv2d(opt.nef // 2, opt.nc , 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.model(input)
        mask = input[:, 3]
        mask = torch.unsqueeze(mask, 1)
        input_image = input[:, :3]
        out = (1 - mask) * output + mask * input_image
        return out


class _netlocalD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, mode = 'GL', norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(_netlocalD, self).__init__()
        self.gpu_ids = gpu_ids
        self.mode = mode

        if 'G' in mode:
            self.model_global = nn.Sequential(
                Conv2d(input_nc, ndf, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.ReLU(),

                Conv2d(ndf, ndf*2, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.ReLU(),

                Conv2d(ndf*2, ndf*4, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.ReLU(),

                Conv2d(ndf * 4, ndf * 8, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.ReLU(),

                Conv2d(ndf * 8, ndf * 8, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.ReLU(),

                Conv2d(ndf * 8, ndf * 8, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.ReLU()
            )

        if 'L' in mode:
            self.model_local = nn.Sequential(# input is (nc) x 64 x 64
                Conv2d(input_nc, ndf, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.ReLU(),

                Conv2d(ndf, ndf * 2, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.ReLU(),

                Conv2d(ndf * 2, ndf * 4, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.ReLU(),

                Conv2d(ndf * 4, ndf * 8, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.ReLU(),

                Conv2d(ndf * 8, ndf * 8, 5, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.ReLU(),
            )

        self.g_fc = nn.Linear(512, 1024)
        self.l_fc = nn.Linear(512, 1024)

        if 'GL' in mode:
            self.fc = nn.Sequential(
                nn.Linear(1024*2, 1),
                nn.Sigmoid(),
            )
        elif 'G' in mode or 'L' in mode:
            self.fc = nn.Sequential(
                nn.Linear(1024, 1),
                nn.Sigmoid(),
            )

    def forward(self, input_global, mask_tensor, MultiExpanTimes=0):
        if 'G' in self.mode:
            out_global = self.model_global(input_global)
            (batch_size, C, H, W) = out_global.data.size()
            out_global = out_global.view(-1, C * H * W)
            # fc_global = nn.Linear(H * W * C, 1024).cuda()
            out_global = self.g_fc(out_global)

        # y_max, x_max = mask_tensor.shape[2], mask_tensor.shape[3]
        if 'L' in self.mode:
            batch_size = input_global.size(0)
            for i in range(batch_size):
                one = mask_tensor[i, :, :, :]
                loc = np.argwhere(np.asarray(1 - mask_tensor[i, :, :, :]) > 0)

                ( _, ystart, xstart) = loc.min(0)
                ( _, ystop, xstop) = loc.max(0) + 1

                if MultiExpanTimes == 0:
                    crop_local = torch.unsqueeze(input_global[i, :, ystart:ystop, xstart:xstop], 0)
                    if i == 0 :
                        batch_local= crop_local
                    else :
                        batch_local = torch.cat((batch_local, crop_local), 0)

            out_local = self.model_local(batch_local)
            (_, C, H, W) = out_local.data.size()
            out_local = out_local.view(-1, C * H * W)
            # print(C*W*H)
            # fc_local = nn.Linear(C * H * W, 1024).cuda()
            out_local = self.l_fc(out_local)
        if 'GL' in self.mode:
            out = torch.cat((out_global, out_local), -1)
        elif 'G' in self.mode:
            out = out_global
        elif 'L' in self.mode:
            out = out_local


        out = self.fc(out)
        return out
