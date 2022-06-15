import torch
import torch.nn as nn
import numpy as np

from torch.nn import Conv2d as Conv2d

class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
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
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], opt=None):
        super(_netlocalD, self).__init__()
        self.gpu_ids = gpu_ids
        self.opt = opt

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
        self.model_local = nn.Sequential(# input is (nc) x 64 x 64
            Conv2d(input_nc + 1, ndf, 5, 2, 1, bias=False),
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
        self.l_fc = nn.Linear(512 * 3 * 3, 1024)


        # MEx loss concatenate
        if self.opt.MultiExpanTimes >= 1:
            self.fc = nn.Sequential(
                nn.Linear(1024*(1 + self.opt.MultiExpanTimes), 1),
                # nn.Linear(1024 * (3), 1),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                # nn.Linear(1024*(1 + self.opt.MultiExpanTimes), 1),
                nn.Linear(1024 * (2), 1),
                nn.Sigmoid(),
            )

    def mask_variable(self, input, mask):
        if not(input.dim()==mask.dim()):
            mask = mask.unsqueeze(1)
        output = input * mask.repeat(1, input.size(1), 1, 1)
        return output

    def forward(self, input_global, gt_global, mask_tensor, MultiExpanTimes=3):
        out_global = self.model_global(input_global)
        (batch_size, C, H, W) = out_global.data.size()
        out_global = out_global.view(-1, C * H * W)
        # fc_global = nn.Linear(H * W * C, 1024).cuda()
        out_global = self.g_fc(out_global)

        y_max, x_max = mask_tensor.shape[2], mask_tensor.shape[3]

        assert MultiExpanTimes > 0

        mask_in_MEx_set = []
        for i in range(MultiExpanTimes):
            mask_in_MEx_set.append(torch.zeros_like(mask_tensor))

        for i in range(batch_size) :
            one = mask_tensor[i, :, :, :]
            loc = np.argwhere(np.asarray(1 - mask_tensor[i, :, :, :]) > 0)

            ( _, ystart, xstart) = loc.min(0)
            ( _, ystop, xstop) = loc.max(0) + 1


            # MEx loss extracting multiple parts
            for m in range(MultiExpanTimes):
                x_min_chosen = xstart - i * self.opt.MultiExpanRadius
                if x_min_chosen < 0:
                    x_min_chosen = 0
                x_max_chosen = xstop + i * self.opt.MultiExpanRadius
                if x_max_chosen >= x_max:
                    x_max_chosen = x_max
                y_min_chosen = ystart - i * self.opt.MultiExpanRadius
                if y_min_chosen < 0:
                    y_min_chosen = 0
                y_max_chosen = ystop + i * self.opt.MultiExpanRadius
                if y_max_chosen >= y_max:
                    y_max_chosen = y_max
                mask_in_MEx_set[m][i, :, y_min_chosen:y_max_chosen, x_min_chosen:x_max_chosen] = 1

        reverse_mask_in = torch.ones_like(mask_tensor).cuda() - mask_tensor
        comb_recon_prob_plus_gen = self.mask_variable(input_global, reverse_mask_in) + self.mask_variable(
            gt_global, mask_tensor)


        for m in range(MultiExpanTimes):
            batch_local = self.mask_variable(comb_recon_prob_plus_gen, mask_in_MEx_set[m].cuda())
            local_input = torch.cat((batch_local, mask_in_MEx_set[m]), 1)
            out_utput = self.model_local(local_input)
            (_, C, H, W) = out_utput.data.size()
            out_utput = out_utput.view(-1, C * H * W)
            # fc_local = self.l_fc().cuda()
            local_loss = self.l_fc(out_utput)

            if m == 0:
                out_local = local_loss
            else:
                out_local = torch.cat((out_local, local_loss), 1)
        out = torch.cat((out_global, out_local), -1)

        #     if m == 0:
        #         out_local = local_loss
        #     elif m == 1:
        #         out_midview = local_loss
        #     elif m > 1:
        #         out_midview = torch.cat((out_midview, local_loss), 1)
        # if MultiExpanTimes >= 1:
        #     out_midview = self.mid_fc(out_midview)
        #     out = torch.cat((out_global, out_local, out_midview), -1)
        # else:
        #     out = torch.cat((out_global, out_local), -1)

        out = self.fc(out)
        return out
