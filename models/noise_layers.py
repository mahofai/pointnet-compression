from typing import OrderedDict
import torch
import torch.nn as nn


class NoiseModule(nn.Module):
    def __init__(self):
        super(NoiseModule, self).__init__()

    def add_noise(self, weight, noise):
        # one = 1.cuda()
        new_w = weight + weight * noise * torch.randn_like(weight)
        # new_w = nn.
        # return nn.Parameter(new_w, requires_grad=False).to(weight.device)
        return new_w.to(weight.device)

    def gen_noise(self, weight, noise):
        new_w = weight * noise * torch.randn_like(weight)
        return new_w.to(weight.device)


class NoiseLinear(NoiseModule):
    def __init__(self, in_features, out_features, sample_noise=False, noise=0, is_train=True):
        super(NoiseLinear, self).__init__()
        self.noise = noise
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.sample_noise = sample_noise
        self.is_train = is_train

    def forward(self, x):
        if not self.noise:
            return self.linear(x)
        # elif self.is_train:
        #     return self.linear(x) + self.noised_foward(x)
        # else:
        #     return self.linear(x) + self.noised_inference(x)
        else:
            return self.linear(x) + self.noised_forward(x)

    def noised_inference(self, x):
        origin_weight = self.linear.weight
        batch, n_points = x.size() 

        x_new = torch.zeros(batch, self.out_features).to(x.device)
        for i in range(batch):
            noise_weight = nn.Parameter(self.gen_noise(origin_weight, self.noise), requires_grad=False)
            x_i = torch.matmul(noise_weight, x[i])
            x_new[i] = x_i
            del noise_weight, x_i

        x_new = x_new.reshape(batch, self.out_features)
        return x_new

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        # x shape: (batch_size, n_points)
        batch_size, n_points = x.size()
        # x = x.reshape(1, -1)

        origin_weight = self.linear.weight
        # x_new = torch.zeros(self.out_features, batch_size*n_points).to(x.device)
        x_new = torch.zeros(batch_size, self.out_features).to(x.device)

        for i in range(x.shape[0]):
            noise_weight = self.gen_noise(origin_weight, self.noise).detach()
            x_i = torch.matmul(noise_weight, x[i, :].unsqueeze(1))
            x_new[i, :] = x_i.squeeze()    # (batch_size, out_features)
            del noise_weight, x_i

        return x_new


class NoiseConv(NoiseModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, sample_noise=False, noise=0):
        super(NoiseConv, self).__init__()
        self.noise = noise
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise

    def forward(self, x):
        if not self.noise:
            return self.conv(x)
        else:
            return self.conv(x) + self.noised_forward(x)

    def noised_inference(self, x):
        origin_weight = self.conv.weight.squeeze()
        batch, nsamples, npoints = x.shape[0], x.shape[2], x.shape[3]

        x = x.reshape(batch, self.in_channels, -1, 1).squeeze() # [channel, number of points]
        x_new = torch.zeros(batch, self.out_channels, x.shape[2], 1).to(x.device)
        for i in range(batch):
            noised_weight = self.gen_noise(origin_weight, self.noise).detach()

            x_i = torch.matmul(noised_weight, x[i])
            x_new[i] = x_i.unsqueeze(-1)
        del noised_weight, x_i
        x_new = x_new.reshape(batch, self.out_channels, nsamples, npoints)
        return x_new.detach()

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        x = x.detach()

        # x shape: (batch_size, in_features, nsamples, npoints)
        # n_points: number of centroids.
        # nsamples: number of points in the neigbor of each centroid.
        batch_size, in_features, nsamples, npoints = x.size()
        # x = x.reshape(1, in_features, 1, -1)
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = self.conv.weight
        x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

        for i in range(x.shape[0]):
            noise_weight= self.gen_noise(origin_weight, self.noise).detach()#.suqeeze()# .detach()
            noise_weight = noise_weight.squeeze()
            # noise_conv = noise_weight
            # del noise_weight
            x_i = x[i, :, :, :].squeeze(-1)#.unsqueeze(-1)
            x_i = torch.matmul(noise_weight, x_i)
            x_new[i, :, :, :] = x_i.unsqueeze(-1)    # (batch_size, out_features)
            del noise_weight, x_i

        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()

class NoiseConv1(NoiseModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, sample_noise=False, noise=0):
        super(NoiseConv1, self).__init__()
        self.noise = noise
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise

    def forward(self, x):
        if not self.noise:
            return self.conv(x)
        else:
            return self.conv(x) + self.noised_forward(x)

    def noised_inference(self, x):
        origin_weight = self.conv.weight.squeeze()
        batch, nsamples, npoints = x.shape[0], x.shape[2], x.shape[3]

        x = x.reshape(batch, self.in_channels, -1, 1).squeeze() # [channel, number of points]
        x_new = torch.zeros(batch, self.out_channels, x.shape[2], 1).to(x.device)
        for i in range(batch):
            noised_weight = self.gen_noise(origin_weight, self.noise).detach()

            x_i = torch.matmul(noised_weight, x[i])
            x_new[i] = x_i.unsqueeze(-1)
        del noised_weight, x_i
        x_new = x_new.reshape(batch, self.out_channels, nsamples, npoints)
        return x_new.detach()

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        x = x.detach()

        # x shape: (batch_size, in_features, nsamples, npoints)
        # n_points: number of centroids.
        # nsamples: number of points in the neigbor of each centroid.
        batch_size, in_features, nsamples, npoints = x.size()
        # x = x.reshape(1, in_features, 1, -1)
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = self.conv.weight
        x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

        for i in range(x.shape[0]):
            noise_weight= self.gen_noise(origin_weight, self.noise).detach()#.suqeeze()# .detach()
            noise_weight = noise_weight.squeeze()
            # noise_conv = noise_weight
            # del noise_weight
            x_i = x[i, :, :, :].squeeze(-1)#.unsqueeze(-1)
            x_i = torch.matmul(noise_weight, x_i)
            x_new[i, :, :, :] = x_i.unsqueeze(-1)    # (batch_size, out_features)
            del noise_weight, x_i

        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()


class OldNoiseLinear(NoiseModule):
    def __init__(self, in_features, out_features, bias=True, origin_weight=False, noise=0):
        super(OldNoiseLinear, self).__init__()
        self.origin_weight = origin_weight
        self.noise = noise
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        if not self.noise:
            return self.linear(x)
        else:
            return self.noised_foward(x)

    def noised_foward(self, x):
        '''
        forward propagation with noise
        '''
        # x shape: (batch_size, n_points)
        batch_size, n_points = x.size()
        # x = x.reshape(1, -1)

        origin_weight = self.linear.weight
        # x_new = torch.zeros(self.out_features, batch_size*n_points).to(x.device)
        x_new = torch.zeros(batch_size, self.out_features).to(x.device)

        for i in range(x.shape[0]):
            self.linear.weight = self.add_noise(origin_weight, self.noise)
            x_i = self.linear(x[i, :].unsqueeze(0))
            x_new[i, :] = x_i.squeeze()    # (batch_size, out_features)

        # x_new = x_new.reshape(batch_size, self.out_features)
        self.linear.weight = origin_weight
        return x_new


class OldNoiseConv(NoiseModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, origin_weight=False, noise=0):
        super(OldNoiseConv, self).__init__()
        self.origin_weight = origin_weight
        self.noise = noise
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.out_channels = out_channels

    def forward(self, x):
        if not self.noise:
            return self.conv(x)
        else:
            return self.noised_foward(x)

    def noised_foward(self, x):
        '''
        forward propagation with noise
        '''
        # x shape: (batch_size, in_features, nsamples, npoints)
        batch_size, in_features, nsamples, npoints = x.size()
        x = x.reshape(1, in_features, 1, -1)

        origin_weight = self.conv.weight
        x_new = torch.zeros(1, self.out_channels, 1, x.shape[-1])

        for i in range(x.shape[-1]):
            noise_weight= self.add_noise(origin_weight, self.noise)# .detach()
            self.conv.weight = noise_weight
            del noise_weight
            x_i = x[:, :, :, i].unsqueeze(-1)
            x_i = self.conv(x_i)
            x_new[:, :, :, i] = x_i.squeeze(-1)    # (batch_size, out_features)

        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        self.conv.weight = origin_weight
        del origin_weight
        return x_new.to(x.device)

