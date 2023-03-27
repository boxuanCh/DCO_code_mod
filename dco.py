import numpy as np
import torch
import torch.nn as nn
from utils import RootRaisedCos_Filter
import sys


class DCO(nn.Module):
    def __init__(self, args):
        super(DCO, self).__init__()
        self.args = args
        self.sample_per_bit = int(self.args.f_sample / self.args.f_bit)
        self.generate_phase_noise_spec()
        spb = int(args.f_sample / args.f_bit)
        self.filter = RootRaisedCos_Filter(args.beta, args.filter_span, spb * 2).to(device=args.device).unsqueeze(0).unsqueeze(0)
    
    def generate_phase_noise_spec(self):
        # max_pack_len = np.max([self.args.pack_len_train, self.args.max_pack_len_train, self.args.pack_len_test, self.args.max_pack_len_test])
        max_pack_len = self.args.block_len
        N_sample_max = self.sample_per_bit * max_pack_len * self.args.code_rate_n / self.args.code_rate_k
        # initialize phase noise spectrum
        self.M = int(N_sample_max / 2 + 1)
        f = torch.arange(self.M) * self.args.f_sample / 2 / (self.M - 1)
        self.df = torch.ones(self.M) * (f[1] - f[0])
        logP = torch.zeros(self.M)
        intptnum = len(self.args.phase_noise_freq)
        realmin = sys.float_info.min
        for idx in range(intptnum):
            f1 = self.args.phase_noise_freq[idx]
            p1 = self.args.phase_noise_pow[idx]
            if idx == intptnum - 1:
                f2 = self.args.f_sample / 2
                p2 = self.args.phase_noise_pow[-1]
            else:
                f2 = self.args.phase_noise_freq[idx + 1]
                p2 = self.args.phase_noise_pow[idx + 1]
            inside = torch.nonzero(torch.bitwise_and(f >= f1, f <= f2))
            logP[inside] = p1 + (np.log10(f[inside] + realmin) - np.log10(f1 + realmin)) / (np.log10(f2 + realmin) - np.log10(f1 + realmin)) * (p2 - p1)
        self.phase_noise_spec = 10 ** (0.1 * logP)

    def generate_phase_noise(self, N_sample, batch_size):
        awgn = np.sqrt(0.5) * (torch.randn(batch_size, self.M) + 1j * torch.randn(batch_size, self.M))
        X = (2 * self.M - 2) * torch.sqrt(self.df * self.phase_noise_spec) * awgn
        X = torch.cat([X, torch.fliplr(X[:, 1:-1])], dim=1)
        X[:, 0] = 0
        x = torch.real(torch.fft.ifft(X, dim=1)[:, :N_sample])
        return x.to(device=self.args.device)

    def forward(self, x, add_delay=False, fix_delay=None):
        batch_size = x.shape[0]
        num_pack = x.shape[1]
        x = x.reshape(batch_size, -1)
        
        # random initial phase for each packet
        rand_phase = torch.rand(x.shape[0], 1) * 2 * np.pi
        
        # scale intermediate frequency (1 -> max_freq, -1 -> min_freq)
        coeff = 2 * np.pi * self.args.mod_idx * self.args.f_bit / self.args.f_sample
        x = x * coeff
        
        # LPF the frequency to restrain spurious emission
        if self.args.if_filter:
            x = x.unsqueeze(1)
            k = self.filter.shape[2]
            x = nn.functional.conv1d(nn.functional.pad(x, [(k-1)//2, k-1-(k-1)//2], mode='circular'), self.filter).squeeze()
            
        # zero-mean the instantaneous frequency
        x = x - x.mean(dim=1, keepdim=True)
        
        ## upsample for finer time resolution
        upsample_rate = self.args.upsample_rate
        if self.args.upsample:
            upsample_kernel = torch.ones(upsample_rate, device=x.device) / upsample_rate
            x = torch.kron(x, upsample_kernel)
            
        # add random carrier frequency offset    
        if self.args.add_freq_offset:    
            rand_cfo = (torch.rand(x.shape[0], 1, device=self.args.device) * 2 - 1) * 2 * np.pi * self.args.max_cfo / self.args.f_sample
            x = x + rand_cfo
            
        # turn instantaneous frequency into phase
        x = torch.cumsum(x, dim=1) + rand_phase.to(device=self.args.device)
        
        # add phase noise
        if self.args.add_phase_noise:
            x = x + self.generate_phase_noise(x.shape[1], x.shape[0])
        sin = torch.sin(x)
        cos = torch.cos(x)
        rand_tio = torch.randint(low=-self.args.max_time_shift, high=self.args.max_time_shift + 1, size=(x.shape[0],), device=self.args.device)
        
        # add time offset to train packet detector
        if add_delay:
            zero_delay = (torch.rand(size=(x.shape[0],), device=self.args.device) < self.args.zero_delay_prob).to(dtype=torch.long)
            rand_tio = rand_tio * (1 - zero_delay)
            sin = self.add_delay(sin, rand_tio, fix_delay)
            cos = self.add_delay(cos, rand_tio, fix_delay)
            
        ## downsample back to original sample rate
        if self.args.upsample:
            sin = sin.reshape(x.shape[0], x.shape[1] // upsample_rate, upsample_rate)
            sin = sin[:, :, -1]
            cos = cos.reshape(x.shape[0], x.shape[1] // upsample_rate, upsample_rate)
            cos = cos[:, :, -1]
        sin = sin.reshape(batch_size, num_pack, -1)
        cos = cos.reshape(batch_size, num_pack, -1)
        y = torch.cat([sin, cos], dim=2)
        return y, rand_cfo, rand_tio
    
    def add_delay(self, x, rand_tio, fix_delay):
        
        if fix_delay is None:
            
            for i in range(x.shape[0]):
                if rand_tio[i] > 0:
                    x[i, rand_tio[i]:] = x[i, :-rand_tio[i]].clone()
                    x[i, :rand_tio[i]] = 0
                elif rand_tio[i] < 0:
                    x[i, :rand_tio[i]] = x[i, -rand_tio[i]:].clone()
                    x[i, rand_tio[i]:] = 0

        else:
            if fix_delay > 0:
                x[:, fix_delay:] = x[:, :-fix_delay].clone()
                x[:, :fix_delay] = 0
            elif fix_delay < 0:
                x[:, :fix_delay] = x[:, -fix_delay:].clone()
                x[:, fix_delay:] = 0
        return x
    
    def generate_buffer(self, x, buffer_size):
        num_instance = 2 * buffer_size + 1
        x = x.reshape(x.shape[0], -1)
        batch_size, packet_len = x.shape
        output_sin = torch.zeros(batch_size, num_instance, buffer_size, device=x.device)
        output_cos = torch.zeros(batch_size, num_instance, buffer_size, device=x.device)
        rand_cfo = (torch.rand(x.shape[0], 1, device=self.args.device) * 2 - 1) * 2 * np.pi * self.args.max_cfo / self.args.f_sample
        coeff = 2 * np.pi * self.args.mod_idx * self.args.f_bit / self.args.f_sample
        rand_phase = torch.rand(x.shape[0], 1) * 2 * np.pi
        x = x * coeff
        if self.args.if_filter:
            x = x.unsqueeze(1)
            # x = nn.functional.conv1d(x, self.filter, padding='same').squeeze()
            k = self.filter.shape[2]
            x = nn.functional.conv1d(nn.functional.pad(x, [(k-1)//2, k-1-(k-1)//2], mode='circular'), self.filter).squeeze()
        x = x - x.mean(dim=1, keepdim=True)
        if self.args.add_freq_offset:
            x = x + rand_cfo
        x = torch.cumsum(x, dim=1) + rand_phase.to(device=self.args.device)
        if self.args.add_phase_noise:
            x = x + self.generate_phase_noise(x.shape[1], x.shape[0])
        sin = torch.sin(x)
        cos = torch.cos(x)
        for i in range(num_instance):
            if i > 0 and i <= buffer_size:
                output_sin[:, i, (buffer_size - i):] = sin[:, :i]
                output_cos[:, i, (buffer_size - i):] = cos[:, :i]
            elif i > buffer_size:
                output_sin[:, i, :] = sin[:, (i - buffer_size):i]
                output_cos[:, i, :] = cos[:, (i - buffer_size):i]
        output = torch.cat((output_sin, output_cos), dim=2)
        return output