import numpy as np
import torch
import math

def RootRaisedCos_Filter(beta, span, sps):
    h = np.zeros(span * sps + 1)
    delay = span * sps / 2
    t = np.linspace(-delay, delay, span * sps + 1) / sps
    idx1 = np.argwhere(t == 0)
    h[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
    idx2 = np.argwhere(np.abs(np.abs(4 * beta * t) - 1) < np.spacing(1))
    h[idx2] = -1 / (2 * np.pi * sps) * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta)) - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta)) + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))
    idx3 = np.argwhere((t != 0) & (np.abs(np.abs(4 * beta * t) - 1) >= np.spacing(1)))
    n = t[idx3]
    h[idx3] = -4 * beta / sps * (np.cos((1 + beta) * np.pi * n) + np.sin((1 - beta) * np.pi * n) / (4 * beta * n)) / (np.pi * ((4 * beta * n) ** 2 - 1))
    h = h / np.sum(h)
    h = torch.Tensor(h)
    return h

def db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)


def sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)

def generate_noise(noise_shape, args, ebno_low, ebno_high = None):
    
    if ebno_high is not None:
        sigma_low = db2sigma(ebno_low)
        sigma_high = db2sigma(ebno_high)
        sigma = (sigma_low - sigma_high) * torch.rand(noise_shape) + sigma_high
    else:
        sigma = db2sigma(ebno_low)
    
    if args.modulation == 'dco':
        sigma *= np.sqrt(int(args.code_rate_n / args.code_rate_k * args.f_sample / args.f_bit / 2))
    
    # generate noise
    if args.channel == 'awgn':
        fwd_noise = sigma * torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 't-dist':
        fwd_noise = sigma * torch.from_numpy(np.sqrt((args.vv-2)/args.vv) * np.random.standard_t(args.vv, size=noise_shape)).type(torch.FloatTensor)
    elif args.channel == 'radar':
        add_pos = np.random.choice([0.0, 1.0], noise_shape, p=[1 - args.radar_prob, args.radar_prob])

        corrupted_signal = args.radar_power * np.random.standard_normal(size=noise_shape) * add_pos
        fwd_noise = sigma * torch.randn(noise_shape, dtype=torch.float) + torch.from_numpy(corrupted_signal).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise = sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise

def model_name(timestamp, args):
    
    if args.enc_bidirectional:
        enc_name = '{}_b_{}_{}'.format(args.enc_rnn, args.enc_rnn_layer, args.enc_hidden_size)
    else:
        enc_name = '{}_s_{}_{}'.format(args.enc_rnn, args.enc_rnn_layer, args.enc_hidden_size)
    if args.dec_bidirectional:
        dec_name = '{}_b_{}_{}'.format(args.dec_rnn, args.dec_rnn_layer, args.dec_hidden_size)
    else:
        dec_name = '{}_s_{}_{}'.format(args.dec_rnn, args.dec_rnn_layer, args.dec_hidden_size)
    sps = args.f_sample / args.f_bit
    name = '{}_{}_{}_{}_{}_{}'.format(timestamp, args.code_rate_k, args.code_rate_n, sps, enc_name, dec_name)
    return name