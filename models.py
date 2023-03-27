import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        sample_per_bit = int(1 / args.code_rate_k * args.code_rate_n * args.f_sample / args.f_bit)

        if args.enc_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.enc_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.enc_rnn = RNN_MODEL(1, args.enc_hidden_size, num_layers=args.enc_rnn_layer, bias=True, batch_first=True,
                                 bidirectional=args.enc_bidirectional)

        self.enc_linear = torch.nn.Linear(2 * args.enc_hidden_size, int(sample_per_bit))


    def forward(self, inputs):
        output, _ = self.enc_rnn(inputs)  
        codes = torch.sigmoid(self.enc_linear(output))
        codes = codes * 2 - 1 # (-1,1)
        return codes, output
    
    
class Decoder(nn.Module):
    
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        sample_per_bit = int(1 / args.code_rate_k * args.code_rate_n * args.f_sample / args.f_bit)
        
        self.dec_rnn = RNN_MODEL(sample_per_bit * 2,  args.dec_hidden_size,
                                num_layers=args.dec_rnn_layer, bias=True, batch_first=True,
                                bidirectional=args.dec_bidirectional)
        self.dec_outputs = torch.nn.Linear(2 * args.dec_hidden_size, 1)

    def forward(self, received):
        out, _ = self.dec_rnn(received)
        fc = self.dec_outputs(out)
        final = torch.sigmoid(fc)
        return final
    
    
class Channel_AutoEncoder(torch.nn.Module):
    
    def __init__(self, args, enc, dec, dco):
        super(Channel_AutoEncoder, self).__init__()

        self.args = args
        self.enc = enc
        self.dec = dec
        self.dco = dco

    def forward(self, input, noise, add_delay=False, fix_delay=None):
        
        codes, _ = self.enc(input)

        dco_out, cfo, delay = self.dco(codes, add_delay, fix_delay)
        
        if self.args.channel in ['awgn', 't-dist', 'radar']:
            received_codes = dco_out + noise
        else:
            print('default AWGN channel')
            received_codes = dco_out + noise
        
        x_dec = self.dec(received_codes)

        return x_dec, codes, dco_out, received_codes, delay