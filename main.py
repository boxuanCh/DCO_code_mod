import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy import arange
from numpy.random import mtrand
import sys
import os
import time

from get_args import get_args
from loops import train, validate, test
from dco import DCO
from models import Encoder, Decoder, Channel_AutoEncoder
from utils import model_name


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    

if __name__ == '__main__':

    ## Pre-Training Preparations ##
    start_time = time.time()
    time.ctime()
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    args = get_args()

    if args.tensorboard:
        writer = SummaryWriter(log_dir='tensorboard')
        
    logfile_name = './logs/' + timestamp + '.txt'
    logfile = open(logfile_name, 'a')
    sys.stdout = Logger(logfile_name, sys.stdout)
    
    print(args)
    
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    print("use_cuda: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(0)
    
    encoder = Encoder(args)
    decoder = Decoder(args)
    dco = DCO(args)
    
    model = Channel_AutoEncoder(args, encoder, decoder, dco).to(device=device)
    
    if args.load_path == 'default':
        pass
    else:
        pretrained_model = torch.load(args.load_path)
        model.load_state_dict(pretrained_model.state_dict())
        
    print(model)
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam
    else:
        optimizer = optim.SGD
    enc_optimizer = optimizer(model.enc.parameters(), lr=args.enc_lr)
    dec_optimizer = optimizer(model.dec.parameters(), lr=args.dec_lr)
    general_optimizer = optimizer(model.parameters(), lr=args.dec_lr)
    
    minber = 1.0
    minloss = np.inf
    record_loss, record_ber, record_bler = [], [], []
    
    ## Training and validation ##
    
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        if args.num_train_enc > 0:
            for idx in range(args.num_train_enc):
                train(epoch, model, enc_optimizer, args, use_cuda=use_cuda, verbose=args.verbose, mode='encoder')

        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                train(epoch, model, dec_optimizer, args, use_cuda=use_cuda, verbose=args.verbose, mode='decoder')

        # if args.num_train_pd > 0:
        #     for idx in range(args.num_train_pd):
        #         train(epoch, model, pd_optimizer, args,
        #               use_cuda=use_cuda, verbose=args.verbose, mode='pd', use_conf=0)

        loss, ber, bler = validate(model, args, verbose=args.verbose, use_cuda=use_cuda)
        if args.tensorboard:
            writer.add_scalar('{}/Loss/Train'.format(timestamp), loss, epoch)
            writer.add_scalar('{}/BER/Train'.format(timestamp), ber, epoch)
            writer.add_scalar('{}/BLER/Train'.format(timestamp), bler, epoch)
        record_loss.append(loss)
        record_ber.append(ber)
        record_bler.append(bler)

        # save the best model
        if (ber < minber) or ((ber == minber) and (loss < minloss)):
            minber, minloss = ber, loss
            modelpath = './saved_models/' + model_name(timestamp, args) + '_best'+'.pt'
            torch.save(model.state_dict(), modelpath)
            print('Model saved, current minimal ber is: ', minber)
    
    modelpath = './saved_models/' + model_name(timestamp, args) + '_last'+'.pt'
    torch.save(model.state_dict(), modelpath)
    
    test(model, args, use_cuda=use_cuda)
    
    print('Total Running Time: {}s'.format(time.time() - start_time))

        
    