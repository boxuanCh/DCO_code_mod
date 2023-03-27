import torch 
import time
import numpy as np
import torch.nn.functional as F
from utils import generate_noise

def train(epoch, model, optimizer, args, use_cuda = False, verbose = False, mode = 'encoder'):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    start_time = time.time()
    train_loss = 0.0

    for _ in range(int(args.num_block / args.batch_size)):

        block_len = args.block_len

        optimizer.zero_grad()
        
        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        
        # generate noise
        if args.modulation == 'dco':
            noise_shape =  (args.batch_size, args.block_len, int(args.code_rate_n * args.f_sample / args.f_bit*2))
        else:
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':           
            noise  = generate_noise(noise_shape, args, ebno_low=args.train_enc_ebno_low, ebno_high=args.train_enc_ebno_high)
        elif mode == 'decoder':
            noise  = generate_noise(noise_shape, args, ebno_low=args.train_dec_ebno_low, ebno_high=args.train_dec_ebno_high)
        else:
            noise  = generate_noise(noise_shape, args, ebno_low=args.train_pd_ebno_low, ebno_high=args.train_pd_ebno_high)
        X_train, noise = X_train.to(device), noise.to(device) 
        if mode in ['encoder', 'decoder']:
            tuple_ = model(X_train, noise, add_delay=False)
        else:
            tuple_ = model(X_train, noise, add_delay=True)
        output, code = tuple_[0],tuple_[1]
        output = torch.clamp(output, 0.0, 1.0)
        delay = tuple_[4]
        packet_det = tuple_[3]
        if mode in ['encoder', 'decoder']:
            loss = F.binary_cross_entropy(output, X_train, reduction='mean')

        else:
            loss = F.binary_cross_entropy(packet_det, (delay==0).to(dtype=torch.float))

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    end_time = time.time()
    train_loss = train_loss / (args.num_block / args.batch_size)
    if verbose:
        print('Training: Epoch: {} Mode: {} Average loss: {:.8f} Running Time: {:.3f}'.format(epoch + 1, mode, train_loss, end_time - start_time))

    return train_loss


def validate(model, args, use_cuda = False ,verbose = False):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, bit_err, block_err = 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block / args.batch_size)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            if args.modulation == 'dco':  ######
                noise_shape =  (args.batch_size, args.block_len, int(args.code_rate_n * args.f_sample / args.f_bit*2))
            else:
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
            noise  = generate_noise(noise_shape, args, ebno_low=args.train_enc_ebno_low, ebno_high=args.train_enc_ebno_low)

            X_test, noise= X_test.to(device), noise.to(device)

            tuple_ = model(X_test, noise, add_delay=False)  ######
            output = tuple_[0]
            output = torch.clamp(output, 0.0, 1.0)

            output = output.detach()
            X_test = X_test.detach()

            test_bce_loss += F.binary_cross_entropy(output, X_test)
            bit_err += torch.sum(torch.ne(torch.round(output), torch.round(X_test)))
            block_err += torch.sum(torch.sum(torch.ne(torch.round(output), torch.round(X_test)), dim=1) > 0)


    test_bce_loss /= num_test_batch
    test_ber  = bit_err / (num_test_batch * args.batch_size * args.block_len)
    test_bler = block_err / (num_test_batch * args.batch_size)

    if verbose:
        print('Validation: Loss: ', float(test_bce_loss), ' BER: ', float(test_ber), ' BLER: ', float(test_bler))

    report_loss = float(test_bce_loss)
    report_ber  = float(test_ber)
    report_bler  = float(test_bler)

    return report_loss, report_ber, report_bler


def test(model, args, block_len = None, use_cuda = False):
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    # signal_pow = 10 ** (args.signal_pow / 10)

    if block_len is None:
        block_len = args.block_len
    else:
        pass

    ber_res, bler_res, erpb_res = [], [], []

    ebnos = np.linspace(args.ebno_test_start, args.ebno_test_end, args.ebno_points, endpoint = True)

    for ebno in ebnos:
        test_ber, test_bler = .0, .0 
        with torch.no_grad():
            num_test_batch = 0
            bit_err = 0
            block_err = 0
            bit_err_per_position = torch.zeros(block_len, device=device)
            while block_err < args.min_block_err:
                X_test     = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
                if args.modulation == 'dco':
                    noise_shape =  (args.batch_size, args.block_len, int(args.code_rate_n * args.f_sample / args.f_bit * 2))
                else:
                    noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                noise  = generate_noise(noise_shape, args, ebno)

                X_test, noise= X_test.to(device), noise.to(device)

                tuple_ = model(X_test, noise)
                X_hat_test, the_codes = tuple_[0], tuple_[1]
                
                bit_err += torch.sum(torch.ne(torch.round(X_hat_test), torch.round(X_test)))
                bit_err_per_position += torch.sum(torch.ne(torch.round(X_hat_test), torch.round(X_test)), dim=0).squeeze()
                block_err += torch.sum(torch.sum(torch.ne(torch.round(X_hat_test), torch.round(X_test)), dim=1) > 0)
                num_test_batch += 1
                if num_test_batch == 1e6:
                    break

        test_erpb = bit_err_per_position / num_test_batch / args.batch_size
        test_ber  = bit_err / num_test_batch / args.batch_size / block_len
        test_bler = block_err / num_test_batch / args.batch_size
        print('Ebno: ', ebno ,' BER: ', float(test_ber), ' BLER: ', float(test_bler))
        ber_res.append(float(test_ber))
        bler_res.append( float(test_bler))
        erpb_res.append(test_erpb)
    # print('Test SNR', this_snr, 'with ber distribution',ber_distribution)
    print('final results on Eb/No ', ebnos)
    print('BER', ber_res)
    print('BLER', bler_res)
    return ber_res, bler_res, erpb_res