import argparse
from distutils.util import strtobool

def strtolist(val):
    val = val[1:-1]
    val = val.split(',')
    res = [float(i) for i in val]
    return res

def get_args():

    parser = argparse.ArgumentParser()

    # Channel Parameters
    parser.add_argument('-channel', choices=['awgn',             # AWGN
                                             't-dist',           # Non-AWGN, ATN, with -vv associated
                                             'radar',            # Non-AWGN, Radar, with -radar_prob, radar_power, associated
                                             ],
                        default='awgn')
    parser.add_argument('-vv', type=float, default=5,
                        help='only for t distribution channel')

    parser.add_argument('-radar_prob', type=float, default=0.05, help='only for radar distribution channel')
    parser.add_argument('-radar_power', type=float, default=5.0, help='only for radar distribution channel')
    
    parser.add_argument('-train_enc_ebno_low', type=float, default=1.0)
    parser.add_argument('-train_enc_ebno_high', type=float, default=1.0)
    parser.add_argument('-train_dec_ebno_low', type=float, default=-1.5)
    parser.add_argument('-train_dec_ebno_high', type=float, default=2.0)
    parser.add_argument('-train_pd_ebno_low', type=float, default=-1.5)
    parser.add_argument('-train_pd_ebno_high', type=float, default=2.0)
    
    parser.add_argument('-test_pd_ebno', type=float, default=2.0)
    parser.add_argument('-ebno_test_start', type=float, default=-1.5)
    parser.add_argument('-ebno_test_end', type=float, default=4.0)
    parser.add_argument('-ebno_points', type=int, default=12)
    
    # Coding Rate k/n
    parser.add_argument('-code_rate_k', type=int, default=1)
    parser.add_argument('-code_rate_n', type=int, default=3)
    
    # Model Structure
    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-enc_rnn_layer', type=int, default=2)
    parser.add_argument('-enc_hidden_size', type=int, default=100)
    parser.add_argument('-enc_bidirectional', type=strtobool, default=True)
    
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn_layer', type=int, default=2)
    parser.add_argument('-dec_hidden_size', type=int, default=25)
    parser.add_argument('-dec_bidirectional', type=strtobool, default=True)
    
    parser.add_argument('-pd_rnn_layer', type=int, default=2)
    parser.add_argument('-pd_hidden_size', type=int, default=100)

    parser.add_argument('-load_path', type=str, default='default')
    
    # DCO Parameters
    parser.add_argument('-f_sample', type=int, default=8e6, help='sampling rate of DCO') 
    parser.add_argument('-f_bit', type=int, default=1e6, help='bit rate')
    parser.add_argument('-mod_idx', type=float, default=0.5, help='modulation index')
    parser.add_argument('-modulation', choices=['dco', 'psk'], default='dco')
    parser.add_argument('-if_filter', type=strtobool, default=True, help='if filtering is applied in DCO')
    parser.add_argument('--beta', type=float, default=0.00001, help='beta of raised or root raised cosine') 
    parser.add_argument('-filter_span', type=int, default=100)
    parser.add_argument('-signal_pow', type=float, default='20', help='transmitting power (dBm)')
    parser.add_argument('-add_phase_noise', type=strtobool, default=False)
    parser.add_argument('-phase_noise_freq', type=strtolist, default=[1e3, 1e4, 1e5, 1e6, 1e7])
    parser.add_argument('-phase_noise_pow', type=strtolist, default=[-84, -100, -96, -109, -122])
    parser.add_argument('-max_cfo', type=int, default=150000)
    parser.add_argument('-add_freq_offset', type=strtobool, default=False)
    parser.add_argument('-add_time_offset', type=strtobool, default=False)
    parser.add_argument('-max_time_shift', type=int, default=5)
    parser.add_argument('-upsample', type=strtobool, default=False)
    parser.add_argument('-upsample_rate', type=int, default=10)
    parser.add_argument('-zero_delay_prob', type=float, default=0.5)
    
    # Training Parameters
    parser.add_argument('-batch_size', type=int, default=500)
    parser.add_argument('-num_epoch', type=int, default=500)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_block', type=int, default=5000)
    parser.add_argument('-device', choices=['cpu', 'cuda'], default='cuda')
    
    parser.add_argument('-num_train_dec', type=int, default=5)
    parser.add_argument('-num_train_enc', type=int, default=1)
    parser.add_argument('-num_train_pd', type=int, default=0)
    
    parser.add_argument('-optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('-dec_lr', type=float, default=0.001)
    parser.add_argument('-enc_lr', type=float, default=0.001)
    parser.add_argument('-pd_lr', type=float, default=0.001)
    
    # MISC
    parser.add_argument('-tensorboard', type=strtobool, default=True, help='if use tensorboard to monitor curve')
    parser.add_argument('-verbose', type=strtobool, default=True)
    parser.add_argument('-min_block_err', type=int, default=1000)

    args = parser.parse_args()

    return args