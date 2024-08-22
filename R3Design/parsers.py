import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4410, type=int)

    # dataset parameters
    parser.add_argument('--data_name', default='', choices=[''])
    parser.add_argument('--data_root', default='./dataset/')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    # method parameters
    parser.add_argument('--method', default='R3Design', choices=['SeqRNN', 'SeqLSTM', 'StructMLP', 'StructGNN', 'GraphTrans', 'HCGNN', 'PiFold', 'RDesignPro'])
    parser.add_argument('--config_file', '-c', default='default.py', type=str)

    # training parameters
    parser.add_argument('--epoch', default=200, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # feature parameters
    parser.add_argument('--node_feat_types', default=['angle', 'distance', 'direction'], type=list)
    parser.add_argument('--edge_feat_types', default=['orientation', 'distance', 'direction'], type=list)
    parser.add_argument('--original', default=0, type=int)

    # model parameters
    parser.add_argument('--nat', default=0, type=int)
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=3, type=int)

    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--k_neighbors', default=30, type=int)
    parser.add_argument('--vocab_size', default=4, type=int)
    parser.add_argument('--shuffle', default=0., type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)

    parser.add_argument('--n_iter', default=3, type=int)

    # TEST
    parser.add_argument('--weigth_clu_con', default=0.5, type=float)
    parser.add_argument('--weigth_sam_con', default=0.5, type=float)
    parser.add_argument('--ss_temp', default=0.5, type=float)
    
    parser.add_argument('--conf_case', default=1, type=int)
    parser.add_argument('--aug_log_case', default=0, type=int)
    parser.add_argument('--ss_case', default=0, type=int)

    parser.add_argument('--wandb', type=int, default=0)
    return parser.parse_args()