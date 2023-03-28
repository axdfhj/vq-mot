import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--eval-batch-size', default=8, type=int, help='eval batch size')

    ## optimization
    parser.add_argument('--warm-up-iter', default=100, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-5, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for vis encoder')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--nodebug', action='store_true', help='set --nodebug while training')
    parser.add_argument('--val-every-epoch', default=1, type=int, help='validation every n train epoch')
    parser.add_argument('--epoch', default=30, type=int, help='training epochs')
    parser.add_argument('--config-path', default='blip2/config/blip2_filetune_proj.yaml', type=str, help='model options')
    
    parser.add_argument('--frame-path', default='dataset/HumanML3D/render_frames', type=str, help='frame path')
    parser.add_argument('--save-path', default='tmp/check_text.txt', type=str, help='frame path')
    
    args = parser.parse_args()
    return args