import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--eval-batch-size', default=32, type=int, help='eval batch size')

    ## optimization
    parser.add_argument('--warm-up-iter', default=100, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-5, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for blip2. if None, get path from args.config_path yaml file finetuned_checkpoint value')
    
    ## other
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--nodebug', action='store_true', help='set --nodebug while training')
    parser.add_argument('--config-path', default='blip2/config/blip2_inference.yaml', type=str, help='model options')
    
    parser.add_argument('--frame-path', default='dataset/HumanML3D/render_frames', type=str, help='frame path')
    
    args = parser.parse_args()
    return args