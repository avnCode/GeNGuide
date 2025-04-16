import json
import argparse
from trainer import train
from evaluator import test

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  
    args.update(param)  
    args['corruption_percent'] = args['noise']
    args['model_name'] = args['model_name']+"_"+args['dataset']

    if 'cifar100' in args['dataset']:
        args['dataset'] = args['dataset']+"_224"
        if args['noise_type'] == 'superclass':
            args["superclass_noise"] = True
        else:
            args["superclass_noise"] = False
        print(args["superclass_noise"])

    elif 'cifar10' in args['dataset']:
        args['dataset'] = args['dataset']+"_224"
        if args['noise_type'] == 'symmetric':
            args["asymmetric_noise"] = False
        else:
            args["asymmetric_noise"] = True

    if args['pretrained']=='moco':
        args['convnet_type'] = args['convnet_type']+"-mocov3"
        args['model_name'] = args['model_name']+"_mocov3"
        
    if args['test_only']:
        test(args)
    else:
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--noise_type', type=str, default='random')
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--pretrained', type=str, default='imagenet')
    return parser


if __name__ == '__main__':
    main()
