import torch
import argparse
import sys
sys.path.append('./options')
from trainer_baseline import Trainer
from Conv_TasNet_wavlm_dwconvFuse import ConvTasNet as Net_dwAtt1
from Conv_TasNet_wavlm_dwconvFuse_att2 import ConvTasNet as Net_dwAtt2
from Conv_TasNet_wavlm_dwconvFuse_film import ConvTasNet as Net_dwAtt_film
from Conv_TasNet_wavlm_film import ConvTasNet as Net_film
from Conv_TasNet_wavlm_repeat_gate import ConvTasNet as Net_gate
from Conv_TasNet_wavlm_up import ConvTasNet as Net_up
from DataLoaders import make_dataloader
from option import parse
from utils import get_logger

def main():
    # Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--model_net', type=str, help='Model Net.')
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)
    
    logger.info('Building the model of Conv-TasNet')
    if(args.model_net == "Net_dwAtt1"):
        net = Net_dwAtt1(**opt['net_conf'])
    elif(args.model_net == "Net_dwAtt2"):
        net = Net_dwAtt2(**opt['net_conf'])
    elif(args.model_net == "Net_dwAtt_film"):
        net = Net_dwAtt_film(**opt['net_conf'])
    elif(args.model_net == "Net_film"):
        net = Net_film(**opt['net_conf'])
    elif(args.model_net == "Net_gate"):
        net = Net_gate(**opt['net_conf'])
    elif(args.model_net == "Net_up"):
        net = Net_up(**opt['net_conf'])
    
    logger.info('Building the trainer of Conv-TasNet')
    gpuid = tuple(opt['gpu_ids'])
    trainer = Trainer(net, **opt['train'], resume=opt['resume'],
                      gpuid=gpuid, optimizer_kwargs=opt['optimizer_kwargs'])

    logger.info('Making the train and test data loader')
    train_loader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])
    val_loader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
                                   ['num_workers'],  batch_size=opt['datasets']['batch_size'])
    logger.info('Train data loader: {}, Test data loader: {}'.format(len(train_loader), len(val_loader)))
    trainer.run(train_loader,val_loader)


if __name__ == "__main__":
    main()
