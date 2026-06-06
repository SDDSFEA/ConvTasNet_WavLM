import os
import torch
import sys
sys.path.append('./options')
from AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from Conv_TasNet_wavlm_dwconvFuse import ConvTasNet as Net_dwAtt1
from Conv_TasNet_wavlm_dwconvFuse_att2 import ConvTasNet as Net_dwAtt2
from Conv_TasNet_wavlm_dwconvFuse_film import ConvTasNet as Net_dwAtt_film
from Conv_TasNet_wavlm_film import ConvTasNet as Net_film
from Conv_TasNet_wavlm_repeat_gate import ConvTasNet as Net_gate
from Conv_TasNet_wavlm_up import ConvTasNet as Net_up
# from Conv_TasNet_wavlm_dwconvFuse_wogate import ConvTasNet as Net_dwAtt1_wogate
from Conv_TasNet_wavlm_dwconvFuse_nogate import ConvTasNet as Net_dwAtt1_nogate
from Conv_TasNet_wavlm_dwconvFuse_woshare import ConvTasNet as Net_dwAtt1_woshare
from utils import get_logger
from option import parse
import tqdm


class Separation():
    def __init__(self, mix_path, yaml_path, model, gpuid, model_net):
        super(Separation, self).__init__()
        self.mix = AudioReader(mix_path, sample_rate=8000)
        opt = parse(yaml_path, is_tain=False)
        # net = ConvTasNet(**opt['net_conf'])
        if(model_net == "Net_dwAtt1"):
            net = Net_dwAtt1(**opt['net_conf'])
        elif(model_net == "Net_dwAtt2"):
            net = Net_dwAtt2(**opt['net_conf'])
        elif(model_net == "Net_dwAtt_film"):
            net = Net_dwAtt_film(**opt['net_conf'])
        elif(model_net == "Net_film"):
            net = Net_film(**opt['net_conf'])
        elif(model_net == "Net_gate"):
            net = Net_gate(**opt['net_conf'])
        elif(model_net == "Net_up"):
            net = Net_up(**opt['net_conf'])
        # elif(args.model_net == "Net_dwAtt1_wogate"):
        #     net = Net_dwAtt1_wogate(**opt['net_conf'])
        elif(model_net == "Net_dwAtt1_nogate"):
            net = Net_dwAtt1_nogate(**opt['net_conf'])
        elif(model_net == "Net_dwAtt1_woshare"):
            net = Net_dwAtt1_woshare(**opt['net_conf'])
        print(net)
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        self.logger = get_logger(__name__)
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid=tuple(gpuid)

    def inference(self, file_path):
        with torch.no_grad():
            for key, egs in tqdm.tqdm(self.mix):
                #self.logger.info("Compute on utterance {}...".format(key))
                egs=egs.to(self.device)
                norm = torch.norm(egs,float('inf'))
                if len(self.gpuid) != 0:
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                else:
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach()) for s in ests]
                index=0
                for s in spks:
                    s = s[:egs.shape[0]]
                    #norm
                    norm = norm.to('cpu')
                    s = s*norm/torch.max(torch.abs(s))
                    s = s.unsqueeze(0)
                    index += 1
                    os.makedirs(file_path+'/spk'+str(index), exist_ok=True)
                    filename=file_path+'/spk'+str(index)+'/'+key
                    write_wav(filename, s, 8000)
            self.logger.info("Compute over {:d} utterances".format(len(self.mix)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='/home/student/zt/Conv-TasNet/Conv_TasNet_Pytorch/data/audio_scp/dev/clean/cv_mix.scp', help='Path to mix scp file.')
        # '-mix_scp', type=str, default='/home/student/zt/Conv-TasNet/Conv_TasNet_Pytorch/data/audio_scp/test/clean/tt_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-yaml', type=str, default='./options/train/train_clean100_WavLM_dwconvFuse_att2_N128.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='/home/student/zt/ConvTasNet_hao/ConvTasNet_Separation_WavLM_CAtt/Conv-TasNet-clean100-WavLM-dwconvFuse-gate-att2-128-Noise100/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./result_C128_wogate_dev', help='save result path')
    parser.add_argument('-model_net', type=str, help='Model Net.')

    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.mix_scp, args.yaml, args.model, gpuid, args.model_net)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()
