import os
import torch
import sys
sys.path.append('./options')
from AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from Conv_TasNet_TCN_concat import ConvTasNet
from utils import get_logger
from DataLoaders_max_text import TranscriptReader
from option import parse
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Separation():
    def __init__(self, mix_path, text_scp1, text_scp2, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        self.mix = AudioReader(mix_path, sample_rate=8000)
        self.text = TranscriptReader(text_scp1, text_scp2)
        opt = parse(yaml_path, is_tain=False)
        # os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpuid[0]}"
        # print("gpuid:",gpuid)   
        net = ConvTasNet(**opt['net_conf'])
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        self.logger = get_logger(__name__)
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        print("self.device:",self.device)
        self.gpuid=tuple(gpuid)

    def inference(self, file_path):
        with torch.no_grad():
            for key, egs in tqdm.tqdm(self.mix):
                #self.logger.info("Compute on utterance {}...".format(key))
                egs=egs.to(self.device)
                # if key not in self.text:
                #     print("key not in self.text:",key)
                # text = self.text[key]
                text = self.text.get(key, "")
                
                norm = torch.norm(egs,float('inf'))
                if len(self.gpuid) != 0:
                    ests=self.net(egs,text)
                    spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                else:
                    ests=self.net(egs,text)
                    spks=[torch.squeeze(s.detach()) for s in ests]
                index=0
                for s in spks:
                    s = s[:egs.shape[0]]
                    #norm
                    norm = norm.to('cpu')
                    s = s*norm/torch.max(torch.abs(s))
                    # print("s.shape:",s.shape)
                    s = s.unsqueeze(0)
                    index += 1
                    os.makedirs(file_path+'/spk'+str(index), exist_ok=True)
                    filename=file_path+'/spk'+str(index)+'/'+key
                    # print("filename:",filename)
                    # print("index:",index)
                    # print("key:",key)
                    write_wav(filename, s, 8000)
            self.logger.info("Compute over {:d} utterances".format(len(self.mix)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='./data/audio_scp_8k/test/clean/tt_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-text_scp1', type=str, default='./data/text/test/test_spk1.csv', help='Path to text scp file.')
    parser.add_argument(
        '-text_scp2', type=str, default='./data/text/test/test_spk1.csv', help='Path to text scp file.')
    parser.add_argument(
        '-yaml', type=str, default='./options/train_text/train_clean100.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./Conv-TasNet-clean100-text/last.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./test_result', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    print("gpuid:",gpuid)
    separation=Separation(args.mix_scp,args.text_scp1, args.text_scp2, args.yaml, args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()