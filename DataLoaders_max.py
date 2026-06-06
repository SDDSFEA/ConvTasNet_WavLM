import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.dataloader import default_collate
from AudioReader import AudioReader
import torch.nn.functional as F
import random

def pad_collate(batch):
    """
    batch: List[dict]
        dict keys: 'mix', 'ref'
        mix: Tensor [T]
        ref: List[Tensor [T]]
    """
    # ===== mix =====
    mix_list = [b['mix'] for b in batch]
    mix_lens = torch.tensor([m.shape[-1] for m in mix_list], dtype=torch.long)
    max_mix_len = max(mix_lens)

    mix_pad = torch.stack([
        F.pad(m, (0, max_mix_len - m.shape[-1]))
        for m in mix_list
    ])

    # ===== ref =====
    num_spk = len(batch[0]['ref'])
    ref_pad = []

    for spk in range(num_spk):
        spk_list = [b['ref'][spk] for b in batch]
        spk_lens = [r.shape[-1] for r in spk_list]
        max_len = max(spk_lens)

        spk_pad = torch.stack([
            F.pad(r, (0, max_len - r.shape[-1]))
            for r in spk_list
        ])
        ref_pad.append(spk_pad)

    return {
        'mix': mix_pad,              # [B, T]
        'ref': ref_pad,              # List([B, T])
        'mix_len': mix_lens           # [B]
    }


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    # chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoader(dataset,
                      shuffle=is_train,
                    #   chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=pad_collate)


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, mix_scp=None, ref_scp=None, sr=8000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(mix_scp, sample_rate=sr)
        self.ref_audio = [AudioReader(r, sample_rate=sr) for r in ref_scp]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        key = self.mix_audio.keys[index]
        mix = self.mix_audio[key]
        ref = [r[key] for r in self.ref_audio]
        return {
            'mix': mix,
            'ref': ref
        }



if __name__ == "__main__":
    datasets = Datasets('/home/likai/data1/create_scp/cv_mix.scp',
                        ['/home/likai/data1/create_scp/cv_s1.scp', '/home/likai/data1/create_scp/cv_s2.scp'])
    dataloaders = DataLoader(datasets, num_workers=0,
                              batch_size=10, is_train=False)
    for eg in dataloaders:
        print(eg)
        import pdb
        pdb.set_trace()
